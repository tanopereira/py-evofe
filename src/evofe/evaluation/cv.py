import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from ..builtin import evo_transformers
from .models import evo_evaluators
from ..utils import call_with_optional_params


# ---------------------------------------------------------------------------
# Metric helpers  (mirrors R's compute_metric / compute_exp_neg_logloss)
# ---------------------------------------------------------------------------

def compute_metric(y_true, y_pred, task, metric="default", num_class=None):
    """
    Compute a scalar fitness score (higher is always better).

    Classification / multiclass default: exp(-log_loss)  — value in (0, 1].
    Regression default: -RMSE.
    """
    metric = metric.lower()

    if task in ("classification", "multiclass"):
        if task == "classification":
            labels = [0, 1]
        else:
            if num_class is not None:
                labels = list(range(num_class))
            else:
                labels = np.unique(y_true)
        if metric in ("ts_refinement", "ts-refinement", "eval-ts-refinement", "eval_ts_refinement"):
            from .metrics import ts_refinement
            loss = ts_refinement(y_true, y_pred, task=task, num_class=num_class)
            return float(np.exp(-loss))
        if metric == "auc":
            if task == "multiclass":
                return roc_auc_score(y_true, y_pred, multi_class="ovr",
                                     average="macro", labels=labels)
            return roc_auc_score(y_true, y_pred)
        if metric == "f1":
            preds_hard = (y_pred >= 0.5).astype(int)
            tp = np.sum((y_true == 1) & (preds_hard == 1))
            fp = np.sum((y_true == 0) & (preds_hard == 1))
            fn = np.sum((y_true == 1) & (preds_hard == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        # default: exp(-log_loss)  [mirrors R]
        ll = log_loss(y_true, y_pred, labels=labels)
        return float(np.exp(-ll))

    else:  # regression
        if metric == "mae":
            return -float(np.mean(np.abs(y_true - y_pred)))
        return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Gene / state helpers
# ---------------------------------------------------------------------------

def gene_to_state_key(gene):
    multi_comp = {"pca", "truncated_svd", "umap",
                  "genie_centroid_dist", "lumbermark_centroid_dist",
                  "genie", "lumbermark"}
    if gene.transformer_name in multi_comp:
        params_str = ""
    else:
        p_dict = {k: v for k, v in gene.params.items() if k != "comp_idx"}
        params_str = f"_{sorted(p_dict.items())}" if p_dict else ""
    return f"{gene.transformer_name}({','.join(gene.input_cols)}){params_str}"


# ---------------------------------------------------------------------------
# apply_individual
# ---------------------------------------------------------------------------

def apply_individual(individual, data, target_col=None, is_train=True,
                     allow_prune=True, state_cache=None, verbose=False):
    """
    Apply the individual's feature pipeline to *data*.

    During training (is_train=True):
      - Stateful transformers are fitted on *data* (or cached via state_cache).
      - Genes that fail (constant, correlated, missing cols) are pruned when
        allow_prune=True.
      - Successfully applied genes are marked gene.tested = True.

    During inference (is_train=False):
      - Uses pre-fitted gene.state — does NOT re-fit.
      - Pruning is disabled (we must apply every gene the model depends on).
    """
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)

    new_genes = []
    if state_cache is None:
        state_cache = {}

    for gene in individual.genes:
        old_data = data
        try:
            # ── Check all input columns exist ──────────────────────────────
            missing = [c for c in gene.input_cols if c not in data.columns]
            if missing:
                raise ValueError(
                    f"Missing input columns for {gene.transformer_name}")

            transformer = evo_transformers[gene.transformer_name]

            # ── State caching / re-use ─────────────────────────────────────
            cache_key = None
            if target_col is not None and target_col in data.columns and state_cache is not None:
                # 1. Training target array's bytes hash
                y_arr = data[target_col].to_numpy()
                import hashlib
                y_hash = hashlib.md5(y_arr.tobytes()).hexdigest()
                # 2. gene_to_state_formula
                from ..evolution.individual import gene_to_state_formula
                state_formula = gene_to_state_formula(gene)
                # 3. md5 key
                cache_key = hashlib.md5(f"{state_formula}_{y_hash}".encode()).hexdigest()

                if cache_key in state_cache and gene.state is None:
                    gene.state = state_cache[cache_key]

            # ── Fit (training only) ────────────────────────────────────────
            if (is_train and transformer.fit_func is not None
                    and target_col is not None and gene.state is None):
                gene.state = call_with_optional_params(
                    transformer.fit_func, data, gene.input_cols,
                    target_col, params=gene.params)
                if cache_key is not None and gene.state is not None:
                    state_cache[cache_key] = gene.state

            # ── Apply ──────────────────────────────────────────────────────
            res = call_with_optional_params(
                transformer.apply_func, data, gene.input_cols,
                gene.state, params=gene.params)

            if isinstance(res, pl.Expr):
                data = data.with_columns(res.alias(gene.output_col))
            elif isinstance(res, pl.Series):
                data = data.with_columns(res.alias(gene.output_col))
            else:
                data = data.with_columns(
                    pl.Series(gene.output_col, res))

            # ── Redundancy checks (training only) ─────────────────────────
            if (allow_prune and is_train
                    and data[gene.output_col].dtype in
                    [pl.Float32, pl.Float64, pl.Int32, pl.Int64]):
                col_s = data[gene.output_col]
                # 1. Constant
                if col_s.drop_nulls().n_unique() <= 1:
                    raise ValueError(
                        f"Gene {gene.output_col} generated a constant column.")
                # 2. High correlation with existing numeric columns
                existing_num = [
                    c for c in old_data.columns
                    if c != target_col
                    and old_data[c].dtype in
                    [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                ]
                if existing_num:
                    corrs = data.select([
                        pl.corr(gene.output_col, c).abs().alias(c)
                        for c in existing_num
                    ]).row(0)
                    corrs = [c if c is not None else 0.0 for c in corrs]
                    if corrs and max(corrs) >= 0.95:
                        raise ValueError(
                            f"Gene {gene.output_col} is highly correlated "
                            f"({max(corrs):.2f}).")

            gene.tested = True
            new_genes.append(gene)

        except Exception as e:
            data = old_data
            if allow_prune:
                if verbose:
                    print(f"  [Prune] Dropped gene {gene.transformer_name}: {e}")
            else:
                raise

    if allow_prune:
        individual.genes = new_genes

    return data


# ---------------------------------------------------------------------------
# evaluate_fitness
# ---------------------------------------------------------------------------

def evaluate_fitness(individual, data, target_col,
                     task="classification",
                     cv_folds=5,
                     evaluator="lightgbm",
                     evaluation_strategy="cv",
                     split_ratio=None,
                     metric="default",
                     complexity_penalty=0.0,
                     split_strategy="cv",
                     train_idx=None,
                     val_idx=None,
                     holdout_idx=None,
                     state_cache=None,
                     **kwargs):
    """
    Evaluate an individual via cross-validation or a single train/val split.

    Returns the Individual with updated .fitness and .importances.
    Fitness is exp(-log_loss) for classification (higher = better, in (0,1]).
    """
    if state_cache is None:
        state_cache = {}
    verbose = kwargs.get('verbose', False)

    if split_ratio is None:
        split_ratio = [0.7, 0.3]

    if evaluator not in evo_evaluators:
        raise ValueError(f"Evaluator '{evaluator}' not found in registry.")

    train_func = evo_evaluators[evaluator]

    num_class = None
    classes = None
    if task == "multiclass":
        y_all = data[target_col].to_numpy()
        classes = np.unique(y_all).tolist()
        num_class = len(classes)

    # ── Helper: prepare X/y from a processed polars DataFrame ──────────────
    def _make_xy(df, feature_cols):
        for col in feature_cols:
            if df[col].dtype in [pl.Utf8, pl.Categorical, pl.String]:
                df = df.with_columns(
                    pl.col(col).cast(pl.Categorical).to_physical())
        X = df.select(feature_cols).to_numpy().astype(np.float64)
        y = df[target_col].to_numpy()
        if task == "multiclass":
            y = np.array([classes.index(v) for v in y], dtype=np.int32)
        return X, y

    # ── Helper: score one train/val pair ───────────────────────────────────
    def _score_split(train_df, val_df):
        ind_copy = _shallow_copy_individual(individual)
        try:
            train_feat = apply_individual(
                ind_copy, train_df, target_col, is_train=True, state_cache=state_cache, verbose=verbose)
        except Exception:
            return None, None, None, None

        # Build a val copy that shares the *fitted* gene states from training
        import copy as _copy
        val_ind = _copy.deepcopy(ind_copy)  # carries fitted states
        val_feat = apply_individual(
            val_ind, val_df, target_col=None, is_train=False, allow_prune=False, state_cache=state_cache)

        raw_cols = (ind_copy.numeric_cols + ind_copy.categorical_cols
                    + [g.output_col for g in ind_copy.genes])
        feature_cols = list(dict.fromkeys(raw_cols))

        X_tr, y_tr = _make_xy(train_feat, feature_cols)
        X_val, y_val = _make_xy(val_feat, feature_cols)

        res = train_func(
            x_train=X_tr, y_train=y_tr,
            x_val=X_val, y_val=y_val,
            task=task, num_class=num_class,
            feature_names=feature_cols,
            verbose=-1
        )
        preds = res["predictions"]
        score = compute_metric(y_val, preds, task, metric, num_class)
        
        # Save best parameters if the tuner set them
        best_params = None
        if "best_params" in res:
            best_params = res["best_params"]
        elif hasattr(individual, "best_params"):
            best_params = individual.best_params

        return score, res.get("importances", {}), ind_copy.genes, best_params

    # ── SPLIT strategy ─────────────────────────────────────────────────────
    if evaluation_strategy == "split" or split_strategy == "split_index":
        if split_strategy == "split_index":
            train_df = data[train_idx]
            val_df   = data[val_idx]
        else:
            n = data.height
            n_train = int(n * split_ratio[0])
            idx = np.random.permutation(n)
            train_df = data[idx[:n_train].tolist()]
            val_df   = data[idx[n_train:].tolist()]
            
        score, importances, genes, best_params = _score_split(train_df, val_df)
        if score is None:
            individual.fitness = -np.inf
            return individual
        individual.fitness = score
        if complexity_penalty > 0:
            individual.fitness -= complexity_penalty * len(individual.genes)
        individual.importances = importances or {}
        if genes is not None:
            individual.genes = genes
        if best_params is not None:
            individual.best_params = best_params
        return individual

    # ── CV strategy (default) ──────────────────────────────────────────────
    y_all = data[target_col].to_numpy()
    if task in ("classification", "multiclass"):
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        splits = list(kf.split(np.zeros(len(y_all)), y_all))
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        splits = list(kf.split(np.zeros(len(y_all))))

    scores = []
    fold_importances = []
    last_genes = None
    last_best_params = None

    for train_idx_fold, val_idx_fold in splits:
        train_df = data[train_idx_fold.tolist()]
        val_df   = data[val_idx_fold.tolist()]
        score, importances, genes, best_params = _score_split(train_df, val_df)
        if score is None:
            continue
        scores.append(score)
        if importances:
            fold_importances.append(importances)
        last_genes = genes
        if best_params is not None:
            last_best_params = best_params

    if not scores:
        individual.fitness = -np.inf
        return individual

    individual.fitness = float(np.mean(scores))
    if complexity_penalty > 0:
        individual.fitness -= complexity_penalty * len(individual.genes)

    # Average importances across folds
    if fold_importances:
        all_feats = set().union(*[d.keys() for d in fold_importances])
        avg = {
            f: float(np.mean([d.get(f, 0.0) for d in fold_importances]))
            for f in all_feats
        }
        total = sum(avg.values())
        if total > 0:
            avg = {k: v / total for k, v in avg.items()}
        individual.importances = avg
    else:
        individual.importances = {}

    # Propagate tested=True genes from last successful fold
    if last_genes is not None:
        tested_cols = {g.output_col for g in last_genes}
        individual.genes = [
            g for g in individual.genes if g.output_col in tested_cols
        ]
    if last_best_params is not None:
        individual.best_params = last_best_params

    return individual


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _shallow_copy_individual(ind):
    """Copy an individual but share transformer registry (not gene states)."""
    import copy
    from ..evolution.individual import Individual, Gene
    new_genes = []
    for g in ind.genes:
        ng = Gene(
            transformer_name=g.transformer_name,
            input_cols=list(g.input_cols),
            params=dict(g.params),
            state=None,          # fresh state per fold
            tested=False
        )
        new_genes.append(ng)
    new_ind = Individual(
        numeric_cols=ind.numeric_cols,
        categorical_cols=ind.categorical_cols,
        genes=new_genes
    )
    new_ind.fitness = ind.fitness
    new_ind.importances = dict(getattr(ind, "importances", {}))
    return new_ind


def evaluate_holdout_fitness(individual, data, split_strategy,
                             train_idx, val_idx, holdout_idx,
                             target_col, task, evaluator, threads=2,
                             state_cache=None, classes=None, num_class=None,
                             metric="default", verbose=False, **kwargs):
    if holdout_idx is None:
        individual.holdout_fitness = None
        return individual

    train_df = data[train_idx]
    val_df = data[val_idx]
    holdout_df = data[holdout_idx]

    ind_copy = _shallow_copy_individual(individual)
    try:
        train_feat = apply_individual(ind_copy, train_df, target_col, is_train=True, state_cache=state_cache, verbose=verbose)
    except Exception:
        individual.holdout_fitness = -np.inf
        return individual

    import copy as _copy
    val_ind = _copy.deepcopy(ind_copy)
    try:
        val_feat = apply_individual(val_ind, val_df, target_col=None, is_train=False, allow_prune=False, state_cache=state_cache)
    except Exception:
        individual.holdout_fitness = -np.inf
        return individual

    raw_cols = (ind_copy.numeric_cols + ind_copy.categorical_cols + [g.output_col for g in ind_copy.genes])
    feature_cols = list(dict.fromkeys(raw_cols))

    if task == "multiclass" and classes is None:
        y_all = data[target_col].to_numpy()
        classes = np.unique(y_all).tolist()
        num_class = len(classes)

    def _make_xy(df, feature_cols):
        for col in feature_cols:
            if df[col].dtype in [pl.Utf8, pl.Categorical, pl.String]:
                df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical())
        X = df.select(feature_cols).to_numpy().astype(np.float64)
        y = df[target_col].to_numpy()
        if task == "multiclass":
            y = np.array([classes.index(v) for v in y], dtype=np.int32)
        return X, y

    X_tr, y_tr = _make_xy(train_feat, feature_cols)
    X_val, y_val = _make_xy(val_feat, feature_cols)

    final_evaluator = evaluator
    if evaluator.endswith("_optuna"):
        final_evaluator = evaluator[:-7]
    elif evaluator.endswith("_tuned"):
        final_evaluator = evaluator[:-6]

    train_func = evo_evaluators.get(final_evaluator, evo_evaluators[evaluator])
    best_params = getattr(individual, "best_params", {}) or {}

    res = train_func(
        x_train=X_tr, y_train=y_tr,
        x_val=X_val, y_val=y_val,
        task=task, num_class=num_class,
        feature_names=feature_cols,
        verbose=-1,
        **best_params
    )

    holdout_ind = _copy.deepcopy(val_ind)
    try:
        holdout_feat = apply_individual(holdout_ind, holdout_df, target_col=None, is_train=False, allow_prune=False, state_cache=state_cache)
    except Exception:
        individual.holdout_fitness = -np.inf
        return individual

    X_holdout, y_holdout = _make_xy(holdout_feat, feature_cols)

    model = res["model"]
    if final_evaluator == "lightgbm":
        preds_holdout = model.predict(X_holdout)
    elif final_evaluator == "xgboost":
        import xgboost as xgb
        dmat = xgb.DMatrix(X_holdout, feature_names=feature_cols)
        preds_holdout = model.predict(dmat)
    else:
        if hasattr(model, "predict"):
            preds_holdout = model.predict(X_holdout)
        else:
            preds_holdout = res["predictions"]

    individual.holdout_fitness = compute_metric(y_holdout, preds_holdout, task, metric, num_class)
    individual.genes = holdout_ind.genes
    return individual
