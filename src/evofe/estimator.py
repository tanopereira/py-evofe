"""
EvoFE — Sklearn-compatible estimator wrapping evolutionary feature engineering.

Usage
-----
    from evofe import EvoFE

    evo = EvoFE(task="multiclass", evaluator="xgboost", pop_size=10, n_generations=5)
    evo.fit(df_train, y_train)           # runs evolution
    df_enriched = evo.transform(df_test) # applies best recipe
    preds = evo.predict(df_test)         # feature-engineer + model inference
    proba = evo.predict_proba(df_test)   # class probabilities
"""
from __future__ import annotations

import numpy as np
import polars as pl
from typing import List, Optional, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .evaluation.cv import evaluate_fitness, apply_individual
from .evolution.engine import evolve_features, EvoRecipe
from .evaluation.models import evo_evaluators


class EvoFE(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible evolutionary feature engineering estimator.

    Parameters
    ----------
    task : str
        "classification", "multiclass", or "regression".
    evaluator : str
        Registered model backend name (e.g. "lightgbm", "xgboost").
    pop_size : int
        Population size.
    n_generations : int
        Maximum number of generations.
    cv_folds : int
        Number of cross-validation folds (used when evaluation_strategy="cv").
    evaluation_strategy : str
        "cv" (cross-validation) or "split" (single train/val split).
    split_ratio : list[float]
        [train_frac, val_frac] — used when evaluation_strategy="split".
    allowed_transformers : str or list
        "all", "basic", "robust", "clustering", or a list of transformer names.
    complexity_penalty : float
        Penalty per gene subtracted from fitness (encourages parsimony).
    metric : str
        Fitness metric: "default", "auc", "f1", "mae".
    mutation_rate : float
        Probability of mutating an offspring.
    early_stopping_rounds : int | None
        Stop evolution if fitness does not improve for this many generations.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        task: str = "classification",
        evaluator: str = "lightgbm",
        pop_size: int = 10,
        n_generations: int = 10,
        cv_folds: int = 5,
        evaluation_strategy: str = "cv",
        split_ratio: Optional[List[float]] = None,
        allowed_transformers="all",
        complexity_penalty: float = 0.0,
        metric: str = "default",
        mutation_rate: float = 0.5,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
        random_state: Optional[int] = None,
    ):
        self.task = task
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cv_folds = cv_folds
        self.evaluation_strategy = evaluation_strategy
        self.split_ratio = split_ratio
        self.allowed_transformers = allowed_transformers
        self.complexity_penalty = complexity_penalty
        self.metric = metric
        self.mutation_rate = mutation_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state

    # ── Internal helpers ────────────────────────────────────────────────────

    def _to_polars(self, X, feature_names=None) -> pl.DataFrame:
        if isinstance(X, pl.DataFrame):
            return X
        if hasattr(X, "to_frame"):  # pandas Series
            return pl.from_pandas(X.to_frame())
        if hasattr(X, "columns"):   # pandas DataFrame
            return pl.from_pandas(X)
        # numpy array
        arr = np.asarray(X)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(arr.shape[1])]
        return pl.DataFrame({n: arr[:, i] for i, n in enumerate(feature_names)})

    def _infer_cols(self, df: pl.DataFrame, target_col: Optional[str] = None):
        num_types = [pl.Float32, pl.Float64, pl.Int8, pl.Int16,
                     pl.Int32, pl.Int64, pl.UInt8, pl.UInt16,
                     pl.UInt32, pl.UInt64]
        cat_types = [pl.Utf8, pl.Categorical, pl.String]
        numeric_cols = [
            c for c in df.columns
            if c != target_col and df[c].dtype in num_types
        ]
        categorical_cols = [
            c for c in df.columns
            if c != target_col and df[c].dtype in cat_types
        ]
        return numeric_cols, categorical_cols

    # ── Sklearn API ─────────────────────────────────────────────────────────

    def fit(self, X, y=None, feature_names=None, target_col: str = "__target__"):
        """
        Run evolutionary feature search.

        Parameters
        ----------
        X : polars.DataFrame | pandas.DataFrame | numpy.ndarray
            Training data including (or excluding) the target column.
        y : array-like | None
            Target values. If X already contains a column named *target_col*
            and y is None, it is used directly.
        feature_names : list[str] | None
            Column names when X is a numpy array.
        target_col : str
            Name of the target column (default "__target__").

        Returns
        -------
        self
        """
        if self.random_state is not None:
            import random
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        df = self._to_polars(X, feature_names)

        if y is not None:
            y_arr = np.asarray(y)
            df = df.with_columns(pl.Series(target_col, y_arr))
        elif target_col not in df.columns:
            raise ValueError(
                "Provide y or include target_col in X.")

        self.target_col_ = target_col
        self.feature_names_in_ = [c for c in df.columns if c != target_col]
        numeric_cols, categorical_cols = self._infer_cols(df, target_col)
        self.numeric_cols_ = numeric_cols
        self.categorical_cols_ = categorical_cols

        self.recipe_: EvoRecipe = evolve_features(
            data=df,
            target_col=target_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            evaluate_fitness=evaluate_fitness,
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            mutation_rate=self.mutation_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            task=self.task,
            evaluator=self.evaluator,
            evaluation_strategy=self.evaluation_strategy,
            split_ratio=self.split_ratio,
            allowed_transformers=self.allowed_transformers,
            complexity_penalty=self.complexity_penalty,
            metric=self.metric,
            cv_folds=self.cv_folds,
            verbose=self.verbose,
        )

        # Train final model on full dataset with the best recipe
        self._fit_final_model(df, target_col)

        return self

    def _fit_final_model(self, df: pl.DataFrame, target_col: str):
        """Fit the best recipe on the full training data and store the model."""
        import copy
        best = copy.deepcopy(self.recipe_.best_individual)
        # Apply genes (fit on full data)
        feat_df = apply_individual(best, df, target_col, is_train=True, verbose=self.verbose)

        raw_cols = (best.numeric_cols + best.categorical_cols
                    + [g.output_col for g in best.genes])
        feature_cols = list(dict.fromkeys(raw_cols))

        for col in feature_cols:
            if feat_df[col].dtype in [pl.Utf8, pl.Categorical, pl.String]:
                feat_df = feat_df.with_columns(
                    pl.col(col).cast(pl.Categorical).to_physical())

        X = feat_df.select(feature_cols).to_numpy().astype(np.float64)
        y = feat_df[target_col].to_numpy()

        classes = self.recipe_.classes
        num_class = len(classes) if classes else None
        if self.task == "multiclass" and classes is not None:
            y = np.array([classes.index(v) for v in y], dtype=np.int32)

        train_func = evo_evaluators[self.evaluator]
        res = train_func(
            x_train=X, y_train=y,
            x_val=None, y_val=None,
            task=self.task,
            num_class=num_class,
            feature_names=feature_cols,
            verbose=-1,
        )
        self.recipe_.best_model = res["model"]
        self.final_feature_cols_ = feature_cols
        self.fitted_individual_ = best   # carries fitted gene states

    def transform_df(self, X, feature_names=None) -> pl.DataFrame:
        """
        Apply the evolved feature recipe to new data.

        Returns a Polars DataFrame with the original columns plus new gene columns.
        """
        check_is_fitted(self, "recipe_")
        import copy
        df = self._to_polars(X, feature_names or self.feature_names_in_)
        ind = copy.deepcopy(self.fitted_individual_)
        result = apply_individual(ind, df, target_col=None,
                                  is_train=False, allow_prune=False)
        return result

    def transform(self, X, feature_names=None) -> np.ndarray:
        """
        Apply the evolved feature recipe to new data and return a numpy array
        of the selected features, suitable for scikit-learn pipelines.
        """
        check_is_fitted(self, "recipe_")
        feat_df = self.transform_df(X, feature_names)
        feature_cols = self.final_feature_cols_

        for col in feature_cols:
            if col in feat_df.columns and feat_df[col].dtype in [
                    pl.Utf8, pl.Categorical, pl.String]:
                feat_df = feat_df.with_columns(
                    pl.col(col).cast(pl.Categorical).to_physical())

        avail = [c for c in feature_cols if c in feat_df.columns]
        X_new = feat_df.select(avail).to_numpy().astype(np.float64)
        return X_new

    def predict(self, X, feature_names=None) -> np.ndarray:
        """
        Apply recipe then run the final model to produce predictions.
        """
        check_is_fitted(self, "recipe_")
        if self.recipe_.best_model is None:
            raise NotFittedError("No final model found. Call fit() first.")

        X_new = self.transform(X, feature_names)
        model = self.recipe_.best_model
        return self._model_predict(model, X_new)

    def predict_proba(self, X, feature_names=None) -> np.ndarray:
        """
        Return class probability matrix (classification / multiclass only).
        """
        if self.task == "regression":
            raise ValueError("predict_proba is not available for regression tasks.")
        preds = self.predict(X, feature_names)
        # LightGBM binary returns 1-D probabilities; wrap into 2-col matrix
        if preds.ndim == 1:
            return np.column_stack([1 - preds, preds])
        return preds

    def _model_predict(self, model, X_new: np.ndarray) -> np.ndarray:
        """Dispatch prediction to the right backend."""
        # LightGBM
        try:
            import lightgbm as lgb
            if isinstance(model, lgb.Booster):
                return model.predict(X_new)
        except ImportError:
            pass
        # XGBoost
        try:
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                feature_names = getattr(model, "feature_names", None)
                dm = xgb.DMatrix(X_new, feature_names=feature_names)
                return model.predict(dm)
        except ImportError:
            pass
        # Generic sklearn-style
        if hasattr(model, "predict"):
            return model.predict(X_new)
        raise TypeError(f"Unsupported model type: {type(model)}")

    def score(self, X, y) -> float:
        """
        Return the mean accuracy or R^2 score on the given test data and labels.
        """
        check_is_fitted(self, "recipe_")
        preds = self.predict(X)
        if self.task == "classification":
            from sklearn.metrics import accuracy_score
            y_pred = (preds >= 0.5).astype(int)
            classes = self.recipe_.classes
            if classes is not None:
                y_mapped = np.array([classes.index(v) for v in y])
            else:
                y_mapped = y
            return accuracy_score(y_mapped, y_pred)
        elif self.task == "multiclass":
            from sklearn.metrics import accuracy_score
            y_pred = preds.argmax(axis=1)
            classes = self.recipe_.classes
            if classes is not None:
                y_mapped = np.array([classes.index(v) for v in y])
            else:
                y_mapped = y
            return accuracy_score(y_mapped, y_pred)
        else: # regression
            from sklearn.metrics import r2_score
            return r2_score(y, preds)

    def get_recipe(self) -> EvoRecipe:
        """Return the EvoRecipe result object."""
        check_is_fitted(self, "recipe_")
        return self.recipe_
