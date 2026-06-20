import random
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Union
from .individual import Individual
from .population import initialize_population

@dataclass
class EvoRecipe:
    """
    Result of an evolve_features run — mirrors R's evo_recipe S3 object.

    Attributes
    ----------
    best_individual : Individual
        The highest-scoring individual found during evolution.
    history : list[Individual]
        All evaluated individuals across every generation.
    task : str
        The learning task ("classification", "multiclass", "regression").
    evaluator : str
        Name of the ML backend used.
    classes : list | None
        Unique class labels for multiclass tasks; None otherwise.
    best_model : Any | None
        Final model trained on the full dataset with the best recipe.
        Populated by EvoFE.fit(); None when using evolve_features directly.
    fitness : float
        Shortcut for best_individual.fitness.
    """
    best_individual: Individual
    history: List[Individual]
    task: str
    evaluator: str
    classes: Optional[List] = None
    best_model: Optional[Any] = None

    @property
    def fitness(self) -> float:
        return self.best_individual.fitness

    @property
    def genes(self):
        return self.best_individual.genes


# Preset allowed_transformers shorthand → list expansion happens in evolve_features
_TRANSFORMER_PRESETS = {
    "basic": [
        "add", "subtract", "multiply", "divide",
        "log", "sqrt", "reciprocal", "power",
        "normalized_difference", "frequency_encode",
        "one_hot_encode", "target_encode", "target_encode_multiclass",
        "rank_transform", "groupby_mean", "groupby_min", "groupby_max",
        "pca",
    ],
    "robust": [
        "log", "sqrt", "reciprocal", "power", "rank_transform",
        "add", "subtract", "multiply", "divide",
        "normalized_difference", "log_ratio",
        "target_encode", "woe_encode", "frequency_encode",
        "groupby_mean", "groupby_median", "groupby_sd",
        "groupby_zscore", "groupby_ratio", "groupby_quantile",
        "groupby_min", "groupby_max",
        "quantile_binning", "pca",
    ],
    "clustering": [
        "genie", "genie_centroid_dist", "lumbermark",
        "lumbermark_centroid_dist", "mst_score", "deadwood",
        "umap", "random_projection", "truncated_svd", "pca",
    ],
}


def is_invalid_individual(c_ind, next_gen, fitness_cache, best_fit):
    # Check 1: Duplicate in current generation
    def get_out(ind):
        if not ind.genes:
            return []
        return sorted([g.output_col for g in ind.genes])
    
    c_out = get_out(c_ind)
    for existing in next_gen:
        e_out = get_out(existing)
        if len(c_out) == len(e_out) and c_out == e_out:
            return True
            
    # Check 2: Taboo search
    from .individual import individual_to_recipe_string
    recipe_str = individual_to_recipe_string(c_ind)
    import hashlib
    cache_key = hashlib.md5(recipe_str.encode()).hexdigest()
    if cache_key in fitness_cache:
        cached_ind = fitness_cache[cache_key]
        known_fit = cached_ind.fitness
        if not np.isnan(known_fit) and not np.isnan(best_fit) and not np.isinf(best_fit):
            taboo_threshold = max(0.00002, 0.1 * (1 - abs(best_fit)))
            if known_fit < best_fit - taboo_threshold:
                return True
                
    return False


def tournament_select(pop, k=3):
    k = min(k, len(pop))
    candidates = random.sample(pop, k)
    def get_fit(ind):
        f = ind.fitness
        if np.isnan(f):
            return -np.inf
        return f
    return max(candidates, key=get_fit)


def evolve_features(
    data: Any,
    target_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    evaluate_fitness: Callable,
    pop_size: int = 10,
    n_generations: int = 10,
    initial_genes: int = 2,
    mutation_rate: float = 0.5,
    tournament_size: int = 3,
    early_stopping_rounds: Optional[int] = None,
    stagnation_limit: Optional[int] = None,
    expansion_factor: float = 1.5,
    task: str = "classification",
    evaluator: str = "lightgbm",
    evaluation_strategy: str = "cv",
    split_ratio: Optional[List[float]] = None,
    split_ids: Optional[List[str]] = None,
    split_strategy: str = "cv",
    train_idx: Optional[List[int]] = None,
    val_idx: Optional[List[int]] = None,
    holdout_idx: Optional[List[int]] = None,
    allowed_transformers: Union[str, List[str]] = "all",
    complexity_penalty: float = 0.0,
    metric: str = "default",
    cv_folds: int = 3,
    model_all_final_genes: bool = False,
    model_all_historical_genes: bool = False,
    verbose: bool = True,
    **kwargs
) -> EvoRecipe:
    """
    Run evolutionary feature engineering.
    """
    from ..builtin import evo_transformers as _all_transformers

    if split_ratio is None:
        split_ratio = [0.7, 0.3]

    # ── Resolve allowed_transformers ────────────────────────────────────────
    all_names = list(_all_transformers.keys())
    if allowed_transformers is None or allowed_transformers == "all":
        allowed_list = all_names
    elif isinstance(allowed_transformers, str):
        preset = _TRANSFORMER_PRESETS.get(allowed_transformers, all_names)
        allowed_list = [t for t in preset if t in all_names]
    else:
        allowed_list = [t for t in allowed_transformers if t in all_names]
    if not allowed_list:
        allowed_list = all_names

    # ── Validate input ──────────────────────────────────────────────────────
    if data is None or (hasattr(data, "height") and data.height == 0):
        raise ValueError("Input data cannot be empty.")
    if hasattr(data, "columns") and target_col not in data.columns:
        raise ValueError(f"target_col '{target_col}' not found in data.")
    if task not in ("classification", "multiclass", "regression"):
        raise ValueError("task must be 'classification', 'multiclass', or 'regression'.")
    if task in ("classification", "multiclass"):
        import polars as pl
        if hasattr(data, "select"):
            unique_vals = data.select(pl.col(target_col).unique()).height
        else:
            unique_vals = len(np.unique(data[target_col]))
        if unique_vals < 2:
            raise ValueError("Classification/multiclass tasks require at least two classes in target column.")

    # Metric task compatibility check
    m_lower = metric.lower()
    if task == "classification":
        valid_metrics = ["default", "auc", "f1", "eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement"]
        if m_lower not in valid_metrics:
            raise ValueError(f"Metric '{metric}' is not compatible with task '{task}'.")
    elif task == "multiclass":
        valid_metrics = ["default", "auc", "eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement"]
        if m_lower not in valid_metrics:
            raise ValueError(f"Metric '{metric}' is not compatible with task '{task}'.")
    elif task == "regression":
        valid_metrics = ["default", "rmse", "mae"]
        if m_lower not in valid_metrics:
            raise ValueError(f"Metric '{metric}' is not compatible with task '{task}'.")

    classes = None
    if task == "multiclass":
        import polars as pl
        classes = sorted(data[target_col].unique().to_list())
    num_class = len(classes) if classes is not None else None

    state_cache = {}
    fitness_cache = {}

    def _eval(ind):
        from .individual import individual_to_recipe_string
        import hashlib
        recipe_str = individual_to_recipe_string(ind)
        cache_key = hashlib.md5(recipe_str.encode()).hexdigest()
        if cache_key in fitness_cache:
            return copy.deepcopy(fitness_cache[cache_key])
            
        evaluated_ind = evaluate_fitness(
            ind, data, target_col,
            task=task, evaluator=evaluator,
            cv_folds=cv_folds,
            evaluation_strategy=evaluation_strategy,
            split_ratio=split_ratio,
            metric=metric,
            complexity_penalty=complexity_penalty,
            split_strategy=split_strategy,
            train_idx=train_idx,
            val_idx=val_idx,
            holdout_idx=holdout_idx,
            state_cache=state_cache,
            verbose=verbose,
            **kwargs
        )
        fitness_cache[cache_key] = copy.deepcopy(evaluated_ind)
        return evaluated_ind

    if verbose:
        print(f"Initializing population of size {pop_size}...")

    population = initialize_population(
        pop_size=pop_size,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        initial_genes=initial_genes,
        task=task,
        allowed_transformers=allowed_list,
    )

    history: List[Individual] = []
    historical_best_genes = []

    if verbose:
        print("Evaluating initial population...")
    for i, ind in enumerate(population):
        population[i] = _eval(ind)
        history.append(copy.deepcopy(population[i]))
        if verbose:
            print(f"  [Init] Individual {i+1}/{pop_size} -> "
                  f"Fitness: {population[i].fitness:.4f}")

    population.sort(key=lambda ind: ind.fitness if not np.isnan(ind.fitness) else -np.inf,
                    reverse=True)
    best_ind = copy.deepcopy(population[0])
    historical_best_genes.extend(copy.deepcopy(best_ind.genes))

    if verbose:
        print(f"Generation 0 - Best Fitness: {best_ind.fitness:.4f}")

    early_stop_counter = 0
    generations_without_improvement = 0
    current_pop_size = pop_size

    for gen in range(1, n_generations + 1):
        if early_stop_counter > 0:
            generations_without_improvement = early_stop_counter
        else:
            generations_without_improvement = 0

        # Stagnation-based population control (growth/decay)
        if stagnation_limit is not None:
            if generations_without_improvement >= stagnation_limit:
                current_pop_size = max(current_pop_size + 1, int(current_pop_size * expansion_factor))
            elif generations_without_improvement == 0:
                current_pop_size = max(pop_size, int(current_pop_size * 0.7))
            target_pop_size = current_pop_size
        else:
            target_pop_size = pop_size

        # Collect tested gene outputs for mutation options
        tested_gene_outputs = []
        for ind in population:
            for g in ind.genes:
                if g.tested:
                    tested_gene_outputs.append(g.output_col)
        tested_gene_outputs = list(set(tested_gene_outputs))

        # Aggregate importances from survivors
        num_survivors = min(len(population), max(2, len(population) // 2))
        survivors = population[:num_survivors]
        
        global_importances = {}
        for s in survivors:
            if hasattr(s, "importances") and s.importances:
                for feat, val in s.importances.items():
                    if feat not in global_importances:
                        global_importances[feat] = []
                    global_importances[feat].append(val)
                    
        global_importances_vec = {}
        if global_importances:
            for feat, vals in global_importances.items():
                global_importances_vec[feat] = float(np.mean(vals))

        # Adaptive mutation rate and temperature
        stagnation_ratio = min(1.0, generations_without_improvement / stagnation_limit) if (stagnation_limit and stagnation_limit > 0) else 0.0
        adaptive_mutation_rate = 0.3 + 0.4 * stagnation_ratio
        temperature = 0.1 + 0.9 * stagnation_ratio

        next_generation = [copy.deepcopy(population[0])]  # elitism

        while len(next_generation) < target_pop_size:
            idx_child = len(next_generation)
            is_expansion = idx_child >= pop_size

            if is_expansion:
                parent = tournament_select(population, k=tournament_size)
                child = copy.deepcopy(parent)
                child.mutate(
                    verbose=False,
                    force_add=True,
                    importances=global_importances_vec,
                    temperature=100.0,
                    allowed_transformers=allowed_list,
                )
            elif random.random() < (1.0 - adaptive_mutation_rate):
                # Crossover
                p1 = tournament_select(population, k=tournament_size)
                p2 = tournament_select(population, k=tournament_size)
                from .individual import crossover, union_crossover
                if random.random() < 0.5:
                    child = crossover(p1, p2, verbose=False)
                else:
                    child = union_crossover(p1, p2, verbose=False)
                
                # 20% chance of mutation
                if random.random() < 0.2:
                    child.mutate(
                        verbose=False,
                        importances=global_importances_vec,
                        temperature=temperature,
                        allowed_transformers=allowed_list,
                    )
            else:
                # Mutate only
                parent = tournament_select(population, k=tournament_size)
                child = copy.deepcopy(parent)
                child.mutate(
                    verbose=False,
                    importances=global_importances_vec,
                    temperature=temperature,
                    allowed_transformers=allowed_list,
                )

            # Taboo search validation
            attempts = 0
            while is_invalid_individual(child, next_generation, fitness_cache, best_ind.fitness) and attempts < 15:
                child.mutate(
                    verbose=False,
                    force_add=True,
                    importances=global_importances_vec,
                    temperature=100.0 if is_expansion else temperature,
                    allowed_transformers=allowed_list,
                )
                attempts += 1

            next_generation.append(child)

        # Evaluate next generation
        for i, ind in enumerate(next_generation):
            if np.isnan(ind.fitness):
                next_generation[i] = _eval(ind)
                history.append(copy.deepcopy(next_generation[i]))

        population = sorted(
            next_generation,
            key=lambda ind: ind.fitness if not np.isnan(ind.fitness) else -np.inf,
            reverse=True,
        )

        if population[0].fitness > best_ind.fitness:
            best_ind = copy.deepcopy(population[0])
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        historical_best_genes.extend(copy.deepcopy(population[0].genes))

        if verbose:
            print(f"Generation {gen} - Best Fitness: {best_ind.fitness:.4f}")

        if (early_stopping_rounds is not None
                and early_stop_counter >= early_stopping_rounds):
            if verbose:
                print(f"Early stopping triggered after {gen} generations.")
            break

    # Final evaluation of new individuals
    for i, ind in enumerate(population):
        population[i] = _eval(ind)
    population.sort(key=lambda ind: ind.fitness if not np.isnan(ind.fitness) else -np.inf, reverse=True)
    best_ind = copy.deepcopy(population[0])

    # model_all_final_genes
    if model_all_final_genes:
        if verbose:
            print("Evaluating pooled features (all final genes)...")
        all_genes = []
        for ind in population:
            all_genes.extend(ind.genes)
        
        seen_cols = set()
        deduped_genes = []
        for gene in all_genes:
            if gene.output_col not in seen_cols:
                seen_cols.add(gene.output_col)
                deduped_genes.append(gene)
        
        super_ind = Individual(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            genes=deduped_genes
        )
        super_ind = _eval(super_ind)
        
        if not getattr(super_ind, "best_params", None) and getattr(best_ind, "best_params", None):
            super_ind.best_params = best_ind.best_params
            
        if super_ind.fitness > best_ind.fitness:
            if verbose:
                print(f"  Pooled features improved validation fitness from {best_ind.fitness:.4f} to {super_ind.fitness:.4f}. Using pooled features.")
            best_ind = super_ind
        else:
            if verbose:
                print(f"  Pooled features (fitness: {super_ind.fitness:.4f}) did not exceed best individual (fitness: {best_ind.fitness:.4f}). Keeping best individual.")

    # model_all_historical_genes
    if model_all_historical_genes:
        if verbose:
            print("Evaluating historical pooled features (best genes from all generations)...")
        historical_best_genes.extend(best_ind.genes)
        if historical_best_genes:
            seen_cols = set()
            deduped_historical_genes = []
            for gene in historical_best_genes:
                if gene.output_col not in seen_cols:
                    seen_cols.add(gene.output_col)
                    deduped_historical_genes.append(gene)
            
            super_ind_hist = Individual(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                genes=deduped_historical_genes
            )
            super_ind_hist = _eval(super_ind_hist)
            
            if not getattr(super_ind_hist, "best_params", None) and getattr(best_ind, "best_params", None):
                super_ind_hist.best_params = best_ind.best_params
                
            if super_ind_hist.fitness > best_ind.fitness:
                if verbose:
                    print(f"  Historical pooled features improved validation fitness from {best_ind.fitness:.4f} to {super_ind_hist.fitness:.4f}. Using historical pooled features.")
                best_ind = super_ind_hist
            else:
                if verbose:
                    print(f"  Historical pooled features (fitness: {super_ind_hist.fitness:.4f}) did not exceed current best fitness (fitness: {best_ind.fitness:.4f}). Keeping current best individual.")

    # Holdout evaluation
    if (evaluation_strategy == "split" or split_strategy == "split_index") and holdout_idx is not None:
        from ..evaluation.cv import evaluate_holdout_fitness
        best_ind = evaluate_holdout_fitness(
            best_ind, data, split_strategy,
            train_idx, val_idx, holdout_idx,
            target_col, task, evaluator,
            state_cache=state_cache, classes=classes, num_class=num_class,
            metric=metric, **kwargs
        )

    if verbose:
        print(f"Evolution complete. Best Fitness: {best_ind.fitness:.4f}")

    return EvoRecipe(
        best_individual=best_ind,
        history=history,
        task=task,
        evaluator=evaluator,
        classes=classes,
        best_model=None,  # populated by EvoFE.fit()
    )
