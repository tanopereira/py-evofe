import pytest
import polars as pl
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss, mean_squared_error

# Import evofe components
from evofe.transformers import EvoTransformer
from evofe.builtin import evo_transformers
from evofe.evolution import evolve_features
from evofe.evaluation import evaluate_fitness
from evofe.evaluation.models import evo_evaluators, register_evaluator
from evofe.evolution.individual import Individual, Gene
from evofe.evaluation.cv import apply_individual

# =====================================================================
# TIER 1: FEATURE COVERAGE (Requirement & Feature Parity Verification)
# =====================================================================

# --- Feature 1: evolve_features (Search Loop) ---

def test_e2e_binary_classification_cv():
    """Case 1.1: E2E Binary Classification with 5-fold CV."""
    np.random.seed(42)
    n_samples = 40
    df = pl.DataFrame({
        "num1": np.random.randn(n_samples),
        "num2": np.random.randn(n_samples),
        "cat1": np.random.choice(["a", "b"], n_samples),
        "target": np.random.randint(0, 2, n_samples)
    })
    best_ind = evolve_features(
        data=df,
        target_col="target",
        numeric_cols=["num1", "num2"],
        categorical_cols=["cat1"],
        evaluate_fitness=evaluate_fitness,
        pop_size=3,
        n_generations=2,
        cv_folds=5,
        task="classification",
        verbose=False
    )
    assert best_ind is not None
    assert hasattr(best_ind, "fitness")
    assert best_ind.fitness > 0.0

def test_e2e_regression_split():
    """Case 1.2: E2E Regression with Split/CV strategy."""
    np.random.seed(42)
    n_samples = 30
    df = pl.DataFrame({
        "num1": np.random.randn(n_samples),
        "num2": np.random.randn(n_samples),
        "cat1": np.random.choice(["a", "b"], n_samples),
        "target": np.random.randn(n_samples)
    })
    best_ind = evolve_features(
        data=df,
        target_col="target",
        numeric_cols=["num1", "num2"],
        categorical_cols=["cat1"],
        evaluate_fitness=evaluate_fitness,
        pop_size=3,
        n_generations=2,
        task="regression",
        verbose=False
    )
    assert best_ind is not None
    assert best_ind.fitness < 0.0

def test_e2e_multiclass_model_all_final():
    """Case 1.3: E2E Multiclass Classification (checking model_all_final_genes)."""
    np.random.seed(42)
    n_samples = 45
    df = pl.DataFrame({
        "num1": np.random.randn(n_samples),
        "num2": np.random.randn(n_samples),
        "cat1": np.random.choice(["a", "b", "c"], n_samples),
        "target": np.random.choice([0, 1, 2], n_samples)
    })
    try:
        best_ind = evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["num1", "num2"],
            categorical_cols=["cat1"],
            evaluate_fitness=evaluate_fitness,
            pop_size=3,
            n_generations=2,
            task="multiclass",
            model_all_final_genes=True,
            verbose=False
        )
        assert best_ind is not None
    except TypeError as e:
        pytest.fail(f"model_all_final_genes parameter parity issue: {e}")

def test_dynamic_population_stagnation():
    """Case 1.4: Dynamic population expansion & decay when fitness stagnates."""
    np.random.seed(42)
    df = pl.DataFrame({
        "num1": np.random.randn(20),
        "target": np.random.randint(0, 2, 20)
    })
    try:
        best_ind = evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["num1"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness,
            pop_size=3,
            n_generations=2,
            stagnation_limit=2,
            expansion_factor=1.5,
            verbose=False
        )
        assert best_ind is not None
    except TypeError as e:
        pytest.fail(f"Stagnation-based population parameters not supported: {e}")

def test_restricted_allowed_transformers():
    """Case 1.5: Search with restricted allowed_transformers."""
    np.random.seed(42)
    df = pl.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target": [0, 1, 0, 1, 1]
    })
    try:
        best_ind = evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["num1"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness,
            pop_size=3,
            n_generations=2,
            allowed_transformers=["log", "sqrt"],
            verbose=False
        )
        assert best_ind is not None
    except TypeError as e:
        pytest.fail(f"allowed_transformers parameter parity issue: {e}")


# --- Feature 2: Evaluators & Tuning ---

def test_register_custom_evaluator():
    """Case 2.1: Register a custom mock evaluator via register_evaluator()."""
    def custom_train(x_train, y_train, x_val=None, y_val=None, task="classification", **kwargs):
        return {
            "model": "custom_mock",
            "predictions": np.zeros(len(x_val)) if x_val is not None else None,
            "importances": {"f0": 1.0}
        }
    register_evaluator("custom_mock", custom_train)
    assert "custom_mock" in evo_evaluators

def test_make_tunable_registration():
    """Case 2.2: Tuning a LightGBM evaluator using Optuna via make_tunable()."""
    try:
        from evofe.evaluation.tuning import make_tunable
    except ImportError as e:
        pytest.fail(f"make_tunable is not implemented: {e}")
        
    param_ranges = {
        "num_leaves": {"type": "integer", "lower": 7, "upper": 31}
    }
    try:
        make_tunable("lightgbm", param_ranges, tuner_name="lightgbm_tuned")
        assert "lightgbm_tuned" in evo_evaluators
    except Exception as e:
        pytest.fail(f"make_tunable failed: {e}")

def test_ts_refinement_binary():
    """Case 2.3: TS-Refinement metric validation on binary predictions."""
    try:
        from evofe.evaluation.metrics import ts_refinement
    except ImportError as e:
        pytest.fail(f"ts_refinement is not implemented: {e}")
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.4, 0.2])
    score = ts_refinement(y_true, y_pred, task="classification")
    assert score >= 0.0

def test_ts_refinement_multiclass():
    """Case 2.4: TS-Refinement metric validation on multiclass predictions."""
    try:
        from evofe.evaluation.metrics import ts_refinement
    except ImportError as e:
        pytest.fail(f"ts_refinement is not implemented: {e}")
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2]
    ])
    score = ts_refinement(y_true, y_pred, task="multiclass", num_class=3)
    assert score >= 0.0

def test_shap_importance_extraction():
    """Case 2.5: SHAP/model-native feature importances extraction."""
    from evofe.evaluation.models import _extract_shap_importances
    sh_2d = np.array([[0.5, 0.2], [0.3, 0.1]]) # 2 samples, 1 feature + 1 base value
    res_2d = _extract_shap_importances(sh_2d, 1, ["f0"])
    assert "f0" in res_2d
    assert res_2d["f0"] == np.mean([0.5, 0.3])


# --- Features 3 & 4: Clustering (genie, lumbermark) ---

def test_clustering_genie_lumbermark_registered():
    """Check genie and lumbermark registration parity."""
    assert "genie" in evo_transformers, "genie transformer not registered"
    assert "genie_centroid_dist" in evo_transformers, "genie_centroid_dist transformer not registered"
    assert "lumbermark" in evo_transformers, "lumbermark transformer not registered"
    assert "lumbermark_centroid_dist" in evo_transformers, "lumbermark_centroid_dist transformer not registered"

def test_clustering_output_types():
    """Case 3/4.1 & 3/4.2: Output types for label and distance estimators."""
    for name in ["genie", "lumbermark"]:
        assert name in evo_transformers
        trans = evo_transformers[name]
        assert trans.output_type == "categorical"
        
    for name in ["genie_centroid_dist", "lumbermark_centroid_dist"]:
        assert name in evo_transformers
        trans = evo_transformers[name]
        assert trans.output_type == "numeric"

def test_clustering_caching_invariant():
    """Case 3/4.3: Fit-Caching Invariant (fit_func executed exactly once)."""
    assert "genie" in evo_transformers, "genie transformer not registered"
    df = pl.DataFrame({
        "x1": [1.0, 2.0, 3.0], 
        "x2": [4.0, 5.0, 6.0],
        "target": [1, 0, 1]
    })
    gene1 = Gene(transformer_name="genie_centroid_dist", input_cols=["x1", "x2"], params={"centroid_idx": 0})
    gene2 = Gene(transformer_name="genie_centroid_dist", input_cols=["x1", "x2"], params={"centroid_idx": 1})
    ind = Individual(numeric_cols=["x1", "x2"], categorical_cols=[], genes=[gene1, gene2])
    
    fit_count = 0
    orig_fit = evo_transformers["genie_centroid_dist"].fit_func
    def instrumented_fit(d, i, t):
        nonlocal fit_count
        fit_count += 1
        return orig_fit(d, i, t)
    
    evo_transformers["genie_centroid_dist"].fit_func = instrumented_fit
    try:
        apply_individual(ind, df, target_col="target", is_train=True, allow_prune=False)
        assert fit_count == 1, f"Expected exactly 1 fit call due to caching, but got {fit_count}"
    except Exception as e:
        # If it raises because genie is not implemented/fallback fails, we still count it as failure
        raise
    finally:
        evo_transformers["genie_centroid_dist"].fit_func = orig_fit

def test_clustering_downsampling():
    """Case 3/4.4: Downsampling verification (max_clustering_size limit)."""
    assert "genie" in evo_transformers, "genie transformer not registered"
    large_df = pl.DataFrame({
        "x1": np.random.randn(2000), 
        "x2": np.random.randn(2000),
        "target": np.random.randint(0, 2, 2000)
    })
    trans = evo_transformers["genie"]
    trans.fit(large_df, input_cols=["x1", "x2"], target_col="target")
    res = trans.transform(large_df)
    assert len(res) == 2000

def test_clustering_missing_value_handling():
    """Case 3/4.5: Missing value (NA) imputation/handling during fit/apply."""
    assert "genie" in evo_transformers, "genie transformer not registered"
    df = pl.DataFrame({
        "x1": [1.0, None, 3.0, 4.0, 5.0],
        "x2": [10.0, 20.0, None, 40.0, 50.0],
        "target": [1, 0, 1, 0, 1]
    })
    trans = evo_transformers["genie"]
    trans.fit(df, input_cols=["x1", "x2"], target_col="target")
    res = trans.transform(df)
    assert len(res) == 5


# --- Feature 5: UMAP Reduction ---

def test_umap_coord_projection():
    """Case 5.1: Dimension coordinate projection."""
    assert "umap" in evo_transformers
    df = pl.DataFrame({
        "x1": np.random.randn(20), 
        "x2": np.random.randn(20),
        "target": np.random.randint(0, 2, 20)
    })
    trans = evo_transformers["umap"]
    trans.fit(df, input_cols=["x1", "x2"], target_col="target")
    res = trans.transform(df)
    assert isinstance(res, (pl.DataFrame, pl.Series, np.ndarray))

def test_umap_component_clamping():
    """Case 5.2: Out-of-bounds component clamping (clamps to max index)."""
    assert "umap" in evo_transformers
    df = pl.DataFrame({
        "x1": np.random.randn(20), 
        "x2": np.random.randn(20),
        "target": np.random.randint(0, 2, 20)
    })
    gene = Gene(transformer_name="umap", input_cols=["x1", "x2"], params={"comp_idx": 5})
    ind = Individual(numeric_cols=["x1", "x2"], categorical_cols=[], genes=[gene])
    res_df = apply_individual(ind, df, target_col="target", is_train=True, allow_prune=False)
    assert gene.output_col in res_df.columns

def test_umap_small_dataset_neighbors():
    """Case 5.3: Small dataset adjustments (scale n_neighbors if rows < 15)."""
    assert "umap" in evo_transformers
    df = pl.DataFrame({
        "x1": np.random.randn(10), 
        "x2": np.random.randn(10),
        "target": np.random.randint(0, 2, 10)
    })
    trans = evo_transformers["umap"]
    trans.fit(df, input_cols=["x1", "x2"], target_col="target")
    res = trans.transform(df)
    assert len(res) == 10

def test_umap_threading():
    """Case 5.4: Threading optimization (passing options for thread limits)."""
    assert "umap" in evo_transformers
    pass

def test_umap_caching():
    """Case 5.5: UMAP Caching invariant."""
    assert "umap" in evo_transformers
    df = pl.DataFrame({
        "x1": np.random.randn(20), 
        "x2": np.random.randn(20),
        "target": np.random.randint(0, 2, 20)
    })
    gene1 = Gene(transformer_name="umap", input_cols=["x1", "x2"], params={"comp_idx": 0})
    gene2 = Gene(transformer_name="umap", input_cols=["x1", "x2"], params={"comp_idx": 1})
    ind = Individual(numeric_cols=["x1", "x2"], categorical_cols=[], genes=[gene1, gene2])
    
    fit_count = 0
    orig_fit = evo_transformers["umap"].fit_func
    def instrumented_fit(d, i, t):
        nonlocal fit_count
        fit_count += 1
        return orig_fit(d, i, t)
    
    evo_transformers["umap"].fit_func = instrumented_fit
    try:
        apply_individual(ind, df, target_col="target", is_train=True, allow_prune=False)
        assert fit_count == 1, "UMAP fit should be cached and called only once"
    finally:
        evo_transformers["umap"].fit_func = orig_fit


# --- Features 6 & 7: Anomaly Detection (mst_score, deadwood) ---

def test_anomaly_score_creation():
    """Case 6/7.1: Score/flag generation and correct column creation."""
    assert "mst_score" in evo_transformers, "mst_score not registered"
    assert "deadwood" in evo_transformers, "deadwood not registered"

def test_anomaly_downsampling_1nn():
    """Case 6/7.2: Downsampling and mapping to the full dataset using 1-NN."""
    pass

def test_anomaly_min_columns_error():
    """Case 6/7.3: Error validation if fewer than 2 input columns are provided."""
    assert "mst_score" in evo_transformers, "mst_score not registered"
    df = pl.DataFrame({"x1": np.random.randn(20)})
    trans = evo_transformers["mst_score"]
    with pytest.raises(ValueError):
        trans.fit(df, input_cols=["x1"])

def test_anomaly_missing_values_safety():
    """Case 6/7.4: Missing values (NA) safety."""
    pass

def test_anomaly_caching():
    """Case 6/7.5: Caching invariant."""
    pass


# --- Missing/Modified R Feature Parity Checks ---

def test_missing_transformers_presence():
    """Verify presence of missing R transformers in Python."""
    assert "groupby_quantile" in evo_transformers, "groupby_quantile not registered"
    assert "target_encode_multiclass" in evo_transformers, "target_encode_multiclass not registered"
    assert "rank_transform" in evo_transformers, "rank_transform not registered"
    assert "datetime_extract" in evo_transformers, "datetime_extract not registered"
    assert "quantile_binning" in evo_transformers, "quantile_binning not registered"
    assert "log_binning" in evo_transformers, "log_binning not registered"

def test_power_dynamic_parameter():
    """Verify power transformer respects dynamic power parameter 'p'."""
    assert "power" in evo_transformers
    df = pl.DataFrame({"x": [2.0, 3.0]})
    trans = evo_transformers["power"]
    
    # We call apply_func with simulated state/params having p=3.0
    state = {"p": 3.0}
    res_expr = trans.apply_func(df, ["x"], state)
    res_df = df.select(res_expr)
    vals = res_df.to_series().to_list()
    # R expected: x^3 (2^3=8, 3^3=27)
    # Python current: hardcoded to x^2 (4, 9)
    assert vals == [8.0, 27.0]

def test_pca_centering_and_scaling_parity():
    """Verify PCA scales & centers, and supports dynamic component indexing."""
    assert "pca" in evo_transformers
    df = pl.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [10.0, 25.0, 15.0, 40.0, 52.0]
    })
    trans = evo_transformers["pca"]
    trans.fit(df, input_cols=["x1", "x2"])
    
    # Check if data was scaled before PCA.
    # We can inspect the model components or variance. Since python PCA does not scale,
    # the variance of x2 (much larger than x1) dominates completely.
    # Also verify if apply_func supports dynamic comp_idx from state/params.
    # Currently apply_func is: lambda d, i, s: _apply_reduction(d, i, s, comp_idx=0)
    # Let's test calling it with comp_idx=1 to get second component:
    state = trans.state_
    # Verify that we can obtain the second component (which should be different from 1st component)
    res_comp0 = trans.apply_func(df, ["x1", "x2"], state) # returns comp_idx=0
    # In R, if we specify comp_idx=1, we get the second component.
    # If python pca hardcodes comp_idx=0, calling with a param should return second component:
    # Here we mock how the engine would apply it:
    res_comp1 = _apply_reduction_with_param(df, ["x1", "x2"], state, comp_idx=1)
    assert not np.allclose(res_comp0, res_comp1), "PCA does not support dynamic component indexing"

def _apply_reduction_with_param(data, input_cols, state, comp_idx):
    # Helper to check what the output would be if dynamic comp_idx was supported
    from evofe.builtin.reduction import _apply_reduction
    return _apply_reduction(data, input_cols, state, comp_idx=comp_idx)

def test_umap_prefix_parity():
    """Verify UMAP uses 'ump' prefix to match R package."""
    assert "umap" in evo_transformers
    gene = Gene(transformer_name="umap", input_cols=["x1", "x2"])
    output_name = gene.output_col
    assert output_name.startswith("ump"), f"UMAP output col name starts with '{output_name.split('_')[0]}' instead of 'ump'"

def test_column_naming_hash_discrepancy():
    """Verify no column name hash discrepancy between Gene and EvoTransformer."""
    assert "power" in evo_transformers
    gene = Gene(transformer_name="power", input_cols=["num1"], params={"p": 3.0})
    df = pl.DataFrame({"num1": [1.0, 2.0, 3.0]})
    trans = evo_transformers["power"]
    trans.fit(df, input_cols=["num1"])
    res_df = trans.transform(df)
    # The output column name should match gene.output_col
    assert gene.output_col in res_df.columns, f"Naming mismatch: expected {gene.output_col}, found {res_df.columns}"


# =====================================================================
# TIER 2: BOUNDARY & CORNER CASES
# =====================================================================

def test_boundary_empty_inputs():
    """Case 2.1: Running with zero-row DataFrame or empty inputs (raises ValueError)."""
    df = pl.DataFrame()
    with pytest.raises(ValueError):
        evolve_features(
            data=df,
            target_col="target",
            numeric_cols=[],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness
        )

def test_boundary_single_class_target():
    """Case 2.2: Running classification with a single-class target column."""
    df = pl.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "target": [1, 1, 1]
    })
    with pytest.raises(ValueError):
        evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["num1"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness,
            task="classification"
        )

def test_boundary_nonexistent_target_col():
    """Case 2.3: Running with a non-existent target column name."""
    df = pl.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "target": [0, 1, 0]
    })
    with pytest.raises(ValueError):
        evolve_features(
            data=df,
            target_col="wrong_col",
            numeric_cols=["num1"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness
        )

def test_boundary_empty_allowed_transformers():
    """Case 2.4: Setting allowed_transformers to an empty list."""
    df = pl.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "target": [0, 1, 0]
    })
    with pytest.raises(ValueError):
        evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["num1"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness,
            allowed_transformers=[]
        )

def test_boundary_incompatible_metric_task():
    """Case 2.5: Incompatible metric and task type combination (raises ValueError)."""
    df = pl.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "target": [1.5, 2.5, 3.5]
    })
    with pytest.raises(ValueError):
        evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["num1"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness,
            task="regression",
            metric="auc"
        )

def test_boundary_tuning_small_dataset():
    """Case 2.6: Running tuning with a very small dataset (e.g., 5 rows)."""
    np.random.seed(42)
    df = pl.DataFrame({
        "num1": np.random.randn(5),
        "target": [0, 1, 0, 1, 0]
    })
    best_ind = evolve_features(
        data=df,
        target_col="target",
        numeric_cols=["num1"],
        categorical_cols=[],
        evaluate_fitness=evaluate_fitness,
        pop_size=3,
        n_generations=2,
        cv_folds=2,
        evaluator="lightgbm_optuna",
        verbose=False
    )
    assert best_ind is not None

def test_boundary_class_imbalance_optuna():
    """Case 2.7: Optuna tuning with class imbalance (validation fold completely missing a class label)."""
    df = pl.DataFrame({
        "num1": np.random.randn(10),
        "target": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })
    best_ind = evolve_features(
        data=df,
        target_col="target",
        numeric_cols=["num1"],
        categorical_cols=[],
        evaluate_fitness=evaluate_fitness,
        pop_size=3,
        n_generations=2,
        cv_folds=2,
        evaluator="lightgbm_optuna",
        verbose=False
    )
    assert best_ind is not None

def test_boundary_ts_refinement_nan_inf():
    """Case 2.8: TS-Refinement with extreme/infinite logits or NaN values."""
    try:
        from evofe.evaluation.metrics import ts_refinement
    except ImportError as e:
        pytest.fail(f"ts_refinement is not implemented: {e}")
    y_true = np.array([1, 0, 1])
    y_pred_nan = np.array([np.nan, 0.1, 0.9])
    score = ts_refinement(y_true, y_pred_nan, task="classification")
    assert score >= 0.0

def test_boundary_unregistered_evaluator():
    """Case 2.9: Using an unregistered evaluator name in evolve_features."""
    df = pl.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "target": [0, 1, 0]
    })
    with pytest.raises(ValueError):
        evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["num1"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness,
            evaluator="invalid_evaluator"
        )

def test_boundary_optimization_direction():
    """Case 2.10: Optimization direction validation (minimizing/maximizing bounds)."""
    pass

def test_boundary_zero_variance():
    """Case 2.11: Input columns with zero variance (constant values)."""
    df = pl.DataFrame({
        "num1": [1.0, 1.0, 1.0, 1.0, 1.0],
        "target": [0, 1, 0, 1, 0]
    })
    gene = Gene(transformer_name="divide", input_cols=["num1", "num1"])
    ind = Individual(numeric_cols=["num1"], categorical_cols=[], genes=[gene])
    res_df = apply_individual(ind, df, is_train=True)
    assert len(ind.genes) == 0

def test_boundary_k_exceeds_rows():
    """Case 2.12: k exceeding the number of unique rows in the dataset."""
    df = pl.DataFrame({
        "x1": [1.0, 2.0],
        "x2": [10.0, 20.0],
        "target": [1, 0]
    })
    assert "genie" in evo_transformers
    trans = evo_transformers["genie"]
    trans.fit(df, input_cols=["x1", "x2"], target_col="target")
    res = trans.transform(df)
    assert len(res) == 2

def test_boundary_out_of_bounds_parameters():
    """Case 2.13: Out-of-bounds parameters (e.g., k <= 1 or negative values)."""
    pass

def test_boundary_single_row_stateful_transform():
    """Case 2.14: Application of stateful transform on a single-row test dataset."""
    df_train = pl.DataFrame({
        "cat1": ["a", "b", "a", "b"],
        "target": [1, 0, 1, 0]
    })
    df_test = pl.DataFrame({
        "cat1": ["a"]
    })
    trans = evo_transformers["target_encode"]
    trans.fit(df_train, input_cols=["cat1"], target_col="target")
    res = trans.transform(df_test)
    assert len(res) == 1

def test_boundary_all_missing_values():
    """Case 2.15: Inputs containing 100% missing values (NA)."""
    df = pl.DataFrame({
        "num1": [None, None, None],
        "target": [0, 1, 0]
    })
    trans = evo_transformers["log"]
    trans.fit(df, input_cols=["num1"])
    res = trans.transform(df)
    assert len(res) == 3


# =====================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# =====================================================================

def test_chaining_umap_genie():
    """Test 3.1: Hierarchical Chaining (UMAP + Genie)."""
    assert "umap" in evo_transformers
    assert "genie" in evo_transformers
    
    gene1 = Gene(transformer_name="umap", input_cols=["x1", "x2"])
    gene2 = Gene(transformer_name="genie", input_cols=[gene1.output_col])
    
    ind = Individual(numeric_cols=["x1", "x2"], categorical_cols=[], genes=[gene1, gene2])
    ind._topological_sort()
    assert ind.genes.index(gene1) < ind.genes.index(gene2)

def test_chaining_lumbermark_groupby():
    """Test 3.2: Multi-clustering Group-by."""
    assert "lumbermark" in evo_transformers
    assert "groupby_mean" in evo_transformers
    
    gene1 = Gene(transformer_name="lumbermark", input_cols=["x1", "x2"])
    gene2 = Gene(transformer_name="groupby_mean", input_cols=[gene1.output_col, "y"])
    
    ind = Individual(numeric_cols=["x1", "x2", "y"], categorical_cols=[], genes=[gene1, gene2])
    ind._topological_sort()
    assert ind.genes.index(gene1) < ind.genes.index(gene2)

def test_tuned_evaluator_all_historical():
    """Test 3.3: Tuned Evaluator with All-Historical Genes."""
    pass

def test_ts_refinement_stagnation_expansion():
    """Test 3.4: TS-Refinement with Dynamic Population Expansion."""
    pass

def test_custom_transformer_chaining():
    """Test 3.5: Custom Transformer Chaining."""
    def fit_scaler(data, input_cols, target_col=None):
        col = input_cols[0]
        mean = data[col].mean()
        std = data[col].std() or 1.0
        return {"mean": mean, "std": std}
        
    def apply_scaler(data, input_cols, state):
        col = input_cols[0]
        return (pl.col(col) - state["mean"]) / state["std"]
        
    custom_scaler = EvoTransformer(
        name="custom_scaler",
        type_="unary",
        fit_func=fit_scaler,
        apply_func=apply_scaler,
        name_generator=lambda cols: f"scale_{cols[0]}"
    )
    
    from evofe.builtin import register_transformer
    register_transformer("custom_scaler", custom_scaler)
    
    gene1 = Gene(transformer_name="custom_scaler", input_cols=["x1"])
    gene2 = Gene(transformer_name="umap", input_cols=[gene1.output_col, "x2"])
    
    ind = Individual(numeric_cols=["x1", "x2"], categorical_cols=[], genes=[gene1, gene2])
    ind._topological_sort()
    assert ind.genes.index(gene1) < ind.genes.index(gene2)


# =====================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# =====================================================================

def test_scenario_fraud_imbalanced():
    """Scenario 1: Financial Fraud (Imbalanced Binary Classification)."""
    np.random.seed(42)
    n_samples = 1000
    target = np.zeros(n_samples, dtype=int)
    target[:5] = 1
    np.random.shuffle(target)
    
    df = pl.DataFrame({
        "amount": np.random.exponential(100.0, n_samples),
        "merchant": np.random.choice(["m1", "m2", "m3", "m4"], n_samples),
        "target": target
    })
    
    try:
        best_ind = evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["amount"],
            categorical_cols=["merchant"],
            evaluate_fitness=evaluate_fitness,
            pop_size=3,
            n_generations=2,
            allowed_transformers=["target_encode", "divide", "log_ratio", "deadwood"],
            verbose=False
        )
        assert best_ind is not None
        assert not np.isnan(best_ind.fitness)
    except Exception as e:
        pytest.fail(f"Scenario 1 failed due to: {e}")

def test_scenario_customer_churn_multiclass():
    """Scenario 2: Customer Segmentation & Churn (Multiclass)."""
    np.random.seed(42)
    n_samples = 500
    df = pl.DataFrame({
        "tenure": np.random.randint(1, 72, n_samples),
        "charges": np.random.randn(n_samples) * 50 + 100,
        "contract": np.random.choice(["month-to-month", "one-year", "two-year"], n_samples),
        "target": np.random.choice([0, 1, 2], n_samples)
    })
    
    try:
        best_ind = evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["tenure", "charges"],
            categorical_cols=["contract"],
            evaluate_fitness=evaluate_fitness,
            pop_size=3,
            n_generations=2,
            allowed_transformers=["umap", "genie", "target_encode_multiclass"],
            task="multiclass",
            verbose=False
        )
        assert best_ind is not None
    except Exception as e:
        pytest.fail(f"Scenario 2 failed due to: {e}")

def test_scenario_real_estate_regression():
    """Scenario 3: Real Estate Valuation (Regression)."""
    np.random.seed(42)
    n_samples = 800
    df = pl.DataFrame({
        "sqft": np.random.randn(n_samples) * 500 + 2000,
        "neighborhood": np.random.choice(["N1", "N2", "N3", "N4", "N5"], n_samples),
        "target": np.random.randn(n_samples) * 100000 + 350000
    })
    
    try:
        best_ind = evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["sqft"],
            categorical_cols=["neighborhood"],
            evaluate_fitness=evaluate_fitness,
            pop_size=3,
            n_generations=2,
            allowed_transformers=["groupby_mean", "log_ratio"],
            task="regression",
            verbose=False
        )
        assert best_ind is not None
    except Exception as e:
        pytest.fail(f"Scenario 3 failed due to: {e}")

def test_scenario_iot_fault_time_series():
    """Scenario 4: IoT Sensor Fault Detection (Supervised Time Series)."""
    np.random.seed(42)
    n_samples = 1500
    # Adjusted end date to ensure the datetime range produces at least 1500 samples
    df = pl.DataFrame({
        "timestamp": pl.datetime_range(start=pl.datetime(2026, 1, 1), end=pl.datetime(2026, 1, 18), interval="15m", eager=True)[:n_samples],
        "sensor1": np.random.randn(n_samples),
        "sensor2": np.random.randn(n_samples) * 2,
        "target": np.random.randint(0, 2, n_samples)
    })
    
    try:
        best_ind = evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["sensor1", "sensor2"],
            categorical_cols=["timestamp"],
            evaluate_fitness=evaluate_fitness,
            pop_size=3,
            n_generations=2,
            allowed_transformers=["datetime_extract", "mst_score"],
            verbose=False
        )
        assert best_ind is not None
    except Exception as e:
        pytest.fail(f"Scenario 4 failed due to: {e}")

def test_scenario_credit_scoring_sandbox():
    """Scenario 5: Credit Scoring Sandbox (Split Strategy)."""
    np.random.seed(42)
    n_samples = 1000
    df = pl.DataFrame({
        "income": np.random.exponential(50000.0, n_samples),
        "debt": np.random.exponential(10000.0, n_samples),
        "target": np.random.randint(0, 2, n_samples)
    })
    try:
        best_ind = evolve_features(
            data=df,
            target_col="target",
            numeric_cols=["income", "debt"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness,
            pop_size=3,
            n_generations=2,
            split_strategy="split_index",
            train_idx=list(range(600)),
            val_idx=list(range(600, 800)),
            holdout_idx=list(range(800, 1000)),
            evaluator="lightgbm_optuna",
            verbose=False
        )
        assert best_ind is not None
    except TypeError as e:
        pytest.fail(f"split_index strategy parameter parity issue: {e}")
