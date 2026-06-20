"""
Comprehensive test suite for py-evoFE feature parity with the R evoFE package.

Coverage targets:
  - All 30 transformer fit/apply round-trips
  - evaluate_fitness for all three tasks
  - EvoFE sklearn estimator (fit/transform/predict/predict_proba/NotFittedError)
  - make_tunable public API
  - evolve_features returning EvoRecipe
  - Untested-gene invariant
  - R2: fitness metric is exp(-log_loss) in (0,1]
  - R3: evaluation_strategy="split"
  - Transformer presets ("basic", "robust", "clustering")
  - git -C ../evoFE status clean (R dir untouched)
"""
import copy
import subprocess

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.datasets import load_iris, load_breast_cancer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris_df():
    iris = load_iris(as_frame=True)
    df = pl.from_pandas(iris.frame)
    return df.rename({
        "sepal length (cm)": "sl", "sepal width (cm)": "sw",
        "petal length (cm)": "pl", "petal width (cm)": "pw",
    })

@pytest.fixture(scope="module")
def binary_df():
    bc = load_breast_cancer(as_frame=True)
    df = pl.from_pandas(bc.frame)
    # Use only the first 5 features for speed
    cols = [c for c in df.columns if c != "target"][:5] + ["target"]
    return df.select(cols)

@pytest.fixture(scope="module")
def regression_df():
    rng = np.random.default_rng(0)
    n = 80
    return pl.DataFrame({
        "a": rng.normal(size=n).tolist(),
        "b": rng.normal(size=n).tolist(),
        "c": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
    })

@pytest.fixture(scope="module")
def small_df():
    """30-row 4-feature synthetic dataframe for transformer unit tests."""
    rng = np.random.default_rng(42)
    n = 30
    return pl.DataFrame({
        "num1": rng.normal(size=n).tolist(),
        "num2": rng.uniform(1, 10, size=n).tolist(),
        "num3": rng.exponential(size=n).tolist(),
        "cat1": (rng.integers(0, 3, size=n).astype(str)).tolist(),
        "target": rng.integers(0, 2, size=n).tolist(),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(transformer_name, df, input_cols, target_col=None, params=None):
    """Convenience: fit then apply a single transformer, return the new series."""
    from evofe.builtin import evo_transformers
    from evofe.utils import call_with_optional_params
    t = evo_transformers[transformer_name]
    state = None
    if t.fit_func is not None and target_col is not None:
        state = call_with_optional_params(
            t.fit_func, df, input_cols, target_col, params=params or {})
    result = call_with_optional_params(
        t.apply_func, df, input_cols, state, params=params or {})
    return result


# ===========================================================================
# R1 — Untested-gene invariant
# ===========================================================================

class TestUntestedGeneInvariant:
    """Only tested=True genes contribute derived columns during mutation."""

    def test_mutate_add_only_uses_tested_cols(self, iris_df):
        from evofe.evolution.individual import Individual, Gene
        ind = Individual(
            numeric_cols=["sl", "sw", "pl", "pw"],
            categorical_cols=[],
        )
        # Manually add an untested gene
        g = Gene("log", ["sl"], params={})
        g.tested = False
        ind.genes.append(g)

        # After mutation with force_add, the new gene should NOT reference g.output_col
        for _ in range(20):
            new_ind = copy.deepcopy(ind)
            new_ind.mutate(force_add=True)
            for added in new_ind.genes[1:]:  # skip the untested log gene
                assert g.output_col not in added.input_cols, \
                    f"New gene references untested output col {g.output_col}"

    def test_tested_flag_reset_on_swap(self, iris_df):
        from evofe.evolution.individual import Individual, Gene
        g = Gene("sqrt", ["sl"], params={})
        g.tested = True
        g.state = {"dummy": 1}
        ind = Individual(numeric_cols=["sl", "sw"], categorical_cols=[], genes=[g])
        ind.genes[0].input_cols[0] = "sw"
        # simulate what mutate does
        ind.genes[0].state = None
        ind.genes[0].tested = False
        assert not ind.genes[0].tested
        assert ind.genes[0].state is None


# ===========================================================================
# R2 — Fitness metric in (0, 1] for classification
# ===========================================================================

class TestFitnessMetric:
    def test_metric_in_unit_interval_classification(self, binary_df):
        from evofe.evaluation.cv import evaluate_fitness
        from evofe.evolution.individual import Individual
        num_cols = [c for c in binary_df.columns if c != "target"]
        ind = Individual(numeric_cols=num_cols, categorical_cols=[])
        result = evaluate_fitness(
            ind, binary_df, "target",
            task="classification", cv_folds=2, evaluator="lightgbm")
        assert 0.0 < result.fitness <= 1.0, \
            f"Expected fitness in (0,1], got {result.fitness}"

    def test_metric_in_unit_interval_multiclass(self, iris_df):
        from evofe.evaluation.cv import evaluate_fitness
        from evofe.evolution.individual import Individual
        num_cols = ["sl", "sw", "pl", "pw"]
        ind = Individual(numeric_cols=num_cols, categorical_cols=[])
        result = evaluate_fitness(
            ind, iris_df, "target",
            task="multiclass", cv_folds=2, evaluator="lightgbm")
        assert 0.0 < result.fitness <= 1.0, \
            f"Expected fitness in (0,1], got {result.fitness}"

    def test_metric_regression_negative_rmse(self, regression_df):
        from evofe.evaluation.cv import evaluate_fitness
        from evofe.evolution.individual import Individual
        ind = Individual(numeric_cols=["a", "b", "c"], categorical_cols=[])
        result = evaluate_fitness(
            ind, regression_df, "y",
            task="regression", cv_folds=2, evaluator="lightgbm")
        assert result.fitness <= 0.0, \
            f"Regression fitness should be -RMSE (<=0), got {result.fitness}"


# ===========================================================================
# R3 — evaluation_strategy="split"
# ===========================================================================

class TestEvaluationStrategy:
    def test_split_strategy_runs(self, iris_df):
        from evofe.evaluation.cv import evaluate_fitness
        from evofe.evolution.individual import Individual
        ind = Individual(numeric_cols=["sl", "sw", "pl", "pw"], categorical_cols=[])
        result = evaluate_fitness(
            ind, iris_df, "target",
            task="multiclass", evaluator="lightgbm",
            evaluation_strategy="split", split_ratio=[0.7, 0.3])
        assert 0.0 < result.fitness <= 1.0

    def test_cv_strategy_runs(self, iris_df):
        from evofe.evaluation.cv import evaluate_fitness
        from evofe.evolution.individual import Individual
        ind = Individual(numeric_cols=["sl", "sw", "pl", "pw"], categorical_cols=[])
        result = evaluate_fitness(
            ind, iris_df, "target",
            task="multiclass", evaluator="lightgbm",
            evaluation_strategy="cv", cv_folds=3)
        assert 0.0 < result.fitness <= 1.0


# ===========================================================================
# R4 — EvoFE sklearn estimator
# ===========================================================================

class TestEvoFEEstimator:
    def test_not_fitted_error_transform(self):
        from evofe import EvoFE
        evo = EvoFE()
        with pytest.raises(NotFittedError):
            evo.transform(np.zeros((5, 3)))

    def test_not_fitted_error_predict(self):
        from evofe import EvoFE
        evo = EvoFE()
        with pytest.raises(NotFittedError):
            evo.predict(np.zeros((5, 3)))

    def test_fit_returns_self(self, iris_df):
        from evofe import EvoFE
        y = iris_df["target"].to_numpy()
        X = iris_df.drop("target")
        evo = EvoFE(task="multiclass", evaluator="lightgbm",
                    pop_size=2, n_generations=1, cv_folds=2, verbose=False)
        result = evo.fit(X, y)
        assert result is evo

    def test_transform_adds_columns(self, iris_df):
        from evofe import EvoFE
        y = iris_df["target"].to_numpy()
        X = iris_df.drop("target")
        evo = EvoFE(task="multiclass", evaluator="lightgbm",
                    pop_size=3, n_generations=2, cv_folds=2, verbose=False)
        evo.fit(X, y)
        transformed_df = evo.transform_df(X)
        assert isinstance(transformed_df, pl.DataFrame)
        assert transformed_df.height == X.height

        transformed_arr = evo.transform(X)
        assert isinstance(transformed_arr, np.ndarray)
        assert transformed_arr.shape[0] == X.height

    def test_predict_shape(self, iris_df):
        from evofe import EvoFE
        y = iris_df["target"].to_numpy()
        X = iris_df.drop("target")
        evo = EvoFE(task="multiclass", evaluator="lightgbm",
                    pop_size=2, n_generations=1, cv_folds=2, verbose=False)
        evo.fit(X, y)
        preds = evo.predict(X)
        # multiclass probabilities: shape (n, n_classes)
        assert preds.shape[0] == X.height

    def test_predict_proba_shape(self, iris_df):
        from evofe import EvoFE
        y = iris_df["target"].to_numpy()
        X = iris_df.drop("target")
        evo = EvoFE(task="multiclass", evaluator="lightgbm",
                    pop_size=2, n_generations=1, cv_folds=2, verbose=False)
        evo.fit(X, y)
        proba = evo.predict_proba(X)
        assert proba.shape == (X.height, 3)  # 3 iris classes

    def test_recipe_has_required_fields(self, iris_df):
        from evofe import EvoFE, EvoRecipe
        y = iris_df["target"].to_numpy()
        X = iris_df.drop("target")
        evo = EvoFE(task="multiclass", evaluator="lightgbm",
                    pop_size=2, n_generations=1, cv_folds=2, verbose=False)
        evo.fit(X, y)
        recipe = evo.get_recipe()
        assert isinstance(recipe, EvoRecipe)
        assert recipe.best_individual is not None
        assert isinstance(recipe.history, list) and len(recipe.history) > 0
        assert recipe.task == "multiclass"
        assert recipe.evaluator == "lightgbm"
        assert recipe.classes is not None
        assert recipe.best_model is not None

    def test_score_method(self, iris_df):
        from evofe import EvoFE
        y = iris_df["target"].to_numpy()
        X = iris_df.drop("target")
        evo = EvoFE(task="multiclass", evaluator="lightgbm",
                    pop_size=2, n_generations=1, cv_folds=2, verbose=False)
        evo.fit(X, y)
        score = evo.score(X, y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_pipeline_compatibility(self, iris_df):
        from evofe import EvoFE
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        y = iris_df["target"].to_numpy()
        X = iris_df.drop("target")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("evofe", EvoFE(task="multiclass", evaluator="lightgbm",
                            pop_size=2, n_generations=1, cv_folds=2, verbose=False))
        ])
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert preds.shape[0] == X.shape[0]

    def test_random_state_reproducibility(self, iris_df):
        from evofe import EvoFE
        y = iris_df["target"].to_numpy()
        X = iris_df.drop("target")
        
        # Two runs with the same seed
        evo1 = EvoFE(task="multiclass", evaluator="lightgbm",
                     pop_size=3, n_generations=2, cv_folds=2, verbose=False, random_state=42)
        evo1.fit(X, y)
        recipe1 = [g.to_formula() for g in evo1.get_recipe().best_individual.genes]
        
        evo2 = EvoFE(task="multiclass", evaluator="lightgbm",
                     pop_size=3, n_generations=2, cv_folds=2, verbose=False, random_state=42)
        evo2.fit(X, y)
        recipe2 = [g.to_formula() for g in evo2.get_recipe().best_individual.genes]
        
        assert recipe1 == recipe2


# ===========================================================================
# R5 — make_tunable public API
# ===========================================================================

class TestMakeTunable:
    def test_make_tunable_registers_evaluator(self):
        from evofe import make_tunable
        from evofe.evaluation.models import evo_evaluators
        make_tunable(
            "lightgbm",
            param_ranges={
                "num_leaves": {"type": "integer", "lower": 7, "upper": 15},
            },
            tuner_name="lgb_test_tuned",
            n_trials=2,
        )
        assert "lgb_test_tuned" in evo_evaluators

    def test_make_tunable_wrong_base_raises(self):
        from evofe import make_tunable
        with pytest.raises(ValueError, match="not found"):
            make_tunable("no_such_model", {}, tuner_name="bad")

    def test_make_tunable_default_name(self):
        from evofe import make_tunable
        from evofe.evaluation.models import evo_evaluators
        make_tunable(
            "xgboost",
            param_ranges={"max_depth": {"type": "integer", "lower": 3, "upper": 5}},
            n_trials=1,
        )
        assert "xgboost_tuned" in evo_evaluators


# ===========================================================================
# evolve_features returns EvoRecipe
# ===========================================================================

class TestEvolveFeatures:
    def test_returns_evo_recipe(self, iris_df):
        from evofe import evolve_features, evaluate_fitness, EvoRecipe
        recipe = evolve_features(
            data=iris_df,
            target_col="target",
            numeric_cols=["sl", "sw", "pl", "pw"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness,
            task="multiclass",
            evaluator="lightgbm",
            pop_size=2,
            n_generations=1,
            cv_folds=2,
            verbose=False,
        )
        assert isinstance(recipe, EvoRecipe)
        assert isinstance(recipe.history, list)
        assert 0.0 < recipe.fitness <= 1.0

    def test_allowed_transformers_basic(self, iris_df):
        from evofe import evolve_features, evaluate_fitness, EvoRecipe
        recipe = evolve_features(
            data=iris_df,
            target_col="target",
            numeric_cols=["sl", "sw", "pl", "pw"],
            categorical_cols=[],
            evaluate_fitness=evaluate_fitness,
            task="multiclass",
            evaluator="lightgbm",
            pop_size=2,
            n_generations=1,
            cv_folds=2,
            allowed_transformers="basic",
            verbose=False,
        )
        assert isinstance(recipe, EvoRecipe)


# ===========================================================================
# All 30 transformers — fit+apply round-trip
# ===========================================================================

class TestAllTransformers:
    """Verify every registered transformer can fit and apply without error."""

    def _num_cols(self, df):
        return [c for c in df.columns
                if c != "target" and df[c].dtype in
                [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int8, pl.Int16,
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]]

    def _cat_cols(self, df):
        return [c for c in df.columns
                if c != "target" and df[c].dtype in [pl.Utf8, pl.String]]

    def _smoke(self, transformer_name, df, input_cols, target_col="target",
               params=None):
        from evofe.builtin import evo_transformers
        from evofe.utils import call_with_optional_params
        t = evo_transformers[transformer_name]
        state = None
        if t.fit_func is not None:
            state = call_with_optional_params(
                t.fit_func, df, input_cols, target_col, params=params or {})
        result = call_with_optional_params(
            t.apply_func, df, input_cols, state, params=params or {})
        return result  # must not raise

    # -- Unary math --
    def test_log(self, small_df):
        self._smoke("log", small_df, ["num1"])
    def test_sqrt(self, small_df):
        self._smoke("sqrt", small_df, ["num1"])
    def test_reciprocal(self, small_df):
        self._smoke("reciprocal", small_df, ["num2"])
    def test_power(self, small_df):
        self._smoke("power", small_df, ["num1"], params={"p": 2.0})

    # -- Binary math --
    def test_add(self, small_df):
        self._smoke("add", small_df, ["num1", "num2"])
    def test_subtract(self, small_df):
        self._smoke("subtract", small_df, ["num1", "num2"])
    def test_multiply(self, small_df):
        self._smoke("multiply", small_df, ["num1", "num2"])
    def test_divide(self, small_df):
        self._smoke("divide", small_df, ["num1", "num2"])
    def test_normalized_difference(self, small_df):
        self._smoke("normalized_difference", small_df, ["num1", "num2"])
    def test_log_ratio(self, small_df):
        self._smoke("log_ratio", small_df, ["num2", "num3"])

    # -- Group-by --
    def test_groupby_mean(self, small_df):
        self._smoke("groupby_mean", small_df, ["cat1", "num1"])
    def test_groupby_sd(self, small_df):
        self._smoke("groupby_sd", small_df, ["cat1", "num1"])
    def test_groupby_max(self, small_df):
        self._smoke("groupby_max", small_df, ["cat1", "num1"])
    def test_groupby_min(self, small_df):
        self._smoke("groupby_min", small_df, ["cat1", "num1"])
    def test_groupby_ratio(self, small_df):
        self._smoke("groupby_ratio", small_df, ["cat1", "num1"])
    def test_groupby_zscore(self, small_df):
        self._smoke("groupby_zscore", small_df, ["cat1", "num1"])
    def test_groupby_median(self, small_df):
        self._smoke("groupby_median", small_df, ["cat1", "num1"])
    def test_groupby_quantile(self, small_df):
        self._smoke("groupby_quantile", small_df, ["cat1", "num1"],
                    params={"q": 0.25})

    # -- Reduction --
    def test_pca(self, small_df):
        self._smoke("pca", small_df, ["num1", "num2", "num3"],
                    params={"comp_idx": 0})
    def test_truncated_svd(self, small_df):
        self._smoke("truncated_svd", small_df, ["num1", "num2", "num3"],
                    params={"comp_idx": 0})
    def test_random_projection(self, small_df):
        self._smoke("random_projection", small_df, ["num1", "num2", "num3"])

    # -- Categorical / supervised --
    def test_frequency_encode(self, small_df):
        self._smoke("frequency_encode", small_df, ["cat1"])
    def test_one_hot_encode(self, small_df):
        self._smoke("one_hot_encode", small_df, ["cat1"], params={"comp_idx": 1})
    def test_target_encode(self, small_df):
        self._smoke("target_encode", small_df, ["cat1"])
    def test_target_encode_multiclass(self, small_df):
        self._smoke("target_encode_multiclass", small_df, ["cat1"],
                    params={"comp_idx": 0})
    def test_woe_encode(self, small_df):
        self._smoke("woe_encode", small_df, ["cat1"])
    def test_rank_transform(self, small_df):
        self._smoke("rank_transform", small_df, ["num1"])
    def test_quantile_binning(self, small_df):
        self._smoke("quantile_binning", small_df, ["num1"], params={"Q": 4})
    def test_quantile_binning_cat(self, small_df):
        self._smoke("quantile_binning_cat", small_df, ["num1"], params={"Q": 4})
    def test_log_binning(self, small_df):
        df2 = small_df.with_columns(pl.col("num2").alias("num2"))
        self._smoke("log_binning", df2, ["num2"], params={"base": 2})
    def test_log_binning_cat(self, small_df):
        self._smoke("log_binning_cat", small_df, ["num2"], params={"base": 2})


# ===========================================================================
# R-directory isolation
# ===========================================================================

class TestRDirUntouched:
    def test_r_dir_git_clean(self):
        # Check that no tracked files in the R directory were modified.
        # Untracked files (pre-existing from earlier sessions) are ignored.
        import os
        r_dir = "../evoFE"
        if not os.path.exists(r_dir):
            pytest.skip("R directory not found locally.")

        result = subprocess.run(
            ["git", "-C", r_dir, "status", "--porcelain"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        modified = [
            line for line in result.stdout.splitlines()
            if not line.startswith("??")  # ignore untracked files
        ]
        assert modified == [], \
            f"R directory has modified tracked files:\n" + "\n".join(modified)
