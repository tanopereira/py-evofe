"""
py-evoFE — Sklearn-Style Usage Example
=======================================

EvoFE is a sklearn-compatible estimator that automatically discovers and
engineers the best features for your dataset using an evolutionary algorithm.

It follows the standard sklearn pattern:

    evo = EvoFE(...)          # configure
    evo.fit(X_train, y_train) # run evolution → finds best feature recipe
    evo.transform(X_test)     # apply recipe → numpy array for sklearn Pipelines
    evo.transform_df(X_test)  # apply recipe → enriched polars DataFrame
    evo.predict(X_test)       # apply recipe + model → predictions
    evo.predict_proba(X_test) # class probabilities (classification only)

EvoFE also plugs into sklearn Pipeline, cross_val_score, GridSearchCV, etc.
"""

import numpy as np
import polars as pl
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evofe import EvoFE, make_tunable


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic usage — fit / transform / predict
# ─────────────────────────────────────────────────────────────────────────────
def example_basic():
    print("=" * 60)
    print("1. BASIC USAGE — fit / transform / predict")
    print("=" * 60)

    # Load the Iris dataset as a Polars DataFrame
    iris = load_iris(as_frame=True)
    df = pl.from_pandas(iris.frame).rename({
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)":  "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)":  "petal_width",
    })

    X = df.drop("target")
    y = df["target"].to_numpy()

    # Split into train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y, test_size=0.25, random_state=42, stratify=y
    )
    X_train_df = pl.DataFrame(X_train, schema=X.columns)
    X_test_df  = pl.DataFrame(X_test,  schema=X.columns)

    # ── Create and configure EvoFE ──────────────────────────────────────────
    evo = EvoFE(
        task="multiclass",       # "classification" | "multiclass" | "regression"
        evaluator="xgboost",     # "lightgbm" | "xgboost" | any registered name
        pop_size=15,              # population size
        n_generations=100,         # max evolutionary generations
        cv_folds=3,              # cross-validation folds per fitness evaluation
        evaluation_strategy="split",# "cv" (default) or "split" for large datasets
        allowed_transformers="all",  # "all" | "basic" | "robust" | "clustering"
        complexity_penalty=0.001,    # penalise long recipes (encourages parsimony)
        metric="default",        # "default" | "auc" | "f1" | "mae"
        early_stopping_rounds=5,
        verbose=True,
        random_state=42,         # deterministic reproducibility seed
    )

    # ── fit(X, y) — runs evolution, finds best feature recipe ───────────────
    print("\n→ Running evolution...")
    evo.fit(X_train_df, y_train)

    recipe = evo.get_recipe()
    print(f"\nBest Fitness (exp(-log_loss)): {recipe.fitness:.4f}")
    print(f"Genes in best recipe:          {len(recipe.genes)}")
    for gene in recipe.genes:
        print(f"  • {gene.to_formula()}  →  {gene.output_col}")

    # ── transform_df(X) — apply recipe, returns enriched DataFrame ─────────
    X_test_enriched = evo.transform_df(X_test_df)
    new_cols = [c for c in X_test_enriched.columns if c not in X_test_df.columns]
    print(f"\ntransform_df(): added {len(new_cols)} new column(s): {new_cols}")
    print(X_test_enriched.head(3))

    # ── transform(X) — returns a 2D numpy array for sklearn Pipelines ──────
    X_test_arr = evo.transform(X_test_df)
    print(f"\ntransform() returns a numpy array of shape: {X_test_arr.shape}")

    # ── predict_proba(X) — class probability matrix ─────────────────────────
    proba = evo.predict_proba(X_test_df)
    print(f"\npredict_proba() shape: {proba.shape}  (samples × classes)")
    print("First 3 rows:")
    print(proba[:3].round(3))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Binary classification with EvoFE
# ─────────────────────────────────────────────────────────────────────────────
def example_binary():
    print("\n" + "=" * 60)
    print("2. BINARY CLASSIFICATION")
    print("=" * 60)

    bc = load_breast_cancer(as_frame=True)
    feature_cols = bc.feature_names[:8].tolist()   # first 8 features for speed
    df = pl.from_pandas(bc.frame[feature_cols + ["target"]])

    X = df.drop("target")
    y = df["target"].to_numpy()

    evo = EvoFE(
        task="classification",
        evaluator="lightgbm",
        pop_size=4,
        n_generations=2,
        cv_folds=3,
        allowed_transformers="basic",   # only simple transformers
        verbose=False,
    )
    evo.fit(X, y)

    proba = evo.predict_proba(X)
    print(f"Fitness (exp(-log_loss)): {evo.get_recipe().fitness:.4f}")
    print(f"predict_proba shape:      {proba.shape}  → columns: [P(0), P(1)]")
    print(f"Sample probabilities:     {proba[:3].round(3)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plugging into a sklearn Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def example_pipeline():
    print("\n" + "=" * 60)
    print("3. SKLEARN PIPELINE USAGE")
    print("=" * 60)

    iris = load_iris(as_frame=True)
    X_np = iris.data
    y    = iris.target

    # EvoFE accepts numpy arrays too — column names default to f0, f1, ...
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("evo",    EvoFE(
            task="multiclass",
            evaluator="lightgbm",
            pop_size=3,
            n_generations=2,
            cv_folds=2,
            verbose=False,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y, test_size=0.2, random_state=0, stratify=y)

    pipe.fit(X_train, y_train)

    # Pipeline.predict runs StandardScaler then EvoFE.predict
    proba = pipe.predict_proba(X_test)
    print(f"Pipeline predict_proba shape: {proba.shape}")
    print(f"EvoFE fitness: {pipe.named_steps['evo'].get_recipe().fitness:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Registering a tuned evaluator with make_tunable
# ─────────────────────────────────────────────────────────────────────────────
def example_make_tunable():
    print("\n" + "=" * 60)
    print("4. CUSTOM TUNED EVALUATOR (make_tunable)")
    print("=" * 60)

    # make_tunable() wraps any registered base evaluator in an Optuna loop
    # and registers the result under a new name — mirrors R's make_tunable().
    make_tunable(
        base_model_name="lightgbm",
        param_ranges={
            "num_leaves":    {"type": "integer", "lower": 7,    "upper": 31},
            "learning_rate": {"type": "numeric", "lower": 0.05, "upper": 0.2},
        },
        tuner_name="lgb_tuned",   # registered under this name
        n_trials=3,               # Optuna trials per fitness call
    )

    iris = load_iris(as_frame=True)
    X = pl.from_pandas(iris.frame).drop("target")
    y = iris.target

    evo = EvoFE(
        task="multiclass",
        evaluator="lgb_tuned",   # ← use the tuned evaluator
        pop_size=10,
        n_generations=10,
        cv_folds=3,
        verbose=True,
    )
    evo.fit(X, y)
    print(f"Fitness with tuned LGB evaluator: {evo.get_recipe().fitness:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Inspecting the recipe
# ─────────────────────────────────────────────────────────────────────────────
def example_inspect_recipe():
    print("\n" + "=" * 60)
    print("5. INSPECTING THE EVOLVED RECIPE")
    print("=" * 60)

    iris = load_iris(as_frame=True)
    X = pl.from_pandas(iris.frame).drop("target")
    y = iris.target

    evo = EvoFE(task="multiclass", evaluator="lightgbm",
                pop_size=4, n_generations=2, cv_folds=2, verbose=False)
    evo.fit(X, y)

    recipe = evo.get_recipe()

    print(f"task:              {recipe.task}")
    print(f"evaluator:         {recipe.evaluator}")
    print(f"classes:           {recipe.classes}")
    print(f"best fitness:      {recipe.fitness:.4f}")
    print(f"individuals tried: {len(recipe.history)}")
    print(f"best model type:   {type(recipe.best_model).__name__}")
    print("\nGenes in best recipe:")
    for gene in recipe.best_individual.genes:
        imp = recipe.best_individual.importances.get(gene.output_col, 0.0)
        print(f"  {gene.to_formula():<40} importance={imp:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Usage with Temperature Scaled Refinement (ts-refinement)
# ─────────────────────────────────────────────────────────────────────────────
def example_ts_refinement():
    print("\n" + "=" * 60)
    print("6. TEMPERATURE SCALED REFINEMENT (ts-refinement) METRIC")
    print("=" * 60)

    # Load breast cancer dataset for binary classification
    bc = load_breast_cancer(as_frame=True)
    feature_cols = bc.feature_names[:8].tolist()   # first 8 features for speed
    df = pl.from_pandas(bc.frame[feature_cols + ["target"]])

    X = df.drop("target")
    y = df["target"].to_numpy()

    # Configure EvoFE with the custom metric "ts_refinement"
    evo = EvoFE(
        task="classification",
        evaluator="lightgbm",
        metric="ts_refinement",   # Use Temperature Scaled Refinement metric
        pop_size=4,
        n_generations=2,
        cv_folds=3,
        allowed_transformers="basic",
        verbose=False,
    )
    evo.fit(X, y)

    print(f"Best Fitness (ts_refinement score): {evo.get_recipe().fitness:.4f}")
    
    # We can also import the metric directly for manual evaluation
    from evofe.evaluation.metrics import ts_refinement
    proba = evo.predict_proba(X)
    # class probabilities for target=1 (second column)
    score = ts_refinement(y, proba[:, 1], task="classification")
    print(f"Calculated ts_refinement score on training data: {score:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_basic()
    example_binary()
    example_pipeline()
    example_make_tunable()
    example_inspect_recipe()
    example_ts_refinement()
    print("\n✓ All examples completed successfully.")
