# py-evofe: Evolutionary Feature Engineering in Python

[![PyPI version](https://img.shields.io/pypi/v/py-evofe)](https://pypi.org/project/py-evofe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/py-evofe)](https://pypi.org/project/py-evofe/)

**py-evofe** is a Python library that uses a genetic algorithm to automatically discover, combine, and optimize feature transformations for tabular datasets. Instead of manually engineering interaction terms, ratios, or binning strategies, `py-evofe` searches the space of possible feature recipes to maximize the predictive performance of LightGBM or XGBoost models.

It implements a scikit-learn compatible interface (`fit`, `transform`, `predict`), allowing seamless integration into standard ML pipelines.

---

## Features

* **Scikit-Learn Interface:** Compatible with scikit-learn's `Pipeline`, `GridSearchCV`, and cross-validation tools.
* **Genetic Algorithm Optimization:** Searches the feature transformation space using selection, crossover, and mutation.
* **Hierarchical Chaining:** Evolved features can build on top of other proven features from previous generations (e.g., `log(ratio(x1, x2))`).
* **Stateful Transformers:** Includes PCA, SVD, UMAP, Genie Clustering, Lumbermark Clustering, and Deadwood Anomaly Detection.
* **Performance Caching:** Features are cached using matrix-hashing to avoid redundant computations (like $K$-NN search or UMAP projections) during cross-validation folds.
* **Flexible Evaluation:** Supports both Cross-Validation (`cv`) and stratified Train/Validation/Holdout Split (`split`) strategies.
* **Alternative & Custom Metrics:** Optimize for standard metrics (LogLoss, AUC, F1, MAE) or use the custom Temperature Scaled Refinement (`ts_refinement`) metric.

---

## Installation

You can install the released version of **py-evofe** from PyPI with:

```bash
pip install py-evofe
```

Or using `uv`:

```bash
uv pip install py-evofe
```

---

## Quick Start

Here is a quick example using the Breast Cancer dataset for a binary classification task:

```python
import polars as pl
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from evofe import EvoFE

# Load dataset and rename columns to be clean
bc = load_breast_cancer(as_frame=True)
feature_cols = bc.feature_names[:8].tolist()  # use first 8 features for speed
df = pl.from_pandas(bc.frame[feature_cols + ["target"]])

X = df.drop("target")
y = df["target"].to_numpy()

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(), y, test_size=0.25, random_state=42, stratify=y
)
X_train_df = pl.DataFrame(X_train, schema=X.columns)
X_test_df = pl.DataFrame(X_test, schema=X.columns)

# 1. Create and configure EvoFE
evo = EvoFE(
    task="classification",          # "classification" | "multiclass" | "regression"
    evaluator="lightgbm",          # "lightgbm" | "xgboost"
    pop_size=10,                   # population size
    n_generations=5,               # max evolutionary generations
    cv_folds=3,                    # CV folds per fitness evaluation
    verbose=True
)

# 2. Fit: Runs evolution to discover best features
evo.fit(X_train_df, y_train)

# 3. Get evolved feature recipe
recipe = evo.get_recipe()
print(f"Best Fitness (exp(-log_loss)): {recipe.fitness:.4f}")
print("Evolved genes:")
for gene in recipe.genes:
    print(f"  • {gene.to_formula()} -> {gene.output_col}")

# 4. Transform: Get numeric features array for downstream sklearn estimators
X_test_arr = evo.transform(X_test_df)
print(f"Selected features array shape: {X_test_arr.shape}")

# Alternatively, get the enriched Polars DataFrame with original + new features:
X_test_enriched = evo.transform_df(X_test_df)
print(f"Enriched DataFrame columns: {X_test_enriched.columns}")

# 5. Predict using the best evolved model
predictions = evo.predict(X_test_df)
probabilities = evo.predict_proba(X_test_df)
```

---

## Supported Transformers

| Category | Transformers |
| :--- | :--- |
| **Arithmetic & Math** | `log`, `sqrt`, `reciprocal`, `power`, `add`, `subtract`, `multiply`, `divide`, `normalized_difference`, `log_ratio` |
| **Group-by Aggregations** | `groupby_mean`, `groupby_median`, `groupby_sd`, `groupby_max`, `groupby_min`, `groupby_ratio`, `groupby_zscore`, `groupby_quantile` |
| **Encoding & Binning** | `target_encode`, `target_encode_multiclass`, `frequency_encode`, `one_hot_encode`, `quantile_binning`, `log_binning`, `rank_transform`, `datetime_extract` |
| **Dimensionality Reduction** | `pca`, `truncated_svd`, `random_projection`, `umap` |
| **Graph & Clustering** | `genie`, `genie_centroid_dist`, `lumbermark`, `lumbermark_centroid_dist`, `mst_score`, `deadwood` |

## Scikit-Learn Pipeline Integration

Since `EvoFE.transform()` returns a standard 2D numeric numpy array of selected features, you can drop it directly into standard scikit-learn Pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from evofe import EvoFE

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("evofe", EvoFE(task="classification", pop_size=10, n_generations=5, random_state=42))
])

# Fits scaler on data, then runs evolutionary feature search on scaled outputs
pipeline.fit(X_train, y_train)

# Scores the entire pipeline on test data
score = pipeline.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")
```

## Reproducibility

Pass the `random_state` parameter to `EvoFE` to ensure evolutionary search runs are deterministic and reproducible:

```python
evo = EvoFE(
    task="classification",
    random_state=42, # seeds random and numpy modules for reproducibility
    pop_size=10,
    n_generations=5
)
```

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
