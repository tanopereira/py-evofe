# Original User Request

## Initial Request — 2026-06-20T09:38:53+02:00

Make the Python library `py-evoFE` achieve full feature parity with the R library `evoFE`. The resulting port must be a production-grade Python package including all feature generators, model backends, tuning strategies, comprehensive documentation, and unit tests.

Working directory: `/home/gustavo/git/py-evoFE`
Integrity mode: `benchmark`

## Requirements

### R1. Complete Feature Transformer Parity
Port all built-in R feature transformers (from `transformers.R` in the R package) to the Python package. The implementation should use Polars expressions and queries where applicable for performance.
This includes:
- **Stateless Unary/Binary**: `log`, `sqrt`, `reciprocal`, `add`, `subtract`, `multiply`, `divide`, `normalized_difference`, `log_ratio`
- **Stateful Group-by**: `groupby_mean`, `groupby_sd`, `groupby_max`, `groupby_min`, `groupby_ratio`, `groupby_zscore`, `groupby_median`, `groupby_quantile`
- **Stateful Multivariate**: `pca`, `truncated_svd`, `umap`, `mst_score`, `genie`, `random_projection`, `lumbermark`, `deadwood`, `genie_centroid_dist`, `lumbermark_centroid_dist`
- **Stateful Categorical/Supervised/Other**: `frequency_encode`, `one_hot_encode`, `target_encode`, `target_encode_multiclass`, `datetime_extract`, `quantile_binning`, `log_binning`, `quantile_binning_cat`, `log_binning_cat`, `power`, `rank_transform`, `woe_encode`

### R2. Model Registry, Metrics, and Tuning Parity
Ensure Python evaluation matches R logic:
- Backends: Support LightGBM and XGBoost, with matching default parameters and objectives (regression, binary classification, multiclass classification).
- Metrics: Implement evaluation metrics corresponding to the R package metrics.
- Tuners: Support Optuna tuning equivalent to R's tuning mechanism, including cross-validation configuration.

### R3. Comprehensive Unit Tests and Documentation
- Provide unit tests verifying individual transformers, model registry, metrics, and tuning logic.
- Ensure the API is fully documented, and a reference/example notebook or vignette is included.

## Acceptance Criteria

### API and Behavior
- All transformers from the R version can be successfully instantiated and applied in Python.
- Evaluators correctly support XGBoost, LightGBM, and standard metric/CV setups.
- Chaining and pruning logic behavior matches R (e.g., topologically dropping highly correlated/constant columns).

### Quality and Verification
- A test suite with at least 80% coverage on new code runs successfully using `pytest`.
- No regression on existing `example.py` workflow.
- An example script or Jupyter notebook demonstrating all features is present.

## Follow-up — 2026-06-20T08:07:51Z

IMPORTANT: Do NOT modify or write any files in the R directory `evoFE` under `../evoFE`. You can read it as reference, but all modifications and changes must be done strictly in `/home/gustavo/git/py-evoFE`. Keep `../evoFE` completely untouched.

## Follow-up — 2026-06-20T13:28:23Z

Bring `py-evoFE` to parity with the `evoFE` R package using a scikit-learn style for the Python version. Please do not write anything in the `../evoFE` directory, which is strictly read-only.

Working directory: /home/gustavo/git/py-evoFE
Integrity mode: development

## Requirements

### R1. Missing Feature Transformers and Estimators
Port and implement all missing or partial feature transformers and clustering/anomaly estimators:
- Clustering & Anomaly: Genie (`genie`, `genie_centroid_dist`), Lumbermark (`lumbermark`, `lumbermark_centroid_dist`), Minimum Spanning Tree score (`mst_score`), and outlier detection (`deadwood`).
- Mathematical & Grouping: Respect parameter `p` in `power`, implement `groupby_quantile` (Q1/Q3 aggregations), and scale/center columns before fitting PCA/UMAP.
- Feature Engineering: Multi-class target encoding (`target_encode_multiclass`), stateful ECDF ranking (`rank_transform`), temporal feature extraction (`datetime_extract`), and stateful binning (`quantile_binning`, `log_binning`).

### R2. Tuning and Metrics
- Implement `make_tunable` in `evofe.evaluation.tuning` using Optuna to dynamically register tuned models in the evaluator registry.
- Port and implement the Temperature Scaled Refinement metric `ts_refinement` for binary and multiclass classification under `evofe.evaluation.metrics`.

### R3. Search Loop and Interface Parity
Ensure the evolution engine `evolve_features` supports and respects:
- Population parameters: `cv_folds`, `model_all_final_genes`, `stagnation_limit`, `expansion_factor`, `allowed_transformers`, `split_strategy` (e.g. `split_index`, `train_idx`, `val_idx`, `holdout_idx`).
- Crossover logic and allowed transformer filtering.

### R4. Integrity and Robustness
Ensure:
- Fit-caching invariant: `fit_func` is called exactly once per unique gene representation.
- Hash and naming parity: `EvoTransformer.transform` must produce columns whose names match `Gene.output_col` via consistent naming/hashing.
- Handled boundary cases: raise custom `ValueError` on empty inputs, zero variance columns, unregistered evaluators, or incompatible metric-task pairs.

## Acceptance Criteria

### E2E Test Suite
- [ ] Running `pytest tests/test_e2e_parity.py` executes all 55 tests successfully with zero failures.
- [ ] No regression on existing unit or integration tests in the `tests/` directory.

### Invariants Verification
- [ ] **Fit-Caching**: The base model/transformer fit runs only once for duplicate/equivalent genes in an individual.
- [ ] **Centering/Scaling**: Centering and scaling (z-score) are applied to PCA/UMAP and distance transformations prior to fitting.
- [ ] **Naming Consistency**: No `KeyError` or mismatch occurs between the output column names in DataFrame and the expected names in the genes.
- [ ] **Performance**: Run time of any single test case does not exceed 45 seconds under `pop_size = 3` and `generations = 2`.

