# TEST_INFRA.md - py-evoFE Test Infrastructure and E2E Parity Plan

This document outlines the Python test philosophy, feature inventory, 4-Tier E2E test suite specifications, and pass/fail criteria for the `py-evoFE` package, designed to align with and verify parity against R's `evoFE`.

---

## 1. Test Philosophy

The `py-evoFE` test suite is designed as a **dual-track E2E verification** system. Rather than just verifying isolated units, the test suite actively tests the entire genetic programming pipeline (representation, fitness evaluation, stateful fit/apply, tuning, crossover/mutation, and execution under boundary pressure).

The testing philosophy is organized into a **4-Tier Hierarchy**:
- **Tier 1 (Feature Coverage):** Individual tests verifying correctness and R-parity for every feature transformer, evaluator, tuner, and custom metric.
- **Tier 2 (Boundary & Corner Cases):** Tests verifying robustness under extreme conditions (empty inputs, single-class targets, NaNs/Infs, invalid configurations, zero variance, and small datasets).
- **Tier 3 (Cross-Feature Combinations):** Chaining tests verifying that transformers can feed into one another (e.g., UMAP + Genie), category keys can drive group-by aggregations, and populations expand dynamically during stagnation.
- **Tier 4 (Real-World Scenarios):** Full pipelines mimicking actual machine learning tasks (Fraud Classification, Customer Churn, Real Estate Regression, IoT Sensor Faults, and Credit Scoring Sandboxes).

---

## 2. Feature Inventory

The following table maps the R `evoFE` features to their corresponding Python `py-evoFE` modules and tracks parity implementation status:

| R Feature Name | Python Module | Parity / Mismatch / Status |
| :--- | :--- | :--- |
| `evolve_features()` | `evofe.evolution.engine` | **Partial Mismatch**. Missing support for `crossover`, `split_strategy` (split indices), `allowed_transformers` restriction, and stagnation-based population expansion/decay parameters. |
| `register_evaluator()` | `evofe.evaluation.models` | **Full Parity**. Supports registering and retrieving training/prediction functions. |
| `make_tunable()` | `evofe.evaluation.tuning` | **Missing**. Bayesian tuning parameter spaces are hardcoded in static wrappers instead of dynamic registration. |
| `ts_refinement` metric | `evofe.evaluation.metrics` | **Missing**. Custom Temperature Scaled Refinement metric is not ported. |
| `power` | `evofe.builtin.math` | **Partial Mismatch**. Python hardcodes power to `2` and does not respect the dynamic parameter `p` passed on the Gene. |
| `pca` | `evofe.builtin.reduction` | **Partial Mismatch**. Python only centers (no scaling) and ignores component index `comp_idx` (hardcoded to `0`). |
| `umap` | `evofe.builtin.reduction` | **Partial Mismatch**. Python uses prefix `"umap"` (R uses `"ump"`), ignores `comp_idx`, and hardcodes components. |
| `genie` / `genie_centroid_dist` | `evofe.builtin.clustering` | **Fallback Mismatch**. Falls back to standard KMeans instead of Genie hierarchical clustering. |
| `lumbermark` / `lumbermark_centroid_dist` | `evofe.builtin.clustering` | **Missing**. Falls back to KMeans instead of Ward/HDBSCAN. |
| `mst_score` | *None* | **Missing**. Minimum Spanning Tree anomaly score not ported. |
| `deadwood` | *None* | **Missing**. Isolation Forest/LOF outlier detection not ported. |
| `groupby_quantile` | *None* | **Missing**. Group-by Q1/Q3 aggregations not ported. |
| `target_encode_multiclass` | *None* | **Missing**. Multiclass target encoding not ported. |
| `rank_transform` | *None* | **Missing**. Stateful ECDF ranking not ported. |
| `datetime_extract` | *None* | **Missing**. Temporal feature extraction not ported. |
| `quantile_binning` / `log_binning` | *None* | **Missing**. Stateful binning not ported. |

---

## 3. E2E Test Suite Specification (Tiers 1-4)

The test suite is implemented in `tests/test_e2e_parity.py` and contains 55 verification test cases:

### Tier 1: Feature Coverage (>=5 test cases per feature area)
- **Feature 1: Evolution Engine (`evolve_features`)**
  - E2E Binary Classification with 5-fold CV.
  - E2E Regression with Split/CV strategy.
  - E2E Multiclass Classification (checking `model_all_final_genes = True`).
  - Dynamic population expansion & decay.
  - Allowed transformers filtering.
- **Feature 2: Evaluators & Tuning**
  - Custom evaluator registration via `register_evaluator()`.
  - Bayesian Optuna tuning via `make_tunable()`.
  - TS-Refinement metric calculation for binary classification.
  - TS-Refinement metric calculation for multiclass classification.
  - SHAP/model-native feature importances extraction.
- **Features 3 & 4: Clustering (`genie`, `lumbermark`)**
  - Presence of genie, genie_centroid_dist, lumbermark, lumbermark_centroid_dist in registry.
  - Label vs centroid distance column output types (categorical vs numeric).
  - Fit-caching invariant (single fit for multiple distance genes).
  - Downsampling verification (max_clustering_size constraints).
  - NA handling and imputation during fitting.
- **Feature 5: UMAP Reduction**
  - Coordinate projection dimensionality verification.
  - Out-of-bounds component index clamping.
  - Small dataset neighbor scaling (adjusting `n_neighbors` if rows < 15).
  - Parallel threading limit validation.
  - Fit-caching validation.
- **Features 6 & 8: Anomaly Detection (`mst_score`, `deadwood`)**
  - Anomaly score/outlier flag creation.
  - Downsampling and 1-NN projection verification.
  - Validation error if columns < 2.
  - NA safety.
  - Fit-caching validation.

### Tier 2: Boundary & Corner Cases (>=5 test cases per feature area)
- Zero-row or empty inputs to `evolve_features()` (should raise `ValueError`).
- Single-class target for classification tasks (should raise `ValueError`).
- Non-existent target column name (should raise `ValueError`).
- Empty `allowed_transformers` list (should raise `ValueError`).
- Incompatible metric and task type (e.g., Regression with AUC, should raise `ValueError`).
- Tuning with extremely small datasets (e.g., 5 rows).
- Validation fold completely missing a class label.
- TS-Refinement with NaNs, Infs, or extreme logits.
- Unregistered evaluator names (should raise `ValueError`).
- Constant/zero-variance input columns (should prune genes).
- Clustering with $k$ exceeding dataset rows.
- Single-row stateful transform application.
- Inputs containing 100% missing values (NA).

### Tier 3: Cross-Feature Combinations
- **Test 3.1: Hierarchical Chaining:** UMAP projection coordinate outputs serving as inputs to Genie clustering.
- **Test 3.2: Multi-clustering Groupby:** Using Lumbermark/Genie cluster cohort column as grouping key for groupby aggregations.
- **Test 3.3: Tuned Evaluator with All-Historical Genes:** Evolving with `lightgbm_optuna` and `model_all_historical_genes = True`.
- **Test 3.4: TS-Refinement with Dynamic Expansion:** Stagnation-based population expansion evaluated via TS-Refinement.
- **Test 3.5: Custom Transformer Chaining:** Registering custom standardization and chaining it into UMAP.

### Tier 4: Real-World Scenarios
- **Scenario 1: Financial Fraud:** 1,000 transactions, 0.5% fraud rate, restricted to division/log ratios, target encoding, and `deadwood` outlier detection.
- **Scenario 2: Customer Churn:** 500 records, 3 classes, using UMAP, Genie, and multiclass target encoding.
- **Scenario 3: Real Estate Valuation:** 800 sales, regression target, neighborhood `groupby_mean`, and square footage log ratios.
- **Scenario 4: IoT Sensor Faults:** 1,500 logs, datetime feature extraction, and `mst_score` anomalies.
- **Scenario 5: Credit Scoring Sandbox:** 1,000 applicants, explicit train/val/holdout split indices with Optuna-tuned LightGBM.

---

## 4. Verification and Pass/Fail Invariants

To pass, the Python verification run must satisfy these invariants:
1. **Zero Errors/Warnings:** The E2E tests run successfully (once implementation parity is complete).
2. **Fit-Caching Invariant:** Underlying `fit_func` must be called exactly once per unique gene representation, even if multiple component/distance columns are generated.
3. **Execution Time Threshold:** No single E2E test should exceed 45 seconds under pop_size = 3 and generations = 2.
4. **Column Naming/Hash Parity:** The hash generated during `transform` (without params) must match the hash generated during `Gene` creation (with params), avoiding downstream column mismatch errors.
5. **Centering and Scaling Invariant:** PCA/UMAP and distance calculations must scale columns (z-score) before fitting to prevent columns with high variance from dominating.
