# TEST_READY.md - py-evoFE Test Readiness Report

This report documents the status of the E2E parity test suite, outlining the test command, coverage summary, and a detailed feature-by-feature parity checklist.

---

## 1. Test Command

To execute the test suite, run the following command from the root of the `py-evoFE` project:

```bash
pytest tests/test_e2e_parity.py
```

Or run all tests in the package:

```bash
pytest
```

---

## 2. Coverage Summary

* **Total Test Cases Executed:** 55
* **Passed:** 19
* **Failed:** 36
* **Status:** Expected Failures (implementation of missing features is planned for subsequent milestones).

---

## 3. Feature Parity Checklist

The following checklist details which tests pass and fail, highlighting the specific architectural gaps between `evoFE` (R) and `py-evoFE` (Python):

### Tier 1: Feature Coverage

#### Feature 1: Evolution Engine (`evolve_features`)
- [x] **Passed** | `test_register_custom_evaluator` — Dynamic registration of custom evaluators works.
- [ ] **Failed** | `test_e2e_binary_classification_cv` — `TypeError: evolve_features() got an unexpected keyword argument 'cv_folds'`.
- [ ] **Failed** | `test_e2e_regression_split` — `TypeError: mean_squared_error() got an unexpected keyword argument 'squared'` (compatibility bug in `cv.py` with newer scikit-learn).
- [ ] **Failed** | `test_e2e_multiclass_model_all_final` — `TypeError: evolve_features() got an unexpected keyword argument 'model_all_final_genes'`.
- [ ] **Failed** | `test_dynamic_population_stagnation` — `TypeError: evolve_features() got an unexpected keyword argument 'stagnation_limit'`.
- [ ] **Failed** | `test_restricted_allowed_transformers` — `TypeError: evolve_features() got an unexpected keyword argument 'allowed_transformers'`.

#### Feature 2: Evaluators & Tuning
- [x] **Passed** | `test_shap_importance_extraction` — Global SHAP importance calculations parsed correctly.
- [ ] **Failed** | `test_make_tunable_registration` — `ImportError: cannot import name 'make_tunable'`.
- [ ] **Failed** | `test_ts_refinement_binary` — Temperature Scaled Refinement metric is missing.
- [ ] **Failed** | `test_ts_refinement_multiclass` — Temperature Scaled Refinement metric is missing.

#### Features 3 & 4: Clustering (`genie`, `lumbermark`)
- [ ] **Failed** | `test_clustering_genie_lumbermark_registered` — Agglomerative Genie/Lumbermark cluster estimators not registered.
- [ ] **Failed** | `test_clustering_output_types` — Missing estimators.
- [ ] **Failed** | `test_clustering_caching_invariant` — Cache lookup logic not reached.
- [ ] **Failed** | `test_clustering_downsampling` — Missing estimators.
- [ ] **Failed** | `test_clustering_missing_value_handling` — Missing estimators.

#### Feature 5: UMAP Reduction
- [x] **Passed** | `test_umap_coord_projection` — Coordinate calculation is functional.
- [x] **Passed** | `test_umap_small_dataset_neighbors` — Automatic neighbor truncation is functional.
- [ ] **Failed** | `test_umap_component_clamping` — Clamping not executed correctly or prunes due to naming hash discrepancies.
- [ ] **Failed** | `test_umap_prefix_parity` — Uses `"umap_"` prefix instead of R's `"ump_"`.
- [ ] **Failed** | `test_umap_caching` — Fit-caching invariant failed (not cached correctly across component columns).

#### Features 6 & 7: Anomaly Detection (`mst_score`, `deadwood`)
- [ ] **Failed** | `test_anomaly_score_creation` — `mst_score` and `deadwood` are not registered.
- [ ] **Failed** | `test_anomaly_min_columns_error` — `mst_score` and `deadwood` are not registered.

#### Missing/Modified Parity Checklists
- [ ] **Failed** | `test_missing_transformers_presence` — `groupby_quantile`, `target_encode_multiclass`, `rank_transform`, `datetime_extract`, `quantile_binning`, and `log_binning` are missing.
- [ ] **Failed** | `test_power_dynamic_parameter` — `power` ignores the parameter `p` passed on the Gene and hardcodes squaring.
- [ ] **Failed** | `test_pca_centering_and_scaling_parity` — PCA centers but does not scale columns; ignores `comp_idx` and hardcodes component 0.
- [ ] **Failed** | `test_column_naming_hash_discrepancy` — `EvoTransformer.transform` generates names without hashing params, causing KeyErrors/mismatches with `Gene.output_col`.

---

### Tier 2: Boundary & Corner Cases
- [x] **Passed** | `test_boundary_single_class_target` — Raises ValueError as expected.
- [x] **Passed** | `test_boundary_nonexistent_target_col` — Raises ValueError as expected.
- [x] **Passed** | `test_boundary_unregistered_evaluator` — Raises ValueError as expected.
- [x] **Passed** | `test_boundary_k_exceeds_rows` — Standard fallbacks clamp/handle k.
- [x] **Passed** | `test_boundary_single_row_stateful_transform` — Applies target encoding to single rows.
- [x] **Passed** | `test_boundary_all_missing_values` — Log transformer handles NAs safely.
- [x] **Passed** | `test_boundary_zero_variance` — Constant check successfully prunes zero-variance columns.
- [ ] **Failed** | `test_boundary_empty_inputs` — Raises Polars `ColumnNotFoundError` instead of a custom `ValueError`.
- [ ] **Failed** | `test_boundary_empty_allowed_transformers` — `TypeError: evolve_features() got an unexpected keyword argument 'allowed_transformers'`.
- [ ] **Failed** | `test_boundary_incompatible_metric_task` — `TypeError: evolve_features() got an unexpected keyword argument 'metric'`.
- [ ] **Failed** | `test_boundary_tuning_small_dataset` — `TypeError: got an unexpected keyword argument 'cv_folds'` (or other signature mismatches in evolve).
- [ ] **Failed** | `test_boundary_class_imbalance_optuna` — `TypeError: got an unexpected keyword argument 'cv_folds'` (or other signature mismatches in evolve).
- [ ] **Failed** | `test_boundary_ts_refinement_nan_inf` — ts_refinement missing.

---

### Tier 3: Cross-Feature Combinations
- [x] **Passed** | `test_chaining_umap_genie` — Topological sort correctly orders UMAP before Genie.
- [x] **Passed** | `test_chaining_lumbermark_groupby` — Topological sort correctly orders clustering before group-by mean.
- [x] **Passed** | `test_custom_transformer_chaining` — Custom transformer successfully registers and chains.
- [ ] **Failed** | `test_ts_refinement_stagnation_expansion` — Population stagnation parameters are not supported.

---

### Tier 4: Real-World Scenarios
- [ ] **Failed** | `test_scenario_fraud_imbalanced` — Missing allowed_transformers and deadwood.
- [ ] **Failed** | `test_scenario_customer_churn_multiclass` — Missing UMAP prefix parity, genie, and target_encode_multiclass.
- [ ] **Failed** | `test_scenario_real_estate_regression` — Missing allowed_transformers and groupby_mean caching.
- [ ] **Failed** | `test_scenario_iot_fault_time_series` — Missing allowed_transformers, datetime_extract, and mst_score.
- [ ] **Failed** | `test_scenario_credit_scoring_sandbox` — `TypeError: evolve_features() got an unexpected keyword argument 'split_strategy'`.
