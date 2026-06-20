# Project: py-evoFE Feature Parity with R's evoFE

## Constraints
- **Strict Working Directory Constraint**: All modifications, source code changes, and test additions must be made strictly under `/home/gustavo/git/py-evoFE`.
- **Read-Only Reference Constraint**: The R directory `../evoFE` is strictly read-only. Do not write or modify any files (including code, tests, documentation, or metadata) under `../evoFE`.

## Architecture
`py-evoFE` is a Python port of the evolutionary feature-engineering package `evoFE`. It uses `polars` for fast expression-based feature generation, supports LightGBM and XGBoost models, and runs genetic programming to evolve optimal feature engineering recipes (individuals) represented as lists of feature transformation genes.

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | E2E Test Suite | Build requirement-driven E2E test suite (Tiers 1-4) in Python verifying all R-equivalent features. Outputs `TEST_READY.md`. | None | DONE |
| 2 | Feature Transformers | Port all missing or incomplete feature transformers from R `transformers.R` to Python (e.g., groupby_quantile, target_encode_multiclass, genie, lumbermark, deadwood, mst_score, power, rank_transform, datetime_extract, quantile/log binning). | M1 | DONE |
| 3 | Evaluation, Metrics & Tuning | Port `ts_refinement` metric, expand Model Registry for LightGBM/XGBoost objectives, implement Optuna-based `make_tunable` parity, and align chaining/pruning logic. | M2 | DONE |
| 4 | Final Integration & Hardening | Run all tests (Tiers 1-4), check `example.py` regression, generate Tier 5 adversarial tests, and finalize package. | M3 | DONE |

## Interface Contracts
### `EvoTransformer` (in `transformers.py`)
- Python class wrapping fit/transform logic compatible with scikit-learn.
- State calculated during `fit(X, y=None, input_cols=None, target_col=None)`.
- Applied using `transform(X)`.

### Evaluator Registry (in `models.py` and `tuning.py`)
- `register_evaluator(name, train_func, predict_func=None)`
- Base models: `lightgbm`, `xgboost`.
- Tuned models registered dynamically via `make_tunable(base_model_name, param_ranges, tuner_name=None)`.

## Code Layout
- `src/evofe/transformers.py`: Base class `EvoTransformer` definition.
- `src/evofe/utils.py`: Hashing and naming helpers.
- `src/evofe/builtin/`: Transformer implementations:
  - `math.py`: Stateless unary and binary mathematical transformers.
  - `grouping.py`: Stateful group-by aggregations.
  - `reduction.py`: Stateful multivariate dimensionality reductions.
  - `clustering.py`: Stateful manifold/graph/clustering estimators.
  - `supervised.py`: Stateful categorical/supervised encodings.
- `src/evofe/evaluation/`:
  - `cv.py`: Individual recipe application, cross-validation, and metrics.
  - `models.py`: Model training functions and SHAP importances.
  - `tuning.py`: Hyperparameter tuning logic.
- `src/evofe/evolution/`: Core genetic algorithm logic.
- `tests/`: pytest suite.
