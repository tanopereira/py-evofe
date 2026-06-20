"""
evoFE — Evolutionary Feature Engineering (Python)
==================================================

Quick start
-----------
    from evofe import EvoFE

    evo = EvoFE(task="multiclass", evaluator="xgboost")
    evo.fit(df_train, y_train)
    df_enriched = evo.transform(df_test)
    preds = evo.predict(df_test)

Lower-level API
---------------
    from evofe.evolution import evolve_features
    from evofe.evaluation import evaluate_fitness, apply_individual
    from evofe.evaluation.tuning import make_tunable
"""

from .transformers import EvoTransformer
from .estimator import EvoFE
from .evolution.engine import evolve_features, EvoRecipe
from .evaluation.cv import evaluate_fitness, apply_individual
from .evaluation.tuning import make_tunable
from .builtin import register_transformer

try:
    from importlib.metadata import version as _version
    __version__ = _version("py-evofe")
except Exception:
    __version__ = "0.1.0"
__all__ = [
    "EvoFE",
    "EvoRecipe",
    "EvoTransformer",
    "evolve_features",
    "evaluate_fitness",
    "apply_individual",
    "make_tunable",
    "register_transformer",
]
