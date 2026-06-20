import polars as pl
import numpy as np

import evofe.builtin  # register all transformers

from evofe.evolution import evolve_features, EvoRecipe
from evofe.evaluation import evaluate_fitness


def test_full_integration():
    np.random.seed(42)
    n_samples = 100
    df = pl.DataFrame({
        "num1": np.random.randn(n_samples),
        "num2": np.random.randn(n_samples) * 5,
        "cat1": np.random.choice(["a", "b", "c"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    })

    recipe = evolve_features(
        data=df,
        target_col="target",
        numeric_cols=["num1", "num2"],
        categorical_cols=["cat1"],
        evaluate_fitness=evaluate_fitness,
        pop_size=3,
        n_generations=2,
        initial_genes=2,
        task="classification",
        verbose=False,
    )

    # evolve_features now returns EvoRecipe
    assert isinstance(recipe, EvoRecipe)
    assert recipe.best_individual is not None
    assert len(recipe.genes) >= 0
    assert hasattr(recipe, "fitness")
    assert isinstance(recipe.history, list) and len(recipe.history) > 0

    # Fitness is exp(-log_loss): value in (0, 1], not negative
    assert 0.0 < recipe.fitness <= 1.0
    assert not np.isnan(recipe.fitness)
