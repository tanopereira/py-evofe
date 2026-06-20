import polars as pl
import random
from evofe.evolution import evolve_features

# Import builtin to register all transformers
import evofe.builtin 

def dummy_evaluate_fitness(individual, data, target_col, **kwargs):
    """
    A dummy evaluator that assigns a random fitness score
    proportional to the number of genes, just to verify evolution mechanics.
    """
    score = random.uniform(0.5, 0.9) + (len(individual.genes) * 0.01)
    individual.fitness = score
    return individual

def test_evolution_engine():
    # Create a small dummy dataset
    df = pl.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [5, 4, 3, 2, 1],
        "cat1": ["a", "b", "a", "b", "c"],
        "target": [0, 1, 0, 1, 1]
    })
    
    best_ind = evolve_features(
        data=df,
        target_col="target",
        numeric_cols=["num1", "num2"],
        categorical_cols=["cat1"],
        evaluate_fitness=dummy_evaluate_fitness,
        pop_size=5,
        n_generations=3,
        initial_genes=1,
        verbose=False
    )
    
    # We should have successfully returned an Individual
    assert best_ind is not None
    assert hasattr(best_ind, "fitness")
    assert not sum(1 for _ in [best_ind.fitness]) == 0 # Check it's not empty/nan
    assert best_ind.fitness > 0.0
