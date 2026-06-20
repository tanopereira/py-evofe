from typing import List, Optional, Dict
from .individual import Individual

def initialize_population(
    pop_size: int, 
    numeric_cols: List[str], 
    categorical_cols: List[str], 
    initial_genes: int = 2, 
    task: str = "classification", 
    importances: Optional[Dict[str, float]] = None,
    allowed_transformers: Optional[List[str]] = None
) -> List[Individual]:
    """
    Initialize a population of candidate feature recipes.
    
    The first individual is always kept as a baseline (0 genes).
    The remaining individuals get randomly initialized genes.
    """
    pop = []
    
    for i in range(pop_size):
        ind = Individual(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
        
        if i > 0:
            attempts = 0
            while len(ind.genes) < initial_genes and attempts < (initial_genes * 10):
                ind.mutate(
                    force_add=True, 
                    importances=importances,
                    allowed_transformers=allowed_transformers
                )
                attempts += 1
                
        pop.append(ind)
        
    return pop
