from .math import create_math_transformers
from .supervised import create_supervised_transformers
from .grouping import create_grouping_transformers
from .reduction import create_reduction_transformers
from .clustering import create_clustering_transformers
from .categorical import create_categorical_transformers

# Global registry of all built-in transformers
evo_transformers = {}

# Register all builtin modules
evo_transformers.update(create_math_transformers())
evo_transformers.update(create_supervised_transformers())
evo_transformers.update(create_grouping_transformers())
evo_transformers.update(create_reduction_transformers())
evo_transformers.update(create_clustering_transformers())
evo_transformers.update(create_categorical_transformers())

def register_transformer(name: str, transformer):
    """
    Registers a custom feature transformer into the global pool.
    
    Args:
        name: Unique string naming the transformer.
        transformer: An object of class EvoTransformer.
    """
    from ..transformers import EvoTransformer
    if not isinstance(transformer, EvoTransformer):
        raise TypeError("transformer must be an instance of EvoTransformer.")
    evo_transformers[name] = transformer
    
__all__ = ["evo_transformers", "register_transformer"]
