import polars as pl
import numpy as np
from functools import partial
from ..transformers import EvoTransformer
from ..utils import gene_col_name

def _fit_target_encode(data, input_cols, target_col):
    if target_col is None:
        raise ValueError("target_col must be provided for supervised transformers.")
        
    x_col = input_cols[0]
    
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
        
    # Calculate global mean
    global_mean = data.select(pl.col(target_col).mean()).item()
    
    # Calculate category means and counts
    stats = data.group_by(x_col).agg([
        pl.col(target_col).mean().alias("mean"),
        pl.col(target_col).count().alias("n")
    ])
    
    # Smoothing parameters
    smoothing = 10.0
    
    # Calculate smoothed target encoding
    # formula: (n * mean + smoothing * global_mean) / (n + smoothing)
    stats = stats.with_columns(
        smoothed = (pl.col("n") * pl.col("mean") + smoothing * global_mean) / (pl.col("n") + smoothing)
    )
    
    # Create mapping dict: {category: smoothed_value}
    mapping = dict(zip(stats[x_col].to_list(), stats["smoothed"].to_list()))
    
    return {"mapping": mapping, "global_mean": global_mean}

def _apply_target_encode(data, input_cols, state):
    x_col = input_cols[0]
    mapping = state["mapping"]
    global_mean = state["global_mean"]
    
    # Map, filling missing categories with global mean
    return pl.col(x_col).replace_strict(mapping, default=global_mean)

def create_supervised_transformers() -> dict:
    """
    Creates and returns a dictionary of built-in supervised stateful transformers.
    """
    transformers = {}

    transformers['target_encode'] = EvoTransformer(
        name="target_encode",
        type_="supervised_unary",
        input_type="categorical",
        fit_func=_fit_target_encode,
        apply_func=_apply_target_encode,
        name_generator=partial(gene_col_name, transformer_name="target_encode", prefix="te")
    )

    return transformers
