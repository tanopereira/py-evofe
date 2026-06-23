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

def _fit_pooled_target_encode(data, input_cols, target_col):
    if target_col is None:
        raise ValueError("target_col must be provided for supervised transformers.")
        
    x_col = input_cols[0]
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
        
    data = data.with_columns(pl.col(target_col).cast(pl.Float64))
        
    global_mean = data.select(pl.col(target_col).drop_nulls().mean()).item()
    if global_mean is None:
        global_mean = 0.0
        
    var_y = data.select(pl.col(target_col).drop_nulls().var()).item()
    if var_y is None or np.isnan(var_y):
        var_y = 0.0

    stats = data.group_by(x_col).agg([
        pl.col(target_col).drop_nulls().mean().alias("mean"),
        pl.col(target_col).drop_nulls().var().alias("var"),
        pl.col(target_col).drop_nulls().count().alias("n")
    ])
    
    stats = stats.with_columns(
        valid_n = pl.when(pl.col("var").is_not_null()).then(pl.col("n")).otherwise(0)
    )
    df_sum = stats.select((pl.col("valid_n") - 1).clip(lower_bound=0).sum()).item()
    
    if df_sum is not None and df_sum > 0:
        var_within = stats.select((((pl.col("valid_n") - 1).clip(lower_bound=0) * pl.col("var")).drop_nulls().sum() / df_sum)).item()
    else:
        var_within = stats.select(pl.col("var").drop_nulls().mean()).item()
        if var_within is None or np.isnan(var_within):
            var_within = var_y
            
    if var_within is None or np.isnan(var_within):
        var_within = var_y
        
    var_between = stats.select(pl.col("mean").drop_nulls().var()).item()
    if var_between is None or np.isnan(var_between):
        var_between = 0.0
        
    if var_between > 0:
        k = var_within / var_between
    else:
        k = float('inf')
        
    if np.isinf(k):
        stats = stats.with_columns(smoothed = pl.lit(global_mean))
    else:
        stats = stats.with_columns(
            smoothed = (pl.col("n") * pl.col("mean") + k * global_mean) / (pl.col("n") + k)
        )
        
    mapping = dict(zip(stats[x_col].to_list(), stats["smoothed"].to_list()))
    return {"mapping": mapping, "global_mean": global_mean}

def _apply_pooled_target_encode(data, input_cols, state):
    x_col = input_cols[0]
    mapping = state["mapping"]
    global_mean = state["global_mean"]
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

    transformers['pooled_target_encode'] = EvoTransformer(
        name="pooled_target_encode",
        type_="supervised_unary",
        input_type="categorical",
        fit_func=_fit_pooled_target_encode,
        apply_func=_apply_pooled_target_encode,
        name_generator=partial(gene_col_name, transformer_name="pooled_target_encode", prefix="pte")
    )

    return transformers
