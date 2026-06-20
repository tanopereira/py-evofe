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

def _fit_woe_encode(data, input_cols, target_col):
    if target_col is None:
        raise ValueError("target_col must be provided for supervised transformers.")
        
    x_col = input_cols[0]
    
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
        
    # WOE is for binary classification (0/1 targets)
    # Check if target is binary
    y_unique = data[target_col].drop_nulls().unique().to_list()
    if len(y_unique) > 2 or not all(val in [0, 1] for val in y_unique):
        # Fallback to zero encoding if not binary
        return {"mapping": {}, "fallback": 0.0, "is_binary": False}
        
    # Global counts
    total_events = data.select(pl.col(target_col).sum()).item()
    total_non_events = data.select((1 - pl.col(target_col)).sum()).item()
    
    # Add Laplace smoothing to global counts
    total_events = max(total_events, 0.5)
    total_non_events = max(total_non_events, 0.5)
    
    # Category counts
    stats = data.group_by(x_col).agg([
        pl.col(target_col).sum().alias("events"),
        pl.col(target_col).count().alias("n")
    ])
    
    stats = stats.with_columns(
        non_events = pl.col("n") - pl.col("events")
    )
    
    # Laplace smoothing for categories (add 0.5 to numerator and 1 to denominator)
    stats = stats.with_columns(
        p_event_given_cat = (pl.col("events") + 0.5) / total_events,
        p_non_event_given_cat = (pl.col("non_events") + 0.5) / total_non_events
    )
    
    # WOE = ln(P(event|cat) / P(non_event|cat))
    stats = stats.with_columns(
        woe = (pl.col("p_event_given_cat") / pl.col("p_non_event_given_cat")).log()
    )
    
    mapping = dict(zip(stats[x_col].to_list(), stats["woe"].to_list()))
    
    return {"mapping": mapping, "fallback": 0.0, "is_binary": True}

def _apply_woe_encode(data, input_cols, state):
    x_col = input_cols[0]
    if not state.get("is_binary", False):
        # Fallback if target was not binary during fit
        return pl.lit(0.0)
        
    mapping = state["mapping"]
    fallback = state["fallback"]
    
    return pl.col(x_col).replace(mapping, default=fallback)

def _fit_target_encode_multiclass(data, input_cols, target_col, params=None):
    if target_col is None:
        raise ValueError("target_col must be provided for supervised transformers.")
        
    x_col = input_cols[0]
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
        
    classes = sorted(data[target_col].drop_nulls().unique().to_list())
    mappings = []
    global_means = []
    smoothing = 10.0
    
    for c in classes:
        y_bin = data.select(pl.when(pl.col(target_col) == c).then(1.0).otherwise(0.0).alias("y_bin"))["y_bin"]
        global_mean = y_bin.mean()
        if global_mean is None:
            global_mean = 0.0
        global_means.append(global_mean)
        
        df_temp = pl.DataFrame({x_col: data[x_col], "y_bin": y_bin})
        stats = df_temp.group_by(x_col).agg([
            pl.col("y_bin").mean().alias("mean"),
            pl.col("y_bin").count().alias("n")
        ])
        
        stats = stats.with_columns(
            smoothed = (pl.col("n") * pl.col("mean") + smoothing * global_mean) / (pl.col("n") + smoothing)
        )
        
        mapping = dict(zip(stats[x_col].to_list(), stats["smoothed"].to_list()))
        mappings.append(mapping)
        
    return {"mappings": mappings, "global_means": global_means, "classes": classes}

def _apply_target_encode_multiclass(data, input_cols, state, params=None):
    x_col = input_cols[0]
    
    comp_idx = 0
    if params is not None and isinstance(params, dict):
        comp_idx = params.get("comp_idx", comp_idx)
        
    mappings = state.get("mappings", [])
    global_means = state.get("global_means", [])
    
    if not mappings:
        return pl.lit(0.0)
        
    if comp_idx >= len(mappings):
        comp_idx = len(mappings) - 1
    if comp_idx < 0:
        comp_idx = 0
        
    mapping = mappings[comp_idx]
    global_mean = global_means[comp_idx]
    
    return pl.col(x_col).replace(mapping, default=global_mean)

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
    
    transformers['woe_encode'] = EvoTransformer(
        name="woe_encode",
        type_="supervised_unary",
        input_type="categorical",
        fit_func=_fit_woe_encode,
        apply_func=_apply_woe_encode,
        name_generator=partial(gene_col_name, transformer_name="woe_encode", prefix="woe")
    )

    transformers['target_encode_multiclass'] = EvoTransformer(
        name="target_encode_multiclass",
        type_="supervised_unary",
        input_type="categorical",
        fit_func=_fit_target_encode_multiclass,
        apply_func=_apply_target_encode_multiclass,
        name_generator=partial(gene_col_name, transformer_name="target_encode_multiclass", prefix="temc")
    )

    return transformers
