import polars as pl
import numpy as np
import math
from functools import partial
from ..transformers import EvoTransformer
from ..utils import gene_col_name

# --- FREQUENCY ENCODING ---

def _fit_frequency_encode(data, input_cols, target_col=None, params=None):
    x_col = input_cols[0]
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
    stats = data.group_by(x_col).len().rename({"len": "N"})
    default_val = stats["N"].median()
    if default_val is None:
        default_val = 0.0
    mapping = dict(zip(stats[x_col].to_list(), stats["N"].to_list()))
    return {"mapping": mapping, "default_val": default_val}

def _apply_frequency_encode(data, input_cols, state, params=None):
    x_col = input_cols[0]
    mapping = state["mapping"]
    default_val = state["default_val"]
    return pl.col(x_col).replace_strict(mapping, default=default_val).cast(pl.Float64)

# --- ONE-HOT ENCODING ---

def _fit_one_hot_encode(data, input_cols, target_col=None, params=None):
    x_col = input_cols[0]
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
    stats = data.select(pl.col(x_col).drop_nulls()).group_by(x_col).len().rename({"len": "count"})
    total_n = stats["count"].sum()
    if total_n == 0:
        return {"top_categories": []}
    stats = stats.with_columns(pct = pl.col("count") / total_n)
    stats = stats.sort("count", descending=True)
    top_cats = stats.filter(pl.col("pct") >= 0.05)[x_col].to_list()
    if len(top_cats) > 5:
        top_cats = top_cats[:5]
    return {"top_categories": top_cats}

def _apply_one_hot_encode(data, input_cols, state, params=None):
    x_col = input_cols[0]
    comp_idx = 1
    if params is not None and isinstance(params, dict):
        comp_idx = params.get("comp_idx", comp_idx)
        
    top_cats = state.get("top_categories", [])
    
    if comp_idx == 6:
        if len(top_cats) == 0:
            return pl.lit(1.0)
        return pl.when(pl.col(x_col).is_in(top_cats).not_() | pl.col(x_col).is_null()).then(1.0).otherwise(0.0)
    else:
        idx_0 = comp_idx - 1
        if 0 <= idx_0 < len(top_cats):
            cat = top_cats[idx_0]
            return pl.when(pl.col(x_col) == cat).then(1.0).otherwise(0.0)
        else:
            return pl.lit(0.0)

# --- DATETIME EXTRACT ---

def _apply_datetime_extract(data, input_cols, state=None, params=None):
    x_col = input_cols[0]
    comp = "month"
    if params is not None and isinstance(params, dict):
        comp = params.get("component", comp)
        
    is_str = False
    if isinstance(data, pl.DataFrame):
        is_str = data[x_col].dtype in [pl.Utf8, pl.String]
    elif hasattr(data, 'dtypes'):
        import pandas as pd
        is_str = pd.api.types.is_string_dtype(data[x_col])
    else:
        is_str = True

    expr = pl.col(x_col)
    if is_str:
        expr = expr.str.to_datetime(strict=False)
        
    if comp == "year":
        res = expr.dt.year()
    elif comp == "month":
        res = expr.dt.month()
    elif comp == "day":
        res = expr.dt.day()
    elif comp == "hour":
        res = expr.dt.hour()
    elif comp == "day_of_week":
        res = expr.dt.weekday()
    elif comp == "weekend":
        res = pl.when(expr.dt.weekday().is_in([6, 7])).then(1.0).otherwise(0.0)
    else:
        res = pl.lit(0.0)
        
    return res.fill_null(0.0).fill_nan(0.0).cast(pl.Float64)

# --- QUANTILE BINNING ---

def _fit_quantile_binning(data, input_cols, target_col=None, params=None):
    x_col = input_cols[0]
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
    x = data[x_col].drop_nulls().to_numpy()
    Q = 5
    if params is not None and isinstance(params, dict):
        Q = params.get("Q", Q)
    if len(x) == 0:
        return {"boundaries": [-float('inf'), float('inf')]}
    try:
        boundaries = np.quantile(x, np.linspace(0, 1, Q + 1))
        boundaries = np.unique(boundaries).tolist()
    except Exception:
        boundaries = [-float('inf'), float('inf')]
    return {"boundaries": boundaries}

def _apply_quantile_binning(data, input_cols, state, params=None):
    x_col = input_cols[0]
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
    x = data[x_col].cast(pl.Float64).to_numpy()
        
    boundaries = state.get("boundaries", [])
    if len(boundaries) <= 1:
        return np.ones(len(x), dtype=np.float64)
        
    res = np.digitize(x, boundaries)
    res = np.clip(res, 1, len(boundaries) - 1).astype(np.float64)
    
    # Replace NaNs/nulls with 0.0
    res[np.isnan(x)] = 0.0
    return res

def _apply_quantile_binning_cat(data, input_cols, state, params=None):
    res = _apply_quantile_binning(data, input_cols, state, params)
    return [str(int(val)) for val in res]

# --- LOG BINNING ---

def _apply_log_binning(data, input_cols, state=None, params=None):
    x_col = input_cols[0]
    base = 2
    if params is not None and isinstance(params, dict):
        base = params.get("base", base)
        
    col = pl.col(x_col).cast(pl.Float64).abs() + 1
    expr = (col.log() / math.log(base)).floor()
    return pl.when(expr.is_infinite() | expr.is_null() | expr.is_nan()).then(0.0).otherwise(expr).cast(pl.Float64)

def _apply_log_binning_cat(data, input_cols, state=None, params=None):
    expr = _apply_log_binning(data, input_cols, state, params)
    return expr.cast(pl.Int64).cast(pl.String)

# --- RANK TRANSFORM ---

def _fit_rank_transform(data, input_cols, target_col=None, params=None):
    x_col = input_cols[0]
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
    x = data[x_col].drop_nulls().to_numpy()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"sorted_x": None}
    return {"sorted_x": np.sort(x)}

def _apply_rank_transform(data, input_cols, state, params=None):
    x_col = input_cols[0]
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
    x = data[x_col].cast(pl.Float64).to_numpy()
        
    sorted_x = state.get("sorted_x")
    if sorted_x is None or len(sorted_x) == 0:
        from scipy.stats import rankdata
        r = rankdata(x, method="average")
        # Handle case where x might contain NaNs in fallback
        r[np.isnan(x)] = 0.5 * len(x)
        return r / len(x)
        
    counts = np.searchsorted(sorted_x, x, side='right')
    res = counts.astype(np.float64) / len(sorted_x)
    res[np.isnan(x)] = 0.5
    return res

# --- WOE ENCODE ---

def _fit_woe_encode(data, input_cols, target_col):
    if target_col is None:
        raise ValueError("target_col must be provided for supervised woe_encode.")
        
    x_col = input_cols[0]
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
        
    y_unique = sorted(data[target_col].drop_nulls().unique().to_list())
    if len(y_unique) != 2:
        return {"mapping": {}, "fallback": 0.0, "is_binary": False}
        
    event_val = y_unique[1]
    y_bin = data.select(pl.when(pl.col(target_col) == event_val).then(1.0).otherwise(0.0).alias("y_bin"))["y_bin"]
    
    total_events = y_bin.sum()
    total_non_events = len(y_bin) - total_events
    
    denom_events = total_events + 1.0
    denom_non_events = total_non_events + 1.0
    
    df_temp = pl.DataFrame({x_col: data[x_col], "y_bin": y_bin})
    stats = df_temp.group_by(x_col).agg([
        pl.col("y_bin").sum().alias("events"),
        pl.col("y_bin").count().alias("n")
    ])
    
    stats = stats.with_columns(
        non_events = pl.col("n") - pl.col("events")
    )
    
    stats = stats.with_columns(
        p_event_given_cat = (pl.col("events") + 0.5) / denom_events,
        p_non_event_given_cat = (pl.col("non_events") + 0.5) / denom_non_events
    )
    
    stats = stats.with_columns(
        woe = (pl.col("p_event_given_cat") / pl.col("p_non_event_given_cat")).log()
    )
    
    mapping = dict(zip(stats[x_col].to_list(), stats["woe"].to_list()))
    return {"mapping": mapping, "fallback": 0.0, "is_binary": True}

def _apply_woe_encode(data, input_cols, state):
    x_col = input_cols[0]
    if not state.get("is_binary", False):
        return pl.lit(0.0)
        
    mapping = state["mapping"]
    fallback = state["fallback"]
    return pl.col(x_col).replace_strict(mapping, default=fallback).cast(pl.Float64)

# --- TARGET ENCODE MULTICLASS ---

def _fit_target_encode_multiclass(data, input_cols, target_col):
    if target_col is None:
        raise ValueError("target_col must be provided for supervised target_encode_multiclass.")
        
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
        
    comp_idx = max(0, min(comp_idx, len(mappings) - 1))
    mapping = mappings[comp_idx]
    global_mean = global_means[comp_idx]
    
    return pl.col(x_col).replace_strict(mapping, default=global_mean).cast(pl.Float64)

# --- CREATE CATEGORICAL TRANSFORMERS ---

def create_categorical_transformers() -> dict:
    """
    Creates and returns a dictionary of built-in categorical and binning transformers.
    """
    transformers = {}
    
    transformers['frequency_encode'] = EvoTransformer(
        name="frequency_encode",
        type_="unary",
        input_type="categorical",
        fit_func=_fit_frequency_encode,
        apply_func=_apply_frequency_encode,
        name_generator=partial(gene_col_name, transformer_name="frequency_encode", prefix="freq")
    )
    
    transformers['one_hot_encode'] = EvoTransformer(
        name="one_hot_encode",
        type_="unary",
        input_type="categorical",
        output_type="numeric",
        fit_func=_fit_one_hot_encode,
        apply_func=_apply_one_hot_encode,
        name_generator=partial(gene_col_name, transformer_name="one_hot_encode", prefix="ohe")
    )
    
    transformers['datetime_extract'] = EvoTransformer(
        name="datetime_extract",
        type_="unary",
        input_type="categorical",
        apply_func=_apply_datetime_extract,
        name_generator=partial(gene_col_name, transformer_name="datetime_extract", prefix="dt")
    )
    
    transformers['quantile_binning'] = EvoTransformer(
        name="quantile_binning",
        type_="unary",
        input_type="numeric",
        fit_func=_fit_quantile_binning,
        apply_func=_apply_quantile_binning,
        name_generator=partial(gene_col_name, transformer_name="quantile_binning", prefix="qb")
    )
    
    transformers['quantile_binning_cat'] = EvoTransformer(
        name="quantile_binning_cat",
        type_="unary",
        input_type="numeric",
        output_type="categorical",
        fit_func=_fit_quantile_binning,
        apply_func=_apply_quantile_binning_cat,
        name_generator=partial(gene_col_name, transformer_name="quantile_binning_cat", prefix="qbc")
    )
    
    transformers['log_binning'] = EvoTransformer(
        name="log_binning",
        type_="unary",
        input_type="numeric",
        apply_func=_apply_log_binning,
        name_generator=partial(gene_col_name, transformer_name="log_binning", prefix="lb")
    )
    
    transformers['log_binning_cat'] = EvoTransformer(
        name="log_binning_cat",
        type_="unary",
        input_type="numeric",
        output_type="categorical",
        apply_func=_apply_log_binning_cat,
        name_generator=partial(gene_col_name, transformer_name="log_binning_cat", prefix="lbc")
    )
    
    transformers['rank_transform'] = EvoTransformer(
        name="rank_transform",
        type_="unary",
        input_type="numeric",
        fit_func=_fit_rank_transform,
        apply_func=_apply_rank_transform,
        name_generator=partial(gene_col_name, transformer_name="rank_transform", prefix="rnk")
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
