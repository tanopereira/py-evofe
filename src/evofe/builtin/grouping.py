import polars as pl
from functools import partial
from ..transformers import EvoTransformer
from ..utils import gene_col_name

def _fit_groupby(data, input_cols, target_col, agg_expr, default_expr):
    """
    Generic fit function for group-by operations.
    """
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
        
    cat_col = input_cols[0]
    num_col = input_cols[1]
    
    # Calculate global default
    default_val = data.select(default_expr(num_col)).item()
    if default_val is None:
        default_val = 0.0
        
    # Calculate group aggregations
    mapping_df = data.group_by(cat_col).agg(agg_expr(num_col).alias("val"))
    
    # Fill nulls in aggregations with the global default
    mapping_df = mapping_df.with_columns(
        pl.col("val").fill_null(default_val).fill_nan(default_val)
    )
    
    mapping = dict(zip(mapping_df[cat_col].to_list(), mapping_df["val"].to_list()))
    
    return {"mapping": mapping, "default_val": default_val}

def _apply_groupby(data, input_cols, state):
    cat_col = input_cols[0]
    mapping = state["mapping"]
    default_val = state["default_val"]
    
    return pl.col(cat_col).replace_strict(mapping, default=default_val).cast(pl.Float64)

# ratio: value / group_mean
def _apply_groupby_ratio(data, input_cols, state):
    cat_col = input_cols[0]
    num_col = input_cols[1]
    group_means = _apply_groupby(data, [cat_col], state)
    return pl.when(group_means == 0).then(0.0).otherwise(pl.col(num_col) / group_means)

# zscore: (value - group_mean) / group_sd
def _apply_groupby_zscore(data, input_cols, state):
    cat_col = input_cols[0]
    num_col = input_cols[1]
    mean_mapping = state["mean_mapping"]
    sd_mapping = state["sd_mapping"]
    default_mean = state["default_mean"]
    default_sd = state["default_sd"]
    
    group_means = pl.col(cat_col).replace_strict(mean_mapping, default=default_mean).cast(pl.Float64)
    group_sds = pl.col(cat_col).replace_strict(sd_mapping, default=default_sd).cast(pl.Float64)
    
    return pl.when(group_sds == 0).then(0.0).otherwise((pl.col(num_col) - group_means) / group_sds)

def _fit_groupby_zscore(data, input_cols, target_col):
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)
    cat_col = input_cols[0]
    num_col = input_cols[1]
    
    mean_res = _fit_groupby(data, input_cols, target_col, lambda c: pl.col(c).mean(), lambda c: pl.col(c).mean())
    sd_res = _fit_groupby(data, input_cols, target_col, lambda c: pl.col(c).std(), lambda c: pl.col(c).std())
    
    return {
        "mean_mapping": mean_res["mapping"],
        "default_mean": mean_res["default_val"],
        "sd_mapping": sd_res["mapping"],
        "default_sd": sd_res["default_val"]
    }

def create_grouping_transformers() -> dict:
    """
    Creates and returns a dictionary of built-in group-by transformers.
    """
    transformers = {}

    def make_gb(name, prefix, agg_expr, default_expr):
        return EvoTransformer(
            name=name,
            type_="mixed_binary",
            input_type="mixed",
            fit_func=lambda d, i, t, params=None: _fit_groupby(d, i, t, agg_expr, default_expr),
            apply_func=_apply_groupby,
            name_generator=partial(gene_col_name, transformer_name=name, prefix=prefix)
        )

    transformers['groupby_mean'] = make_gb("groupby_mean", "gbm", lambda c: pl.col(c).mean(), lambda c: pl.col(c).mean())
    transformers['groupby_sd'] = make_gb("groupby_sd", "gbsd", lambda c: pl.col(c).std(), lambda c: pl.col(c).std())
    transformers['groupby_max'] = make_gb("groupby_max", "gbmx", lambda c: pl.col(c).max(), lambda c: pl.col(c).max())
    transformers['groupby_min'] = make_gb("groupby_min", "gbmn", lambda c: pl.col(c).min(), lambda c: pl.col(c).min())
    transformers['groupby_median'] = make_gb("groupby_median", "gbmed", lambda c: pl.col(c).median(), lambda c: pl.col(c).median())

    transformers['groupby_ratio'] = EvoTransformer(
        name="groupby_ratio",
        type_="mixed_binary",
        input_type="mixed",
        fit_func=lambda d, i, t, params=None: _fit_groupby(d, i, t, lambda c: pl.col(c).mean(), lambda c: pl.col(c).mean()),
        apply_func=_apply_groupby_ratio,
        name_generator=partial(gene_col_name, transformer_name="groupby_ratio", prefix="gbr")
    )
    
    transformers['groupby_zscore'] = EvoTransformer(
        name="groupby_zscore",
        type_="mixed_binary",
        input_type="mixed",
        fit_func=lambda d, i, t, params=None: _fit_groupby_zscore(d, i, t),
        apply_func=_apply_groupby_zscore,
        name_generator=partial(gene_col_name, transformer_name="groupby_zscore", prefix="gbz")
    )

    def _fit_groupby_quantile(data, input_cols, target_col, params=None):
        if params is None:
            params = {}
        q = params.get('q', 0.25)
        return _fit_groupby(data, input_cols, target_col, 
                            lambda c: pl.col(c).quantile(q), 
                            lambda c: pl.col(c).quantile(q))

    transformers['groupby_quantile'] = EvoTransformer(
        name="groupby_quantile",
        type_="mixed_binary",
        input_type="mixed",
        fit_func=_fit_groupby_quantile,
        apply_func=_apply_groupby,
        name_generator=partial(gene_col_name, transformer_name="groupby_quantile", prefix="gbq")
    )

    return transformers
