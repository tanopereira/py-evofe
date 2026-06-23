import polars as pl
from functools import partial
from ..transformers import EvoTransformer
from ..utils import gene_col_name

def create_math_transformers() -> dict:
    """
    Creates and returns a dictionary of the built-in mathematical transformers.
    These all operate lazily using Polars expressions for maximum performance.
    """
    transformers = {}

    # --- UNARY TRANSFORMERS ---

    transformers['log'] = EvoTransformer(
        name="log",
        type_="unary",
        input_type="numeric",
        apply_func=lambda data, cols, state: pl.col(cols[0]).cast(pl.Float64).abs().log1p(),
        name_generator=partial(gene_col_name, transformer_name="log", prefix="log")
    )

    transformers['sqrt'] = EvoTransformer(
        name="sqrt",
        type_="unary",
        input_type="numeric",
        apply_func=lambda data, cols, state: pl.col(cols[0]).cast(pl.Float64).abs().sqrt(),
        name_generator=partial(gene_col_name, transformer_name="sqrt", prefix="sqrt")
    )

    transformers['reciprocal'] = EvoTransformer(
        name="reciprocal",
        type_="unary",
        input_type="numeric",
        apply_func=lambda data, cols, state: pl.when(pl.col(cols[0]).cast(pl.Float64) == 0).then(0.0).otherwise(1 / pl.col(cols[0]).cast(pl.Float64)),
        name_generator=partial(gene_col_name, transformer_name="reciprocal", prefix="rec")
    )

    # --- BINARY / MULTIVARIATE TRANSFORMERS ---

    transformers['add'] = EvoTransformer(
        name="add",
        type_="multivariate",
        input_type="numeric",
        apply_func=lambda data, cols, state: pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in cols]),
        name_generator=partial(gene_col_name, transformer_name="add", prefix="add", sort_cols=True),
        allow_replace=True
    )

    transformers['subtract'] = EvoTransformer(
        name="subtract",
        type_="binary",
        input_type="numeric",
        apply_func=lambda data, cols, state: pl.col(cols[0]).cast(pl.Float64) - pl.col(cols[1]).cast(pl.Float64),
        name_generator=partial(gene_col_name, transformer_name="subtract", prefix="sub")
    )

    def _multiply_expr(cols):
        expr = pl.col(cols[0]).cast(pl.Float64)
        for c in cols[1:]:
            expr = expr * pl.col(c).cast(pl.Float64)
        return expr

    transformers['multiply'] = EvoTransformer(
        name="multiply",
        type_="multivariate",
        input_type="numeric",
        apply_func=lambda data, cols, state: _multiply_expr(cols),
        name_generator=partial(gene_col_name, transformer_name="multiply", prefix="mul", sort_cols=True),
        allow_replace=True
    )

    transformers['divide'] = EvoTransformer(
        name="divide",
        type_="binary",
        input_type="numeric",
        apply_func=lambda data, cols, state: pl.when(pl.col(cols[1]).cast(pl.Float64) == 0).then(0.0).otherwise(pl.col(cols[0]).cast(pl.Float64) / pl.col(cols[1]).cast(pl.Float64)),
        name_generator=partial(gene_col_name, transformer_name="divide", prefix="div")
    )

    def _power_apply(data, cols, state, params=None):
        p = 2
        if params and 'p' in params:
            p = params['p']
        elif state and isinstance(state, dict) and 'p' in state:
            p = state['p']
        val = pl.col(cols[0]).cast(pl.Float64)
        res = val.sign() * (val.abs() ** p)
        return pl.when(res.is_nan() | res.is_infinite() | res.is_null()).then(0.0).otherwise(res)

    transformers['power'] = EvoTransformer(
        name="power",
        type_="unary",
        input_type="numeric",
        apply_func=_power_apply,
        name_generator=partial(gene_col_name, transformer_name="power", prefix="pow")
    )

    transformers['normalized_difference'] = EvoTransformer(
        name="normalized_difference",
        type_="binary",
        input_type="numeric",
        apply_func=lambda data, cols, state: (pl.col(cols[0]).cast(pl.Float64) - pl.col(cols[1]).cast(pl.Float64)) / (pl.col(cols[0]).cast(pl.Float64).abs() + pl.col(cols[1]).cast(pl.Float64).abs() + 1e-8),
        name_generator=partial(gene_col_name, transformer_name="normalized_difference", prefix="nd")
    )

    transformers['log_ratio'] = EvoTransformer(
        name="log_ratio",
        type_="binary",
        input_type="numeric",
        apply_func=lambda data, cols, state: pl.col(cols[0]).cast(pl.Float64).abs().log1p() - pl.col(cols[1]).cast(pl.Float64).abs().log1p(),
        name_generator=partial(gene_col_name, transformer_name="log_ratio", prefix="lr")
    )

    def _displaced_log_apply(data, cols, state, params=None):
        displacement = 100
        if params and 'displacement' in params:
            displacement = params['displacement']
        elif state and isinstance(state, dict) and 'displacement' in state:
            displacement = state['displacement']
        val = pl.col(cols[0]).cast(pl.Float64)
        res = (val + displacement).abs().log1p()
        return pl.when(res.is_nan() | res.is_infinite() | res.is_null()).then(0.0).otherwise(res)

    transformers['displaced_log'] = EvoTransformer(
        name="displaced_log",
        type_="unary",
        input_type="numeric",
        apply_func=_displaced_log_apply,
        name_generator=partial(gene_col_name, transformer_name="displaced_log", prefix="dlog")
    )

    return transformers
