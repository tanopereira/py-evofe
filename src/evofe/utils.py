import hashlib
import json
from typing import List, Any, Dict, Optional

def gene_col_name(input_cols: List[str], transformer_name: str, prefix: str, params: Optional[Dict[str, Any]] = None, sort_cols: bool = False) -> str:
    """
    Produce a short, stable column name for a generated feature: "{prefix}_{6-char hash}".
    The hash covers the transformer name, input columns, and parameters, ensuring that
    identical feature recipes always get identical names (for deduplication).
    
    Args:
        input_cols: List of input column names used
        transformer_name: Name of the transformer (e.g., 'log', 'add')
        prefix: Prefix for the new column (e.g., 'log', 'sub')
        params: Optional dictionary of additional parameters
        sort_cols: Whether to sort input_cols before hashing (useful for commutative ops)
        
    Returns:
        A string like "log_a1b2c3"
    """
    if params is None:
        params = {}
        
    # Create a stable representation of the inputs
    cols_to_hash = sorted(input_cols) if sort_cols else input_cols
    state_dict = {
        "name": transformer_name,
        "cols": cols_to_hash,
        "params": {k: params[k] for k in sorted(params.keys())}
    }
    
    state_str = json.dumps(state_dict, sort_keys=True).encode('utf-8')
    h = hashlib.md5(state_str).hexdigest()[:6]
    
    return f"{prefix}_{h}"


def call_with_optional_params(func, *args, **kwargs):
    import inspect
    if func is None:
        return None
    try:
        sig = inspect.signature(func)
    except ValueError:
        try:
            return func(*args, **kwargs)
        except TypeError:
            return func(*args)
            
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    valid_kwargs = {}
    for k, v in kwargs.items():
        if has_var_keyword or k in sig.parameters:
            valid_kwargs[k] = v
    return func(*args, **valid_kwargs)

