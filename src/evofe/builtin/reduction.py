import polars as pl
import numpy as np
import os
from functools import partial
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import umap
from ..transformers import EvoTransformer
from ..utils import gene_col_name

def _extract_numpy(data, input_cols):
    cols = list(dict.fromkeys(input_cols))
    if isinstance(data, pl.DataFrame):
        return data.select(cols).fill_null(0).fill_nan(0).to_numpy()
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        return data[cols].fillna(0).to_numpy()
    return data

def _fit_reduction(data, input_cols, target_col, model_class, params=None, **kwargs):
    x = _extract_numpy(data, input_cols)
    try:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x = scaler.fit_transform(x)
            
        model = model_class(**kwargs)
        model.fit(x)
        return {"model": model, "scaler": scaler, "valid": True}
    except Exception:
        return {"model": None, "scaler": None, "valid": False}

def _fit_svd(data, input_cols, target_col=None, params=None):
    x = _extract_numpy(data, input_cols)
    try:
        n_features = x.shape[1]
        C = max(2, int(np.round(np.log2(n_features))))
        n_components = min(C, n_features)
        if n_features > 1:
            n_components = min(n_components, n_features - 1)
        else:
            n_components = 1
            
        model = TruncatedSVD(n_components=n_components)
        model.fit(x)
        return {"model": model, "valid": True}
    except Exception:
        return {"model": None, "valid": False}

def _fit_umap(data, input_cols, target_col=None, params=None):
    x = _extract_numpy(data, input_cols)
    try:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x)
        
        n_samples = x_scaled.shape[0]
        n_neighbors = 15
        if n_samples < 15:
            n_neighbors = max(2, n_samples - 1)
            
        n_features = x_scaled.shape[1]
        C = max(2, int(np.round(np.log2(n_features))))
        if C >= n_samples:
            C = max(1, n_samples - 1)
            
        num_threads = int(os.environ.get("EVOFE_THREADS", 1))
        
        model = umap.UMAP(n_components=C, n_neighbors=n_neighbors, init="random", n_jobs=num_threads)
        model.fit(x_scaled)
        return {"model": model, "scaler": scaler, "valid": True}
    except Exception:
        return {"model": None, "scaler": None, "valid": False}

def _apply_reduction(data, input_cols, state, params=None, comp_idx=0):
    if not state.get("valid", False):
        return pl.lit(0.0)
        
    idx = comp_idx
    if params is not None and isinstance(params, dict):
        idx = params.get("comp_idx", idx)
        
    x = _extract_numpy(data, input_cols)
    scaler = state.get("scaler")
    if scaler is not None:
        x = scaler.transform(x)
        
    model = state["model"]
    preds = model.transform(x)
    
    C = preds.shape[1]
    idx = int(np.clip(idx, 0, C - 1))
        
    return preds[:, idx]

def _fit_random_projection(data, input_cols, target_col=None, params=None):
    x = _extract_numpy(data, input_cols)
    try:
        P = x.shape[1]
        w = np.random.randn(P)
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            w = np.ones(P) / np.sqrt(P)
        else:
            w = w / w_norm
        return {"w": w, "valid": True}
    except Exception:
        return {"w": None, "valid": False}

def _apply_random_projection(data, input_cols, state, params=None):
    if not state.get("valid", False) or "w" not in state or state["w"] is None:
        return pl.lit(0.0)
    x = _extract_numpy(data, input_cols)
    w = state["w"]
    return x @ w

def create_reduction_transformers() -> dict:
    """
    Creates and returns a dictionary of built-in dimensionality reduction transformers.
    """
    transformers = {}
    
    transformers['pca'] = EvoTransformer(
        name="pca",
        type_="multivariate",
        input_type="numeric",
        fit_func=lambda d, i, t, params=None: _fit_reduction(d, i, t, PCA, n_components=min(5, len(i))),
        apply_func=lambda d, i, s, params=None: _apply_reduction(d, i, s, params=params, comp_idx=0),
        name_generator=partial(gene_col_name, transformer_name="pca", prefix="pca")
    )
    
    transformers['truncated_svd'] = EvoTransformer(
        name="truncated_svd",
        type_="multivariate",
        input_type="numeric",
        fit_func=_fit_svd,
        apply_func=lambda d, i, s, params=None: _apply_reduction(d, i, s, params=params, comp_idx=0),
        name_generator=partial(gene_col_name, transformer_name="truncated_svd", prefix="svd")
    )

    transformers['random_projection'] = EvoTransformer(
        name="random_projection",
        type_="multivariate",
        input_type="numeric",
        fit_func=_fit_random_projection,
        apply_func=_apply_random_projection,
        name_generator=partial(gene_col_name, transformer_name="random_projection", prefix="rp")
    )
    
    transformers['umap'] = EvoTransformer(
        name="umap",
        type_="multivariate",
        input_type="numeric",
        fit_func=_fit_umap,
        apply_func=lambda d, i, s, params=None: _apply_reduction(d, i, s, params=params, comp_idx=0),
        name_generator=partial(gene_col_name, transformer_name="umap", prefix="ump")
    )

    return transformers
