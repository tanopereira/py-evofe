import polars as pl
import numpy as np
import hashlib
from functools import partial
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from ..transformers import EvoTransformer
from ..utils import gene_col_name
from .reduction import _extract_numpy

def _cluster_prep_x(x, min_rows=6):
    # Deduplicate input rows first
    unique_rows = np.unique(x, axis=0)
    if len(unique_rows) < min_rows:
        return None
        
    max_size = 5000
    if len(unique_rows) > max_size:
        indices = np.random.choice(len(unique_rows), max_size, replace=False)
        unique_rows = unique_rows[indices]
        
    if len(unique_rows) < min_rows:
        return None
        
    return unique_rows

def _cluster_knn_apply(x_test, state, get_preds):
    if not state.get("valid", False) or state.get("x_train") is None:
        return np.zeros(x_test.shape[0])
        
    if "preds_cache" not in state:
        state["preds_cache"] = {}
        
    arr_hash = hashlib.md5(np.ascontiguousarray(x_test).tobytes()).hexdigest()
    if arr_hash in state["preds_cache"]:
        return state["preds_cache"][arr_hash]
        
    x_train = state["x_train"]
    if x_test.shape == x_train.shape and np.allclose(x_test, x_train, rtol=1e-5, atol=1e-8):
        idx = np.arange(x_train.shape[0])
        preds = get_preds(idx, state)
    else:
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(x_train)
        distances, indices = nn.kneighbors(x_test)
        idx = indices.flatten()
        preds = get_preds(idx, state)
        
    state["preds_cache"][arr_hash] = preds
    return preds

# --- GENIE ---

def _fit_genie(data, input_cols, target_col=None, params=None):
    if len(input_cols) < 2:
        raise ValueError("genie requires at least 2 input columns.")
    x = _extract_numpy(data, input_cols)
    k = 2
    if params is not None and isinstance(params, dict):
        k = params.get("k", k)
        
    try:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x)
        
        x_s = _cluster_prep_x(x_scaled, min_rows=max(6, k))
        if x_s is None:
            return {"valid": False}
            
        model = AgglomerativeClustering(n_clusters=k, linkage='single')
        labels = model.fit_predict(x_s)
        return {"scaler": scaler, "x_train": x_s, "labels": labels, "valid": True}
    except Exception:
        return {"valid": False}

def _apply_genie(data, input_cols, state, params=None):
    if not state.get("valid", False):
        return np.ones(data.shape[0] if hasattr(data, 'shape') else len(data))
    x_test = _extract_numpy(data, input_cols)
    if "scaler" in state and state["scaler"] is not None:
        x_test = state["scaler"].transform(x_test)
    return _cluster_knn_apply(x_test, state, lambda idx, s: s["labels"][idx])

# --- GENIE CENTROID DISTANCE ---

def _fit_genie_centroid_dist(data, input_cols, target_col=None, params=None):
    if len(input_cols) < 2:
        raise ValueError("genie_centroid_dist requires at least 2 input columns.")
    x = _extract_numpy(data, input_cols)
    k = 2
    if params is not None and isinstance(params, dict):
        k = params.get("k", k)
        
    try:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x)
        
        x_s = _cluster_prep_x(x_scaled, min_rows=max(6, k))
        if x_s is None:
            return {"centroids": None, "valid": False}
            
        model = AgglomerativeClustering(n_clusters=k, linkage='single')
        labels = model.fit_predict(x_s)
        centroids = []
        for j in range(k):
            pts = x_s[labels == j]
            if len(pts) == 0:
                centroids.append(x_s.mean(axis=0))
            else:
                centroids.append(pts.mean(axis=0))
        return {"scaler": scaler, "centroids": centroids, "valid": True}
    except Exception:
        return {"centroids": None, "valid": False}

def _apply_centroid_dist(data, input_cols, state, params=None):
    if not state.get("valid", False) or "centroids" not in state or state["centroids"] is None:
        return np.zeros(data.shape[0] if hasattr(data, 'shape') else len(data))
        
    comp_idx = 0
    if params is not None and isinstance(params, dict):
        comp_idx = params.get("comp_idx", params.get("centroid_idx", comp_idx))
        
    centroids = state["centroids"]
    comp_idx = max(0, min(comp_idx, len(centroids) - 1))
    c = centroids[comp_idx]
    
    x_test = _extract_numpy(data, input_cols)
    if "scaler" in state and state["scaler"] is not None:
        x_test = state["scaler"].transform(x_test)
        
    return np.sqrt(np.sum((x_test - c) ** 2, axis=1))

# --- LUMBERMARK ---

def _fit_lumbermark(data, input_cols, target_col=None, params=None):
    if len(input_cols) < 2:
        raise ValueError("lumbermark requires at least 2 input columns.")
    x = _extract_numpy(data, input_cols)
    k = 2
    if params is not None and isinstance(params, dict):
        k = params.get("k", k)
        
    try:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x)
        
        x_s = _cluster_prep_x(x_scaled, min_rows=max(6, 2 * k))
        if x_s is None:
            return {"valid": False}
            
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(x_s)
        return {"scaler": scaler, "x_train": x_s, "labels": labels, "valid": True}
    except Exception:
        return {"valid": False}

def _apply_lumbermark(data, input_cols, state, params=None):
    if not state.get("valid", False):
        return np.ones(data.shape[0] if hasattr(data, 'shape') else len(data))
    x_test = _extract_numpy(data, input_cols)
    if "scaler" in state and state["scaler"] is not None:
        x_test = state["scaler"].transform(x_test)
    return _cluster_knn_apply(x_test, state, lambda idx, s: s["labels"][idx])

# --- LUMBERMARK CENTROID DISTANCE ---

def _fit_lumbermark_centroid_dist(data, input_cols, target_col=None, params=None):
    if len(input_cols) < 2:
        raise ValueError("lumbermark_centroid_dist requires at least 2 input columns.")
    x = _extract_numpy(data, input_cols)
    k = 2
    if params is not None and isinstance(params, dict):
        k = params.get("k", k)
        
    try:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x)
        
        x_s = _cluster_prep_x(x_scaled, min_rows=max(6, 2 * k))
        if x_s is None:
            return {"centroids": None, "valid": False}
            
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(x_s)
        centroids = []
        for j in range(k):
            pts = x_s[labels == j]
            if len(pts) == 0:
                centroids.append(x_s.mean(axis=0))
            else:
                centroids.append(pts.mean(axis=0))
        return {"scaler": scaler, "centroids": centroids, "valid": True}
    except Exception:
        return {"centroids": None, "valid": False}

# --- DEADWOOD ---

def _fit_deadwood(data, input_cols, target_col=None, params=None):
    if len(input_cols) < 2:
        raise ValueError("deadwood requires at least 2 input columns.")
    x = _extract_numpy(data, input_cols)
    try:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x)
        
        x_s = _cluster_prep_x(x_scaled, min_rows=6)
        if x_s is None:
            return {"valid": False}
            
        model = IsolationForest(random_state=42)
        preds = model.fit_predict(x_s)
        outliers = (preds == -1).astype(np.float64)
        return {"scaler": scaler, "x_train": x_s, "labels": outliers, "valid": True}
    except Exception:
        return {"valid": False}

def _apply_deadwood(data, input_cols, state, params=None):
    if not state.get("valid", False):
        return np.zeros(data.shape[0] if hasattr(data, 'shape') else len(data))
    x_test = _extract_numpy(data, input_cols)
    if "scaler" in state and state["scaler"] is not None:
        x_test = state["scaler"].transform(x_test)
    return _cluster_knn_apply(x_test, state, lambda idx, s: s["labels"][idx])

# --- MST SCORE ---

def _fit_mst_score(data, input_cols, target_col=None, params=None):
    if len(input_cols) < 2:
        raise ValueError("mst_score requires at least 2 input columns.")
    x = _extract_numpy(data, input_cols)
    try:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x)
        
        x_s = _cluster_prep_x(x_scaled, min_rows=6)
        if x_s is None:
            return {"valid": False}
            
        dists = squareform(pdist(x_s))
        mst = minimum_spanning_tree(dists).toarray()
        mst_undirected = np.maximum(mst, mst.T)
        scores = mst_undirected.max(axis=1)
        return {"scaler": scaler, "x_train": x_s, "scores": scores, "valid": True}
    except Exception:
        return {"valid": False}

def _apply_mst_score(data, input_cols, state, params=None):
    if not state.get("valid", False):
        return np.zeros(data.shape[0] if hasattr(data, 'shape') else len(data))
    x_test = _extract_numpy(data, input_cols)
    if "scaler" in state and state["scaler"] is not None:
        x_test = state["scaler"].transform(x_test)
    return _cluster_knn_apply(x_test, state, lambda idx, s: s["scores"][idx])

# --- CREATE CLUSTERING TRANSFORMERS ---

def create_clustering_transformers() -> dict:
    """
    Creates and returns a dictionary of built-in clustering and anomaly transformers.
    """
    transformers = {}
    
    transformers['genie'] = EvoTransformer(
        name="genie",
        type_="multivariate",
        input_type="numeric",
        output_type="categorical",
        fit_func=_fit_genie,
        apply_func=_apply_genie,
        name_generator=partial(gene_col_name, transformer_name="genie", prefix="gnie")
    )
    
    transformers['genie_centroid_dist'] = EvoTransformer(
        name="genie_centroid_dist",
        type_="multivariate",
        input_type="numeric",
        fit_func=_fit_genie_centroid_dist,
        apply_func=_apply_centroid_dist,
        name_generator=partial(gene_col_name, transformer_name="genie_centroid_dist", prefix="gncd")
    )
    
    transformers['lumbermark'] = EvoTransformer(
        name="lumbermark",
        type_="multivariate",
        input_type="numeric",
        output_type="categorical",
        fit_func=_fit_lumbermark,
        apply_func=_apply_lumbermark,
        name_generator=partial(gene_col_name, transformer_name="lumbermark", prefix="lmb")
    )
    
    transformers['lumbermark_centroid_dist'] = EvoTransformer(
        name="lumbermark_centroid_dist",
        type_="multivariate",
        input_type="numeric",
        fit_func=_fit_lumbermark_centroid_dist,
        apply_func=_apply_centroid_dist,
        name_generator=partial(gene_col_name, transformer_name="lumbermark_centroid_dist", prefix="lmcd")
    )
    
    transformers['deadwood'] = EvoTransformer(
        name="deadwood",
        type_="multivariate",
        input_type="numeric",
        output_type="categorical",
        fit_func=_fit_deadwood,
        apply_func=_apply_deadwood,
        name_generator=partial(gene_col_name, transformer_name="deadwood", prefix="dwd")
    )
    
    transformers['mst_score'] = EvoTransformer(
        name="mst_score",
        type_="multivariate",
        input_type="numeric",
        fit_func=_fit_mst_score,
        apply_func=_apply_mst_score,
        name_generator=partial(gene_col_name, transformer_name="mst_score", prefix="mst")
    )
    
    return transformers
