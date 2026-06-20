import numpy as np
from scipy.optimize import minimize_scalar

def ts_refinement(y_true, y_pred, task="classification", num_class=None, alpha=1, is_logits=False):
    """
    Computes the Temperature Scaled Refinement (TS-Refinement) metric for binary or multiclass classification.
    The metric finds the temperature T that minimizes the Laplace-smoothed log-loss of the
    temperature-scaled prediction margins (logits).
    """
    if task not in ["classification", "multiclass"]:
        raise ValueError("TS-Refinement metric is only supported for 'classification' and 'multiclass' tasks.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return 0.0

    if task == "classification":
        if is_logits:
            z = y_pred.copy()
        else:
            # Reconstruct logits (margins) from probabilities
            p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
            z = np.log(p / (1.0 - p))
        
        # Clamp infinite values and impute NA/NaN
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.clip(z, -35.0, 35.0)

        # Laplace smooth the labels based on true class count
        N1 = np.sum(y_true == 1)
        N0 = np.sum(y_true == 0)

        y_smooth = np.zeros_like(y_true, dtype=float)
        mask_1 = (y_true == 1)
        mask_0 = (y_true == 0)
        
        if N1 > 0:
            y_smooth[mask_1] = (N1 + alpha) / (N1 + 2 * alpha)
        else:
            y_smooth[mask_1] = 0.5
            
        if N0 > 0:
            y_smooth[mask_0] = alpha / (N0 + 2 * alpha)
        else:
            y_smooth[mask_0] = 0.5

        def obj_fn(temp):
            if temp <= 0:
                temp = 1e-5
            probs_T = 1.0 / (1.0 + np.exp(-z / temp))
            probs_T = np.clip(probs_T, 1e-15, 1.0 - 1e-15)
            ll = -np.mean(y_smooth * np.log(probs_T) + (1.0 - y_smooth) * np.log(1.0 - probs_T))
            return ll

        res = minimize_scalar(obj_fn, bounds=(0.001, 10.0), method='bounded')
        best_temp = res.x

        # Return the un-smoothed log-loss at the optimal temperature
        probs_T = 1.0 / (1.0 + np.exp(-z / best_temp))
        probs_T = np.clip(probs_T, 1e-15, 1.0 - 1e-15)
        ll_unsmoothed = -np.mean(y_true * np.log(probs_T) + (1.0 - y_true) * np.log(1.0 - probs_T))
        return float(ll_unsmoothed)

    elif task == "multiclass":
        if num_class is None:
            raise ValueError("num_class must be specified for multiclass TS-Refinement.")
        
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape((-1, num_class), order='F')

        if is_logits:
            z = y_pred.copy()
        else:
            p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
            z = np.log(p)

        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.clip(z, -35.0, 35.0)

        n = len(y_true)
        # Class counts
        N_vec = np.bincount(y_true, minlength=num_class)
        N_k_row = N_vec[y_true]

        true_target = (N_k_row + alpha) / (N_k_row + 2 * alpha)
        leftover_mass = alpha / (N_k_row + 2 * alpha)

        denom = n - N_k_row
        mass_per_item = np.where(denom == 0, 0.0, leftover_mass / denom)

        # Build y_smooth matrix
        y_smooth = np.zeros((n, num_class), dtype=float)
        for c in range(num_class):
            y_smooth[:, c] = N_vec[c] * mass_per_item
        y_smooth[np.arange(n), y_true] = true_target

        def obj_fn(temp):
            if temp <= 0:
                temp = 1e-5
            z_scaled = z / temp
            z_max = np.max(z_scaled, axis=1, keepdims=True)
            z_stable = z_scaled - z_max
            exp_z = np.exp(z_stable)
            sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
            probs_T = exp_z / sum_exp_z
            probs_T = np.clip(probs_T, 1e-15, 1.0 - 1e-15)
            
            ll = -np.mean(np.sum(y_smooth * np.log(probs_T), axis=1))
            return ll

        res = minimize_scalar(obj_fn, bounds=(0.001, 10.0), method='bounded')
        best_temp = res.x

        z_scaled = z / best_temp
        z_max = np.max(z_scaled, axis=1, keepdims=True)
        z_stable = z_scaled - z_max
        exp_z = np.exp(z_stable)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        probs_T = exp_z / sum_exp_z
        probs_T = np.clip(probs_T, 1e-15, 1.0 - 1e-15)

        y_hard = np.zeros((n, num_class), dtype=float)
        y_hard[np.arange(n), y_true] = 1.0
        ll_unsmoothed = -np.mean(np.sum(y_hard * np.log(probs_T), axis=1))
        return float(ll_unsmoothed)
