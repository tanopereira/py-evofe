import optuna
import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss, mean_squared_error
from .models import evo_evaluators, register_evaluator

def tune_evaluator(
    X, y, 
    task="classification", 
    evaluator="lightgbm",
    n_trials=10, 
    cv_folds=3, 
    num_class=None
):
    """
    Hyperparameter tuning for registered evaluators using Optuna.
    Equivalent to the 'lightgbm_mbo' tuner in evoFE.
    """
    
    if evaluator not in evo_evaluators:
        raise ValueError(f"Evaluator '{evaluator}' not found in registry.")
    
    train_func = evo_evaluators[evaluator]
    
    # Disable optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    if task == "classification" or task == "multiclass":
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
    def objective(trial):
        if evaluator == "lightgbm":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", -1, 10),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 20, 100)
            }
        elif evaluator == "xgboost":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 20, 100)
            }
        else:
            params = {}
            
        scores = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            res = train_func(
                x_train=X_train, y_train=y_train, 
                x_val=X_val, y_val=y_val, 
                task=task, 
                num_class=num_class,
                feature_names=kwargs.get('feature_names') if 'kwargs' in locals() else None,
                **params
            )
            
            preds = res["predictions"]
            
            if task == "classification" or task == "multiclass":
                score = -log_loss(y_val, preds, labels=np.unique(y))
            else:
                score = -np.sqrt(mean_squared_error(y_val, preds))
                
            scores.append(score)
            
        return np.mean(scores)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def _make_optuna_evaluator(base_evaluator_name, n_trials=5):
    """
    Creates an inline Optuna evaluator that tunes hyperparameters per fold.
    Equivalent to the R package's `lightgbm_mbo` inline evaluator.
    """
    def train_func(x_train, y_train, x_val=None, y_val=None, task="classification", num_threads=2, num_class=None, **kwargs):
        base_train = evo_evaluators[base_evaluator_name]
        
        # We need a robust class list for log_loss
        labels = None
        if task == "classification":
            labels = [0, 1]
        elif task == "multiclass":
            if num_class is not None:
                labels = list(range(num_class))
            else:
                # Combine y_train and y_val to find all possible classes
                y_comb = y_train if y_val is None else np.concatenate([y_train, y_val])
                labels = np.unique(y_comb)
            
        def objective(trial):
            if base_evaluator_name == "lightgbm":
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", -1, 10),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "n_estimators": trial.suggest_int("n_estimators", 20, 100)
                }
            elif base_evaluator_name == "xgboost":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "n_estimators": trial.suggest_int("n_estimators", 20, 100)
                }
            else:
                params = {}
                
            res = base_train(
                x_train, y_train, x_val, y_val, 
                task=task, num_threads=num_threads, num_class=num_class, **kwargs, **params
            )
            
            preds = res["predictions"]
            if preds is None:
                return 0.0 # Should not happen since cv.py provides x_val
                
            if task == "classification" or task == "multiclass":
                score = -log_loss(y_val, preds, labels=labels)
            else:
                score = -np.sqrt(mean_squared_error(y_val, preds))
            return score
            
        def optuna_callback(study, trial):
            print(f"    [Optuna Trial {trial.number}] params={trial.params} -> Score: {trial.value:.4f}")
            
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        print(f"  [MBO] Starting {base_evaluator_name} Hyperparameter Tuning (Iters: {n_trials})...")
        study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
        
        # Train final model with best params
        best_params = study.best_params
        return base_train(
            x_train, y_train, x_val, y_val, 
            task=task, num_threads=num_threads, num_class=num_class, **kwargs, **best_params
        )
    return train_func

# Register the inline Optuna evaluators
register_evaluator("lightgbm_optuna", _make_optuna_evaluator("lightgbm", n_trials=5))
register_evaluator("xgboost_optuna", _make_optuna_evaluator("xgboost", n_trials=5))


def make_tunable(
    base_model_name: str,
    param_ranges: dict,
    tuner_name: str = None,
    n_trials: int = 10,
):
    """
    Wrap any registered evaluator in an Optuna tuning loop and register it.

    Mirrors R's ``make_tunable(base_model_name, param_ranges, tuner_name)``.

    Parameters
    ----------
    base_model_name : str
        Name of an already-registered base evaluator (e.g. ``"lightgbm"``).
    param_ranges : dict
        Dict of parameter definitions. Each value is a dict with keys:
        - ``type``: ``"numeric"``, ``"integer"``, or ``"discrete"``
        - ``lower`` / ``upper``: bounds (for numeric / integer)
        - ``values``: list of choices (for discrete)
    tuner_name : str | None
        Name under which to register the tuned evaluator.
        Defaults to ``base_model_name + "_tuned"``.
    n_trials : int
        Number of Optuna trials per evaluation call.

    Examples
    --------
    >>> make_tunable(
    ...     "lightgbm",
    ...     param_ranges={
    ...         "num_leaves": {"type": "integer", "lower": 7, "upper": 63},
    ...         "learning_rate": {"type": "numeric", "lower": 0.01, "upper": 0.3},
    ...     },
    ...     tuner_name="lgb_tuned",
    ... )
    """
    if base_model_name not in evo_evaluators:
        raise ValueError(
            f"Base evaluator '{base_model_name}' not found in registry. "
            f"Registered evaluators: {list(evo_evaluators.keys())}"
        )
    if tuner_name is None:
        tuner_name = f"{base_model_name}_tuned"

    base_train = evo_evaluators[base_model_name]

    def _suggest(trial, name, spec):
        t = spec.get("type", "numeric")
        if t == "integer":
            return trial.suggest_int(name, int(spec["lower"]), int(spec["upper"]))
        elif t == "discrete":
            return trial.suggest_categorical(name, spec["values"])
        else:  # numeric / float
            return trial.suggest_float(
                name, float(spec["lower"]), float(spec["upper"]),
                log=spec.get("log", False)
            )

    def train_func(x_train, y_train, x_val=None, y_val=None,
                   task="classification", num_threads=2, num_class=None,
                   feature_names=None, **kwargs):
        from sklearn.metrics import log_loss
        import optuna

        labels = None
        if task == "classification":
            labels = [0, 1]
        elif task == "multiclass":
            import numpy as np
            if num_class is not None:
                labels = list(range(num_class))
            else:
                y_comb = y_train if y_val is None else np.concatenate([y_train, y_val])
                labels = np.unique(y_comb)

        def objective(trial):
            import numpy as np
            params = {name: _suggest(trial, name, spec)
                      for name, spec in param_ranges.items()}
            res = base_train(
                x_train, y_train, x_val, y_val,
                task=task, num_threads=num_threads,
                num_class=num_class, feature_names=feature_names,
                **kwargs, **params
            )
            preds = res["predictions"]
            if preds is None:
                return 0.0
            if task in ("classification", "multiclass"):
                return float(np.exp(-log_loss(y_val, preds, labels=labels)))
            return -float(np.sqrt(np.mean((y_val - preds) ** 2)))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        return base_train(
            x_train, y_train, x_val, y_val,
            task=task, num_threads=num_threads,
            num_class=num_class, feature_names=feature_names,
            **kwargs, **best_params
        )

    register_evaluator(tuner_name, train_func)
