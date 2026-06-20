import polars as pl
import numpy as np

# Global registry
evo_evaluators = {}

def register_evaluator(name, train_func):
    evo_evaluators[name] = train_func

def _extract_shap_importances(sh, num_feats, feature_names):
    """
    Extracts global feature importances by averaging absolute SHAP values.
    Handles different shapes returned by LightGBM and XGBoost for binary/multiclass.
    """
    importances = np.zeros(num_feats)
    
    if len(sh.shape) == 3:
        # 3D Array: [N, num_classes, num_feats + 1] (XGBoost Multiclass)
        sh_feats = sh[:, :, :num_feats]
        # Sum across classes, then mean across samples
        sh_sum = np.sum(np.abs(sh_feats), axis=1)
        importances = np.mean(sh_sum, axis=0)
        
    elif len(sh.shape) == 2:
        if sh.shape[1] == num_feats + 1:
            # 2D Matrix: [N, num_feats + 1] (Regression/Binary)
            sh_feats = sh[:, :num_feats]
            importances = np.mean(np.abs(sh_feats), axis=0)
        else:
            # 2D Matrix: [N, (num_feats + 1) * num_classes] (LightGBM Multiclass)
            num_classes = sh.shape[1] // (num_feats + 1)
            for c in range(num_classes):
                cols = slice(c * (num_feats + 1), c * (num_feats + 1) + num_feats)
                sh_feats = sh[:, cols]
                importances += np.mean(np.abs(sh_feats), axis=0)
                
    return dict(zip(feature_names, importances))

def _train_lightgbm(x_train, y_train, x_val=None, y_val=None, task="classification", num_threads=2, num_class=None, feature_names=None, **kwargs):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is not installed. Please install it.")
        
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(x_train.shape[1])]
        
    train_data = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
    valid_sets = [train_data]
    if x_val is not None and y_val is not None:
        valid_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
        valid_sets.append(valid_data)
        
    params = {
        "objective": "multiclass" if task == "multiclass" else ("binary" if task == "classification" else "regression"),
        "metric": "multi_logloss" if task == "multiclass" else ("binary_logloss" if task == "classification" else "rmse"),
        "num_leaves": 15,
        "learning_rate": 0.1,
        "verbose": -1,
        "num_threads": num_threads,
        "seed": 42
    }
    
    if task == "multiclass" and num_class is not None:
        params["num_class"] = num_class
        
    # Update with kwargs
    for k, v in kwargs.items():
        if k not in ["n_estimators", "early_stopping_rounds"]:
            params[k] = v
            
    model = lgb.train(
        params,
        train_data,
        num_boost_round=kwargs.get("n_estimators", 50),
        valid_sets=valid_sets,
    )
    
    # Calculate SHAP feature importances using the validation set if available
    if x_val is not None:
        sh = model.predict(x_val, pred_contrib=True)
        importances = _extract_shap_importances(sh, x_train.shape[1], model.feature_name())
    else:
        # Fallback to Gain if no validation set
        importances = dict(zip(model.feature_name(), model.feature_importance(importance_type="gain")))
    
    preds = None
    if x_val is not None:
        preds = model.predict(x_val)
        
    return {
        "model": model,
        "predictions": preds,
        "importances": importances
    }

def _train_xgboost(x_train, y_train, x_val=None, y_val=None, task="classification", num_threads=2, num_class=None, feature_names=None, **kwargs):
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost is not installed. Please install it.")
        
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(x_train.shape[1])]
        
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    evals = [(dtrain, 'train')]
    if x_val is not None and y_val is not None:
        dval = xgb.DMatrix(x_val, label=y_val, feature_names=feature_names)
        evals.append((dval, 'eval'))
        
    params = {
        "objective": "multi:softprob" if task == "multiclass" else ("binary:logistic" if task == "classification" else "reg:squarederror"),
        "eval_metric": "mlogloss" if task == "multiclass" else ("logloss" if task == "classification" else "rmse"),
        "max_depth": 6,
        "eta": 0.1,
        "verbosity": 0,
        "nthread": num_threads,
        "seed": 42
    }
    
    if task == "multiclass" and num_class is not None:
        params["num_class"] = num_class
        
    # Update with kwargs
    for k, v in kwargs.items():
        if k not in ["n_estimators", "early_stopping_rounds"]:
            params[k] = v
            
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=kwargs.get("n_estimators", 50),
        evals=evals,
        verbose_eval=False
    )
    
    # Calculate SHAP feature importances using the validation set
    if x_val is not None:
        sh = model.predict(dval, pred_contribs=True)
        importances = _extract_shap_importances(sh, x_train.shape[1], model.feature_names)
    else:
        # Fallback to Gain
        importances = model.get_score(importance_type="gain")
    
    preds = None
    if x_val is not None:
        preds = model.predict(dval)
        
    return {
        "model": model,
        "predictions": preds,
        "importances": importances
    }

register_evaluator("lightgbm", _train_lightgbm)
register_evaluator("xgboost", _train_xgboost)
