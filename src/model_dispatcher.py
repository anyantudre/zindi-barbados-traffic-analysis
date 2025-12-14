# Model dispatcher for easy model swapping
"""
Define models and their hyperparameters here.
Usage: from src.model_dispatcher import MODELS, get_model
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, accuracy_score

def competition_score(y_true, y_pred) -> float:
    """
    Score officiel de la compétition : 0.7 * macro-F1 + 0.3 * accuracy.
    Accepte y_pred sous forme de labels (1D) ou de proba/logits (2D).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return 0.7 * f1 + 0.3 * acc


def competition_metric_xgb(y_pred, dtrain):
    """
    Métrique custom pour l'API native XGBoost (eval_metric).
    y_pred est aplati; on le reshape en (n_samples, n_classes) si besoin.
    Retourne (name, value).
    """
    labels = dtrain.get_label()
    preds = np.asarray(y_pred)
    n = labels.shape[0]
    if preds.ndim == 1 and n > 0:
        num_class = max(1, preds.size // n)
        preds = preds.reshape(n, num_class)
    if preds.ndim > 1:
        pred_labels = preds.argmax(axis=1)
    else:
        pred_labels = preds
    score = competition_score(labels, pred_labels)
    return "competition", score


def competition_metric_sklearn(y_true, y_pred) -> float:
    """
    Métrique pour l'interface scikit-learn (eval_metric dans XGBClassifier
    ou cross_val_score). Retourne un float directement.
    """
    return competition_score(y_true, y_pred)


def get_model(model_name: str, device: str = "cpu"):
    """
    Get a model instance with proper device configuration.
    
    Args:
        model_name: Name of the model.
        device: Device to use ('cuda' or 'cpu').
        
    Returns:
        Model instance.
    """
    if model_name == "random_forest":
        return RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            bootstrap=True,
            **{'n_estimators': 500, 'max_depth': 13, 'min_samples_split': 16, 'min_samples_leaf': 8, 'max_features': None}
        )
    
    elif model_name == "xgb":
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_jobs=-1,
                device='cuda' if device == 'cuda' else 'cpu',
                objective='multi:softmax',
                eval_metric=competition_metric_sklearn,
                random_state=42,
                **{'learning_rate': 0.01061515021837236, 'max_depth': 7, 'min_child_weight': 3, 'subsample': 0.7303160466926395, 'colsample_bytree': 0.7424793969110655, 'reg_alpha': 0.04224199746665908, 'reg_lambda': 0.003608443241043876}
            )
            # Set the estimator type for scikit-learn compatibility
            model._estimator_type = "classifier"
            return model
        except ImportError:
            raise ImportError("xgboost not installed")
    
    elif model_name == "lgbm":
        try:
            import lightgbm as lgbm
            return lgbm.LGBMClassifier(
                n_jobs=-1,
                verbose=-1,
                device='gpu' if device == 'cuda' else 'cpu',
                random_state=42,
                class_weight='balanced',
                **{
                    'num_leaves': 159, 'max_depth': 10, 
                    'learning_rate': 0.07201934773717454, 
                    'n_estimators': 1000, 
                    'min_child_samples': 11, 
                    'subsample': 0.6724360662104916, 
                    'colsample_bytree': 0.9689043505892945, 
                    'reg_alpha': 2.7261162591155736, 
                    'reg_lambda': 5.16531384161136
                }
            )
        except ImportError:
            raise ImportError("lightgbm not installed")
    
    elif model_name == "catboost":
        try:
            import catboost

            params = {
                'depth': 9, 
                'learning_rate': 0.022515921040338078, 
                'iterations': 262, 
                'l2_leaf_reg': 0.0002240008599075843, 
                'bagging_temperature': 0.9243734060325552, 
                'random_strength': 5.811210933306866
            }
            return catboost.CatBoostClassifier(
                        **params,
                        loss_function="MultiClass",
                        eval_metric="TotalF1",      # CatBoost’s multiclass F1
                        task_type="GPU",
                        devices="0",
                        auto_class_weights="Balanced",
                        verbose=False,
                        random_state=42
                    )
        except ImportError:
            raise ImportError("catboost not installed")
    
    elif model_name == "ensemble_soft":
        # Build base estimators with device
        base_estimators = [
            ("rf", get_model("random_forest", device))
        ]
        try:
            base_estimators.append(("xgb", get_model("xgb", device)))
        except ImportError:
            pass
        try:
            base_estimators.append(("lgbm", get_model("lgbm", device)))
        except ImportError:
            pass
        try:
            base_estimators.append(("catboost", get_model("catboost", device)))
        except ImportError:
            pass
        
        return VotingClassifier(
            estimators=base_estimators,
            voting='soft',
            n_jobs=-1
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Legacy MODELS dict for backward compatibility (uses CPU by default)
MODELS = {
    "random_forest": RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        **{'n_estimators': 1000, 'max_depth': 13, 'min_samples_split': 16, 'min_samples_leaf': 8, 'max_features': None}
    ),
}

