# Model dispatcher for easy model swapping
"""
Define models and their hyperparameters here.
Usage: from src.model_dispatcher import MODELS
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


MODELS = {
    "random_forest": RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,

        **{'n_estimators': 187, 'max_depth': 13, 'min_samples_split': 16, 'min_samples_leaf': 8, 'max_features': None}

    ),
}

# Store base estimators for ensemble
_BASE_ESTIMATORS = [("rf", MODELS["random_forest"])]

# Optional: Add GPU models if available
try:
    import xgboost as xgb
    MODELS["xgb"] = xgb.XGBClassifier(
        n_jobs=-1,
        device='cuda',
        objective='multi:softmax',
        eval_metric=competition_metric_sklearn,
        random_state=42,
        **{
            'learning_rate': 0.01, 
            'max_depth': 10, 'min_child_weight': 1, 
            'subsample': 0.8, 
            'colsample_bytree': 0.8, 
            'reg_alpha': 0.01, 'reg_lambda': 0.01, 
            'n_estimators': 500
        }
    )
    _BASE_ESTIMATORS.append(("xgb", MODELS["xgb"]))
except ImportError:
    pass

try:
    import lightgbm as lgbm
    MODELS["lgbm"] = lgbm.LGBMClassifier(
        n_jobs=-1,
        verbose=-1,
        device='gpu',
        random_state=42,
        class_weight='balanced',
        **{
            'num_leaves': 159, 'max_depth': 4, 
            'learning_rate': 0.07201934773717454, 
            'n_estimators': 332, 
            'min_child_samples': 11, 
            'subsample': 0.6724360662104916, 
            'colsample_bytree': 0.9689043505892945, 
            'reg_alpha': 2.7261162591155736, 
            'reg_lambda': 5.16531384161136
        }
    )
    _BASE_ESTIMATORS.append(("lgbm", MODELS["lgbm"]))
except ImportError:
    pass

try:
    import catboost
    MODELS["catboost"] = catboost.CatBoostClassifier(
        verbose=False,
        task_type="GPU",
        devices='0',
        random_state=42,
        auto_class_weights='Balanced',
        **{
            'depth': 9, 
            'learning_rate': 0.022515921040338078, 
            'iterations': 262, 
            'l2_leaf_reg': 0.0002240008599075843, 
            'bagging_temperature': 0.9243734060325552, 
            'random_strength': 5.811210933306866
        }
    )
    #_BASE_ESTIMATORS.append(("catboost", MODELS["catboost"]))
except ImportError:
    pass

# =============================================================================
# Ensemble Models (combine all available base models)
# =============================================================================

# Hard Voting: majority vote
MODELS["ensemble_hard"] = VotingClassifier(
    estimators=_BASE_ESTIMATORS.copy(),
    voting='hard',
    n_jobs=-1
)

# Soft Voting: average probabilities (usually better)
MODELS["ensemble_soft"] = VotingClassifier(
    estimators=_BASE_ESTIMATORS.copy(),
    voting='soft',
    n_jobs=-1
)

