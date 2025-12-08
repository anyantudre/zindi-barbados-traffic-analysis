# Model dispatcher for easy model swapping
"""
Define models and their hyperparameters here.
Usage: from src.model_dispatcher import MODELS
"""

from sklearn.ensemble import RandomForestClassifier

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
}

# Optional: Add GPU models if available
try:
    import xgboost as xgb
    MODELS["xgb"] = xgb.XGBClassifier(
        n_jobs=-1,
        device='cuda',
        objective='multi:softmax',
        eval_metric='mlogloss',
        random_state=42,
        learning_rate=0.03,
        max_depth=10,
        min_child_weight=13,
        subsample=0.85,
        colsample_bytree=0.68,
        reg_alpha=0.05,
        reg_lambda=0.006,
        n_estimators=374,
    )
except ImportError:
    pass

try:
    import lightgbm as lgbm
    MODELS["lgbm"] = lgbm.LGBMClassifier(
        n_jobs=-1,
        verbose=-1,
        device='gpu',
        random_state=42,
        num_leaves=61,
        max_depth=14,
        learning_rate=0.07,
        n_estimators=324,
        min_child_samples=59,
        subsample=0.86,
        colsample_bytree=0.80,
        reg_alpha=9.3,
        reg_lambda=0.004,
        class_weight='balanced',
    )
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
    )
except ImportError:
    pass

