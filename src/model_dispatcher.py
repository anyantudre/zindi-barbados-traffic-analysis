# Model dispatcher for easy model swapping
"""
Define models and their hyperparameters here.
Usage: from src.model_dispatcher import MODELS
"""

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
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
        eval_metric='mlogloss',
        random_state=42,
        **{
            'learning_rate': 0.19227540215073882, 
            'max_depth': 3, 'min_child_weight': 19, 
            'subsample': 0.8527684825662769, 
            'colsample_bytree': 0.9133703751855122, 
            'reg_alpha': 0.03406409161937166, 'reg_lambda': 4.405288594138643, 
            'n_estimators': 123
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

