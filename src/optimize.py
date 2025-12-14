# Hyperparameter optimization with Optuna
"""
Usage: 
  from src.optimize import optimize_model
  best_params = optimize_model(training_df, config, model_name="lgbm", n_trials=50)
"""

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight


from src.model_dispatcher import competition_metric_xgb
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_param_space(trial, model_name: str) -> dict:
    """Define hyperparameter search space for each model."""
    
    if model_name == "xgb":
        import xgboost as xgb

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            
        }
        return {
            'model': xgb.XGBClassifier(
                **params,
                n_jobs=-1, device='cuda', random_state=42, eval_metric=competition_metric_xgb,
                objective='multi:softmax',
                n_estimators=500,
                num_class=4,
            )
        }
    
    elif model_name == "lgbm":
        import lightgbm as lgbm
        params = {
                    "num_leaves": trial.suggest_int("num_leaves", 20, 2000, step=20),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),

                    # regularization
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 8000, step=100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),

                    # sampling
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "bagging_freq": 1,

                    # structure
                    "max_bin": trial.suggest_int("max_bin", 200, 300),

                    # quality
                    "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 10),
                }
        return {
            'model': lgbm.LGBMClassifier(
                                            **params,
                                            n_estimators=500,
                                            n_jobs=-1,
                                            verbose=-1,
                                            device='gpu',
                                            random_state=42,
                                            class_weight='balanced'
                                        )
        }
    
    elif model_name == "catboost":
        import catboost

        params = {
                "depth": trial.suggest_int("depth", 4, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

                # regularization
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),

                # tree & sampling behavior
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
                "random_strength": trial.suggest_float("random_strength", 0.5, 20.0),

                # iterations
                "iterations": trial.suggest_int("iterations", 100, 500),
            }
        return {
            'model': catboost.CatBoostClassifier(
                **params,
                loss_function="MultiClass",
                eval_metric="TotalF1",      # CatBoost’s multiclass F1
                task_type="GPU",
                devices="0",
                auto_class_weights="Balanced",
                verbose=False,
                random_state=42
            )
        }
    
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),

            # tree structure
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),

            # feature sampling
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

        return {
            'model': RandomForestClassifier(
                **params,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
                bootstrap=True
            )
        }
    
    raise ValueError(f"Unknown model: {model_name}")


def objective(trial, X, y, model_name, n_folds, random_state):
    """Objective function - maximize Macro-F1."""
    params = get_param_space(trial, model_name)
    model = params['model']
    
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model_clone = clone(model)
        if model_name == 'xgb':
            sample_weight = compute_sample_weight('balanced', y_train)
            model_clone.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)
        
        f1 = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def optimize_model(
    training_df,
    config,
    model_name: str = "lgbm",
    n_trials: int = 50,
    n_folds: int = 5,
    random_state: int = 42
) -> dict:
    """
    Optimize hyperparameters for a model using Optuna.

    Args:
        training_df: Training DataFrame with features and 'congestion_rating'.
        config: Config object with configuration settings.
        model_name: Model to optimize ("xgb", "lgbm", "catboost", "random_forest").
        n_trials: Number of Optuna trials.
        n_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        Dict with best_params, best_score, study.
    """
    # Dynamically determine feature columns (exclude non-feature columns)
    feature_cols = [col for col in training_df.columns
                   if col not in config.ID_COLS + ['ID', 'cycle_phase', 'ID_type', 'video_path', 'congestion_rating']]

    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER OPTIMIZATION: {model_name.upper()}")
    print(f"Metric: Macro-F1 | Folds: {n_folds} | Trials: {n_trials}")
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    print(f"{'='*60}")

    X = training_df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(training_df['congestion_rating'])
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y, model_name, n_folds, random_state),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE!")
    print(f"Best Macro-F1: {study.best_value:.4f}")
    print(f"Best Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"{'='*60}\n")
    
    return {
        "model_name": model_name,
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study
    }

