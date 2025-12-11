# Model training and prediction
"""
This module contains model classes for traffic congestion prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.base import clone
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.utils.class_weight import compute_sample_weight
from .config import Config
from .model_dispatcher import MODELS


class TrafficCongestionModel:
    """Model for predicting traffic congestion levels."""
    
    def __init__(self, config: Config = None, model_name: str = "random_forest"):
        """
        Initialize the model.
        
        Args:
            config: Configuration object.
            model_name: Name of model from MODELS dispatcher.
        """
        self.config = config or Config()
        self.model_name = model_name
        self.model = clone(MODELS.get(model_name, MODELS["random_forest"]))
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names = None
        self.cv_scores = []
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        feature_cols: List[str] = None
    ) -> 'TrafficCongestionModel':
        """
        Train the model on full data.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            feature_cols: List of feature column names (optional).
            
        Returns:
            self
        """
        self.feature_names = feature_cols or list(X_train.columns)
        y_encoded = self.label_encoder.fit_transform(y_train)
        self.model.fit(X_train[self.feature_names], y_encoded)
        self.is_fitted = True
        return self
    
    def fit_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str] = None,
        n_splits: int = 5,
        random_state: int = 42
    ) -> 'TrafficCongestionModel':
        """
        Train with Stratified K-Fold cross-validation.
        
        Args:
            X: Features.
            y: Labels.
            feature_cols: Feature column names.
            n_splits: Number of folds.
            random_state: Random seed for reproducibility.
            
        Returns:
            self (trained on full data after CV)
        """
        self.feature_names = feature_cols or list(X.columns)
        y_encoded = self.label_encoder.fit_transform(y)
        X_feat = X[self.feature_names].values
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.cv_scores = []

        if self.model_name == 'xgb':
            sample_weight = compute_sample_weight('balanced', y_encoded)

        print(f"Training {self.model_name} with {n_splits}-fold Stratified CV...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y_encoded)):
            model_fold = clone(self.model)
            if self.model_name == 'xgb':
                model_fold.fit(X_feat[train_idx], y_encoded[train_idx], sample_weight=sample_weight)
            else:
                model_fold.fit(X_feat[train_idx], y_encoded[train_idx])
            
            score = model_fold.score(X_feat[val_idx], y_encoded[val_idx])
            self.cv_scores.append(score)
            print(f"  Fold {fold+1}: {score:.4f}")
        
        print(f"  Mean CV: {np.mean(self.cv_scores):.4f} (+/- {np.std(self.cv_scores):.4f})")
        
        # Final fit on all data
        self.model.fit(X_feat, y_encoded)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Array of predicted labels (decoded).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        y_pred_encoded = self.model.predict(X[self.feature_names])
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_encoded(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions (encoded format).
        
        Args:
            X: Features to predict on.
            
        Returns:
            Array of predicted labels (encoded).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        return self.model.predict(X[self.feature_names])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Array of class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        return self.model.predict_proba(X[self.feature_names])
    
    def evaluate(
        self, 
        X_val: pd.DataFrame, 
        y_val: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Evaluate using competition metric: 0.7*Macro-F1 + 0.3*Accuracy.
        
        Args:
            X_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            Tuple of (y_val_encoded, y_pred_encoded, metrics_dict)
        """
        y_val_encoded = self.label_encoder.transform(y_val)
        y_pred_encoded = self.predict_encoded(X_val)
        
        # Competition metrics
        macro_f1 = f1_score(y_val_encoded, y_pred_encoded, average='macro')
        accuracy = accuracy_score(y_val_encoded, y_pred_encoded)
        competition_score = 0.7 * macro_f1 + 0.3 * accuracy
        
        report = classification_report(
            y_val_encoded, 
            y_pred_encoded, 
            target_names=self.label_encoder.classes_
        )
        
        metrics = {
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'competition_score': competition_score,
            'report': report
        }
        
        return y_val_encoded, y_pred_encoded, metrics
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances from the model.
        
        Returns:
            Array of feature importances (or None if not available).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance.")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_classes(self) -> np.ndarray:
        """
        Get the class labels.
        
        Returns:
            Array of class labels.
        """
        return self.label_encoder.classes_
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model.
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str, config: Config = None) -> 'TrafficCongestionModel':
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from.
            config: Configuration object.
            
        Returns:
            Loaded TrafficCongestionModel.
        """
        model_data = joblib.load(path)
        
        instance = cls(config=config)
        instance.model = model_data['model']
        instance.label_encoder = model_data['label_encoder']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        return instance

