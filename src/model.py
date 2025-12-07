# Model training and prediction
"""
This module contains model classes for traffic congestion prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple, Optional
import joblib

from .config import Config


class TrafficCongestionModel:
    """Model for predicting traffic congestion levels."""
    
    def __init__(self, config: Config = None, **model_params):
        """
        Initialize the model.
        
        Args:
            config: Configuration object.
            **model_params: Additional parameters to pass to RandomForestClassifier.
        """
        self.config = config or Config()
        
        # Use config params as defaults, override with any passed params
        params = {**self.config.MODEL_PARAMS, **model_params}
        
        self.model = RandomForestClassifier(**params)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names = None
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        feature_cols: List[str] = None
    ) -> 'TrafficCongestionModel':
        """
        Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            feature_cols: List of feature column names (optional).
            
        Returns:
            self
        """
        self.feature_names = feature_cols or list(X_train.columns)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Train model
        self.model.fit(X_train[self.feature_names], y_encoded)
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
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Evaluate the model on validation data.
        
        Args:
            X_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            Tuple of (y_val_encoded, y_pred_encoded, classification_report_str)
        """
        y_val_encoded = self.label_encoder.transform(y_val)
        y_pred_encoded = self.predict_encoded(X_val)
        
        report = classification_report(
            y_val_encoded, 
            y_pred_encoded, 
            target_names=self.label_encoder.classes_
        )
        
        return y_val_encoded, y_pred_encoded, report
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances from the model.
        
        Returns:
            Array of feature importances.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance.")
        
        return self.model.feature_importances_
    
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

