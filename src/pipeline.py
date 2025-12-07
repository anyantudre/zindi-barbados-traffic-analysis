# Training and inference pipelines
"""
This module contains the main pipelines for training and inference.
Provides a unified interface for the entire workflow.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from IPython.display import display

from .config import Config
from .data_loader import DataLoader
from .feature_extraction import FeatureExtractor, get_or_extract_features
from .preprocessing import DataPreprocessor
from .model import TrafficCongestionModel
from .visualization import Visualizer


class TrainingPipeline:
    """Pipeline for training the traffic congestion model."""
    
    def __init__(self, config: Config = None):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.visualizer = Visualizer()
        self.model = None
        
        # Data containers
        self.train_df = None
        self.test_df = None
        self.videos_df = None
        self.features_df = None
        self.training_df = None
        self.validation_df = None
        self.testing_df = None
    
    def setup(self) -> 'TrainingPipeline':
        """
        Setup the environment.
        
        Returns:
            self
        """
        self.config.setup_environment()
        # Update feature columns based on active flags
        self.config.FEATURES_COLS = self.config.get_features_cols()
        return self
    
    def load_data(self) -> 'TrainingPipeline':
        """
        Load all required data.
        
        Returns:
            self
        """
        print("Loading data...")
        self.videos_df, self.train_df, self.test_df, self.features_df = \
            self.data_loader.load_all_data()
        print(f"Loaded {len(self.train_df)} training samples and {len(self.test_df)} test samples")
        return self
    
    def preprocess(self) -> 'TrainingPipeline':
        """
        Preprocess the data.
        
        Returns:
            self
        """
        print("Preprocessing data...")
        self.training_df, self.validation_df, self.testing_df = \
            self.preprocessor.prepare_full_dataset(
                self.train_df, 
                self.test_df, 
                self.videos_df, 
                self.features_df
            )
        print(f"Training: {len(self.training_df)}, Validation: {len(self.validation_df)}, Testing: {len(self.testing_df)}")
        return self
    
    def train(self, **model_params) -> 'TrainingPipeline':
        """
        Train the model.
        
        Args:
            **model_params: Additional model parameters.
            
        Returns:
            self
        """
        print("Training model...")
        
        X_train = self.training_df[self.config.FEATURES_COLS]
        y_train = self.training_df['congestion_rating']
        
        self.model = TrafficCongestionModel(self.config, **model_params)
        self.model.fit(X_train, y_train, self.config.FEATURES_COLS)
        
        print("Model training complete!")
        return self
    
    def evaluate(self, show_plots: bool = True) -> Dict:
        """
        Evaluate the model on validation data.
        
        Args:
            show_plots: Whether to display evaluation plots.
            
        Returns:
            Dictionary with evaluation results.
        """
        print("Evaluating model...")
        
        X_val = self.validation_df[self.config.FEATURES_COLS]
        y_val = self.validation_df['congestion_rating']
        
        y_val_encoded, y_pred_encoded, report = self.model.evaluate(X_val, y_val)
        
        print(report)
        
        if show_plots:
            # Confusion matrix
            self.visualizer.plot_confusion_matrix(
                y_val_encoded, 
                y_pred_encoded, 
                self.model.get_classes()
            )
            
            # Feature importance
            self.visualizer.plot_feature_importance(
                self.model.get_feature_importance(),
                self.config.FEATURES_COLS
            )
        
        return {
            'y_true': y_val_encoded,
            'y_pred': y_pred_encoded,
            'report': report
        }
    
    def run(self, show_plots: bool = True, **model_params) -> 'TrainingPipeline':
        """
        Run the full training pipeline.
        
        Args:
            show_plots: Whether to display evaluation plots.
            **model_params: Additional model parameters.
            
        Returns:
            self
        """
        self.setup()
        self.load_data()
        self.preprocess()
        self.train(**model_params)
        self.evaluate(show_plots)
        return self
    
    def save_model(self, path: str = 'model.joblib') -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save. Run training first.")
        self.model.save(path)
        print(f"Model saved to {path}")


class InferencePipeline:
    """Pipeline for making predictions with a trained model."""
    
    def __init__(self, config: Config = None, model: TrafficCongestionModel = None):
        """
        Initialize the inference pipeline.
        
        Args:
            config: Configuration object.
            model: Trained model (optional, can be loaded later).
        """
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.model = model
        
        # Data containers
        self.testing_df = None
    
    def load_model(self, path: str) -> 'InferencePipeline':
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model.
            
        Returns:
            self
        """
        self.model = TrafficCongestionModel.load(path, self.config)
        print(f"Model loaded from {path}")
        return self
    
    def set_model(self, model: TrafficCongestionModel) -> 'InferencePipeline':
        """
        Set the model directly.
        
        Args:
            model: Trained model.
            
        Returns:
            self
        """
        self.model = model
        return self
    
    def prepare_test_data(
        self, 
        testing_df: pd.DataFrame = None
    ) -> 'InferencePipeline':
        """
        Prepare test data for inference.
        
        Args:
            testing_df: Pre-processed testing dataframe (optional).
            
        Returns:
            self
        """
        if testing_df is not None:
            self.testing_df = testing_df
        else:
            # Load and preprocess from scratch
            print("Loading and preprocessing test data...")
            videos_df, train_df, test_df, features_df = self.data_loader.load_all_data()
            _, _, self.testing_df = self.preprocessor.prepare_full_dataset(
                train_df, test_df, videos_df, features_df
            )
        return self
    
    def predict(self) -> pd.DataFrame:
        """
        Make predictions on test data.
        
        Returns:
            DataFrame with predictions.
        """
        if self.model is None:
            raise ValueError("No model available. Load or set a model first.")
        if self.testing_df is None:
            raise ValueError("No test data. Call prepare_test_data first.")
        
        print("Making predictions...")
        
        X_test = self.testing_df[self.config.FEATURES_COLS]
        predictions = self.model.predict(X_test)
        
        # Create submission dataframe
        submission_df = self.testing_df.copy()
        submission_df['Target'] = predictions
        submission_df['Target_Accuracy'] = predictions
        submission_df = submission_df[['ID', 'Target', 'Target_Accuracy']]
        
        return submission_df
    
    def predict_and_save(
        self, 
        output_path: str = 'submission.csv'
    ) -> pd.DataFrame:
        """
        Make predictions and save to CSV.
        
        Args:
            output_path: Path for the output CSV file.
            
        Returns:
            DataFrame with predictions.
        """
        submission_df = self.predict()
        submission_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        display(submission_df.shape, submission_df.head())
        return submission_df
    
    def run(
        self, 
        model_path: str = None, 
        testing_df: pd.DataFrame = None,
        output_path: str = 'submission.csv'
    ) -> pd.DataFrame:
        """
        Run the full inference pipeline.
        
        Args:
            model_path: Path to saved model (if not already loaded).
            testing_df: Pre-processed testing dataframe (optional).
            output_path: Path for the output CSV file.
            
        Returns:
            DataFrame with predictions.
        """
        if model_path and self.model is None:
            self.load_model(model_path)
        
        self.prepare_test_data(testing_df)
        return self.predict_and_save(output_path)


class ExplorationPipeline:
    """Pipeline for data exploration and visualization."""
    
    def __init__(self, config: Config = None):
        """
        Initialize the exploration pipeline.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.visualizer = Visualizer()
    
    def explore_dataset_stats(self) -> None:
        """Show dataset statistics."""
        print("Collecting dataset statistics...")
        self.data_loader.collect_stats(self.config.COMPRESSED_PATHS_DICT, "compressed")
        self.data_loader.collect_stats(self.config.FULL_PATHS_DICT, "full")
    
    def explore_label_distribution(self, train: pd.DataFrame) -> None:
        """
        Explore label distribution.
        
        Args:
            train: Training dataframe.
        """
        self.visualizer.plot_label_distribution(train)
    
    def explore_temporal_patterns(self, train: pd.DataFrame) -> None:
        """
        Explore temporal patterns.
        
        Args:
            train: Training dataframe with time features.
        """
        train = self.preprocessor.add_time_features(train)
        print("Max hour in train:", train["hour"].max())
        print("Unique Dates:", train["Date"].unique().tolist())
        self.visualizer.plot_hourly_distribution(train)
    
    def explore_signaling(self, train: pd.DataFrame) -> None:
        """
        Explore signaling patterns.
        
        Args:
            train: Training dataframe.
        """
        self.visualizer.plot_signaling_distribution(train)
        self.visualizer.plot_signaling_vs_congestion(train)
    
    def show_sample_video(self, train: pd.DataFrame, index: int = 0) -> None:
        """
        Show a sample video.
        
        Args:
            train: Training dataframe with video_path column.
            index: Index of the video to show.
        """
        video_path = train['video_path'].iloc[index]
        self.visualizer.show_video(video_path)

