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
from .preprocessing import DataPreprocessor
from .model import TrafficCongestionModel
from .visualization import Visualizer
from .video_models import VideoTransformerModel


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

        # Store the dynamic feature columns for use in training/evaluation
        # Dynamically determine feature columns from the training data
        # Exclude metadata columns but keep time_segment_id for grouping
        exclude_cols = ['ID', 'cycle_phase', 'ID_type', 'video_path', 'congestion_rating', 
                        'video_time', 'date', 'Date', 'view_label']
        feature_cols = [col for col in self.training_df.columns
                       if col not in exclude_cols]
        self.dynamic_feature_cols = [col for col in feature_cols if col in self.training_df.columns]

        print(f"Training: {len(self.training_df)}, Validation: {len(self.validation_df)}, Testing: {len(self.testing_df)}")
        print(f"Using {len(self.dynamic_feature_cols)} feature columns: {self.dynamic_feature_cols}")
        return self
    
    def train(self, model_name: str = "random_forest", use_cv: bool = False, 
              n_splits: int = 5, use_wandb: bool = False, 
              video_col: str = "video_path", **kwargs) -> 'TrainingPipeline':
        """
        Train the model.
        
        Args:
            model_name: Name of model ("random_forest", "lgbm", "videomae", etc.)
            use_cv: Whether to use Stratified K-Fold CV (ML models only).
            n_splits: Number of CV folds.
            use_wandb: Enable W&B logging (video models).
            video_col: Column with video paths (video models).
            **kwargs: Additional args for video models (num_epochs, batch_size).
            
        Returns:
            self
        """
        y_train = self.training_df['congestion_rating']
        
        # Video transformer models
        if model_name == "videomae":
            self.model = VideoTransformerModel(
                config=self.config,
                num_epochs=kwargs.get('num_epochs', 4),
                batch_size=kwargs.get('batch_size', 4),
            )
            
            X_val = self.validation_df if self.validation_df is not None else None
            y_val = self.validation_df['congestion_rating'] if X_val is not None else None
            
            self.model.fit(
                self.training_df, y_train,
                X_val=X_val, y_val=y_val,
                video_col=video_col,
                use_wandb=use_wandb,
            )
        else:
            # ML models
            X_train = self.training_df[self.dynamic_feature_cols]
            self.model = TrafficCongestionModel(self.config, model_name=model_name)

            if use_cv:
                # Use GroupKFold with time_segment_id to prevent data leakage across views
                groups = self.training_df['time_segment_id'] if 'time_segment_id' in self.training_df.columns else None
                self.model.fit_cv(X_train, y_train, self.dynamic_feature_cols, n_splits, groups=groups)
            else:
                print(f"Training {model_name}...")
                self.model.fit(X_train, y_train, self.dynamic_feature_cols)
        
        print("Model training complete!")
        return self
    
    def evaluate(self, show_plots: bool = True, video_col: str = "video_path") -> Dict:
        """
        Evaluate using competition metric: 0.7*Macro-F1 + 0.3*Accuracy.
        
        Args:
            show_plots: Whether to display evaluation plots.
            video_col: Column with video paths (for video models).
            
        Returns:
            Dictionary with evaluation results.
        """
        print("Evaluating model...")
        
        y_val = self.validation_df['congestion_rating']
        
        # Video transformer models
        if isinstance(self.model, VideoTransformerModel):
            y_val_encoded, y_pred_encoded, metrics = self.model.evaluate(
                self.validation_df, y_val, video_col=video_col
            )
        else:
            # ML models
            X_val = self.validation_df[self.dynamic_feature_cols]
            y_val_encoded, y_pred_encoded, metrics = self.model.evaluate(X_val, y_val)
        
        # Display competition metrics
        print(f"\n{'='*50}")
        print(f"COMPETITION METRICS")
        print(f"{'='*50}")
        print(f"Macro-F1 (70%):      {metrics['macro_f1']:.4f}")
        print(f"Accuracy (30%):      {metrics['accuracy']:.4f}")
        print(f"{'='*50}")
        print(f"COMPETITION SCORE:   {metrics['competition_score']:.4f}")
        print(f"{'='*50}\n")
        print(metrics['report'])
        
        if show_plots:
            # Confusion matrix
            self.visualizer.plot_confusion_matrix(
                y_val_encoded, 
                y_pred_encoded, 
                self.model.get_classes()
            )
            
            # Feature importance (ML models only)
            if hasattr(self.model, 'get_feature_importance'):
                feat_imp = self.model.get_feature_importance()
                if feat_imp is not None:
                    self.visualizer.plot_feature_importance(
                        feat_imp,
                        self.dynamic_feature_cols,
                        model_name=getattr(self.model, 'model_name', 'model')
                    )
        
        return metrics
    
    def run(self, model_name: str = "random_forest", use_cv: bool = False, 
            n_splits: int = 5, show_plots: bool = True) -> 'TrainingPipeline':
        """
        Run the full training pipeline.
        
        Args:
            model_name: Name of model from MODELS dispatcher.
            use_cv: Whether to use Stratified K-Fold CV.
            n_splits: Number of CV folds.
            show_plots: Whether to display evaluation plots.
            
        Returns:
            self
        """
        if any(getattr(self.config, extract_flag) for extract_flag in self.config.EXTRACT_LIST):
            print("Please extract features first. Run the feature extraction pipeline first.")
            print("Setting all extract flags to False")
            for extract_flag in self.config.EXTRACT_LIST:
                if getattr(self.config, extract_flag):
                    setattr(self.config, extract_flag, False)

        self.setup()
        self.load_data()
        self.preprocess()
        self.train(model_name=model_name, use_cv=use_cv, n_splits=n_splits)
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
        self.validation_df = None  # For video models
    
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
        testing_df: pd.DataFrame = None,
        validation_df: pd.DataFrame = None
    ) -> 'InferencePipeline':
        """
        Prepare test data for inference.

        Args:
            testing_df: Pre-processed testing dataframe (for submission IDs).
            validation_df: Validation dataframe with video paths (for video models).

        Returns:
            self
        """
        if testing_df is not None:
            self.testing_df = testing_df
        if validation_df is not None:
            self.validation_df = validation_df

        if self.testing_df is None:
            # Load and preprocess from scratch
            print("Loading and preprocessing test data...")
            videos_df, train_df, test_df, features_df = self.data_loader.load_all_data()
            _, self.validation_df, self.testing_df = self.preprocessor.prepare_full_dataset(
                train_df, test_df, videos_df, features_df
            )

        # Determine dynamic feature columns from the test data
        if hasattr(self, 'testing_df') and self.testing_df is not None:
            exclude_cols = ['ID', 'cycle_phase', 'ID_type', 'video_path', 'congestion_rating', 
                           'video_time', 'date', 'Date', 'view_label']
            feature_cols = [col for col in self.testing_df.columns
                           if col not in exclude_cols]
            self.dynamic_feature_cols = [col for col in feature_cols if col in self.testing_df.columns]
            print(f"Using {len(self.dynamic_feature_cols)} feature columns: {self.dynamic_feature_cols}")

        return self
    
    def _extract_group(self, id_col: str) -> tuple:
        """Extract view_label and rating_type from ID for grouping."""
        parts = id_col.split('_')
        # ID format: time_segment_X_View_Label_congestion_enter/exit_rating
        rating_type = parts[-2] + '_' + parts[-1]
        view_parts = parts[3:-3]
        view_label = '_'.join(view_parts) if view_parts else 'unknown'
        return (view_label, rating_type)
    
    def _predict_video_model(self, video_col: str = "video_path") -> pd.DataFrame:
        """
        Predict using video model.
        
        Video models require actual video files. Since testing_df contains
        future predictions (no videos), we predict on validation_df (which has
        videos) and map predictions to testing_df.
        """
        from collections import Counter
        
        if self.validation_df is None:
            raise ValueError("Video models require validation_df with video paths.")
        
        # Get valid video paths from validation data
        val_df = self.validation_df.dropna(subset=[video_col]).copy()
        print(f"Predicting on {len(val_df)} videos from validation set...")
        
        # Predict on validation videos
        predictions = self.model.predict(val_df, video_col=video_col)
        val_df['predicted'] = predictions
        
        # Group predictions and take most common per (view_label, rating_type)
        val_df['group'] = val_df['ID'].apply(self._extract_group)
        group_preds = val_df.groupby('group')['predicted'].agg(
            lambda x: Counter(x).most_common(1)[0][0] if len(x) > 0 else 'free flowing'
        ).to_dict()
        
        # Map predictions to testing_df
        test_df = self.testing_df.copy()
        test_df['group'] = test_df['ID'].apply(self._extract_group)
        test_df['Target'] = test_df['group'].map(group_preds).fillna('free flowing')
        test_df['Target_Accuracy'] = test_df['Target']
        
        return test_df[['ID', 'Target', 'Target_Accuracy']]
    
    def predict(self, video_col: str = "video_path") -> pd.DataFrame:
        """
        Make predictions on test data.
        
        Args:
            video_col: Column with video paths (for video models).
        
        Returns:
            DataFrame with predictions.
        """
        if self.model is None:
            raise ValueError("No model available. Load or set a model first.")
        if self.testing_df is None:
            raise ValueError("No test data. Call prepare_test_data first.")
        
        print("Making predictions...")
        
        # Video transformer models need special handling
        if isinstance(self.model, VideoTransformerModel):
            return self._predict_video_model(video_col)
        
        # ML models - direct prediction
        X_test = self.testing_df[self.dynamic_feature_cols]
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

