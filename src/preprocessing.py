# Data preprocessing and transformation
"""
This module contains functions for preprocessing and transforming data.
Includes time feature extraction, data merging, and encoding.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from .config import Config


class DataPreprocessor:
    """Class for preprocessing the traffic analysis data."""
    
    def __init__(self, config: Config = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config: Configuration object. Uses default if None.
        """
        self.config = config or Config()
    
    @staticmethod
    def extract_video_id(video_column: pd.Series) -> pd.Series:
        """
        Extract video ID from video path column.
        
        Args:
            video_column: Series containing video paths.
            
        Returns:
            Series with extracted video IDs.
        """
        return video_column.apply(lambda x: x.split('/')[1].replace('.mp4', ''))
    
    @staticmethod
    def add_time_features(df: pd.DataFrame, time_col: str = "video_time") -> pd.DataFrame:
        """
        Add time-based features to the dataframe.
        
        Args:
            df: Input dataframe.
            time_col: Name of the time column to process.
            
        Returns:
            DataFrame with added time features.
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df["day_week"] = df[time_col].dt.day_of_week   # 0=Mon, 6=Sun
        df["hour"] = df[time_col].dt.hour              # 0–23
        df["Date"] = df[time_col].dt.date
        return df
    
    @staticmethod
    def encode_signaling(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode signaling column as categorical codes.
        
        Args:
            train: Training dataframe.
            test: Test dataframe.
            
        Returns:
            Tuple of (train, test) with encoded signaling.
        """
        train = train.copy()
        test = test.copy()
        
        # Define categories from combined data
        categories = pd.concat([train['signaling'], test['signaling']]).unique()
        train['signaling'] = pd.Categorical(train['signaling'], categories=categories).codes
        test['signaling'] = pd.Categorical(test['signaling'], categories=categories).codes
        
        return train, test
    
    def merge_with_video_paths(
        self, 
        train: pd.DataFrame, 
        test: pd.DataFrame, 
        videos_df: Optional[pd.DataFrame] = None,
        features: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge train and test with video paths.
        
        Args:
            train: Training dataframe.
            test: Test dataframe.
            videos_df: DataFrame with video paths (optional, will be created from features if None/empty).
            features: DataFrame with extracted features (used to build videos_df if videos_df is None/empty).
            
        Returns:
            Tuple of (train, test) merged with video paths.
        """
        train = train.copy()
        test = test.copy()
        
        # Extract video IDs
        train['videos'] = self.extract_video_id(train['videos'])
        test['videos'] = self.extract_video_id(test['videos'])
        
        # Build videos_df from features if not provided or empty
        if videos_df is None or len(videos_df) == 0:
            if features is not None and 'video_path' in features.columns:
                # Extract video ID from video_path in features
                videos_df = features[['video_path']].copy()
                videos_df['videos'] = videos_df['video_path'].apply(
                    lambda x: os.path.basename(x).replace('.mp4', '') if pd.notna(x) else None
                )
                videos_df = videos_df.dropna(subset=['videos']).drop_duplicates(subset=['videos'])
            else:
                # Fallback: create videos_df from train/test videos column
                # Extract video IDs and create minimal video_path (using original path from train/test)
                all_videos = pd.concat([train[['videos']], test[['videos']]]).drop_duplicates()
                # Reconstruct video_path from videos column (before extract_video_id was called)
                # Since we already called extract_video_id above, we need to work with what we have
                # Create a minimal videos_df - video_path will be None/NaN for local runs without actual videos
                videos_df = all_videos.copy()
                videos_df['video_path'] = None  # Will be filled from features merge later if available
        
        # Ensure videos_df has the required column
        if 'videos' not in videos_df.columns:
            raise ValueError("videos_df must have a 'videos' column")
        
        # Merge with video paths
        train = train.merge(videos_df, on='videos', how='left')
        test = test.merge(videos_df, on='videos', how='left')
        
        return train, test
    
    def preprocess_train_test(
        self, 
        train: pd.DataFrame, 
        test: pd.DataFrame, 
        videos_df: Optional[pd.DataFrame] = None, 
        features: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full preprocessing pipeline for train and test data.
        
        Args:
            train: Training dataframe.
            test: Test dataframe.
            videos_df: DataFrame with video paths (optional, will be created from features if None/empty).
            features: DataFrame with extracted features.
            
        Returns:
            Tuple of (train_df, test_df) fully preprocessed.
        """
        # Merge with video paths (videos_df can be None/empty if features already have video_path)
        train, test = self.merge_with_video_paths(train, test, videos_df, features)
        
        # Add time features
        train = self.add_time_features(train)
        test = self.add_time_features(test)
        
        # Encode signaling
        train, test = self.encode_signaling(train, test)
        
        # Merge with features
        train_df = train.merge(features, on='video_path', how='left')
        test_df = test.merge(features, on='video_path', how='left')
        
        return train_df, test_df
    
    @staticmethod
    def create_long_format(final_df: pd.DataFrame) -> pd.DataFrame:
        """
        Melt ID_enter and ID_exit into a single 'ID' column (long format).
        
        Args:
            final_df: Combined train and test dataframe.
            
        Returns:
            DataFrame in long format with single congestion_rating column.
        """
        long_df = final_df.melt(
            id_vars=[col for col in final_df.columns if col not in ['ID_enter', 'ID_exit']],
            value_vars=['ID_enter', 'ID_exit'],
            var_name='ID_type',
            value_name='ID'
        )
        
        long_df['congestion_rating'] = long_df.apply(
            lambda row: row['congestion_enter_rating'] 
                        if row['ID_type'] == 'ID_enter' 
                        else row['congestion_exit_rating'],
            axis=1
        )
        long_df = long_df.drop(columns=['congestion_enter_rating', 'congestion_exit_rating'])
        
        return long_df
    
    @staticmethod
    def prepare_sample_submission(submission_path: str) -> pd.DataFrame:
        """
        Prepare sample submission dataframe with extracted features.
        
        Args:
            submission_path: Path to sample submission CSV.
            
        Returns:
            Prepared submission dataframe.
        """
        ss = pd.read_csv(submission_path)
        ss['view_label'] = ss['ID'].apply(lambda x: x.split('_')[3])
        ss['time_segment_id'] = ss['ID'].apply(lambda x: int(x.split('_')[2]))
        ss['ID_type'] = ss['ID'].apply(lambda x: 'ID_' + x.split('#')[1].split('_')[2])
        ss['cycle_phase'] = "test_output_5"
        ss = ss.drop(columns=['Target', 'Target_Accuracy'])
        return ss
    
    def create_shifted_features(
        self, 
        merged_df: pd.DataFrame, 
        features_cols: List[str], 
        lag_shift: int = 7
    ) -> pd.DataFrame:
        """
        Create lagged features by shifting within groups.
        
        Args:
            merged_df: Merged dataframe.
            features_cols: List of feature column names.
            lag_shift: Number of time steps to shift.
            
        Returns:
            DataFrame with shifted features.
        """
        shifted_df = merged_df.groupby(
            ['view_label', 'ID_type'], 
            as_index=True
        )[features_cols].shift(lag_shift)
        
        shifted_df['ID'] = merged_df['ID']
        shifted_df['cycle_phase'] = merged_df['cycle_phase']
        shifted_df['congestion_rating'] = merged_df['congestion_rating']
        # Keep video_path for video transformer models
        if 'video_path' in merged_df.columns:
            shifted_df['video_path'] = merged_df['video_path']
        shifted_df = shifted_df.dropna(subset=features_cols)
        
        return shifted_df
    
    def prepare_full_dataset(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        videos_df: Optional[pd.DataFrame] = None,
        features: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare the full dataset for modeling.

        Args:
            train: Training dataframe.
            test: Test dataframe.
            videos_df: DataFrame with video paths (optional, will be created from features if None/empty).
            features: DataFrame with extracted features (required).

        Returns:
            Tuple of (training_df, validation_df, testing_df).
        """
        if features is None or len(features) == 0:
            raise ValueError("features DataFrame is required and cannot be empty")
        
        # Dynamically determine feature columns from the loaded features DataFrame
        # Exclude 'video_path' column as it's not a feature
        feature_cols = [col for col in features.columns if col != 'video_path']

        # Combine with temporal features (always included)
        all_feature_cols = self.config.TEMPORAL_FEATURES + feature_cols

        # Preprocess train and test
        train_df, test_df = self.preprocess_train_test(train, test, videos_df, features)

        # Combine into final dataframe
        final_df = pd.concat([train_df, test_df])

        # Create long format
        long_df = self.create_long_format(final_df)

        # Prepare sample submission
        ss = self.prepare_sample_submission(self.config.SAMPLE_SUBMISSION_PATH)

        # Merge all data
        merged_df = pd.concat([long_df, ss])
        merged_df = merged_df.sort_values(by=['view_label', 'ID_type', 'time_segment_id'])
        # Keep video_path for video models
        keep_cols = self.config.ID_COLS + ['ID', 'cycle_phase', 'ID_type', 'video_path'] + \
                    ['congestion_rating'] + all_feature_cols
        merged_df = merged_df[[c for c in keep_cols if c in merged_df.columns]]
        merged_df = merged_df.reset_index(drop=True)

        #Create shifted features
        shifted_df = self.create_shifted_features(
            merged_df,
            all_feature_cols,
            self.config.LAG_SHIFT
        )
        
        # Split into train, validation, and test
        training_df = shifted_df[shifted_df['cycle_phase'] == 'train']
        training_df = training_df.sample(frac=1)

        validation_df = shifted_df[shifted_df['cycle_phase'] == 'test_input_15']
        testing_df = shifted_df[shifted_df['cycle_phase'] == 'test_output_5']
        
        
        return training_df, validation_df, testing_df

