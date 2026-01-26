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
        Add time-based features to the dataframe with cyclical encoding.
        
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
        df["minute"] = df[time_col].dt.minute          # 0–59
        df["Date"] = df[time_col].dt.date
        
        # Cyclical encoding for hour (crucial for temporal patterns)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_week_sin"] = np.sin(2 * np.pi * df["day_week"] / 7)
        df["day_week_cos"] = np.cos(2 * np.pi * df["day_week"] / 7)
        
        # Peak hour indicators (critical for traffic patterns)
        df["is_peak_am"] = df["hour"].between(6, 9).astype(int)
        df["is_peak_pm"] = df["hour"].between(16, 19).astype(int)
        df["is_peak"] = (df["is_peak_am"] | df["is_peak_pm"]).astype(int)
        
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
        Build long format for entry ratings only (exit removed from evaluation).
        
        Args:
            final_df: Combined train and test dataframe.
            
        Returns:
            DataFrame in long format with single congestion_rating column (enter only).
        """
        # Keep only enter IDs/labels (exit ratings removed from evaluation)
        cols = [c for c in final_df.columns if c not in ['ID_exit', 'congestion_exit_rating']]
        df = final_df[cols].copy()
        df = df.rename(columns={'ID_enter': 'ID'})
        df['ID_type'] = 'ID_enter'
        df['congestion_rating'] = df['congestion_enter_rating']
        df = df.drop(columns=['congestion_enter_rating'])
        return df
    
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
        # Only enter ratings are evaluated; set ID_type accordingly
        ss['ID_type'] = 'ID_enter'
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
        
        # Preserve metadata columns
        shifted_df['ID'] = merged_df['ID']
        shifted_df['cycle_phase'] = merged_df['cycle_phase']
        shifted_df['congestion_rating'] = merged_df['congestion_rating']
        shifted_df['time_segment_id'] = merged_df['time_segment_id']
        shifted_df['view_label'] = merged_df['view_label']
        if 'video_path' in merged_df.columns:
            shifted_df['video_path'] = merged_df['video_path']
        shifted_df = shifted_df.dropna(subset=features_cols)
        
        return shifted_df
    
    def create_multi_view_features(
        self,
        merged_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create multi-view aggregated features by time_segment_id.
        Aggregates features across the 4 camera views and adds asymmetry features.
        
        Args:
            merged_df: DataFrame with features and time_segment_id, view_label.
            feature_cols: List of numeric feature columns to aggregate.
            
        Returns:
            DataFrame with added multi-view aggregated features.
        """
        df = merged_df.copy()
        
        # Identify numeric feature columns (exclude metadata)
        meta_cols = ['time_segment_id', 'view_label', 'ID', 'cycle_phase', 'ID_type', 
                     'video_path', 'congestion_rating', 'video_time', 'date', 'Date']
        # Exclude high-dimensional embeddings from aggregation (keep them as-is)
        embed_prefixes = ('vmae_', 'pe_', 'emb_', 'PCA_', 'yoloflow_')
        num_cols = [
            c for c in feature_cols
            if c in df.columns
            and df[c].dtype in ['float64', 'int64']
            and not c.startswith(embed_prefixes)
        ]
        
        if len(num_cols) == 0:
            return df
        
        # Aggregate by time_segment_id (across 4 views)
        agg_funcs = ['mean', 'std', 'min', 'max', 'sum']
        grouped = df.groupby('time_segment_id')[num_cols].agg(agg_funcs)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        # Merge aggregated features back
        df = df.merge(grouped, on='time_segment_id', how='left', suffixes=('', '_agg'))
        
        # Create asymmetry features (difference vs mean, ratio)
        for col in num_cols:
            mean_col = f'{col}_mean'
            if mean_col in df.columns:
                df[f'{col}_vs_mean'] = df[col] - df[mean_col]
                df[f'{col}_ratio_mean'] = df[col] / (df[mean_col] + 1e-6)
            
            max_col = f'{col}_max'
            min_col = f'{col}_min'
            if max_col in df.columns and min_col in df.columns:
                df[f'{col}_range'] = df[max_col] - df[min_col]
        
        return df
    
    def create_temporal_lag_features(
        self,
        merged_df: pd.DataFrame,
        feature_cols: List[str],
        lags: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ) -> pd.DataFrame:
        """
        Create temporal lag features both per view and globally.
        
        Args:
            merged_df: DataFrame sorted by time_segment_id.
            feature_cols: List of feature columns to create lags for.
            lags: List of lag values to create.
            
        Returns:
            DataFrame with added lag features.
        """
        df = merged_df.copy()
        
        # Sort by time_segment_id to ensure temporal order
        df = df.sort_values(['time_segment_id', 'view_label']).reset_index(drop=True)
        
        # Select key features for lagging (avoid too many)
        key_features = [c for c in feature_cols if any(x in c.lower() for x in 
                       ['total', 'count', 'mean', 'vehicle', 'object', 'flow', 'brightness'])]
        if len(key_features) == 0:
            key_features = feature_cols[:10]  # Limit to first 10 if no matches
        
        # Lag features per view_label (camera-specific history)
        for lag in lags:
            for col in key_features:
                if col in df.columns:
                    df[f'{col}_view_lag{lag}'] = df.groupby('view_label')[col].shift(lag).fillna(0)
        
        # Global lag features (aggregated across views, by time_segment_id)
        # First get one row per time_segment_id with aggregated values
        global_df = df.groupby('time_segment_id').first().reset_index()
        global_df = global_df.sort_values('time_segment_id').reset_index(drop=True)
        
        # Create global lags on aggregated mean features
        for lag in lags:
            for col in key_features:
                mean_col = f'{col}_mean'
                if mean_col in global_df.columns:
                    global_df[f'{mean_col}_global_lag{lag}'] = global_df[mean_col].shift(lag).fillna(0)
        
        # Merge global lags back
        global_lag_cols = [c for c in global_df.columns if '_global_lag' in c] + ['time_segment_id']
        df = df.merge(global_df[global_lag_cols], on='time_segment_id', how='left')
        
        # Create trend features
        for col in key_features:
            mean_col = f'{col}_mean'
            if mean_col in df.columns and f'{mean_col}_global_lag2' in df.columns:
                df[f'{mean_col}_trend'] = df[mean_col] - df[f'{mean_col}_global_lag2']
        
        return df
    
    def create_traffic_dynamics_features(
        self,
        merged_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create traffic dynamics features: delta, growth rate, persistence.
        These capture traffic flow changes between time segments.
        
        Args:
            merged_df: DataFrame sorted by time_segment_id.
            
        Returns:
            DataFrame with added traffic dynamics features.
        """
        df = merged_df.copy()
        
        # Sort by time_segment_id to ensure temporal order
        df = df.sort_values('time_segment_id').reset_index(drop=True)
        
        # Get aggregated features per time_segment_id (one row per segment)
        seg_df = df.groupby('time_segment_id').first().reset_index()
        seg_df = seg_df.sort_values('time_segment_id').reset_index(drop=True)
        
        # Key aggregated features for dynamics
        key_agg_features = [c for c in seg_df.columns if c.endswith('_mean') and 
                           any(x in c.lower() for x in ['total_objects', 'object_count', 'vehicle_count', 'total_bbox'])]
        
        # Create delta and growth rate features
        for col in key_agg_features:
            if col in seg_df.columns:
                # Delta (change from previous segment)
                seg_df[f'{col}_delta'] = seg_df[col].diff().fillna(0)
                
                # Growth rate (percentage change)
                seg_df[f'{col}_growth_rate'] = seg_df[col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
                
                # Acceleration (change in delta)
                seg_df[f'{col}_acceleration'] = seg_df[f'{col}_delta'].diff().fillna(0)
        
        # Merge dynamics features back to full dataframe
        dynamics_cols = [c for c in seg_df.columns if any(x in c for x in ['_delta', '_growth_rate', '_acceleration'])] + ['time_segment_id']
        df = df.merge(seg_df[dynamics_cols], on='time_segment_id', how='left')
        
        return df
    
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
        # Update temporal features to include cyclical encoding and peak indicators
        temporal_features = ['signaling', 'day_week', 'hour', 'minute', 'hour_sin', 'hour_cos', 
                            'day_week_sin', 'day_week_cos', 'is_peak_am', 'is_peak_pm', 'is_peak']
        all_feature_cols = temporal_features + feature_cols

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
        merged_df = merged_df.sort_values(by=['time_segment_id', 'view_label', 'ID_type'])
        
        # Keep video_path for video models
        keep_cols = self.config.ID_COLS + ['ID', 'cycle_phase', 'ID_type', 'video_path'] + \
                    ['congestion_rating'] + all_feature_cols
        merged_df = merged_df[[c for c in keep_cols if c in merged_df.columns]]
        merged_df = merged_df.reset_index(drop=True)

        # Create multi-view aggregated features
        # merged_df = self.create_multi_view_features(merged_df, all_feature_cols)
        
        # Update feature columns to include new aggregated features
        new_feature_cols = [c for c in merged_df.columns 
                           if c not in self.config.ID_COLS + ['ID', 'cycle_phase', 'ID_type', 
                                                              'video_path', 'congestion_rating', 
                                                              'video_time', 'date', 'Date']]
        
        # Create temporal lag features
        # merged_df = self.create_temporal_lag_features(merged_df, new_feature_cols, lags=[1, 2, 3])
        
        # Create traffic dynamics features (delta, growth rate, etc.)
        # merged_df = self.create_traffic_dynamics_features(merged_df)
        
        # Replace inf values that can arise from pct_change/divisions
        # merged_df = merged_df.replace([np.inf, -np.inf], 0)
        
        # Update feature columns again after lag and dynamics features
        final_feature_cols = [c for c in merged_df.columns 
                             if c not in self.config.ID_COLS + ['ID', 'cycle_phase', 'ID_type', 
                                                                'video_path', 'congestion_rating',
                                                                'video_time', 'date', 'Date']]
        
        # Create shifted features (original lag approach for backward compatibility)
        shifted_df = self.create_shifted_features(
            merged_df,
            final_feature_cols,
            self.config.LAG_SHIFT
        )
        
        # Split into train, validation, and test
        training_df = shifted_df[shifted_df['cycle_phase'] == 'train']
        training_df = training_df.sample(frac=1)

        validation_df = shifted_df[shifted_df['cycle_phase'] == 'test_input_15']
        testing_df = shifted_df[shifted_df['cycle_phase'] == 'test_output_5']
        
        
        return training_df, validation_df, testing_df

