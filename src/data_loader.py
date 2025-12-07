# Data loading and dataset building utilities
"""
This module handles all data loading operations including:
- Building dataframes from video files
- Loading CSV files
- Downloading datasets from Kaggle
"""

import os
import pandas as pd
import kagglehub
from typing import Dict, Tuple, Optional

from .config import Config


class DataLoader:
    """Class for loading and building datasets."""
    
    def __init__(self, config: Config = None):
        """
        Initialize the DataLoader.
        
        Args:
            config: Configuration object. Uses default if None.
        """
        self.config = config or Config()
    
    @staticmethod
    def folder_stats(path: str) -> Tuple[int, float]:
        """
        Calculate folder statistics for video files.
        
        Args:
            path: Path to the folder to analyze.
            
        Returns:
            Tuple of (total_files, total_size_gb)
        """
        total_files = 0
        total_size = 0  # in bytes
        
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".mp4"):
                    total_files += 1
                    fp = os.path.join(root, f)
                    total_size += os.path.getsize(fp)
        
        # Convert bytes → GB
        total_gb = total_size / (1024 * 1024 * 1024)
        return total_files, total_gb
    
    @staticmethod
    def collect_stats(paths: Dict[str, str], label: str) -> None:
        """
        Collect and print statistics for multiple paths.
        
        Args:
            paths: Dictionary mapping labels to paths.
            label: Label for this collection of paths.
        """
        print(f"\n--- {label.upper()} DATASETS ---")
        grand_files, grand_gb = 0, 0
        
        for name, path in paths.items():
            files, gb = DataLoader.folder_stats(path)
            print(f"{name}: {files} files, {gb:.2f} GB")
            grand_files += files
            grand_gb += gb
        
        print(f"TOTAL: {grand_files} files, {grand_gb:.2f} GB")
    
    @staticmethod
    def build_dataframe(paths_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Build a dataframe from video files in the given paths.
        
        Args:
            paths_dict: Dictionary mapping labels to base paths.
            
        Returns:
            DataFrame with video IDs and paths.
        """
        records = []
        
        for label, base_path in paths_dict.items():
            for root, dirs, files in os.walk(base_path):
                for f in files:
                    if f.lower().endswith(".mp4"):
                        file_id = os.path.splitext(f)[0]  # filename without extension
                        file_path = os.path.join(root, f)  # full path
                        records.append({"videos": file_id, "video_path": file_path})
        
        return pd.DataFrame(records)
    
    def load_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, test, and sample submission CSVs.
        
        Returns:
            Tuple of (train_df, test_df, submission_df)
        """
        train = pd.read_csv(self.config.TRAIN_CSV_PATH)
        test = pd.read_csv(self.config.TEST_CSV_PATH)
        sub = pd.read_csv(self.config.SAMPLE_SUBMISSION_PATH)
        
        return train, test, sub
    
    def load_features(self) -> pd.DataFrame:
        """
        Load precomputed features from CSV.
        
        Returns:
            DataFrame with video features.
        """
        return pd.read_csv(self.config.FEATURES_CSV_PATH)
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required data: video paths, train, test, and features.
        
        Returns:
            Tuple of (videos_df, train_df, test_df, features_df)
        """
        from .feature_extraction import get_all_features
        
        # Build video paths dataframe
        paths_dict = self.config.get_paths_dict()
        videos_df = self.build_dataframe(paths_dict)
        
        # Load CSVs
        train, test, _ = self.load_train_test_data()
        
        # Load/extract all enabled features
        video_paths = videos_df['video_path'].tolist() if len(videos_df) > 0 else []
        features = get_all_features(self.config, video_paths)
        
        return videos_df, train, test, features
    
    @staticmethod
    def download_datasets():
        """Download all required datasets from Kaggle."""
        # Download the compressed dataset
        kagglehub.dataset_download("juliusmwangi/brb-traffic-compressed-n1")
        kagglehub.dataset_download("juliusmwangi/brb-traffic-compressed-n2")
        kagglehub.dataset_download("juliusmwangi/brb-traffic-compressed-n3")
        kagglehub.dataset_download("juliusmwangi/brb-traffic-compressed-n4")
        kagglehub.dataset_download("juliusmwangi/brb-traffic-compressed-added")
        
        # Download the un-compressed dataset
        kagglehub.dataset_download("kipngetichv/normanniles1-barbados-traffic")
        kagglehub.dataset_download("kipngetichv/normanniles2-barbados-traffic")
        kagglehub.dataset_download("kipngetichv/normanniles3-barbados-traffic")
        kagglehub.dataset_download("kipngetichv/normanniles4-barbados-traffic")
        
        # Download processed feature CSV
        kagglehub.dataset_download("juliusmwangi/simple-features-for-traffic-analysis")

