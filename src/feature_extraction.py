# Video feature extraction functions
"""
This module contains functions for extracting features from videos.
Includes simple statistical features like brightness, color histograms, etc.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional, List

from .config import Config


class FeatureExtractor:
    """Class for extracting features from video files."""
    
    def __init__(self, sample_interval: int = 30, target_size: int = 224):
        """
        Initialize the FeatureExtractor.
        
        Args:
            sample_interval: Frame sampling interval (process every Nth frame).
            target_size: Size to resize frames to before processing.
        """
        self.sample_interval = sample_interval
        self.target_size = target_size
    
    def extract_video_features(self, video_path: str) -> Optional[Dict]:
        """
        Extract lightweight features from a single video.
        
        Uses Welford's online algorithm for computing running mean/std
        to be memory efficient for long videos.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Dictionary with extracted features, or None if video couldn't be processed.
            
        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        sample_count = 0
        
        # Welford online mean/std accumulators
        mean = np.zeros(3, dtype=np.float64)
        M2 = np.zeros(3, dtype=np.float64)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_interval == 0:
                # Resize sample
                if self.target_size:
                    frame = cv2.resize(frame, (self.target_size, self.target_size))
                
                # Compute mean per channel on this sample
                sample_mean = frame.mean(axis=(0, 1))  # (B, G, R)
                
                sample_count += 1
                delta = sample_mean - mean
                mean += delta / sample_count
                M2 += delta * (sample_mean - mean)
            
            frame_idx += 1
        
        cap.release()
        
        if sample_count == 0:
            return None
        
        # Final std per channel
        std = np.sqrt(M2 / sample_count)
        
        brightness = mean.mean()  # overall luminance approx
        
        return {
            "video_path": video_path,
            "num_frames_sampled": sample_count,
            "mean_r": mean[2],
            "mean_g": mean[1],
            "mean_b": mean[0],
            "std_r": std[2],
            "std_g": std[1],
            "std_b": std[0],
            "brightness": brightness,
        }
    
    def extract_features_from_paths(self, paths_list: List[str]) -> pd.DataFrame:
        """
        Extract features from all videos in the given paths.
        
        Args:
            paths_list: List of base paths to search for videos.
            
        Returns:
            DataFrame with extracted features for all videos.
        """
        records = []
        
        for base_path in paths_list:
            for root, dirs, files in os.walk(base_path):
                for f in tqdm(files, total=len(files), desc=f"Processing {base_path}"):
                    if f.lower().endswith(".mp4"):
                        video_path = os.path.join(root, f)
                        features = self.extract_video_features(video_path)
                        if features:
                            records.append(features)
        
        return pd.DataFrame(records)
    
    def extract_and_save(self, paths_list: List[str], output_path: str = 'simple_features.csv') -> pd.DataFrame:
        """
        Extract features and save to CSV.
        
        Args:
            paths_list: List of base paths to search for videos.
            output_path: Path for the output CSV file.
            
        Returns:
            DataFrame with extracted features.
        """
        features_df = self.extract_features_from_paths(paths_list)
        features_df.to_csv(output_path, index=False)
        return features_df


def get_or_extract_features(config: Config) -> pd.DataFrame:
    """
    Get features either by extracting them or loading from CSV.
    
    Args:
        config: Configuration object.
        
    Returns:
        DataFrame with video features.
    """
    if config.EXTRACT_FEATURES:
        extractor = FeatureExtractor(
            sample_interval=config.SAMPLE_INTERVAL,
            target_size=config.TARGET_SIZE
        )
        features = extractor.extract_and_save(
            config.COMPRESSED_PATHS_LIST,
            'simple_features.csv'
        )
    else:
        features = pd.read_csv(config.FEATURES_CSV_PATH)
    
    return features

