# Configuration parameters and paths for Barbados Traffic Analysis
"""
This module contains all configuration settings, paths, and constants.
"""

import warnings
import pandas as pd
import seaborn as sns


class Config:
    """Configuration class for the traffic analysis pipeline."""
    
    # Feature extraction flag
    EXTRACT_FEATURES: bool = False
    
    # Use full (uncompressed) dataset or compressed
    USE_FULL: bool = False
    
    # Paths for compressed datasets
    COMPRESSED_PATHS_DICT: dict = {
        "n1": "/kaggle/input/brb-traffic-compressed-n1/normanniles1",
        "n2": "/kaggle/input/brb-traffic-compressed-n2/normanniles2",
        "n3": "/kaggle/input/brb-traffic-compressed-n3/normanniles3",
        "n4": "/kaggle/input/brb-traffic-compressed-n4/normanniles4",
        "added": "/kaggle/input/brb-traffic-compressed-added/added"
    }
    
    COMPRESSED_PATHS_LIST: list = [
        "/kaggle/input/brb-traffic-compressed-n1/normanniles1",
        "/kaggle/input/brb-traffic-compressed-n2/normanniles2",
        "/kaggle/input/brb-traffic-compressed-n3/normanniles3",
        "/kaggle/input/brb-traffic-compressed-n4/normanniles4",
        "/kaggle/input/brb-traffic-compressed-added/added"
    ]
    
    # Paths for full (uncompressed) datasets
    FULL_PATHS_DICT: dict = {
        "n1": "/kaggle/input/normanniles1-barbados-traffic/normanniles1",
        "n2": "/kaggle/input/normanniles2-barbados-traffic/normanniles2",
        "n3": "/kaggle/input/normanniles3-barbados-traffic/normanniles3",
        "n4": "/kaggle/input/normanniles4-barbados-traffic/normanniles4",
    }
    
    # CSV file paths
    TRAIN_CSV_PATH: str = "/kaggle/input/normanniles1-barbados-traffic/Train.csv"
    TEST_CSV_PATH: str = "/kaggle/input/normanniles1-barbados-traffic/TestInputSegments.csv"
    SAMPLE_SUBMISSION_PATH: str = "/kaggle/input/normanniles1-barbados-traffic/SampleSubmission.csv"
    FEATURES_CSV_PATH: str = "/kaggle/input/simple-features-for-traffic-analysis/simple_features.csv"
    
    # Column definitions
    FEATURES_COLS: list = [
        'signaling', 'day_week', 'hour',
        'num_frames_sampled', 'mean_r', 'mean_g', 'mean_b', 
        'std_r', 'std_g', 'std_b', 'brightness'
    ]
    
    LABELS_COLS: list = ['congestion_enter_rating', 'congestion_exit_rating']
    
    ID_COLS: list = ['view_label', 'time_segment_id']
    
    DELETE_COLS: list = [
        'responseId', 'view_label', 'videos',
        'video_time', 'datetimestamp_start', 'datetimestamp_end', 'date'
    ]
    
    # Model parameters
    MODEL_PARAMS: dict = {
        'n_estimators': 200,
        'max_depth': None,
        'random_state': 42,
        'class_weight': 'balanced'
    }
    
    # Video feature extraction parameters
    SAMPLE_INTERVAL: int = 30
    TARGET_SIZE: int = 224
    
    # Time shift for lagged features
    LAG_SHIFT: int = 7
    
    @classmethod
    def setup_environment(cls):
        """Setup pandas display options and warnings."""
        pd.set_option("display.max_rows", 70)
        pd.set_option("display.width", None)
        pd.set_option("display.max_columns", None)
        warnings.filterwarnings('ignore')
        sns.set_theme(style="whitegrid")
    
    @classmethod
    def get_paths_dict(cls) -> dict:
        """Get the appropriate paths dictionary based on USE_FULL setting."""
        return cls.FULL_PATHS_DICT if cls.USE_FULL else cls.COMPRESSED_PATHS_DICT


# Create a default config instance
config = Config()

