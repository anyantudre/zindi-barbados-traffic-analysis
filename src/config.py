# Configuration parameters and paths for Barbados Traffic Analysis
"""
This module contains all configuration settings, paths, and constants.
"""

import warnings
import pandas as pd
import seaborn as sns


def get_device() -> str:
    """Auto-detect best available device (cuda > cpu)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Config:
    """Configuration class for the traffic analysis pipeline."""
    
    # Feature extraction flags (True = extract, False = load from CSV)
    EXTRACT_SIMPLE: bool = False     # Simple features (brightness, colors)
    EXTRACT_YOLO: bool = False       # YOLO vehicle detection features
    EXTRACT_FLOW: bool = False       # Optical flow features
    EXTRACT_EMBEDDINGS: bool = False # EfficientNet embeddings
    EXTRACT_VIDEOMAE: bool = False   # VideoMAE spatio-temporal embeddings
    EXTRACT_PE: bool = False        # Perception Encoder spatio-temporal embeddings
    
    # Use full (uncompressed) dataset or compressed
    USE_FULL: bool = False
    
    # Device for GPU models (auto-detected)
    DEVICE: str = get_device()
    
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
    
    # Feature CSV paths (Kaggle datasets or local). Set to None to disable.
    SIMPLE_FEATURES_PATH: str = "/kaggle/input/simple-features-for-traffic-analysis/simple_features.csv"
    YOLO_FEATURES_PATH: str = None  # e.g. "/kaggle/input/yolo-features-traffic/yolo_features.csv"
    FLOW_FEATURES_PATH: str = None  # e.g. "/kaggle/input/flow-features-traffic/flow_features.csv"
    EMBEDDING_FEATURES_PATH: str = None  # e.g. "/kaggle/input/embedding-features-traffic/embedding_features.csv"
    VIDEOMAE_FEATURES_PATH: str = None  # e.g. "/kaggle/input/videomae-features-traffic/videomae_features.csv"
    PE_FEATURES_PATH: str = None  # e.g. "/kaggle/input/pe-features-traffic/pe_features.csv"

    # Column definitions - Temporal features (from preprocessing)
    TEMPORAL_FEATURES: list = ['signaling', 'day_week', 'hour']
    
    # Simple video features
    SIMPLE_FEATURES: list = [
        'num_frames_sampled', 'mean_r', 'mean_g', 'mean_b', 
        'std_r', 'std_g', 'std_b', 'brightness'
    ]
    
    # YOLO features
    YOLO_FEATURES: list = [
        'vehicle_count_mean', 'vehicle_count_std', 'vehicle_count_max',
        'avg_bbox_area', 'car_count', 'truck_count', 'bus_count', 'motorcycle_count'
    ]
    
    # Optical flow features
    FLOW_FEATURES: list = [
        'flow_mean', 'flow_std', 'flow_max', 'flow_angle_mean', 'flow_angle_std'
    ]
    
    # Embedding dimension (first N components)
    EMBEDDING_DIM: int = 64
    
    # VideoMAE parameters
    VIDEOMAE_DIM: int = 768       # Number of embedding dimensions to keep
    VIDEOMAE_NUM_FRAMES: int = 16  # Number of frames to sample per video

    # Perception Encoder parameters
    PE_DIM: int = 128             # or 256 or 512 if you can afford more
    PE_NUM_FRAMES: int = 16       # same as VideoMAE for consistency

    
    def get_features_cols(self) -> list:
        """Get active feature columns based on defined paths (path=None → disabled)."""
        cols = self.TEMPORAL_FEATURES.copy()
        if self.SIMPLE_FEATURES_PATH:
            cols.extend(self.SIMPLE_FEATURES)
        if self.YOLO_FEATURES_PATH:
            cols.extend(self.YOLO_FEATURES)
        if self.FLOW_FEATURES_PATH:
            cols.extend(self.FLOW_FEATURES)
        if self.EMBEDDING_FEATURES_PATH:
            cols.extend([f'emb_mean_{i}' for i in range(self.EMBEDDING_DIM)])
            cols.extend([f'emb_std_{i}' for i in range(self.EMBEDDING_DIM)])
        if self.VIDEOMAE_FEATURES_PATH:
            cols.extend([f'vmae_{i}' for i in range(self.VIDEOMAE_DIM)])
        if self.PE_FEATURES_PATH:
            cols.extend([f'pe_{i}' for i in range(self.PE_DIM)])
        return cols
    
    # Default features (updated dynamically in pipeline.setup())
    FEATURES_COLS: list = TEMPORAL_FEATURES + SIMPLE_FEATURES
    
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

