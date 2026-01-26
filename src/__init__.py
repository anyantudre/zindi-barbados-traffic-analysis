# Barbados Traffic Analysis Challenge - Source Package
"""
This package contains modular components for traffic congestion prediction.

Modules:
    - config: Configuration parameters and paths
    - data_loader: Data loading and dataset building utilities
    - feature_extraction: Video feature extraction functions
    - model_dispatcher: Model dispatcher for different models
    - model: Model training and prediction
    - optimize: Hyperparameter optimization
    - pipeline: Training and inference pipelines
    - preprocessing: Data preprocessing and transformation
    - utils: Utility functions
    - video_models: Video models for feature extraction
    - visualization: Plotting and visualization utilities
"""

from . import config
from . import data_loader
from . import feature_extraction
from . import model_dispatcher
from . import preprocessing
from . import optimize
from . import visualization
from . import model
from . import pipeline
from . import utils
from . import video_models

