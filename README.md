# Barbados Traffic Analysis Challenge
Modular pipeline for the [Zindi Barbados Traffic Analysis Challenge](https://zindi.africa/competitions/barbados-traffic-analysis-challenge).

## Usage on Kaggle

In your Kaggle notebook, run:

```python
# Clone the repository
!git clone https://github.com/anyantudre/barbados-traffic-analysis-challenge.git /kaggle/working/barbados-traffic-analysis-challenge

# Add to Python path
import sys
sys.path.insert(0, "/kaggle/working/barbados-traffic-analysis-challenge")

# Import modules
from src.config import Config
from src.pipeline import TrainingPipeline, InferencePipeline
```

## Project Structure

```
├── src/
│   ├── config.py             # Configuration and parameters
│   ├── data_loader.py        # Data loading utilities
│   ├── feature_extraction.py # Video feature extraction
│   ├── preprocessing.py      # Data preprocessing
│   ├── visualization.py      # Plotting functions
│   ├── model.py              # Model class
│   └── pipeline.py           # Training & inference pipelines
├── notebook-barbados-challenge.ipynb
├── requirements.txt
└── README.md
```

## Quick Start

```python
from src.config import Config
from src.pipeline import TrainingPipeline, InferencePipeline

# Train
config = Config()
pipeline = TrainingPipeline(config)
pipeline.run()

# Predict
inference = InferencePipeline(config, model=pipeline.model)
inference.prepare_test_data(pipeline.testing_df)
inference.predict_and_save('submission.csv')
```

