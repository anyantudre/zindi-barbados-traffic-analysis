# Barbados Traffic Analysis Challenge
Modular pipeline for the [Zindi Barbados Traffic Analysis Challenge](https://zindi.africa/competitions/barbados-traffic-analysis-challenge).

## Usage on Kaggle
In the Kaggle notebook, you need to generate a PAT to be able to clone our GitHub repo since it is private:

#### Generate a Personal Access Token
On GitHub:
1. Settings
1. Developer settings
1. Personal access tokens
1. Create Fine-grained token or Classic token
1. Permissions required:
    Contents: Read

Copy the token. Guard it like your passport and your dignity :)

#### Add a Kaggle secret
On Kaggle, click on Adds-on then Secrets then create a new secret called GITHUB_PAT and paste your PAT.   
Now you're good to execute the notebook.


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
