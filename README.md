# Project Title
MELI Financial Transaction Prediction

## Description
The goal of this analysis is to understand the dataset's structure, identify patterns, and uncover insights that will inform subsequent modeling efforts.
The ML Project will be a prediction of account balance with Regression Algorithms.

## Installation
- **Prerequisites**: Python 3.11
- **Setup**: 
    1. git clone https://github.com/akatowolf/meli-financial-test.git
    2. Set up a new environment "python -m venv myvenv"
    3. Activate venv "myvenv\Scripts\activate"
    4. Install requirements.txt "pip install -r requirements.txt"
    5. Initialize mlflow server "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"

## For training new models
For train new models run the script "main.py". This will execute the train pipeline.
To try new models, parameters and metrics edit "training_pipeline.py"

## For testing predictions
There is a json in data/predict/pred.json with dummie data to test the production model.
The model is retrieved from the MLFlow model registry and it is named as "financial-xgboost"
Run the script "predict.py". This will execute the predictions and will store them in a csv file in "data/predictions/predictions.csv"
Other runs are stored locally to reduce github weight

# EDA
The Exploratory Data Analysis is a notebook and is stored in Notebooks folder
The Notebook contains the insights extracted from the dataset

## Business problem
The goal was to predict the balance of the accounts after a transaction event.

## Modeling
The registred model is an XGBoost Regressor, for performance details go to the mlflow artifact
The choosed metric is R2
Other regressors were used and the performance is stored locally (Lasso, Ridge, RandomForest)

## Experiments
MLFlow is used for tracking experiments, metrics stored are r2, rmse, mae and cv r2. 
The model is stored as an artifact, another artifact stored is a chart comparing predictions vs real data.

## Results
The performance of the choosed model is: r2->0.54
