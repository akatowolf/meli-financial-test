import os
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from feature_engineering import FeatureEngineering
from load_data import DataLoader
import matplotlib.pyplot as plt
class ModelTrainer:
    def __init__(self, data_path, json_path, processed_path):
        # Set up MLflow
        mlflow.set_tracking_uri('sqlite:///../mlflow.db')
        mlflow.artifact_location = "../models/mlruns"
        mlflow.set_experiment('financial-regressor')
        
        # Load data
        data_loader = DataLoader(filepath=data_path)
        self.df = data_loader.load_data()
        
        # Initialize Feature Engineering
        self.feature_engineering = FeatureEngineering(self.df)
        self.df_processed = self.feature_engineering.create_features(json_path=json_path, processed_path=processed_path)

        # Prepare data
        self.X = self.df_processed.drop(columns=['balance_amt'])
        self.y = self.df_processed['balance_amt']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Define pipelines
        self.pipelines = {
            'Lasso': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Lasso())
            ]),
            'Ridge': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Ridge())
            ]),
            'XGBoost': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', xgb.XGBRegressor())
            ]),
            'RandomForest': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor())
            ])
        }

    def train_and_log_models(self):
        # Define cross-validation settings
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train and log models
        for name, pipeline in self.pipelines.items():
            with mlflow.start_run(run_name=name):
                # Perform cross-validation
                cv_scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring='r2')
                mean_rmse = cv_scores.mean() ** 0.5  # Convert negative MSE to RMSE

                # Fit the model
                pipeline.fit(self.X_train, self.y_train)
                predictions = pipeline.predict(self.X_test)
                rmse = mean_squared_error(self.y_test, predictions, squared=False)
                r2 = r2_score(self.y_test, predictions)
                mae = mean_absolute_error(self.y_test, predictions)
                
                # Log model and metrics
                mlflow.sklearn.log_model(pipeline, name)
                mlflow.log_metric('cv_rmse', mean_rmse)
                mlflow.log_metric('test_rmse', rmse)
                mlflow.log_metric('r2', r2)
                mlflow.log_metric('mae', mae)
                self._plot_predictions_vs_true(predictions, self.y_test, name)
                print(f'{name} Model CV RMSE: {mean_rmse}')
                print(f'{name} Model Test RMSE: {rmse}')
                print(f'{name} Model R2: {r2}')
                print(f'{name} Model MAE: {mae}')
        
    def _plot_predictions_vs_true(self, predictions, y_test, model_name):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5, label='Predictions vs True Values')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'{model_name} - Predictions vs True Values')
        plt.legend()
        plt.grid(True)
        plot_path = f'predictions_vs_true_{model_name}.png'
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)