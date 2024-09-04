import os
import mlflow
import mlflow.sklearn
import pandas as pd
import json

class Predictor:
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME', 'financial-xgboost')
        self.model_version = os.getenv('MODEL_VERSION', '1')
        #self.processed_data_path = os.getenv('PROCESSED_DATA_PATH', '/app/pred.json')
        self.processed_data_path = os.getenv('PROCESSED_DATA_PATH', '../data/prediction/pred.json')
        #self.predictions_output_path = os.getenv('PREDICTIONS_OUTPUT_PATH', '/app/predictions.csv')
        self.predictions_output_path = os.getenv('PREDICTIONS_OUTPUT_PATH', '../data/prediction/predictions.csv')

        # Set up MLflow
        mlflow.set_tracking_uri('sqlite:///../mlflow.db')
        mlflow.artifact_location = "/mlruns"

    def predict_and_save(self):
        # Load processed data
        with open(self.processed_data_path, 'r') as file:
            data = json.load(file)

        # Load model from MLflow
        model_uri = f"models:/{self.model_name}/{self.model_version}"
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded")
        # Make predictions
        predictions = model.predict(pd.DataFrame(data))
        
        # Save predictions
        predictions_df = pd.DataFrame(predictions, columns=['predictions'])
        predictions_df.to_csv(self.predictions_output_path, index=False)

        print(f"Predictions saved to {self.predictions_output_path}")

if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict_and_save()

