import mlflow
import mlflow.xgboost
from sklearn.metrics import recall_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
from load_data import DataLoader
from feature_engineering import FeatureEngineering
from preprocessor import Preprocessor
import re
import os
import numpy as np

tracking_uri = "sqlite:///../mlflow.db"
artifact_location = os.path.abspath("../models/mlruns")
mlflow.set_tracking_uri(tracking_uri)

class FineTuningPipeline:
    def __init__(self, data_path, experiment_name='Diabetes_Classy_tuning'):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.data_loader = DataLoader(filepath=self.data_path)
        self.feature_engineer = FeatureEngineering()
        self.preprocessor = None

        # Set experiment
        mlflow.set_experiment(self.experiment_name)
        mlflow.get_experiment_by_name(self.experiment_name)

    def preprocess_data(self):
        # Load data
        df = self.data_loader.load_data()
        
        # Feature Engineering
        df_preprocessed = self.feature_engineer.create_features(df)
        
        # Separate features and target
        X = df_preprocessed.drop('readmitted', axis=1)
        y = df_preprocessed['readmitted']
        
        # Select numerical and categorical features for scaling and encoding
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Initialize the Preprocessor
        self.preprocessor = Preprocessor(numerical_features, categorical_features, apply_saving=True)
        
        # Apply preprocessing
        X_preprocessed = self.preprocessor.fit_transform(X)
        X_preprocessed.columns = [
            re.sub(r'[^A-Za-z0-9_]+', '', str(col)) for col in X_preprocessed.columns
        ]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, stratify=y, random_state=42)
        
        sample = y_train[y_train==0].shape

        # Apply ROSE to balance the training data
        sampling_strategy = {0:sample[0], 1:sample[0], 2:sample[0]}
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
        noise = np.random.normal(0, 0.01, X_train_balanced.shape)
        X_train_balanced += noise
        
        return X_train_balanced, X_val, y_train_balanced, y_val

    def fine_tuning(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],  # Reduced max_depth
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 0.9],  # Reduced subsample
            'lambda': [1, 10],  # Added L2 regularization
            'alpha': [0, 1]  # Added L1 regularization
        }

        model = xgb.XGBClassifier(eval_metric='mlogloss')

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='recall_weighted',
            cv=5,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, grid_search.cv_results_

    def run(self):
        X_train, X_val, y_train, y_val = self.preprocess_data()
        
        # Fine-tuning the XGBoost model
        best_model, best_params, best_score, cv_results = self.fine_tuning(X_train, y_train)
        
        with mlflow.start_run(run_name="Fine-Tuned_XGBoost"):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_recall_score", best_score)

            # Evaluate on training set
            y_train_pred = best_model.predict(X_train)
            train_recall = recall_score(y_train, y_train_pred, average='weighted')
            train_report = classification_report(y_train, y_train_pred)

            # Evaluate on validation set
            y_val_pred = best_model.predict(X_val)
            val_recall = recall_score(y_val, y_val_pred, average='weighted')
            val_report = classification_report(y_val, y_val_pred)

            # Log metrics and model
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("val_recall", val_recall)
            mlflow.log_text(train_report, "train_classification_report.txt")
            mlflow.log_text(val_report, "val_classification_report.txt")
            mlflow.log_text("This train use ROSE", "Rose.txt")
            mlflow.log_text("Added params to avoid overfitting", "Additional.txt")
            mlflow.log_text(f"Grid Search CV Results: {cv_results}", "cv_results.txt")
            mlflow.xgboost.log_model(best_model, "Fine-Tuned_XGBoost")

            print(f"Best Parameters: {best_params}")
            print(f"Best Recall Score: {best_score:.4f}")
            print(f"Training Recall: {train_recall:.4f}")
            print(f"Validation Recall: {val_recall:.4f}")
            print(f"Training Classification Report:\n{train_report}")
            print(f"Validation Classification Report:\n{val_report}")

