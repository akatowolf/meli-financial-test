import os
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

class Preprocessor:
    def __init__(self, numerical_features, categorical_features, apply_scaling=True, apply_encoding=True, apply_saving=False, encoder_file='../models/encoders/encoder.pkl'):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.apply_scaling = apply_scaling
        self.apply_encoding = apply_encoding
        self.apply_saving = apply_saving
        self.encoder_file = encoder_file
        self.scaler = StandardScaler() if apply_scaling else None
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) if apply_encoding else None

        if self.apply_saving and self.encoder_file:
            self.load_encoder()

    def fit(self, X):
        if self.apply_scaling and self.numerical_features:
            self.scaler.fit(X[self.numerical_features])
        
        if self.apply_encoding and self.categorical_features:
            self.encoder.fit(X[self.categorical_features])

    def transform(self, X):
        result = pd.DataFrame()
        
        if self.apply_scaling and self.numerical_features:
            scaled_data = self.scaler.transform(X[self.numerical_features])
            result = pd.DataFrame(scaled_data, columns=self.numerical_features, index=X.index)
        
        if self.apply_encoding and self.categorical_features:
            encoded_data = self.encoder.transform(X[self.categorical_features])
            encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(self.categorical_features), index=X.index)
            result = pd.concat([result, encoded_df], axis=1)
        
        return result

    def fit_transform(self, X):
        self.fit(X)
        if self.apply_saving:
            self.save_encoder()
        return self.transform(X)

    def save_preprocessor(self, preprocessor_file='../models/preprocessors/preprocessor.pkl'):
        os.makedirs(os.path.dirname(preprocessor_file), exist_ok=True)
        with open(preprocessor_file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_preprocessor(preprocessor_file='../models/preprocessors/preprocessor.pkl'):
        with open(preprocessor_file, 'rb') as f:
            return pickle.load(f)
