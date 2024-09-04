import pandas as pd
import numpy as np
import utils
import json
class FeatureEngineering:
    def __init__(self, data):
        self.data = data
    
    def create_features(self, json_path, processed_path):
        """
        Create new features from the existing data.
        """
        df = self.data.copy()

        # Drop columns
        df = df.drop(['account_id', 'transaction_details', 'value_date', 'city', 'device'], axis=1)

        # Date to features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5)*1
        df = df.drop(['date'], axis=1)

        # Mapping check
        df['chq_no'] = np.where(df['chq_no'].isna(),0,1)

        # Mapping category
        category_mapping = {
            'Miscellaneous': 'Lifestyle',
            'Transfer': 'Financial',
            'Investment': 'Financial',
            'Subscriptions': 'Lifestyle',
            'Utility Bill': 'Obligation',
            'Charity & Donations': 'Lifestyle',
            'Food & Dining': 'Lifestyle',
            'Loan Payment': 'Obligation',
            'Electronics & Gadgets': 'Lifestyle',
            'Shopping': 'Lifestyle',
            'Pets & Pet Care': 'Lifestyle',
            'Travel': 'Lifestyle',
            'Insurance': 'Obligation',
            'Transportation': 'Lifestyle',
            'Health & Wellness': 'Lifestyle',
            'Entertainment': 'Lifestyle',
            'Education': 'Lifestyle',
            'Childcare & Parenting': 'Obligation'
        }

        df['category'] = df['category'].map(category_mapping)
        

        df['total_transfer'] = df['deposit_amt'].fillna(0) - df['withdrawal_amt'].fillna(0)
        df = df.drop(['deposit_amt','withdrawal_amt'],axis=1)

        # Clip outliers
        cols_to_clip = ['total_transfer']
        for column in cols_to_clip:
            df[column] = utils.clip_upper(df, column)

        dummies = pd.get_dummies(df['category'], prefix='category', drop_first=False)*1
        df_final = pd.concat([df, dummies], axis=1)
        df_final = df_final.drop('category', axis=1)
        df_final = df_final.drop('chq_no', axis=1)
        column_names = self.data.columns.tolist()
        with open(json_path, 'w') as file:
            json.dump(column_names, file, indent=4)
        print(f"Column names saved to {json_path}")
        df_final.to_csv(processed_path, index=False)
        print(f"Processed data saved to {processed_path}")

        return df_final
