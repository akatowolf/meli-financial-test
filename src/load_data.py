# Script for load data
import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        data = pd.read_parquet(self.filepath)
        return data