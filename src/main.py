from training_pipeline import ModelTrainer

# Main to run train pipelinegit 
data_path = '../data/raw/bank_transactions.parquet'
json_path = '../data/processed/column_names.json'
processed_path = '../data/processed/processed.csv'

trainer = ModelTrainer(data_path, json_path, processed_path)
trainer.train_and_log_models()
