from data_loader.load_datasets import load_all_datasets
from training.merge_datasets import merge_datasets
from training.train_model import train_model

datasets = load_all_datasets()

merged_dataset = merge_datasets(datasets)

train_model(merged_dataset)