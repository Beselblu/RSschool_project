import logging 
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from skleatn.model_selection import KFold

def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    dataset.columns = dataset.columns.str.lower()
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop('cover_type', axis=1)
    target = dataset['cover_type']
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    return features_train, features_val, target_train, target_val

def get_KFdataset(
    csv_path: Path, n_splits: int, shuffle: bool,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    dataset.columns = dataset.columns.str.lower()
    click.echo(f"Dataset shape: {dataset.shape}.")

    kf = KFold(n_splits, shuffle, random_state)
    for train_index, val_index in kf.splits(dataset):
        features_train, features_val = dataset[train_index], dataset[val_index]
        target_train, target_val = dataset[train_index], dataset[val_index]
    
    features_train, features_val = features_train.drop('cover_type', axis=1), features_val.drop('cover_type', axis=1)
    target_train, target_val = target_train['cover_type'], target_val['cover_type']
    return features_train, features_val, target_train, target_val