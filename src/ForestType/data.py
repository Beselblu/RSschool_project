import logging 
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split

def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    dataset.columns = dataset.columns.srt.lower()
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop('cover_type', axis=1)
    target = dataset['cover_type']
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    return features_train, features_val, target_train, target_val