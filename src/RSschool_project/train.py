from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

from .data import get_dataset

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def train(
    dataset_path: Path,
    random_state: int,
    test_split_ratio: float
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        csv_path, 
        random_state, 
        test_split_ratio)
    print('Done')