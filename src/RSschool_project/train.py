from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

from .data import get_dataset
from .pipeline import create_pipeline

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-feature-selection",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-standart-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-minmax-scaler",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_feature_selection: bool,
    use_standart_scaler: bool,
    use_minmax_scaler: bool,
    max_iter: int,
    logreg_c: float
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path, 
        random_state, 
        test_split_ratio)
    print('Done')
    with mlflow.start_run():
        pipeline = create_pipeline(use_feature_selection, use_standart_scaler, 
        use_minmax_scaler, max_iter, logreg_c, random_state)
        pipeline.fit(features_train, target_train)
        accuracy = accuracy_score(target_val, pipeline.predict(features_val))
        mlflow.log_param('FeatureSelection', use_feature_selection)
        mlflow.log_param('StandartScaler', use_standart_scaler)
        mlflow.log_param('MinMaxScaler', use_minmax_scaler)
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('logreg_c', logreg_c)
        mlflow.log_param('accuracy', accuracy)
        click.echo(f"Accuracy: {round(accuracy, 5)}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")

