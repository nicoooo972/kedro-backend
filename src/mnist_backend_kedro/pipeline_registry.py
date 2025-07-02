"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from mnist_backend_kedro.pipelines import data_preparation
from mnist_backend_kedro.pipelines import training


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_preparation_pipeline = data_preparation.create_pipeline()
    training_pipeline = training.create_pipeline()

    return {
        "dp": data_preparation_pipeline,
        "train": training_pipeline,
        "__default__": data_preparation_pipeline + training_pipeline,
    }
