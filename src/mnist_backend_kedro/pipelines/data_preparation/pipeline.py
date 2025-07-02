"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.13
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_dataloaders


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_dataloaders,
                inputs=None,
                outputs=["train_loader", "test_loader"],
                name="create_dataloaders_node",
            )
        ]
    )
