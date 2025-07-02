"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.13
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_permutation, train_model, evaluate_model, package_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_permutation,
                inputs=None,
                outputs="permutation",
                name="create_permutation_node",
            ),
            node(
                func=train_model,
                inputs=["train_loader", "permutation", "params:model_params"],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "test_loader", "permutation"],
                outputs=None,
                name="evaluate_model_node",
            ),
            node(
                func=package_model,
                inputs=["trained_model", "permutation", "params:model_params"],
                outputs="packaged_model",
                name="package_model_node",
            ),
        ]
    )
