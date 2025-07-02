"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.13
"""

import torch
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from typing import Dict, Any

from mnist_backend_kedro.model.convnet import ConvNet


def create_permutation():
    """Create a fixed permutation for the MNIST dataset."""
    return torch.randperm(784)


def train_model(
    train_loader: Any, permutation: torch.Tensor, parameters: Dict[str, Any]
):
    """Trains the model.

    Args:
        train_loader: Training data loader.
        permutation: The permutation to apply to the images.
        parameters: Dictionary of parameters for training.

    Returns:
        Trained model.
    """
    mlflow.start_run()
    mlflow.log_params(parameters)

    # Initialize model
    model = ConvNet(
        input_size=parameters["input_size"],
        n_kernels=parameters["n_kernels"],
        output_size=parameters["output_size"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perm = permutation.to(device)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"])

    for epoch in range(parameters["epochs"]):
        print(f"--- Epoch {epoch+1}/{parameters['epochs']} ---")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            batch_size = data.shape[0]
            data_flattened = data.view(batch_size, -1)
            data_permuted = data_flattened[:, perm]
            data_reshaped = data_permuted.view(batch_size, 1, 28, 28)

            optimizer.zero_grad()
            logits = model(data_reshaped)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                loss_val = loss.item()
                print(
                    f"  Batch: {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss_val:.4f}"
                )
                mlflow.log_metric(
                    "train_loss", loss_val, step=epoch * len(train_loader) + batch_idx
                )

    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

    return model


def evaluate_model(model: torch.nn.Module, test_loader: Any, permutation: torch.Tensor):
    """Calculates and logs the loss and accuracy of the model on the test set.

    Args:
        model: Trained model.
        test_loader: Test data loader.
        permutation: The permutation to apply to the images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perm = permutation.to(device)
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            data_flattened = data.view(batch_size, -1)
            data_permuted = data_flattened[:, perm]
            data_reshaped = data_permuted.view(batch_size, 1, 28, 28)

            logits = model(data_reshaped)
            test_loss += F.cross_entropy(logits, target, reduction="sum").item()
            pred = torch.argmax(logits, dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"Test: Loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({accuracy:.2f}%)"
    )

    # Log metrics to MLflow
    # This assumes we are in an active MLflow run context.
    # A better way would be to pass the run_id, but for simplicity...
    with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id):
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", accuracy)


def package_model(
    model: torch.nn.Module, permutation: torch.Tensor, parameters: Dict[str, Any]
):
    """Packages the model and permutation into a single file.

    Args:
        model: Trained model.
        permutation: The permutation used.
        parameters: Dictionary of model parameters.
    """
    model_data = {
        "model_state_dict": model.state_dict(),
        "permutation": permutation,
        "n_kernels": parameters["n_kernels"],
        "input_size": parameters["input_size"],
        "output_size": parameters["output_size"],
    }
    return model_data
