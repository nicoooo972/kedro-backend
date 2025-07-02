"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.13
"""

import torch
from torchvision import datasets, transforms

def create_dataloaders():
    """
    Downloads the MNIST dataset and creates DataLoader objects for train and test sets.
    """
    # Chargement des données
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/01_raw", download=True, train=True, transform=tf),
        batch_size=64,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/01_raw", download=True, train=False, transform=tf),
        batch_size=64,
        shuffle=True,
    )

    return train_loader, test_loader 