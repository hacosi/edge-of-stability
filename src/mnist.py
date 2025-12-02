import os
import torch
from torch.utils.data.dataset import TensorDataset
from torchvision.datasets import MNIST
from cifar import flatten, center, standardize, unflatten, make_labels

DATASETS_FOLDER = "./data"
os.makedirs(DATASETS_FOLDER, exist_ok=True)


def load_mnist(loss: str) -> (TensorDataset, TensorDataset):
    mnist_train = MNIST(root=DATASETS_FOLDER, download=True, train=True)
    mnist_test = MNIST(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = (
        flatten(mnist_train.data.unsqueeze(-1) / 255),
        flatten(mnist_test.data.unsqueeze(-1) / 255),
    )
    y_train, y_test = (
        make_labels(torch.tensor(mnist_train.targets), loss),
        make_labels(torch.tensor(mnist_test.targets), loss),
    )
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(
        center_X_train, center_X_test)
    train = TensorDataset(
        torch.from_numpy(unflatten(standardized_X_train,
                         (28, 28, 1)).transpose((0, 3, 1, 2))).float(),
        y_train,
    )
    test = TensorDataset(
        torch.from_numpy(unflatten(standardized_X_test,
                         (28, 28, 1)).transpose((0, 3, 1, 2))).float(),
        y_test,
    )
    return train, test
