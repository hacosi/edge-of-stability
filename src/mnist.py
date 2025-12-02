import os
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torchvision.datasets import MNIST
from cifar import flatten, center, standardize, unflatten, make_labels

DATASETS_FOLDER = "./data"
os.makedirs(DATASETS_FOLDER, exist_ok=True)


def load_mnist(loss: str) -> (TensorDataset, TensorDataset):
    mnist_train = MNIST(root=DATASETS_FOLDER, download=True, train=True)
    mnist_test = MNIST(root=DATASETS_FOLDER, download=True, train=False)
    print(mnist_train.data.numpy())
    X_train_np = mnist_train.data.numpy().astype(np.float32) / 255.0
    X_test_np = mnist_test.data.numpy().astype(np.float32) / 255.0

    # add channel-last axis like CIFAR (H,W,C) so your flatten/unflatten stay consistent
    X_train_np = np.expand_dims(X_train_np, axis=-1)  # (N,28,28,1)
    X_test_np = np.expand_dims(X_test_np, axis=-1)
    # X_train, X_test = (
    #     flatten(mnist_train.data.unsqueeze(-1) / 255),
    #     flatten(mnist_test.data.unsqueeze(-1) / 255),
    # )
    X_train = flatten(X_train_np)
    X_test = flatten(X_test_np)
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
