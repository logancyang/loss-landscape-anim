import os
import pickle

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

SAMPLE_DATA_PATH = "./sample_data/data_2d_3class.p"


class SampleDataModule(pl.LightningDataModule):
    def __init__(self):
        self.datadict = pickle.load(open(SAMPLE_DATA_PATH, "rb"))
        self.input_dim = self.datadict["X_train"].shape[1]
        self.n_classes = len(set(self.datadict["y_train"]))
        self.X = torch.Tensor(self.datadict["X_train"])
        self.y = torch.LongTensor(self.datadict["y_train"])
        self.dataset = TensorDataset(self.X, self.y)

    def train_dataloader(self, batch_size=None, num_workers=0):
        if not batch_size:
            batch_size = len(self.X)
        return DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, n_examples=3000):
        # transforms for images
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.mnist_train = MNIST(
            os.getcwd(), train=True, download=True, transform=transform
        )
        self.input_dim = 28 * 28
        self.n_classes = 10
        x_flat = torch.flatten(
            self.mnist_train.data, start_dim=1
        )  # (60000, 28, 28) -> (60000, 784)
        x_float = x_flat.float() / 255
        randrow = torch.randperm(x_flat.size(0))[:n_examples]
        self.X = x_float[randrow, :]
        assert self.X.shape[1] == 784
        assert all(self.X[0].size(0) == example.size(0) for example in self.X)
        self.y = self.mnist_train.targets[randrow]
        self.dataset = TensorDataset(self.X, self.y)

    def train_dataloader(self, batch_size, num_workers=0):
        if not batch_size:
            raise Exception("batch_size must be specified for MNIST dataset.")
        return DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)
