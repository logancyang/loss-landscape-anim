import os
import pickle

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

SPIRALS_DATA_PATH = "./sample_data/spirals.p"


class SpiralsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=None):
        self.datadict = pickle.load(open(SPIRALS_DATA_PATH, "rb"))
        self.input_dim = self.datadict["X_train"].shape[1]
        self.num_classes = len(set(self.datadict["y_train"]))
        self.X = torch.Tensor(self.datadict["X_train"])
        self.y = torch.LongTensor(self.datadict["y_train"])
        self.dataset = TensorDataset(self.X, self.y)
        self.batch_size = batch_size if batch_size else len(self.X)

    def train_dataloader(self, num_workers=0):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=num_workers
        )


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, n_examples=3000):
        super().__init__()
        self.batch_size = batch_size if batch_size else 1
        self.num_classes = 10
        self.input_dim = (1, 28, 28)
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.mnist_train = MNIST(
            os.getcwd(), train=True, download=True, transform=transform
        )
        # Note: do not use mnist_train.data, it's the original data
        # *before* transformations! The transformations are applied in __getitem__
        randrow = torch.randperm(len(self.mnist_train))[:n_examples]
        subset = Subset(self.mnist_train, randrow)
        assert len(subset) == n_examples
        self.X, self.y = self._extract_features_targets(subset)
        self.dataset = TensorDataset(self.X, self.y)

    def train_dataloader(self, num_workers=0):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=num_workers
        )

    def _extract_features_targets(self, subset):
        X, y = [], []
        for tup in subset:
            X.append(tup[0])
            y.append(tup[1])
        return torch.stack(X), torch.LongTensor(y)
