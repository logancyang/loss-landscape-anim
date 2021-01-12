import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def setup(self, stage):
        # transforms for images
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # prepare transforms standard to MNIST
        self.mnist_train = MNIST(
            os.getcwd(), train=True, download=True, transform=transform
        )
        self.mnist_test = MNIST(
            os.getcwd(), train=False, download=True, transform=transform
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=64)
