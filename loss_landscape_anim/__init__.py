"""Package init."""
from .datamodule import MNISTDataModule, SpiralsDataModule
from .loss_landscape import LossGrid
from .main import loss_landscape_anim
from .model import MLP, GenericModel, LeNet

__version__ = "0.1.10"

__all__ = [
    "loss_landscape_anim",
    "GenericModel",
    "MLP",
    "LeNet",
    "LossGrid",
    "SpiralsDataModule",
    "MNISTDataModule",
]
