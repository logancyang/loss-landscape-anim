import pickle

import numpy as np
import torch
from loss_landscape_anim import __version__
from loss_landscape_anim.loss_landscape import LossGrid
from pytest import fixture
from torch.utils.data import TensorDataset

SEED = 180224


def test_version():
    assert __version__ == "0.1.9"


@fixture
def test_model():
    return torch.load("./tests/test_models/model.pt")


@fixture
def test_dataset():
    datadict = pickle.load(open("./tests/test_data/data_2d_3class.p", "rb"))
    X_train = torch.Tensor(datadict["X_train"])
    y_train = torch.LongTensor(datadict["y_train"])
    return TensorDataset(X_train, y_train)


@fixture
def test_loss_grid():
    loss_values_2d, argmin, loss_min = pickle.load(
        open("./tests/test_data/lossgrid.p", "rb")
    )
    return loss_values_2d, argmin, loss_min


def test_loss_grid_coords(test_loss_grid):
    loss_values_2d, argmin, loss_min = test_loss_grid

    argmin = np.unravel_index(loss_values_2d.argmin(), loss_values_2d.shape)
    assert np.min(loss_values_2d) == loss_values_2d[argmin] == loss_min
