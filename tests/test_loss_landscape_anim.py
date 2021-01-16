import pickle

import numpy as np
import torch
from loss_landscape_anim import __version__
from loss_landscape_anim.loss_landscape import LossGrid
from pytest import fixture
from torch.utils.data import TensorDataset

SEED = 180224


def test_version():
    assert __version__ == "0.1.6"


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
def test_loss_grid(test_model, test_dataset):
    optim_path, _, _ = zip(
        *[
            (path["flat_w"], path["loss"], path["accuracy"])
            for path in test_model.optim_path
        ]
    )
    return LossGrid(
        optim_path=optim_path,
        model=test_model,
        data=test_dataset.tensors,
        seed=SEED,
        tqdm_disable=True,
    )


def test_loss_grid_coords(test_loss_grid):
    loss_2d = test_loss_grid.loss_values_2d
    coords_x, coords_y = test_loss_grid.coords
    assert type(coords_x) == np.ndarray == type(loss_2d)
    assert len(loss_2d) == len(coords_x) == len(coords_y)

    argmin_2d = np.unravel_index(loss_2d.argmin(), loss_2d.shape)
    assert np.min(loss_2d) == loss_2d[argmin_2d] == test_loss_grid.loss_min
