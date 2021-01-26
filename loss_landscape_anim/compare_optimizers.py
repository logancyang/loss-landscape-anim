import pathlib
import pickle
import json

import numpy as np
import pytorch_lightning as pl
import torch

from loss_landscape_anim.datamodule import SpiralsDataModule
from loss_landscape_anim.loss_landscape import DimReduction, LossGrid
from loss_landscape_anim.model import MLP
from loss_landscape_anim.plot import animate_paths, sample_frames


def train_models(
    n_epochs,
    optimizers,
    learning_rate,
    datamodule=None,
    model_dirpath="checkpoints/",
    weight_init=None,
    gpus=0,
    seed=None,
):
    """Train the same neural net with different optimizers on the same data."""
    if not datamodule:
        datamodule = SpiralsDataModule()

    param_count = None

    # Train models
    for i, optimizer in enumerate(optimizers):
        print(f"\nTraining MLP with {optimizer}\n")
        if seed:
            torch.manual_seed(seed)

        model = MLP(
            input_dim=datamodule.input_dim,
            num_classes=datamodule.num_classes,
            num_hidden_layers=1,
            hidden_dim=20,
            learning_rate=learning_rate,
            optimizer=optimizer,
            weight_init=weight_init,
            gpus=gpus,
        )

        model.gpus = gpus
        train_loader = datamodule.train_dataloader()
        trainer = pl.Trainer(
            progress_bar_refresh_rate=5, max_epochs=n_epochs, gpus=gpus
        )
        print(f"Training for {n_epochs} epochs...")
        trainer.fit(model, train_loader)
        file_path = f"./{model_dirpath}/model_{optimizer}_{i}.pt"
        torch.save(model, file_path)
        print(f"Model saved at {pathlib.Path(file_path).absolute()}.")
        if not param_count:
            param_count = model.get_param_count()
    print("All models trained successfully.")
    return param_count


def get_optimizer_paths(
    optimizers, custom_directions, model_dirpath="checkpoints/", seed=None
):
    """Make one plot to compare the paths of different optimizers.

    Load from pretrained models. Each pretrained model has info on what optimizer it
    used in model.optimizer.

    Note that this function needs a list of *pretrained* model paths as input
    """
    optim_paths_dict = {}
    # Set the directions
    dim_reduction = DimReduction(
        reduction_method="custom",
        custom_directions=custom_directions,
        seed=seed,
    )
    for i, optimizer in enumerate(optimizers):
        # Try loading models, getting the paths one by one
        model_path = f"./{model_dirpath}/model_{optimizer}_{i}.pt"
        model_file = pathlib.Path(model_path)
        if not model_file.is_file():
            raise Exception("Model file not found!")

        model = torch.load(model_path)
        # Sample from full path
        sampled_optim_path = sample_frames(model.optim_path, max_frames=300)
        optim_path = [path["flat_w"] for path in sampled_optim_path]
        # optim_paths_dict[optimizer] = dim_reduction.reduce(optim_path)
        reduced_dict = dim_reduction.reduce(optim_path)
        optim_paths_dict[optimizer] = {}
        optim_paths_dict[optimizer]["path_2d"] = reduced_dict["path_2d"]
        optim_paths_dict[optimizer]["optim_path"] = optim_path
    return optim_paths_dict


def plot_optimizers(
    optimizers,
    optim_paths_dict,
    custom_directions,
    model_dirpath="checkpoints/",
    datamodule=None,
):
    # TODO: Experiment with different end points
    N_OPTIM = 3
    if not datamodule:
        datamodule = SpiralsDataModule()
    # Takes multiple paths, compute loss grid, draw contour and paths
    # loss grid needs full-network param path for setting the center point,
    optim_dict = optim_paths_dict[optimizers[N_OPTIM]]
    # needs model to compute loss grid
    model_path = f"./{model_dirpath}/model_{optimizers[N_OPTIM]}_{N_OPTIM}.pt"
    model_file = pathlib.Path(model_path)
    if not model_file.is_file():
        raise Exception("Model file not found!")
    model = torch.load(model_path)

    assert model.get_param_count() == custom_directions[0].shape[0]
    # path_2d is needed only to set margin for loss grid
    path_2d = optim_paths_dict[optimizers[N_OPTIM]]["path_2d"]
    loss_grid = LossGrid(
        optim_path=optim_dict["optim_path"],
        model=model,
        data=datamodule.dataset.tensors,
        path_2d=path_2d,
        directions=custom_directions,
    )

    optim_paths_2d = [
        optim_paths_dict[optimizer]["path_2d"].tolist() for optimizer in optimizers
    ]

    animate_paths(
        optimizers=optimizers,
        optim_paths_2d=optim_paths_2d,
        loss_grid=loss_grid.loss_values_log_2d,
        coords=loss_grid.coords,
        true_optim_point=loss_grid.true_optim_point,
    )


def compare_optimizers(
    optimizers,
    learning_rate,
    param_count=None,
    model_dirpath="checkpoints/",
    train_new=True,
    seed=None,
):
    if train_new:
        weight_init = pickle.load(open("checkpoints/weight.p", "rb"))
        print("\n Weight init loaded from disk.")
        param_count = train_models(
            learning_rate=learning_rate,
            n_epochs=200,
            optimizers=optimizers,
            model_dirpath="checkpoints/",
            weight_init=weight_init,
            gpus=0,
            seed=seed,
        )
        print(f"New models trained with {param_count} parameters.")
    else:
        assert param_count is not None, "Must enter # params when loading model."

    np.random.seed(seed)
    u_gen = np.random.normal(size=param_count)
    u = u_gen / np.linalg.norm(u_gen)
    v_gen = np.random.normal(size=param_count)
    v = v_gen / np.linalg.norm(v_gen)
    optim_paths_dict = get_optimizer_paths(
        optimizers=optimizers, custom_directions=[u, v], seed=seed
    )

    # inspect_model(optimizers, optim_paths_dict)
    # print("Model inspection done!")
    plot_optimizers(
        optimizers=optimizers,
        optim_paths_dict=optim_paths_dict,
        custom_directions=[u, v],
        model_dirpath=model_dirpath,
    )


def save_weight_init(model_path="checkpoints/model_adam_0.pt"):
    model = torch.load(model_path)
    optim_path = [path["flat_w"] for path in model.optim_path]
    weight_init = optim_path[0]
    pickle.dump(weight_init, open("checkpoints/weight.p", "wb"))


def inspect_model(optimizers, optim_paths_dict):
    for i, optimizer in enumerate(optimizers):
        model_path = f"checkpoints/model_{optimizer}_{i}.pt"
        model = torch.load(model_path)
        optim_path_converted = _convert_tensor_to_np(model.optim_path)
        optim_path_json = []
        for j, item in enumerate(optim_path_converted):
            new_item = {}
            new_item["path_2d"] = optim_paths_dict[optimizer]["path_2d"][j].tolist()
            new_item["loss"] = item["loss"]
            new_item["accuracy"] = item["accuracy"]
            optim_path_json.append(new_item)
        with open(f"checkpoints/model_{optimizer}_{i}.json", "w") as f:
            json.dump(optim_path_json, f)


def _convert_tensor_to_np(optim_path):
    res = []
    for item in optim_path:
        res_item = {}
        res_item["flat_w"] = item["flat_w"].numpy().tolist()
        res_item["loss"] = item["loss"].numpy().tolist()
        res_item["accuracy"] = item["accuracy"].numpy().tolist()
        res.append(res_item)
    return res
