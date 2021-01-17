"""
Steps:
1. Load data
2. Create a pytorch lightning model
3. Record the parameters during training
4. Use PCA to project the parameters to 2D
5. Collect the values in 2D:
    a. A list of 2D values as the trajectory obtained by projecting the
       parameters down to the 2D space spanned by the top 2 PC.
    b. A grid of 2D values that capture (a) and some more for visual
       aesthetics.
"""
import pathlib

import pytorch_lightning as pl
import torch

from loss_landscape_anim.datamodule import MNISTDataModule, SpiralsDataModule
from loss_landscape_anim.loss_landscape import LossGrid, DimReduction
from loss_landscape_anim.model import MLP, LeNet
from loss_landscape_anim.plot import animate_contour, sample_frames


def loss_landscape_anim(
    n_epochs,
    datamodule=None,
    model=None,
    optimizer="adam",
    reduction_method="pca",  # "pca", "random", "custom" are supported
    custom_directions=None,
    model_dirpath="checkpoints/",
    model_filename="model.pt",
    gpus=0,
    load_model=False,
    output_to_file=True,
    output_filename="sample.gif",
    giffps=15,
    sampling=False,
    n_frames=300,
    seed=None,
    return_data=False,
):
    if seed:
        torch.manual_seed(seed)

    if not datamodule:
        print("Data module not provided, using sample data: spirals dataset")
        datamodule = SpiralsDataModule()

    if not model and not load_model:
        print(
            "Model not provided, using default classifier: "
            "MLP with 1 hidden layer of 50 neurons"
        )
        model = MLP(
            input_dim=datamodule.input_dim,
            num_classes=datamodule.num_classes,
            learning_rate=5e-3,
            optimizer=optimizer,
            gpus=gpus,
        )

    model_dir = pathlib.Path(model_dirpath)
    if not model_dir.is_dir():
        (model_dir.parent / model_dirpath).mkdir(parents=True, exist_ok=True)
        print(f"Model directory {model_dir.absolute()} does not exist, creating now.")
    file_path = model_dirpath + model_filename

    if gpus > 0:
        print("======== Using GPU for training ========")

    # Train model
    if not load_model:
        model.gpus = gpus
        train_loader = datamodule.train_dataloader()
        trainer = pl.Trainer(
            progress_bar_refresh_rate=5, max_epochs=n_epochs, gpus=gpus
        )
        print(f"Training for {n_epochs} epochs...")
        trainer.fit(model, train_loader)
        torch.save(model, file_path)
        print(f"Model saved at {pathlib.Path(file_path).absolute()}.")
    else:
        print(f"Loading model from {pathlib.Path(file_path).absolute()}")

    model_file = pathlib.Path(file_path)
    if not model_file.is_file():
        raise Exception("Model file not found!")

    model = torch.load(file_path)
    # Sample from full path
    sampled_optim_path = sample_frames(model.optim_path, max_frames=n_frames)
    optim_path, loss_path, accu_path = zip(
        *[
            (path["flat_w"], path["loss"], path["accuracy"])
            for path in sampled_optim_path
        ]
    )

    print(f"\n# sampled steps in optimization path: {len(optim_path)}")

    """Dimensionality reduction and Loss Grid"""
    print(f"Dimensionality reduction method specified: {reduction_method}")
    dim_reduction = DimReduction(
        params_path=optim_path,
        reduction_method=reduction_method,
        custom_directions=custom_directions,
        seed=seed,
    )
    reduced_dict = dim_reduction.reduce()
    path_2d = reduced_dict["path_2d"]
    directions = reduced_dict["reduced_dirs"]
    pcvariances = reduced_dict.get("pcvariances")

    loss_grid = LossGrid(
        optim_path=optim_path,
        model=model,
        data=datamodule.dataset.tensors,
        path_2d=path_2d,
        directions=directions,
    )

    animate_contour(
        param_steps=path_2d.tolist(),
        loss_steps=loss_path,
        acc_steps=accu_path,
        loss_grid=loss_grid.loss_values_log_2d,
        coords=loss_grid.coords,
        true_optim_point=loss_grid.true_optim_point,
        true_optim_loss=loss_grid.loss_min,
        pcvariances=pcvariances,
        giffps=giffps,
        sampling=sampling,
        output_to_file=output_to_file,
        filename=output_filename,
    )
    if return_data:
        return list(optim_path), list(loss_path), list(accu_path)
