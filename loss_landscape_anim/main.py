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
from loss_landscape_anim.loss_landscape import LossGrid
from loss_landscape_anim.model import MLP, LeNet
from loss_landscape_anim.plot import animate_contour, sample_frames


SEED = 180224


def loss_landscape_anim(
    n_epochs,
    datamodule=None,
    model=None,
    optimizer="adam",
    model_dirpath="checkpoints/",
    model_filename="model.pt",
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

    if not model:
        print(
            "Model not provided, using default classifier: "
            "MLP with 1 hidden layer of 50 neurons"
        )
        model = MLP(
            input_dim=datamodule.input_dim,
            num_classes=datamodule.num_classes,
            learning_rate=5e-3,
            optimizer=optimizer,
        )

    train_loader = datamodule.train_dataloader()

    # Train model
    model_dir = pathlib.Path(model_dirpath)
    if not model_dir.is_dir():
        (model_dir.parent / model_dirpath).mkdir(parents=True, exist_ok=True)
        print(f"Model directory {model_dir.absolute()} does not exist, creating now.")
    file_path = model_dirpath + model_filename

    if not load_model:
        trainer = pl.Trainer(progress_bar_refresh_rate=5, max_epochs=n_epochs)
        print(f"Training for {n_epochs} epochs...")
        trainer.fit(model, train_loader)
        torch.save(model, file_path)
        print(f"Model saved at {pathlib.Path(file_path).absolute()}.")

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

    print(f"Total steps in optimization path: {len(optim_path)}")

    """PCA and Loss Grid"""

    loss_grid = LossGrid(
        optim_path=optim_path,
        model=model,
        data=datamodule.dataset.tensors,
        seed=SEED,
    )

    loss_log_2d = loss_grid.loss_values_log_2d
    steps = loss_grid.path_2d.tolist()

    animate_contour(
        param_steps=steps,
        loss_steps=loss_path,
        acc_steps=accu_path,
        loss_grid=loss_log_2d,
        coords=loss_grid.coords,
        true_optim_point=loss_grid.true_optim_point,
        true_optim_loss=loss_grid.loss_min,
        pcvariances=loss_grid.pcvariances,
        giffps=giffps,
        sampling=sampling,
        output_to_file=output_to_file,
        filename=output_filename,
    )
    if return_data:
        return list(optim_path), list(loss_path), list(accu_path)
