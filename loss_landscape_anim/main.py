import pathlib
import warnings

import pytorch_lightning as pl
import torch

from loss_landscape_anim.datamodule import MNISTDataModule, SampleDataModule
from loss_landscape_anim.model import MLP, LossGrid
from loss_landscape_anim.plot import animate_contour, sample_frames

warnings.filterwarnings("ignore")

SEED = 180224


def loss_landscape_anim(
    learning_rate,
    datamodule=None,
    custom_model=None,
    n_epochs=50,
    batch_size=None,
    optimizer="adam",
    model_path="./models/model.pt",
    load_model=False,
    output_to_file=True,
    output_filename="sample.gif",
    giffps=15,
    sampling=False,
    n_frames=300,
    seed=None,
):
    if seed:
        torch.manual_seed(seed)

    if not datamodule:
        # datamodule = SampleDataModule()
        datamodule = MNISTDataModule(n_examples=2000)

    train_loader = datamodule.train_dataloader(batch_size=batch_size)

    # Train model
    if not load_model:
        n_hidden_dim = 100
        n_hidden_layers = 2

        model = MLP(
            input_dim=datamodule.input_dim,
            hidden_dim=n_hidden_dim,
            num_classes=datamodule.n_classes,
            num_hidden_layers=n_hidden_layers,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )
        trainer = pl.Trainer(max_epochs=n_epochs)
        print(f"Training for {n_epochs} epochs...")
        trainer.fit(model, train_loader)
        torch.save(model, model_path)

    model_file = pathlib.Path(model_path)
    if not model_file.is_file():
        raise Exception("Model file not found!")

    model = torch.load(model_path)
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

    return list(optim_path), list(loss_path), list(accu_path)


if __name__ == "__main__":
    optim_path, loss_path, accu_path = loss_landscape_anim(
        learning_rate=1e-2,
        batch_size=1,
        optimizer="adam",
        n_epochs=10,
        giffps=15,
        seed=SEED,
        load_model=False,
        output_to_file=True,
    )
