import pathlib
import pickle

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from loss_landscape_anim.model import MLP, LossGrid
from loss_landscape_anim.plot import animate_contour


SEED = 180224


def loss_landscape_anim(
    learning_rate,
    dataset=None,
    custom_model=None,
    n_epochs=50,
    batch_size=None,
    optimizer="adam",
    model_path="./models/model.pt",
    load_model=False,
    output_to_file=True,
    output_filename="test.gif",
    giffps=15,
    sampling=False,
    max_frames=300,
    seed=None,
):
    if seed:
        torch.manual_seed(seed)

    # TODO: Generic data loader
    """Load data"""
    datadict = pickle.load(open("./sample_data/data_2d_3class.p", "rb"))
    # Convert np array to tensor
    X_train = torch.Tensor(datadict["X_train"])
    y_train = torch.LongTensor(datadict["y_train"])
    dataset = TensorDataset(X_train, y_train)

    if not batch_size:
        batch_size = len(X_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    # dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    # train_loader = DataLoader(dataset)

    INPUT_DIM = X_train.shape[1]  # X is 2-dimensional
    NUM_CLASSES = 3  # TODO: Get # classes from train_y

    # TODO: Generic model definition, default to MLP
    """Define model"""

    # TODO: Optional training, skip if load trained model
    """Train model"""
    if not load_model:
        HIDDEN_DIM = 0
        NUM_HIDDEN_LAYERS = 0

        model = MLP(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )
        trainer = pl.Trainer(max_epochs=n_epochs)
        trainer.fit(model, train_loader)
        torch.save(model, model_path)

    model_file = pathlib.Path(model_path)
    if not model_file.is_file():
        raise Exception("Model file not found!")

    model = torch.load(model_path)
    optim_path, loss_path, accu_path = zip(
        *[(path["flat_w"], path["loss"], path["accuracy"]) for path in model.optim_path]
    )

    print(f"Total steps in optimization path: {len(optim_path)}")

    """PCA and Loss Grid"""

    loss_grid = LossGrid(
        optim_path=optim_path,
        model=model,
        data=dataset.tensors,
        loss_fn=F.cross_entropy,
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
        max_frames=max_frames,
        sampling=sampling,
        output_to_file=output_to_file,
        filename=output_filename,
    )

    return list(optim_path), list(loss_path), list(accu_path)


if __name__ == "__main__":
    optim_path, loss_path, accu_path = loss_landscape_anim(
        learning_rate=1e-3,
        optimizer="adam",
        n_epochs=100,
        batch_size=64,
        giffps=15,
        seed=None,
        load_model=False,
        output_to_file=True,
        sampling=True,
    )
