import numpy as np
import torch
import pickle
from loss_landscape_anim import (
    loss_landscape_anim,
    MNISTDataModule,
    LeNet,
    compare_optimizers,
)
from loss_landscape_anim.compare_optimizers import inspect_model


if __name__ == "__main__":
    optimizers = ["adam", "sgd", "adagrad", "rmsprop"]
    compare_optimizers(
        optimizers,
        learning_rate=1e-2,
        train_new=True,
        seed=180224,
    )

    # model = torch.load("checkpoints/model_adam_0.pt")
    # optim_path = [path["flat_w"] for path in model.optim_path]
    # weight_init = optim_path[0]
    # pickle.dump(weight_init, open("checkpoints/weight.p", "wb"))

    # loss_landscape_anim(n_epochs=300)

    """
    u_gen = np.random.normal(size=61706)
    u = u_gen / np.linalg.norm(u_gen)
    v_gen = np.random.normal(size=61706)
    v = v_gen / np.linalg.norm(v_gen)

    bs = 16
    lr = 1e-3
    datamodule = MNISTDataModule(batch_size=bs, n_examples=3000)
    model = LeNet(learning_rate=lr)

    loss_landscape_anim(
        n_epochs=10,
        model=model,
        datamodule=datamodule,
        optimizer="adam",
        reduction_method="custom",
        custom_directions=(u, v),
        giffps=15,
        seed=180224,
        load_model=False,
        output_to_file=True,
        gpus=0,  # Set to # gpus if available
    )
    """
