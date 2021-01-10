import pickle

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from loss_landscape_anim.model import MLP, LossGrid
from loss_landscape_anim.plot import animate_contour


"""CLI Arguments"""
NUM_EPOCHS = 10
BATCH_SIZE = 32
LOAD_MODEL = True

# Optional seed
SEED = 180224
torch.manual_seed(SEED)


# TODO: Generic data loader
"""Load data"""
datadict = pickle.load(open("./sample_data/data_2d_3class.p", "rb"))
# Convert np array to tensor
X_train = torch.Tensor(datadict["X_train"])
y_train = torch.LongTensor(datadict["y_train"])
dataset = TensorDataset(X_train, y_train)

bs = 32
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
# dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
# train_loader = DataLoader(dataset)

INPUT_DIM = X_train.shape[1]  # X is 2-dimensional
NUM_CLASSES = 3

# TODO: Generic model definition, default to MLP
"""Define model"""

# TODO: Optional training, skip if load trained model
"""Train model"""
if not LOAD_MODEL:
    HIDDEN_DIM = 0
    NUM_HIDDEN_LAYERS = 0
    torch.manual_seed(SEED)

    model = MLP(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        optimizer="adam",
    )
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model, train_loader)
    torch.save(model, "./models/model.pt")

model = torch.load("./models/model.pt")

optim_path, loss_path, accu_path = zip(
    *[(path["flat_w"], path["loss"], path["accuracy"]) for path in model.optim_path]
)

print(f"Total points in optim_path: {len(optim_path)}")

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
    pcvariances=loss_grid.pcvariances,
)
