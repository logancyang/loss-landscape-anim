import pickle

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from loss_landscape_anim.model import MLP, LossGrid


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

optim_path, loss_path = zip(
    *[(path["flat_w"], path["loss"]) for path in model.optim_path]
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
coords_x, coords_y = loss_grid.coords
pcvariances = loss_grid.pcvariances

xc = list(range(0, 200))
yc = list(range(0, 200))
steps = loss_grid.path_2d.tolist()

fig, ax = plt.subplots(figsize=(6, 4))
cp = ax.contourf(coords_x, coords_y, loss_log_2d, levels=35, alpha=0.9, cmap="YlGnBu")
w1s = [step[0] for step in steps]
w2s = [step[1] for step in steps]
(pathline,) = ax.plot(w1s, w2s, color="r", lw=1)

ax.set_title("MLP")
ax.set_xlabel(f"principal component 0, {pcvariances[0]:.1%}")
ax.set_ylabel(f"principal component 1, {pcvariances[1]:.1%}")
plt.savefig("test.png")
