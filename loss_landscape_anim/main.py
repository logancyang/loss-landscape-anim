import pickle

import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader

from loss_landscape_anim.model import MLP


"""CLI Arguments"""
NUM_EPOCHS = 50
BATCH_SIZE = 32
LOAD_MODEL = False

# Optional seed
SEED = 180224
torch.manual_seed(SEED)


# TODO: Generic data loader
"""Load data"""
datadict = pickle.load(open("./sample_data/data_2d_3class.p", "rb"))
# Convert np array to tensor
X_train = torch.Tensor(datadict['X_train'])
y_train = torch.LongTensor(datadict['y_train'])
dataset = TensorDataset(X_train, y_train)

bs = 32
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
# dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
# train_loader = DataLoader(dataset)

INPUT_DIM = X_train.shape[1]  # X is 2-dimensional
NUM_CLASSES = 3

# TODO: Generic model definition, default to MLP
"""Define model"""
HIDDEN_DIM = 50
NUM_HIDDEN_LAYERS = 1
torch.manual_seed(SEED)

model = MLP(
    input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
    num_hidden_layers=NUM_HIDDEN_LAYERS, optimizer='adam'
)
print(model.named_parameters)

trainer = pl.Trainer(max_epochs=NUM_EPOCHS)

# TODO: Optional training, skip if load trained model
"""Train model"""
trainer.fit(model, train_loader)
torch.save(model, './models/model.pt')
model = torch.load('./models/model.pt')

optim_path, loss_path = zip(*[
    (path['flat_w'], path['loss']) for path in model.optim_path
])

print(f"Total points in optim_path: {len(optim_path)}")

"""PCA"""
