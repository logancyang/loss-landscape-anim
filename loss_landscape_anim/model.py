"""GenericModel class and its child classes.

The GenericModel class enables flattening of the model parameters for tracking.
MLP and LeNet are example models. Add your own PyTorch model by inheriting
from GenericModel and organizing it into the pytorch lightning style.
"""
# pylint: disable = no-member
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam, Adagrad, RMSprop


class GenericModel(pl.LightningModule):
    """GenericModel class that enables flattening of the model parameters."""

    def __init__(self, optimizer, learning_rate, custom_optimizer=None, gpus=0):
        """Init a new GenericModel.

        Args:
            optimizer: optimizer to use, such as "adam", "sgd", etc.
            learning_rate: learning rate to use.
            custom_optimizer (optional): Custom optimizer object. Defaults to None.
            gpus (optional): GPUs to use for training. Defaults to 0.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.custom_optimizer = custom_optimizer
        self.gpus = gpus
        self.optim_path = []
        self.accuracy = pl.metrics.Accuracy()

    def configure_optimizers(self):
        """Configure the optimizer for Pytorch Lightning.

        Raises:
            Exception: Optimizer not recognized.
        """
        if self.custom_optimizer:
            return self.custom_optimizer(self.parameters(), self.learning_rate)
        elif self.optimizer == "adam":
            return Adam(self.parameters(), self.learning_rate)
        elif self.optimizer == "sgd":
            return SGD(self.parameters(), self.learning_rate)
        elif self.optimizer == "adagrad":
            return Adagrad(self.parameters(), self.learning_rate)
        elif self.optimizer == "rmsprop":
            return RMSprop(self.parameters(), self.learning_rate)
        else:
            raise Exception(
                f"custom_optimizer supplied is not supported: {self.custom_optimizer}"
            )

    def get_flat_params(self):
        """Get flattened and concatenated params of the model."""
        params = self._get_params()
        flat_params = torch.Tensor()
        if torch.cuda.is_available() and self.gpus > 0:
            flat_params = flat_params.cuda()
        for _, param in params.items():
            flat_params = torch.cat((flat_params, torch.flatten(param)))
        return flat_params

    def init_from_flat_params(self, flat_params):
        """Set all model parameters from the flattened form."""
        if not isinstance(flat_params, torch.Tensor):
            raise AttributeError(
                "Argument to init_from_flat_params() must be torch.Tensor"
            )
        shapes = self._get_param_shapes()
        state_dict = self._unflatten_to_state_dict(flat_params, shapes)
        self.load_state_dict(state_dict, strict=True)

    def _get_param_shapes(self):
        shapes = []
        for name, param in self.named_parameters():
            shapes.append((name, param.shape, param.numel()))
        return shapes

    def _get_params(self):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.data
        return params

    def _unflatten_to_state_dict(self, flat_w, shapes):
        state_dict = {}
        counter = 0
        for shape in shapes:
            name, tsize, tnum = shape
            param = flat_w[counter : counter + tnum].reshape(tsize)
            state_dict[name] = torch.nn.Parameter(param)
            counter += tnum
        assert counter == len(flat_w), "counter must reach the end of weight vector"
        return state_dict


class MLP(GenericModel):
    """A Multilayer Perceptron model.

    Default is 1 hidden layer with 50 neurons.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        learning_rate,
        num_hidden_layers=1,
        hidden_dim=50,
        optimizer="adam",
        custom_optimizer=None,
        gpus=0,
    ):
        """Init an MLP model.

        Args:
            input_dim: Number of input dimensions.
            num_classes: Number of classes or output dimensions.
            learning_rate: The learning rate to use.
            num_hidden_layers (optional): Number of hidden layers. Defaults to 1.
            hidden_dim (optional): Number of neurons in each layer. Defaults to 50.
            optimizer (optional): The optimizer to use. Defaults to "adam".
            custom_optimizer (optional): The custom optimizer to use. Defaults to None.
            gpus (optional): GPUs to use if available. Defaults to 0.
        """
        super().__init__(
            optimizer=optimizer,
            learning_rate=learning_rate,
            custom_optimizer=custom_optimizer,
            gpus=gpus,
        )
        # NOTE: nn.ModuleList is not the same as Sequential,
        # the former doesn't have forward implemented
        if num_hidden_layers == 0:
            self.layers = nn.Linear(input_dim, num_classes)
        else:
            self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            n_layers = 2
            for _ in range(num_hidden_layers - 1):
                self.layers.add_module(
                    name=f"{n_layers}", module=nn.Linear(hidden_dim, hidden_dim)
                )
                self.layers.add_module(name=f"{n_layers+1}", module=nn.ReLU())
                n_layers += 2

            self.layers.add_module(
                name=f"{n_layers}", module=nn.Linear(hidden_dim, num_classes)
            )

    def forward(self, x_in, apply_softmax=False):
        """Forward pass."""
        # Pytorch lightning recommends using forward for inference, not training
        y_pred = self.layers(x_in)
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred

    def loss_fn(self, y_pred, y):
        """Loss function."""
        return F.cross_entropy(y_pred, y)

    def training_step(self, batch, batch_idx):
        """Training step for a batch of data.

        The model computes the loss and save it along with the flattened model params.
        """
        X, y = batch
        y_pred = self(X)
        # Get model weights flattened here to append to optim_path later
        flat_w = self.get_flat_params()
        loss = self.loss_fn(y_pred, y)

        preds = y_pred.max(dim=1)[1]  # class
        accuracy = self.accuracy(preds, y)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "accuracy": accuracy, "flat_w": flat_w}

    def training_epoch_end(self, training_step_outputs):
        """Only save the last step in each epoch.

        Args:
            training_step_outputs: all the steps in this epoch.
        """
        # Only record the last step in each epoch
        self.optim_path.append(training_step_outputs[-1])


class LeNet(GenericModel):
    """LeNet-5 convolutional neural network."""

    def __init__(self, learning_rate, optimizer="adam", custom_optimizer=None, gpus=0):
        """Init a LeNet model.

        Args:
            learning_rate: learning rate to use.
            optimizer (optional): optimizer to use. Defaults to "adam".
            custom_optimizer (optional): custom optimizer to use. Defaults to None.
            gpus (optional): Number of GPUs for training if available. Defaults to 0.
        """
        super().__init__(optimizer, learning_rate, custom_optimizer, gpus=gpus)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))  # (n_examples, 120, 1, 1) -> (n_examples, 120)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def loss_fn(self, y_pred, y):
        """Loss function."""
        return F.cross_entropy(y_pred, y)

    def training_step(self, batch, batch_idx):
        """Training step for a batch of data.

        The model computes the loss and save it along with the flattened model params.
        """
        X, y = batch
        y_pred = self(X)
        # Get model weights flattened here to append to optim_path later
        flat_w = self.get_flat_params()
        loss = self.loss_fn(y_pred, y)

        preds = y_pred.max(dim=1)[1]  # class
        accuracy = self.accuracy(preds, y)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "accuracy": accuracy, "flat_w": flat_w}

    def training_epoch_end(self, training_step_outputs):
        """Only save the last step in each epoch.

        Args:
            training_step_outputs: all the steps in this epoch.
        """
        self.optim_path.extend(training_step_outputs)
