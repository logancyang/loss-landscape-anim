"""
Steps:
1. Load data
2. Create a model in torch
3. Record the parameters during training
4. Use PCA to project the parameters to 2D
5. Collect the values in 2D:
    a. A list of 2D values as the trajectory obtained by projecting the
       parameters down to the 2D space spanned by the top 2 PC.
    b. A grid of 2D values that capture (a) and some more for visual
       aesthetics.
"""
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.optim import SGD, Adam
from tqdm import tqdm


RES = 50
# Controls the margin from the optim starting point to the edge of the graph.
# The value is a multiplier on the distance between the optim start and end
MARGIN = 0.3


class GenericModel(pl.LightningModule):
    def __init__(self, optimizer, learning_rate, custom_optimizer=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.custom_optimizer = custom_optimizer
        self.optim_path = []
        self.accuracy = pl.metrics.Accuracy()

    def configure_optimizers(self):
        if self.custom_optimizer:
            return self.custom_optimizer(self.parameters(), self.learning_rate)
        elif self.optimizer == "adam":
            return Adam(self.parameters(), self.learning_rate)
        elif self.optimizer == "sgd":
            return SGD(self.parameters(), self.learning_rate)
        else:
            raise Exception(
                f"custom_optimizer supplied is not supported, "
                f"try torch.optim.Adam or torch.optim.SGD: "
                f"{self.custom_optimizer}"
            )

    def get_flat_params(self):
        """Get flattened and concatenated params of the model"""
        params = self._get_params()
        flat_params = torch.Tensor()
        for _, param in params.items():
            flat_params = torch.cat((flat_params, torch.flatten(param)))
        return flat_params

    def init_from_flat_params(self, flat_params):
        """Set all model parameters from flattened and concat version"""
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
        assert counter == len(
            flat_w
        ), "counter must reach the end of flat weight vector"
        return state_dict


class MLP(GenericModel):
    """
    Multilayer Perceptron with specified number of hidden layers
    with equal number of hidden dimensions in each layer
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        learning_rate,
        num_hidden_layers=1,
        optimizer="adam",
        custom_optimizer=None,
    ):
        super().__init__(optimizer, learning_rate, custom_optimizer)
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
        """
        Pytorch lightning recommends using forward for inference,
        not training
        """
        y_pred = self.layers(x_in)
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        # Get model weights flattened here to append to optim_path later
        flat_w = self.get_flat_params()
        loss = F.cross_entropy(y_pred, y)

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
        """Only record optimization path on epoch level"""
        self.optim_path.extend(training_step_outputs)


class DimReduction:
    def __init__(self, params_path, seed):
        """params_path: List of params from each optimization step"""
        self.matrix_to_reduce = self._transform(params_path)
        self.seed = seed

    def pca(self):
        pca = PCA(n_components=2, random_state=self.seed)
        optim_path = self.matrix_to_reduce.T
        n_steps, _ = optim_path.shape
        pca.fit(optim_path)
        path_2d = pca.transform(optim_path)
        reduced_dirs = pca.components_
        assert path_2d.shape == (n_steps, 2)
        return {
            "optim_path": optim_path,
            "path_2d": path_2d,
            "reduced_dirs": reduced_dirs,
            "pcvariances": pca.explained_variance_ratio_,
        }

    def _transform(self, model_params):
        npvectors = []
        for tensor in model_params:
            npvectors.append(np.array(tensor))
        return np.vstack(npvectors).T


class LossGrid:
    def __init__(
        self, optim_path, model, data, loss_fn, seed, res=RES, tqdm_disable=False
    ):
        dim_reduction = DimReduction(params_path=optim_path, seed=seed)
        reduced_dict = dim_reduction.pca()

        self.optim_point = reduced_dict["optim_path"][-1]
        self.optim_point_2d = reduced_dict["path_2d"][-1]
        self.path_2d = reduced_dict["path_2d"]
        self.dir0, self.dir1 = reduced_dict["reduced_dirs"]
        self.pcvariances = reduced_dict["pcvariances"]

        alpha = self._compute_stepsize(res)
        self.params_grid = self.build_params_grid(res, alpha)
        self.loss_values_2d, self.argmin, self.loss_min = self.compute_loss_2d(
            model, data, loss_fn, tqdm_disable=tqdm_disable
        )
        self.loss_values_log_2d = np.log(self.loss_values_2d)
        self.coords = self.convert_coords(res, alpha)
        # True optim in loss grid
        self.true_optim_point = self.indices_to_coords(self.argmin, res, alpha)

    def build_params_grid(self, res, alpha):
        """
        Produce the grid for the contour plot. Start from the optimal point,
        span directions of the pca result with stepsize alpha, resolution res.
        """
        grid = []
        for i in range(-res, res):
            row = []
            for j in range(-res, res):
                w_new = self.optim_point + i * alpha * self.dir0 + j * alpha * self.dir1
                row.append(w_new)
            grid.append(row)
        assert (grid[res][res] == self.optim_point).all()
        return grid

    def compute_loss_2d(self, model, data, loss_fn, tqdm_disable=False):
        """
        Compute loss values for each weight vector in grid for the model
        and data
        """
        X, y = data
        loss_2d = []
        n = len(self.params_grid)
        m = len(self.params_grid[0])
        loss_min = float("inf")
        argmin = ()
        print("Generating loss values for the contour plot...")
        with tqdm(total=n * m, disable=tqdm_disable) as pbar:
            for i in range(n):
                loss_row = []
                for j in range(m):
                    w_ij = torch.Tensor(self.params_grid[i][j])
                    # Load flattened weight vector into model
                    model.init_from_flat_params(w_ij)
                    y_pred = model(X)
                    loss_val = loss_fn(y_pred, y).item()
                    if loss_val < loss_min:
                        loss_min = loss_val
                        argmin = (i, j)
                    loss_row.append(loss_val)
                    pbar.update(1)
                loss_2d.append(loss_row)
        # This transpose below is very important for a correct contour plot because
        # originally in loss_2d, dir1 (y) is row-direction, dir0 (x) is column
        loss_2darray = np.array(loss_2d).T
        print("Loss values generated.")
        return loss_2darray, argmin, loss_min

    def _convert_coord(self, i, ref_point_coord, alpha):
        """
        Given a reference point coordinate (1D), find the value i steps away
        with step size alpha
        """
        return i * alpha + ref_point_coord

    def convert_coords(self, res, alpha):
        """
        Convert the coordinates from (i, j) indices to (x, y) values with
        unit vectors as the top 2 principal components.

        Original path_2d has PCA output, i.e. the 2D projections of each W step
        onto the 2D space spanned by the top 2 PCs.
        We need these steps in (i, j) terms with unit vectors
        reduced_w1 = (1, 0) and reduced_w2 = (0, 1) in the 2D space.

        We center the plot on optim_point_2d, i.e.
        let center_2d = optim_point_2d

        ```
        i = (x - optim_point_2d[0]) / alpha
        j = (y - optim_point_2d[1]) / alpha

        i.e.

        x = i * alpha + optim_point_2d[0]
        y = j * alpha + optim_point_2d[1]
        ```

        where (x, y) is the 2D points in path_2d from PCA. Again, the unit
        vectors are reduced_w1 and reduced_w2.
        Return the grid coordinates in terms of (x, y) for the loss values
        """
        converted_coord_xs = []
        converted_coord_ys = []
        for i in range(-res, res):
            x = self._convert_coord(i, self.optim_point_2d[0], alpha)
            y = self._convert_coord(i, self.optim_point_2d[1], alpha)
            converted_coord_xs.append(x)
            converted_coord_ys.append(y)
        return np.array(converted_coord_xs), np.array(converted_coord_ys)

    def indices_to_coords(self, indices, res, alpha):
        grid_i, grid_j = indices
        i, j = grid_i - res, grid_j - res
        x = i * alpha + self.optim_point_2d[0]
        y = j * alpha + self.optim_point_2d[1]
        return x, y

    def _compute_stepsize(self, res):
        dist_2d = self.path_2d[-1] - self.path_2d[0]
        dist = (dist_2d[0] ** 2 + dist_2d[1] ** 2) ** 0.5
        return dist * (1 + MARGIN) / res
