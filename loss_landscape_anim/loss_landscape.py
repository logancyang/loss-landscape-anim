import pathlib
import pickle

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

RES = 50
# Controls the margin from the optim starting point to the edge of the graph.
# The value is a multiplier on the distance between the optim start and end
MARGIN = 0.3


class DimReduction:
    def __init__(self, params_path, reduction_method, custom_directions, seed):
        """params_path: List of params from each optimization step"""
        self.optim_path_matrix = self._transform(params_path)
        self.n_steps, self.n_dim = self.optim_path_matrix.shape
        self.custom_directions = custom_directions
        self.reduction_method = reduction_method
        self.seed = seed

    def reduce(self):
        if self.reduction_method == "pca":
            return self.pca()
        elif self.reduction_method == "random":
            return self.reduce_to_random_directions()
        elif self.reduction_method == "custom":
            return self.reduce_to_custom_directions()
        else:
            raise Exception(f"Unrecognized reduction method {self.reduction_method}")

    def pca(self):
        pca = PCA(n_components=2, random_state=self.seed)
        path_2d = pca.fit_transform(self.optim_path_matrix)
        reduced_dirs = pca.components_
        assert path_2d.shape == (self.n_steps, 2)
        return {
            "optim_path": self.optim_path_matrix,
            "path_2d": path_2d,
            "reduced_dirs": reduced_dirs,
            "pcvariances": pca.explained_variance_ratio_,
        }

    def reduce_to_random_directions(self):
        """
        Produce 2 random flat unit vectors of dim <dim_params> as the directions.
        Since 2 high-dimensional vectors are almost always orthogonal,
        it's no problem to use them as the axes for the 2D slice of
        loss landscape.
        """
        print("Generating random axes...")
        # Generate 2 random unit vectors (u, v)
        if self.seed:
            print(f"seed={self.seed}")
            np.random.seed(self.seed)
        u_gen = np.random.normal(size=self.n_dim)
        u = u_gen / np.linalg.norm(u_gen)
        v_gen = np.random.normal(size=self.n_dim)
        v = v_gen / np.linalg.norm(v_gen)
        return self._project(np.array([u, v]))

    def reduce_to_custom_directions(self):
        """
        Manually pick two direction vectors dir0, dir1 of dim <dim_params>
        and use them as the axes for the 2D slice of loss landscape.
        """
        print("Using custom axes...")
        dir0, dir1 = self.custom_directions
        dir0_exists = dir0 is not None
        dir1_exists = dir1 is not None
        if not (dir0_exists and dir1_exists):
            raise Exception(
                "Custom directions not provided, please provide 2 vectors of "
                f"dim={self.n_dim}"
            )
        # Normalize given direction vectors
        u = dir0 / np.linalg.norm(dir0)
        v = dir1 / np.linalg.norm(dir1)
        # Transform all step params into the coordinates of (u, v)
        return self._project(np.array([u, v]))

    def _project(self, reduced_dirs):
        """Project self.optim_path_matrix onto (u, v)"""
        path_projection = self.optim_path_matrix.dot(reduced_dirs.T)
        assert path_projection.shape == (self.n_steps, 2)
        return {
            "optim_path": self.optim_path_matrix,
            "path_2d": path_projection,
            "reduced_dirs": reduced_dirs,
        }

    def _transform(self, model_params):
        npvectors = []
        for tensor in model_params:
            npvectors.append(np.array(tensor.cpu()))
        return np.vstack(npvectors)


class LossGrid:
    def __init__(
        self,
        optim_path,
        model,
        data,
        path_2d,
        directions,
        res=RES,
        tqdm_disable=False,
        save_grid=True,
        load_grid=False,
        filepath="./checkpoints/lossgrid.p",
    ):
        self.dir0, self.dir1 = directions
        self.path_2d = path_2d
        self.optim_point = optim_path[-1]
        self.optim_point_2d = path_2d[-1]

        alpha = self._compute_stepsize(res)
        self.params_grid = self.build_params_grid(res, alpha)

        if load_grid:
            self.loss_values_2d, self.argmin, self.loss_min = pickle.load(
                open(filepath, "rb")
            )
            print("\nLoss grid loaded from disk.")
        else:
            self.loss_values_2d, self.argmin, self.loss_min = self.compute_loss_2d(
                model, data, tqdm_disable=tqdm_disable
            )

        if save_grid:
            loss_2d_tup = (self.loss_values_2d, self.argmin, self.loss_min)
            pickle.dump(loss_2d_tup, open(filepath, "wb"))
            print(f"\nLoss grid saved at {pathlib.Path(filepath).absolute()}.")

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
                w_new = self.optim_point.cpu() + i * alpha * self.dir0 + j * alpha * self.dir1
                row.append(w_new)
            grid.append(row)
        assert (grid[res][res] == self.optim_point.cpu()).all()
        return grid

    def compute_loss_2d(self, model, data, tqdm_disable=False):
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
                    w_ij = torch.Tensor(self.params_grid[i][j].float())
                    # Load flattened weight vector into model
                    model.init_from_flat_params(w_ij)
                    y_pred = model(X)
                    loss_val = model.loss_fn(y_pred, y).item()
                    if loss_val < loss_min:
                        loss_min = loss_val
                        argmin = (i, j)
                    loss_row.append(loss_val)
                    pbar.update(1)
                loss_2d.append(loss_row)
        # This transpose below is very important for a correct contour plot because
        # originally in loss_2d, dir1 (y) is row-direction, dir0 (x) is column
        loss_2darray = np.array(loss_2d).T
        print("\nLoss values generated.")
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
