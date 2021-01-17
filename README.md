# Animating the Optimization Trajectory of Neural Nets

`loss-landscape-anim` lets you create animated optimization path in a 2D slice of the loss landscape of your neural networks. It is based on [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), please follow its suggested style if you want to add your own model.

Check out my article [Visualizing Optimization Trajectory of Neural Nets](https://towardsdatascience.com/from-animation-to-intuition-visualizing-optimization-trajectory-in-neural-nets-726e43a08d85?sk=dae85760fb921ecacddbe1af903e3c69) for more examples and some intuitive explanations.

## 0. Installation

From PyPI:

```sh
pip install loss-landscape-anim
```

From source, you need [Poetry](https://python-poetry.org/docs/#installation). Once you cloned this repo, run the command below to install the dependencies.

```sh
poetry install
```

## 1. Basic Examples

With the provided [spirals dataset](https://github.com/logancyang/loss-landscape-anim/blob/master/sample_images/spirals-dataset.png) and the default multilayer perceptron `MLP` model, you can directly call `loss_landscape_anim` to get a sample animated GIF like this:

```py
# Use default MLP model and sample spirals dataset
loss_landscape_anim(n_epochs=300)
```

<img src="./sample_images/sample_mlp_2l_50n.gif" alt="sample gif 1" title="MLP with two 50-node hidden layers on the Spirals dataset, PCA" align="middle"/>

Note: if you are using it in a notebook, don't forget to include the following at the top:

```py
%matplotlib notebook
```

Here's another example – the LeNet5 convolutional network on the MNIST dataset. There are many levers you can tune: learning rate, batch size, epochs, frames per second of the GIF output, a seed for reproducible results, whether to load from a trained model, etc. Check out the function signature for more details.

```py
bs = 16
lr = 1e-3
datamodule = MNISTDataModule(batch_size=bs, n_examples=3000)
model = LeNet(learning_rate=lr)

optim_path, loss_steps, accu_steps = loss_landscape_anim(
    n_epochs=10,
    model=model,
    datamodule=datamodule,
    optimizer="adam",
    giffps=15,
    seed=SEED,
    load_model=False,
    output_to_file=True,
    return_data=True,  # Optional return values if you need them
    gpus=1  # Enable GPU training if available
)
```

GPU training is supported. Just pass `gpus` into `loss_landscape_anim` if they are available.

The output of LeNet5 on the MNIST dataset looks like this:

<img src="./sample_images/lenet-1e-3.gif" alt="sample gif 2" title="LeNet5 on the MNIST dataset, PCA" align="middle"/>

## 2. Why PCA?

The optimization path almost always fall into a low-dimensional space <sup>[[1]](#reference)</sup>. For visualizing the most movement, PCA is the best approach. However, it is not the best approach for all use cases. For instance, if you would like to compare the paths of different optimizers, PCA is not viable because its 2D slice depends on the path itself. The fact that different paths result in different 2D slices makes it impossible to plot them in the same space. In that case, 2 fixed directions are needed.

## 3. Random and Custom Directions

You can also set 2 fixed directions, either generated at random or handpicked.

For 2 random directions, set `reduction_method` to `"random"`, e.g.

```py
loss_landscape_anim(n_epochs=300, load_model=False, reduction_method="random")
```

For 2 fixed directions of your choosing, set `reduction_method` to `"custom"`, e.g.

```py
import numpy as np

n_params = ... # number of parameters your model has
u_gen = np.random.normal(size=n_params)
u = u_gen / np.linalg.norm(u_gen)
v_gen = np.random.normal(size=n_params)
v = v_gen / np.linalg.norm(v_gen)

loss_landscape_anim(
    n_epochs=300, load_model=False, reduction_method="custom", custom_directions=(u, v)
)
```

Here is an sample GIF produced by two random directions:

<img src="./sample_images/random_directions.gif" alt="sample gif 3" title="MLP with 1 50-node hidden layer on the Spirals dataset, random directions" align="middle"/>

By default, `reduction_method="pca"`.

## 4. Custom Dataset and Model

1. Prepare your `DataModule`. Refer to [datamodule.py](https://github.com/logancyang/loss-landscape-anim/blob/master/loss_landscape_anim/datamodule.py) for examples.
2. Define your custom model that inherits `model.GenericModel`. Refer to [model.py](https://github.com/logancyang/loss-landscape-anim/blob/master/loss_landscape_anim/model.py) for examples.
3. Once you correctly setup your custom `DataModule` and `model`, call the function as shown below to train the model and plot the loss landscape animation.

```py
bs = ...
lr = ...
datamodule = YourDataModule(batch_size=bs)
model = YourModel(learning_rate=lr)

loss_landscape_anim(
    n_epochs=10,
    model=model,
    datamodule=datamodule,
    optimizer="adam",
    seed=SEED,
    load_model=False,
    output_to_file=True
)
```

## Reference

[1] [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913v3)
