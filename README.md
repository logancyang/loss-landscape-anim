# Animating the Optimization Trajectory of Neural Nets

`loss-landscape-anim` lets you create animated optimization path in a 2D slice of the loss landscape of your neural networks. It is based on [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), please follow its suggested style if you want to add your own model.

Check out my article [Visualizing Optimization Trajectory of Neural Nets](https://towardsdatascience.com/from-animation-to-intuition-visualizing-optimization-trajectory-in-neural-nets-726e43a08d85?sk=dae85760fb921ecacddbe1af903e3c69) for more examples and some intuitive explanations.

## 1. Basic Examples

With the provided [spirals dataset](./sample_images/spirals-dataset.png) and the default multilayer perceptron `MLP` model, you can directly call `loss_landscape_anim` to get a sample animated GIF like this:

```py
# Use default MLP model and sample spirals dataset
loss_landscape_anim(n_epochs=300)
```

<img src="/sample_images/sample_mlp_2l_50n.gif" alt="sample gif 1" align="middle"/>

Here's another example – the LeNet5 convolutional network on the MNIST dataset. There are many levers you can tune: learning rate, batch size, epochs, frames per second of the GIF output, a seed for reproducible results, whether to load from a trained model, etc. Check out the function signature for more details.

```py
bs = 16
lr = 1e-3
datamodule = MNISTDataModule(batch_size=bs, n_examples=3000)
model = LeNet(learning_rate=lr)

# Optional return values if you need them
optim_path, loss_steps, accu_steps = loss_landscape_anim(
    n_epochs=10,
    model=model,
    datamodule=datamodule,
    optimizer="adam",
    giffps=15,
    seed=SEED,
    load_model=False,
    output_to_file=True
)
```

The output looks like this:

<img src="/sample_images/lenet-1e-3.gif" alt="sample gif 2" align="middle"/>

## 2. Why PCA?

The optimization path almost always fall into a low-dimensional space <sup>[[1]](#reference)</sup>. For visualizing the most movement, PCA is the best approach. However, it is not the best approach for all use cases. For instance, if you would like to compare the paths of different optimizers, PCA is not viable because its 2D slice depends on the path itself. The fact that different paths result in different 2D slices makes it impossible to plot them in the same space. In that case, 2 fixed directions are needed.

## 3. Custom Directions


## 4. Custom Dataset and Model

1. Prepare your `DataModule`. Refer to [datamodule.py](./loss_landscape_anim/datamodule.py) for examples.
2. Define your custom model that inherits `model.GenericModel`. Refer to [model.py](./loss_landscape_anim/model.py) for examples.
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
