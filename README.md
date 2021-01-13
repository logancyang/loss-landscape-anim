# Animating the Optimization Trajectory of Neural Nets

`loss-landscape-anim` lets you create animated optimization path in a 2D slice of the loss landscape of your neural networks. It is based on [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), please follow its suggested style if you want to add your own model.

Check out my article [Visualizing Optimization Trajectory of Neural Nets](https://towardsdatascience.com/from-animation-to-intuition-visualizing-optimization-trajectory-in-neural-nets-726e43a08d85?sk=dae85760fb921ecacddbe1af903e3c69) for more examples and some intuitive explanations.

## 1. Basic Example

With the provided dataset and the default multilayer perceptron `MLP` model, you can directly call `loss_landscape_anim` to get a sample animated GIF like this.

<img src="/sample_images/sample_mlp_2l_50n.gif" alt="sample gif" align="middle"/>

```py
# TODO: Add example for python and cli using fastcore.script
```

## 2. Why PCA?

The optimization path almost always fall into a low-dimensional space <sup>[[1]](#reference)</sup>. For visualizing the most movement, PCA is the best approach. However, it is not the best approach for all use cases. For instance, if you would like to compare the paths of different optimizers, PCA is not viable because its 2D slice depends on the path itself. The fact that different paths result in different 2D slices makes it impossible to plot them in the same space. In that case, 2 fixed directions are needed.

## 3. Custom Directions


## 4. Custom Dataset and Model

1. Prepare your `DataModule`
2. Define your custom model that inherits `model.GenericModel`.
3. Call the function as shown below

```py
# TODO: Add example for custom dataset and model
```

## Reference

[1] [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913v3)
