# Animating the Optimization Trajectory of Neural Nets

`loss-landscape-anim` lets you create animated optimization path in a 2D slice of the loss landscape of your neural networks.

Check out my article [Visualizing Optimization Trajectory of Neural Nets](https://towardsdatascience.com/from-animation-to-intuition-visualizing-optimization-trajectory-in-neural-nets-726e43a08d85?sk=dae85760fb921ecacddbe1af903e3c69) for more examples and some intuitive explanations.

## 1. Basic Example

With the provided dataset and the default multilayer perceptron `MLP` model, you can directly call `todo` to get a sample animated GIF like this.

```py
# TODO: Add example for python and cli using fastcore.script
```

## 2. Why PCA?

The optimization path almost always fall into a 1-2 dimensional space <sup>[[1]](#reference)</sup>. For visualizing the most movement, PCA is the best approach. However, it is not the best approach for all use cases. For instance, if you would like to compare the paths of different optimizors, PCA is not viable because its 2D slice depends on the path itself. The fact that different paths have different 2D slices makes it impossible to plot them in the same space. In that case, 2 fixed directions are needed.

## 3. Use Your Own Dimensions


## 4. Use Your Custom Dataset and Model

1. Prepare your `torch.Dataset`
2. Define your custom model that inherits `model.GenericModel`.
3. Call the function in Python or use the CLI as shown below

```py
# TODO: Add example for custom dataset and model
```

## 5. Implementation

This library is using PyTorch Lightning, please follow its suggested style if you want to add your own model. The command line interface is using `fastcore.script` which is a `fastai` library for simpler CLI development than popular alternatives such as `click` or `fire`.

## Reference

[1] [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913v3)
