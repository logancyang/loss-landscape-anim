# Animating the Optimization Trajectory of Neural Nets

`loss-landscape-anim` lets you create animated optimization path in a 2D slice of the loss landscape of your neural networks.

Check out my article [Visualizing Optimization Trajectory of Neural Nets](https://towardsdatascience.com/from-animation-to-intuition-visualizing-optimization-trajectory-in-neural-nets-726e43a08d85?sk=dae85760fb921ecacddbe1af903e3c69) for more examples and some intuitive explanations.

## Basic Example

With the provided dataset and the default multilayer perceptron `MLP` model, you can directly call `todo` to get a sample animated GIF like this.

```py
# TODO: Add example for python and cli using fastcore.script
```

## Use Your Custom Dataset and Model

1. Prepare your `torch.Dataset`
2. Define your custom model that inherits `model.GenericModel`.
3. Call the function in Python or use the CLI as shown below

```py
# TODO: Add example for custom dataset and model
```

## Implementation

This library is based on PyTorch Lightning, matplotlib and used `fastcore.script` to build the command line interface.
