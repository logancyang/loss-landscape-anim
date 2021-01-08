import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_multiclass_decision_boundary(model, X, y, ax=None):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101)
    )

    X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    y_pred = model(X_test, apply_softmax=True)
    _, y_pred = y_pred.max(dim=1)
    y_pred = y_pred.reshape(xx.shape)
    if not ax:
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
    else:
        ax.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        ax.xlim(xx.min(), xx.max())
        ax.ylim(yy.min(), yy.max())


def animate_decision_area(
    model, X_train, y_train, steps, giffps, write2gif=False, file="nn_decision"
):
    # (W, loss, acc) in steps
    print(f'frames: {len(steps)}')
    weight_steps = [step[0] for step in steps]
    loss_steps = [step[1] for step in steps]
    acc_steps = [step[2] for step in steps]

    fig, ax = plt.subplots(figsize=(9, 6))
    W = weight_steps[0]
    model.init_from_flat_params(W)
    plot_multiclass_decision_boundary(model, X_train, y_train)

    ax.set_title("DECISION BOUNDARIES")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # animation function. This is called sequentially
    def animate(i):
        W = weight_steps[i]
        model.init_from_flat_params(W)
        ax.clear()
        # This line is key!!
        ax.collections = []

        X = X_train
        y = y_train
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101)
        )

        X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
        y_pred = model(X_test, apply_softmax=True)
        _, y_pred = y_pred.max(dim=1)
        y_pred = y_pred.reshape(xx.shape)
        ax.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)

        ax.set_title("DECISION BOUNDARIES")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        step_text = ax.text(
            0.05, 0.9, '', fontsize=10,
            ha='left', va='center', transform=ax.transAxes
        )
        value_text = ax.text(
            0.05, 0.8, '', fontsize=10,
            ha='left', va='center', transform=ax.transAxes
        )
        step_text.set_text(f"epoch: {i}")
        value_text.set_text(
            f"loss: {loss_steps[i]: .2f}\nacc: {acc_steps[i]: .2f}"
        )

    # call the animator.  blit=True means only
    # re-draw the parts that have changed.
    # NOTE this anim must be global to work
    global anim
    anim = FuncAnimation(
        fig, animate, frames=len(steps), interval=200, blit=False
    )
    plt.ioff()

    # Write to gif
    if write2gif:
        anim.save(f"./{file}.gif", writer='imagemagick', fps=giffps)

    plt.show()
