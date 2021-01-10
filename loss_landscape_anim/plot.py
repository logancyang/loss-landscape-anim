import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_multiclass_decision_boundary(model, data, ax=None):
    X, y = data
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

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
    print(f"frames: {len(steps)}")
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
            0.05, 0.9, "", fontsize=10, ha="left", va="center", transform=ax.transAxes
        )
        value_text = ax.text(
            0.05, 0.8, "", fontsize=10, ha="left", va="center", transform=ax.transAxes
        )
        step_text.set_text(f"epoch: {i}")
        value_text.set_text(f"loss: {loss_steps[i]: .2f}\nacc: {acc_steps[i]: .2f}")

    # call the animator.  blit=True means only
    # re-draw the parts that have changed.
    # NOTE this anim must be global to work
    global anim
    anim = FuncAnimation(fig, animate, frames=len(steps), interval=200, blit=False)
    plt.ioff()

    # Write to gif
    if write2gif:
        anim.save(f"./{file}.gif", writer="imagemagick", fps=giffps)

    plt.show()


def static_contour(steps, loss_grid, coords, pcvariances, filename="test.png"):
    _, ax = plt.subplots(figsize=(6, 4))
    coords_x, coords_y = coords
    ax.contourf(coords_x, coords_y, loss_grid, levels=35, alpha=0.9, cmap="YlGnBu")
    w1s = [step[0] for step in steps]
    w2s = [step[1] for step in steps]
    (pathline,) = ax.plot(w1s, w2s, color="r", lw=1)

    ax.set_title("MLP")
    ax.set_xlabel(f"principal component 0, {pcvariances[0]:.1%}")
    ax.set_ylabel(f"principal component 1, {pcvariances[1]:.1%}")
    plt.savefig(filename)
    print(f"{filename} created.")


def animate_contour(
    param_steps,
    loss_steps,
    acc_steps,
    loss_grid,
    coords,
    pcvariances,
    giffps=30,
    figsize=(9, 6),
    filename="test.gif",
):
    print(f"\nTotal frames to process: {len(param_steps)}")

    fig, ax = plt.subplots(figsize=figsize)
    coords_x, coords_y = coords
    ax.contourf(coords_x, coords_y, loss_grid, levels=35, alpha=0.9, cmap="YlGnBu")

    ax.set_title("Loss Landscape")
    ax.set_xlabel(f"principal component 0, {pcvariances[0]:.1%}")
    ax.set_ylabel(f"principal component 1, {pcvariances[1]:.1%}")

    W0 = param_steps[0]
    w1s = [W0[0]]
    w2s = [W0[1]]
    (pathline,) = ax.plot(w1s, w2s, color="r", lw=1)
    (point,) = ax.plot(W0[0], W0[1], "ro")

    step_text = ax.text(
        0.05, 0.9, "", fontsize=10, ha="left", va="center", transform=ax.transAxes
    )
    value_text = ax.text(
        0.05, 0.8, "", fontsize=10, ha="left", va="center", transform=ax.transAxes
    )

    def animate(i):
        W = param_steps[i]
        w1s.append(W[0])
        w2s.append(W[1])
        pathline.set_data(w1s, w2s)
        point.set_data(W[0], W[1])
        step_text.set_text(f"epoch: {i}")
        value_text.set_text(f"loss: {loss_steps[i]: .2f}\nacc: {acc_steps[i]: .2f}")

    # Call the animator. blit=True means only re-draw the parts that have changed.
    # NOTE: anim must be global for the the animation to work
    global anim
    anim = FuncAnimation(
        fig,
        animate,
        frames=len(param_steps),
        interval=100,
        blit=False,
    )
    plt.ioff()
    print(f"Writing {filename}.")
    anim.save(
        f"./{filename}",
        writer="imagemagick",
        fps=giffps,
        progress_callback=lambda i, n: print(
            "\r"
            + f"Processing frame {i+1}/{n}... Once processing is done, the conversion "
            "will take a while...",
            end="",
        ),
    )
    print(f"\n{filename} created successfully.")
