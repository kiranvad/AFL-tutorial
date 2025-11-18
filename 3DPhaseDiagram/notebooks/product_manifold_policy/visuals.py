import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize as MPLNormalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 kept for side effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import numpy as np
import torch 
from modules import PosteriorModel
import ast, json

scatter_kwargs = {
    "s": 40,
    "edgecolors": "k",
    "linewidths": 0.6,
    "alpha": 0.8,
    "zorder": 4,
}

def make_gridspec(bounds):
    fig = plt.figure(layout="constrained", figsize=(4*2, 4*2))

    # Outer structure: 2 rows (top, bottom), 1 column
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1,1])

    # Top row: 1×3
    gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0])
    ax1 = fig.add_subplot(gs_top[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs_top[0, 1], projection="3d")
    ax3 = fig.add_subplot(gs_top[0, 2], projection="3d")
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(bounds[:,0].numpy())
        ax.set_ylim(bounds[:,1].numpy())
        ax.set_zlim(bounds[:,2].numpy())

    # Bottom row: 1×2 spanning full width
    gs_bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1])
    ax4 = fig.add_subplot(gs_bottom[0, 0])
    ax4.set_xlim(bounds[:,0].numpy())
    ax4.set_ylim(bounds[:,1].numpy())
    ax5 = fig.add_subplot(gs_bottom[0, 1])

    return fig, [[ax1, ax2, ax3], [ax4, ax5]]

def plot_model_posterior(model, grid_xyz, ground_truth_labels, ax=None, return_stats=False):
    posterior = PosteriorModel(grid_xyz, model)
    pred_samples = posterior.sample(torch.Size((256,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
    entropy = torch.special.entr(probabilities).sum(dim=0)
    labels = posterior.loc.max(0)[1]
    accuracy = (labels==ground_truth_labels).sum()/len(ground_truth_labels)

    if ax is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    else:
        ax1 = ax[0]
        ax2 = ax[1]
        fig = plt.gcf()
    for l in labels.unique():
        flags = labels==l
        ax1.scatter(
            grid_xyz[flags, 0], grid_xyz[flags, 1], grid_xyz[flags, 2],
            s=40,
            depthshade=False,
            edgecolors="k",
            linewidths=0.6,
            alpha=0.2,
            zorder=4,
        )
    ax1.set_title(f"Accuracy {accuracy:.2f}")

    img = ax2.scatter(
        grid_xyz[:, 0], grid_xyz[:, 1], grid_xyz[:, 2],
        c=entropy,
        s=40,
        depthshade=False,
        edgecolors="k",
        linewidths=0.6,
        cmap="magma"
    )

    cb = ax2.figure.colorbar(img, use_gridspec=True, shrink=0.3, pad=0.2)
    cb.set_label(r"$\mathbb{H}(x)$")

    if return_stats:
        return labels, fig, [ax1, ax2], {"entropy": entropy, "accuracy": accuracy}
    return labels, fig, [ax1, ax2]

def plot_xy_optimization(acqf, grid_xy, xy_star, trajectory, ax=None, return_scores=False):
    score_xy = np.asarray([acqf(xy).item() for xy in grid_xy])
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    img = ax.tricontourf(
        grid_xy[:,0], 
        grid_xy[:,1], 
        score_xy, 
        cmap="magma"
    )
    ax.plot(
        trajectory[:,0], 
        trajectory[:,1], 
        "o-", 
        color="w", 
        lw=1.0, 
        markersize=2
    )
    ax.scatter(
        xy_star[:,0], 
        xy_star[:,1], 
        marker="^", 
        color="w", 
        s=50
    )
    fig.colorbar(img, ax=ax, label=r"$\pi(c)$")

    if return_scores:
        return fig, ax, score_xy
    return fig, ax

def plot_function_optimization(manifold, temperature, trajectory, bounds_t, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    cmap = plt.get_cmap("viridis")
    norm = MPLNormalize(vmin=0, vmax=trajectory.shape[0] - 1)
    for i in range(trajectory.shape[0]):
        fx = manifold.srsf_to_function(trajectory[i,...].reshape(1, -1, 1))
        color = "tab:red" if i==trajectory.shape[0]-1 else cmap(norm(i))
        ls = "--" if i==trajectory.shape[0]-1 else "-"
        ax.plot(
            np.linspace(0, 1, manifold.n_sampling_points),
            bounds_t[0] + (bounds_t[1]-bounds_t[0])*fx.squeeze(),
            color=color,
            ls = ls,
        )
    ax.scatter(
        temperature.x, 
        temperature.fx.squeeze(), 
        zorder=2,
        color="tab:red"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Trajectory index")

    return ax

def plot_active_learning_data(train_X, train_Y, xy_star, t_values, ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = plt.gcf()
    for l in train_Y.unique().squeeze():
        flags = train_Y.squeeze()==l
        ax.scatter(
            train_X[flags, 0], 
            train_X[flags, 1], 
            train_X[flags, 2],
            **scatter_kwargs
        )
    for t in t_values:
        ax.scatter(
            xy_star[:, 0], xy_star[:, 1], t,
            color="k",
            s=50,
            depthshade=False,
            edgecolors="k",
            linewidths=0.6,
        )
    return fig, ax

def plot_geodesic_progress(batch_ids, file):
    baselines = {}
    baselines[(0,1)] = 40.0
    baselines[(1,2)] = 15.0
    # --- Load data ---
    records = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                records.append(json.loads(line))

    # --- Extract phase boundary keys dynamically ---
    keys = sorted(list(records[0].keys()))
    iterations = np.arange(len(records))
    measurements = np.asarray([(batch_ids<=i).sum().item() for i in iterations])

    # Convert data into arrays with None -> np.nan
    data = {k: np.array([r.get(k, None) for r in records], dtype=float) for k in keys}

    # --- Valid mask for both curves existing ---
    valid_mask = np.isfinite(data[keys[0]]) & np.isfinite(data[keys[1]])

    # --- First iteration where both exist till the end ---
    first_both_idx = None
    for idx in range(len(valid_mask)):
        if np.all(valid_mask[idx:]):
            first_both_idx = idx
            break

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(4, 4))

    color1 = "tab:blue"
    color2 = "tab:orange"

    # Plot first boundary (shifted by baseline)
    y = data[keys[0]] - baselines[ast.literal_eval(keys[0])]
    ax.plot(measurements, 
             y/np.nanmax(y),
             color=color1, 
             label="Solubility"
        )
    ax.set_xlabel("Number of Measurements")
    ax.set_ylabel(f"Normalized Boundary Score")

    y = data[keys[1]] - baselines[ast.literal_eval(keys[1])]
    ax.plot(measurements, 
             y/np.nanmax(y),
             color=color2, 
             label="Liquid-Liquid"
        )

    # Vertical line: first iteration where both boundaries are observed till end
    if first_both_idx is not None:
        ax.axvline(measurements[first_both_idx], color="grey", linestyle="--", lw=1.5)
        ax.text(measurements[first_both_idx] + 0.5, ax.get_ylim()[1]*0.9,
                 f"Both observed after {measurements[first_both_idx]} measurements",
                 rotation=0, va="top", color="grey")

    # Baseline lines
    ax.axhline(0.0, color="k", linestyle="--", lw=1.5)
    ax.legend()

    plt.title("Phase boundary scores over iterations")
    fig.tight_layout()
    return fig, ax
