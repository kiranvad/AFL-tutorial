import warnings, logging
import torch
import torch.optim as optim
import numpy as np 
import xarray as xr 
from funcshape.surface import PointCloudSurface
from funcshape.transforms import SRNF
from funcshape.networks import SurfaceReparametrizer
from funcshape.layers.sinefourier import SineFourierLayer
from funcshape.loss import SurfaceDistance
from funcshape.reparametrize import reparametrize
from funcshape.logging import Logger 
from sklearn.neighbors import KDTree
from collections import defaultdict
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt 
from funcshape.interpolation import linear_interpolate
from funcshape.visual import plot_surface
import shutil, uuid, gc
from pathlib import Path

def find_boundary_points(points, labels, k=10):
    """
    Identify boundary points in a point cloud with labels using KDTree.
    
    Parameters
    ----------
    points : ndarray of shape (N, 3)
        3D coordinates of points
    labels : ndarray of shape (N,)
        Class labels (e.g. +1/-1 or multiclass)
    k : int
        Number of neighbors to check for label differences
    
    Returns
    -------
    boundaries : dict
        Dictionary where keys are (label_a, label_b) tuples and values 
        are arrays of boundary points lying between those classes.
    """
    tree = KDTree(points)
    boundaries = defaultdict(list)

    for i, p in enumerate(points):
        _, idx = tree.query([p], k=k)
        neighbor_labels = labels[idx[0]]
        
        # Compare with current point’s label
        current_label = labels[i]
        diff_labels = set(neighbor_labels) - {current_label}
        
        for other_label in diff_labels:
            key = tuple(sorted((current_label.item(), other_label.item())))
            boundaries[key].append(p)

    # Convert lists to numpy arrays
    return {k: np.array(v) for k, v in boundaries.items()}

def get_ground_truth_boundaries():
    DIR = "/Users/knv2/Documents/codebase/AFL-tutorial/3DPhaseDiagram"
    boundary_dataset = xr.load_dataset(DIR+'/dataset/250610-extrap_expand_dataset.nc')
    composition = boundary_dataset.composition
    # Define per-component limits
    limits = {'protein': (0.0, 80.0), 'glycerol': (0.0, 12.0), 'temperature': (0.0, 50.0)}
    # Build a mask for each component
    masks = []
    for c in composition.component.values:
        min_val, max_val = limits[c]
        masks.append(
            (composition.sel(component=c) >= min_val) & (composition.sel(component=c) <= max_val)
        )
    # Combine masks across components: keep only samples where ALL components satisfy their limits
    combined_mask = xr.concat(masks, dim="component").all(dim="component")

    # Select only the valid samples
    xyz = composition.sel(sample=combined_mask)
    ell = boundary_dataset.labels.sel(sample=combined_mask)

    ref_boundaries = find_boundary_points(xyz.values.T[:,[0,2,1]], ell.values, k=16)
    ref_01 = PointCloudSurface(ref_boundaries[(0,1)])
    ref_12 = PointCloudSurface(ref_boundaries[(1,2)])
    ground_truth = {}
    ground_truth[(0,1)] = ref_01
    ground_truth[(1,2)] = ref_12

    return ref_boundaries, ground_truth

def get_logger(verbose):
    # Create logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    if verbose:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_boundaries_geodesic_distance(grid: np.ndarray, labels: np.ndarray, is_calibrate: bool=False, make_animation: bool=False, verbose:bool = False):
    logger = get_logger(verbose)

    ref_boundaries, ground_truth = get_ground_truth_boundaries()
    if make_animation:
        temp_dir = Path.cwd() / "tmp_boundary_score"
        
        # Clean and recreate each time
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    if not is_calibrate:
        boundaries = find_boundary_points(grid, labels, k=16)
    else:
        boundaries = ref_boundaries.copy()

    for key, val in boundaries.items():
        logger.info(f"Boundary {key}: {len(val)} points")
    
    if not ref_boundaries.keys() == boundaries.keys():
        warnings.warn("Some phase boundaries are not resolved...")
    
    dists = {}
    for key, ref in ground_truth.items():
        logger.info(f"Evaluating boundary {key}")
        try:
            query = PointCloudSurface(boundaries[key])
        except:
            logger.info(f"Phase boundary {key} is not resolved")
            dists[key] = None
            continue

        q = SRNF(query)
        r = SRNF(ref)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define reparametrization-network
        RN = SurfaceReparametrizer(
            [SineFourierLayer(10) for _ in range(10)]
        ).to(device)
        loss_func = SurfaceDistance(q, r, k=32, h=1e-4).to(device)

        optimizer = optim.LBFGS(RN.parameters(), max_iter=30, line_search_fn="strong_wolfe")
        errors = reparametrize(RN, loss_func, optimizer, 30, Logger(1 if is_calibrate else 0))
        RN.eval()
        RN.to("cpu").detach()
        dists[key] = errors[-1].item()
        if make_animation:
            tag = f"boundary{''.join(str(i) for i in key)}_{uuid.uuid4().hex[:6]}"
            gif_path = temp_dir / f"animation_{tag}.gif"
            logger.info(f"Saving animation to {gif_path}")
            make_geodesic_animation(
                ref,
                query,
                RN,
                path=gif_path
            )
        
        del RN 
        gc.collect()

    return dists 

def get_baseline_geodesics(n_runs=1):
    all_dists = []
    logger = get_logger(True)
    for i in range(n_runs):
        torch.manual_seed(i*273)
        logger.info(f"Baseline geodesic computation run {i}/{n_runs}")
        dists = get_boundaries_geodesic_distance([], [], is_calibrate=True, verbose=True)
        all_dists.append(dists)

    return all_dists

def make_geodesic_animation(
        start, 
        end, 
        warping, 
        path="./tmp/geodesics/animation.gif",
        num_steps = 10
        ):

    # --- Your existing setup ---
    k = 30
    X, Y = torch.meshgrid(torch.linspace(0, 1, k), torch.linspace(0, 1, k))
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    X = torch.cat((X, Y), dim=1)

    # Precompute all surfaces
    surfaces = []
    for h in linear_interpolate(lambda x: end(warping(x)), start, num_steps):
        pts = h(X).detach().cpu().numpy()
        surfaces.append(PointCloudSurface(pts))

    # --- Animation setup ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    def init():
        ax.set_xlim([0.0, 80.0])
        ax.set_ylim([0.0, 12.0])
        ax.set_zlim([0.0, 50.0])
        return []

    def update(frame):
        ax.cla()  # Clear axis for new frame
        ax.set_xlim([0.0, 80.0])
        ax.set_ylim([0.0, 12.0])
        ax.set_zlim([0.0, 50.0])
        ax.set_title(f"Surface Evolution (t={frame})")

        # Plot initial surface (t=0) in grey
        plot_surface(surfaces[0], k=20, ax=ax, colormap=plt.get_cmap("gray"))

        # Plot current surface (t=frame) in color
        cmap = plt.get_cmap("jet")
        plot_surface(surfaces[frame], k=20, ax=ax, colormap=cmap)

        return []

    # --- Create animation ---
    anim = FuncAnimation(
        fig, update, frames=range(num_steps), init_func=init, blit=False
    )

    # --- Save as GIF ---
    anim.save(path, writer=PillowWriter(fps=2))
    plt.close(fig)