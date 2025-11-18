import torch 
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 
import yaml 
from pathlib import Path
from gpytorch.likelihoods import DirichletClassificationLikelihood
from botorch.models.transforms import Normalize

from modules import * 
from visuals import *
import sys, os, json, shutil

from artifact_saver import (
    ExperimentArtifactSaver,
    PosteriorArtifacts,
    AcquisitionArtifacts,
    FunctionOptimizationArtifacts,
    ActiveLearningArtifacts,
)

DIR = "/Users/knv2/Documents/codebase/AFL-tutorial/3DPhaseDiagram"
folders = [DIR + "/modules/", DIR + "/dataset/", DIR + "/benchmarking/"]
for path in folders:
    if path not in sys.path:
        sys.path.append(path)

from helpers import get_simulator
from evaluate import get_boundaries_geodesic_distance

sim = get_simulator(DIR + "/dataset/")

def oracle(xy, t):
    labels = []
    for ti in t:
        ds = sim.simple_expose(
            {'protein':xy.numpy()[0],
            'temperature':ti.numpy().item(),
            'glycerol':xy.numpy()[1]
        })
        labels.append(int(ds.attrs['labels']))
    return torch.tensor(labels).to(xy)


with open("./config.yaml") as stream:
    cfg = yaml.safe_load(stream)


bounds = torch.tensor(
    [
        [0.0, 80.0], 
        [0.0, 12.0],
        [0.0, 50.0]
    ]
    ).T

bounds_c = bounds.clone()[:,:2]
bounds_t = bounds.clone()[:,-1]

grid_xy, grid_t, grid_xyz = make_grid_point_cloud(bounds_c, bounds_t, **cfg["grid"])
ground_truth_labels = torch.cat([oracle(xyz[:2], xyz[-1].reshape(-1,1)) for xyz in grid_xyz])

manifold = SRSFHilbertSphere(**cfg["manifold"])

n_iterations = cfg["n_iterations"]
n_init = cfg["n_init"]
n_init_temps = cfg["n_init_temps"]
save_path = Path(cfg["save_path"]).resolve()
# Clean and recreate each time
if save_path.exists():
    shutil.rmtree(save_path)
save_path.mkdir(parents=True, exist_ok=True)
saver = ExperimentArtifactSaver(save_path, cfg, bounds, grid_xy, grid_t, grid_xyz)

train_X, train_Y = generate_random_samples(oracle, n_init, bounds_c, n_init_temps, bounds_t)
batch_sample_indx = torch.zeros(int(n_init*n_init_temps))
dists_file_json = save_path / "dists.json"
if os.path.isfile(dists_file_json):
    os.remove(dists_file_json)

pbar = tqdm(range(n_iterations), desc="Active Learning")

for i in pbar:
    fig, axs = make_gridspec(bounds)
    likelihood = DirichletClassificationLikelihood(
        train_Y.long().squeeze(), 
        learn_additional_noise=True
    )
    model = DirichletGPModel(
        train_X, 
        likelihood.transformed_targets, 
        likelihood, 
        num_classes = likelihood.num_classes,
        transform = Normalize(d=3, bounds = bounds)
        )
    model, likelihood = fit_dirchlet_gp(train_X, model, likelihood, n_iterations=cfg["gp"]["n_iterations"])
    saver.save_gp_state(i + 1, model, likelihood)
    plot_outputs = plot_model_posterior(
        model, grid_xyz, ground_truth_labels, ax=[axs[0][0], axs[0][1]], return_stats=True
    )
    grid_labels, _, _, posterior_stats = plot_outputs
    
    # optimize composition space in R^2
    acqf = lambda X: batch_acqusition_xy(X, model, manifold, bounds_t, n_samples=cfg["optimize_xy"]["n_samples"])
    xy_star, trajectory_xy = optimize_xy(
        bounds_c, acqf, n_iterations=cfg["optimize_xy"]["n_iterations"], n_restarts=cfg["optimize_xy"]["n_restarts"]
        )
    _, _, score_xy = plot_xy_optimization(
        acqf, grid_xy, xy_star, trajectory_xy, ax=axs[1][0], return_scores=True
    )
    
    # optimize temperature profile in S_{\inf}
    q_star, f_star, trajectory_f, loss_traj_f, ind_f = optimize_f(
        model, manifold, xy_star, bounds_t, n_iterations=cfg["optimize_f"]["n_iterations"], n_restarts=cfg["optimize_f"]["n_restarts"]
        )
    profile = get_temperature_profile(
        f_star, bounds_t, dt=cfg["optimize_f"]["dt"], dT=cfg["optimize_f"]["dT"]
        )
    temperatures = profile.fx.reshape(-1,1)
    plot_function_optimization(manifold, profile, trajectory_f[ind_f,...], bounds_t, ax=axs[1][1])
    
    plot_active_learning_data(train_X, train_Y, xy_star, temperatures, ax=axs[0][2])
    train_X_snapshot = train_X.clone()
    train_Y_snapshot = train_Y.clone()
    batch_snapshot = batch_sample_indx.clone()

    # update dataset
    new_Y = oracle(xy_star.squeeze(), temperatures)
    xy_star_expanded = xy_star.view(-1, 1, 2).expand(-1, len(temperatures), -1).squeeze(0)
    new_X = torch.cat([xy_star_expanded.squeeze(), temperatures], dim=1)
    
    train_X = torch.cat((train_X, new_X))
    train_Y = torch.cat((train_Y, new_Y.reshape(-1, 1)))
    batch_sample_indx = torch.cat([batch_sample_indx, (i+1)*torch.ones(new_X.shape[0])], dim=0)
    fig.suptitle(f"Active Learning iteration {i+1}/{n_iterations}")
    fig_path = saver.figure_path(i+1)
    plt.savefig(fig_path)
    plt.close()

    # Save distances
    dists = get_boundaries_geodesic_distance(grid_xyz.numpy(), grid_labels.numpy(), make_animation=False)
    with open(dists_file_json, "a") as f:
        f.write(json.dumps({str(k): v for k, v in dists.items()}) + "\n")
        f.flush()

    posterior_artifacts = PosteriorArtifacts(
        labels=grid_labels,
        entropy=posterior_stats["entropy"],
        accuracy=float(posterior_stats["accuracy"]),
    )
    acquisition_artifacts = AcquisitionArtifacts(
        scores=np.asarray(score_xy, dtype=np.float32),
        xy_star=xy_star,
        trajectory=trajectory_xy,
    )
    trajectory_slice = trajectory_f[ind_f, ...]
    profile_time = profile.x
    profile_temp = profile.fx
    if torch.is_tensor(profile_time):
        profile_time = profile_time.detach().cpu().numpy()
    else:
        profile_time = np.asarray(profile_time)
    if torch.is_tensor(profile_temp):
        profile_temp = profile_temp.detach().cpu().numpy()
    else:
        profile_temp = np.asarray(profile_temp)
    function_opt_artifacts = FunctionOptimizationArtifacts(
        trajectory=trajectory_slice,
        profile_time=profile_time,
        profile_temperature=profile_temp,
    )
    active_learning_artifacts = ActiveLearningArtifacts(
        train_X=train_X_snapshot,
        train_Y=train_Y_snapshot,
        batch_indices=batch_snapshot,
        xy_star=xy_star,
        proposed_temperatures=temperatures,
    )
    saver.log_iteration(
        i + 1,
        posterior=posterior_artifacts,
        acquisition=acquisition_artifacts,
        function_opt=function_opt_artifacts,
        active_learning=active_learning_artifacts,
        extras={
            "figure": str(fig_path),
            "gp_state": str(saver.gp_model_path(i + 1)),
        },
    )

plot_geodesic_progress(dists_file_json)
plt.savefig(saver.figure_dir / "score_tracker.png")
plt.close()
if dists_file_json.exists():
    saver.save_geodesic_history(dists_file_json)
