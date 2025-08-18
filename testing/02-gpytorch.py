import xarray as xr
xr.set_options(display_expand_data=False)
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from matplotlib import colormaps 
import mpltern

from AFL.double_agent_tutorial.instruments.tutorial import get_virtual_instrument
from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent import SavgolFilter, Similarity, SpectralClustering, BarycentricGrid
from AFL.double_agent.PyTorchExtrapolator import DirichletGPExtrapolator

import pdb 

instrument = get_virtual_instrument()

def generate_random_point():
    """Generate a random point (a, b, c) such that a + b + c = 1.0 and all values are non-negative"""
    # Generate two random values between 0 and 1
    r = np.random.random(2)
    r1, r2 = np.sort(r)
    
    # Convert to barycentric coordinates
    a = r1
    b = r2 - r1
    c = 1 - r2
    
    return {'a': a, 'b': b, 'c': c}

composition_list = []
np.random.seed(42)  # For reproducible results
for i in range(100):
    composition_list.append(generate_random_point())

input_dataset = instrument.measure_multiple(composition_list)
input_dataset['composition'] = input_dataset[['c','a','b']].to_array('component').transpose('sample',...)
input_dataset = input_dataset.drop_vars(['c','a','b'])
print(input_dataset)

with Pipeline() as test_pipeline:

    ## Preprocess the SAS measurements
    SavgolFilter(
        input_variable='sas',
        output_variable='derivative',
        dim='q',
        derivative=1
        )

    ## Calculate the pairwise similarity between each measurement
    Similarity(
        input_variable='derivative',
        output_variable='similarity',
        sample_dim='sample',
        params={'metric': 'laplacian','gamma':1e-4}
        )

    ## Label/cluster the SAS measurements by numerical similarity
    SpectralClustering(
        input_variable='similarity',
        output_variable='labels',
        dim='sample',
        params={'n_phases': 2}
        )

    ## Create a barycentric (ternary) grid to extrapolate onto
    BarycentricGrid(
        output_variable='composition_grid',
        components = ['a','b','c'],
        sample_dim='grid',
        pts_per_row=50
    )

test_pipeline.print()
result_dataset = test_pipeline.calculate(input_dataset)
mcmc_params = {"num_samples":100,
               "num_warmup":100,
               "verbose":True,
               "method":"mcmc"
            }
mll_params = {"learning_rate": 1e-1, 
            "n_iterations": 500, 
            "verbose": True,
            "method":"mll"
        }

extrapolator = DirichletGPExtrapolator(
    feature_input_variable="composition",
    predictor_input_variable="labels", 
    output_prefix="gp",
    grid_variable="composition_grid",
    grid_dim="grid",
    sample_dim="sample",
    params= mll_params
)
result = extrapolator.calculate(result_dataset)
print(result.output)
print("Labels : ", result.output["gp_mean"].shape)
print("Probabilities: ", result.output["gp_y_prob"].shape)
print("Entropy: ", result.output["gp_entropy"].shape)
print("Gradient of Entropy: ", result.output["gp_entropy_gradient"].shape)

fig = plt.figure(figsize=(4*2, 4*2))
fig.subplots_adjust(wspace=0.5, hspace=0.5)
cmap = colormaps["tab10"]
norm = Normalize(vmin=0, vmax = 1)

ax = fig.add_subplot(2,2,1, projection = 'ternary')
ax.scatter(result_dataset["composition"][:,0], 
               result_dataset["composition"][:,1],
               result_dataset["composition"][:,2],
               c = result_dataset["labels"],
               cmap = cmap,
               norm = norm
            )
ax.set_title("Labeled data")

ax = fig.add_subplot(2,2,2)
for label in range(2):
    flags = np.argwhere(result_dataset["labels"].values==label).squeeze()
    color = cmap(norm(label))
    for i in flags:
        ax.loglog(result_dataset["q"], 
                      result_dataset["sas"][i, :],
                      color = color
                    )
ax.set_xlabel("q")
ax.set_ylabel("I(q)")
ax.set_title("SAS classification")

ax = fig.add_subplot(2,2,3, projection = 'ternary')
ax.scatter(result_dataset["composition_grid"][:,0], 
               result_dataset["composition_grid"][:,1],
               result_dataset["composition_grid"][:,2],
               c = result.output["gp_mean"],
               cmap = cmap,
               norm = norm
            )
ax.set_title("Labeles on a grid")

ax = fig.add_subplot(2,2,4, projection = 'ternary')
ax.scatter(result_dataset["composition_grid"][:,0], 
           result_dataset["composition_grid"][:,1],
           result_dataset["composition_grid"][:,2],
           c = result.output["gp_entropy"],
           cmap = "Blues"
        )

norms = np.linalg.norm(result.output["gp_entropy_gradient"], axis=1, keepdims=True)  # shape (N, 1)
grad = result.output["gp_entropy_gradient"].values 
mean_component = grad.sum(axis=1, keepdims=True) / 3.0
grad -= mean_component
scale_factor = 0.1  
disp_unit = grad/ (norms + 1e-12)  # add epsilon to avoid division by zero
disp_scaled = disp_unit * (norms * scale_factor)
ax.quiver(result_dataset["composition_grid"][:,0],
          result_dataset["composition_grid"][:,1],
          result_dataset["composition_grid"][:,2],
          disp_scaled[:,0],
          disp_scaled[:,1],
          disp_scaled[:,2],
        )

ax.set_title("Entropy")
plt.savefig(f"02_gpytorch_{extrapolator.params['method']}.png", dpi=300)
plt.close()