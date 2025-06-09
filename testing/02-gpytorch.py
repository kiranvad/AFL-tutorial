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
from AFL.double_agent.GPyTorchExtrapolator import DirichletGPExtrapolator

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
    )

test_pipeline.print()
result_dataset = test_pipeline.calculate(input_dataset)
print(result_dataset)

extrapolator = DirichletGPExtrapolator(
    feature_input_variable="composition",
    predictor_input_variable="labels", 
    output_prefix="gp",
    grid_variable="composition_grid",
    grid_dim="grid",
    sample_dim="sample",
    params={"learning_rate": 1e-1, "n_iterations": 500, "verbose": True}
)
result = extrapolator.calculate(result_dataset)
print(result.output)
print("Labels : ", result.output["gp_mean"].shape)
print("Probabilities: ", result.output["gp_y_prob"].shape)
print("Entropy: ", result.output["gp_entropy"].shape)

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
               c = result.output["gp_entropy"]
            )
ax.set_title("Entropy")
plt.savefig("1.png")
plt.close()