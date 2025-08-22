
import xarray as xr
xr.set_options(display_expand_data=False)

from AFL.double_agent.data import example_dataset2
from AFL.double_agent.Preprocessor import LogLogTransform
from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.AmplitudePhaseDistance import AmplitudePhaseDistance

import matplotlib.pyplot as plt 
import numpy as np 
import time 
from scipy.spatial.distance import squareform

example_dataset = example_dataset2()
print(example_dataset)

n_samples = 3
sample_indices = example_dataset.sample.values 
selected_samples = np.random.choice(sample_indices, size=n_samples, replace=False)
input_dataset = example_dataset.sel(sample=selected_samples)
print(input_dataset)

# test log-log preprocessor
with Pipeline() as test_pipeline:
       LogLogTransform(
           input_variable='I',
           output_variable='log_iq',
           dim='q',
        )

test_pipeline.print()

result_dataset = test_pipeline.calculate(input_dataset)
print(result_dataset)

x = result_dataset["log_q"].to_numpy()
y = result_dataset["log_iq"].to_numpy()
print(x.shape, y.shape)

# test apdist pairmetric
for method in ["discrete", "continuous"]:
    print("testing for %s..."%method)
    start = time.time()
    with test_pipeline.copy() as tp:
        AmplitudePhaseDistance(
            input_variable="log_iq",
            output_variable='similarity',
            method=method,
            sample_dim='sample',
            )

    result_dataset = tp.calculate(input_dataset)
    D = result_dataset.similarity.values
    print(D)
    assert np.isfinite(D).all(), "Distances are not finite"
    assert (D>=0).all(), "Distances are not positive"
    end = time.time()
    print("total time elapsed : ", end-start)

    fig, ax = plt.subplots()
    for i in range(result_dataset.sizes["sample"]):
        ax.scatter(x, y[i,:], label="%d"%i)

    # Get condensed distances (1D vector)
    dists = squareform(D, force='tovector')

    # Generate (i, j) index pairs for upper triangle (excluding diagonal)
    N = D.shape[0]
    ij_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # Combine into a string
    dist_str = "Pairwise distances:\n" + ";".join(
        f"({i},{j}): {d:.2f}" for (i, j), d in zip(ij_pairs, dists)
    )
    plt.title(dist_str, fontsize=10)

    ax.legend()
    plt.savefig("%s.png"%method)
    plt.close()






