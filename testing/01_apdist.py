
import xarray as xr
xr.set_options(display_expand_data=False)

from AFL.double_agent_tutorial.instruments.tutorial import get_virtual_instrument
from AFL.double_agent.Preprocessor import LogLogTransform
from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.AmplitudePhaseDistance import AmplitudePhaseDistance

import matplotlib.pyplot as plt 
import numpy as np 
import time 
from scipy.spatial.distance import squareform

instrument = get_virtual_instrument()

composition_list = [
    {'a':1/3,'b':1/3,'c':1/3},
    {'a':0.0,'b':0.5,'c':0.5},
    {'a':0.5,'b':0.0,'c':0.5},
]

input_dataset = instrument.measure_multiple(composition_list)
print(input_dataset)

# test log-log preprocessor
with Pipeline() as test_pipeline:
       LogLogTransform(
           input_variable='sas',
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
    with test_pipeline:
        AmplitudePhaseDistance(
            input_variable="log_iq",
            output_variable='similarity',
            method=method,
            sample_dim='log_q',
            )

    test_pipeline.print()

    result_dataset = test_pipeline.calculate(input_dataset)
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
    ax.legend()
    plt.title(dist_str, fontsize=10)
    plt.savefig("%s.png"%method)
    plt.close()






