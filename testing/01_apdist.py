
import xarray as xr
xr.set_options(display_expand_data=False)

from AFL.double_agent_tutorial.instruments.tutorial import get_virtual_instrument
from AFL.double_agent.Preprocessor import SAXSLogLogTransform
from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.PairMetric import AmplitudePhaseDistance

import matplotlib.pyplot as plt 

instrument = get_virtual_instrument()

composition_list = [
    {'a':1/3,'b':1/3,'c':1/3},
    {'a':0.0,'b':0.5,'c':0.5},
    {'a':0.5,'b':0.0,'c':0.5},
    {'a':0.5,'b':0.5,'c':0.0},
]

input_dataset = instrument.measure_multiple(composition_list)
print(input_dataset)


with Pipeline() as test_pipeline:
       SAXSLogLogTransform(
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

fig, ax = plt.subplots()
for i in range(result_dataset.sizes["sample"]):
    ax.scatter(x, y[i,:])
plt.savefig("1.png")
plt.close()

with test_pipeline:
    AmplitudePhaseDistance(
        input_variable="log_iq",
        output_variable='similarity',
        sample_dim='log_q',
        )

test_pipeline.print()

result_dataset = test_pipeline.calculate(input_dataset)
print(result_dataset)




