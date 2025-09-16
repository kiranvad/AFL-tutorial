import xarray as xr
import matplotlib.pyplot as plt
import sys, os, shutil
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--budget', type=int, default= 50, required=True, help="total number of measurements budget")
parser.add_argument('--runs', type=int, default=3, required=True, help="total number of repeats runs of any pipeline")
parser.add_argument('--test', type=bool, default=False, required=False, help="variable indicating whether the run is a test")

args = parser.parse_args()
DIR = "/Users/knv2/Documents/codebase/AFL-tutorial/3DPhaseDiagram"
N_RUNS = args.runs
N_SAMPLES = args.budget
if args.test:
    PLOT_DIR = f"./test/n_samples/"
else:
    PLOT_DIR = f"./results_{N_SAMPLES:d}_samples/"

folders = [DIR+"/modules/", DIR+ "/dataset/", DIR+"/benchmarking/v2.0/"]
for path in folders:
    if path not in sys.path:
        sys.path.append(path)

from helpers import *
from pipelines import get_runner, ExperimentCost

expense = ExperimentCost(
        input_variable="design_space",
        input_dim="ds_dim",
        composition_dims_cost={"protein":10.0, "glycerol" :5.0},
        temperature_dim_cost={"temperature" : 20.0},
        output_variable="expense",
        name="ExperimentCost"
    ) 
sim = get_simulator(DIR+"/dataset/")

case_studies = [
    # ("P1", {}),
    # ("P2", {}),
    # ("P3", {}),
    ("P4", {"n_temperatures":5}),
    ("P5", {"n_temperatures":5})
]
with tqdm(case_studies, desc="Case Studies") as pbar:
    for case in pbar:
        pbar.set_postfix({"case": case[0]})
        runner = get_runner(case[0], **case[1])
        for run in tqdm(range(N_RUNS), desc="Runs", leave=False):
            idirec = PLOT_DIR+f"/{case[0]}/{run:d}/"
            if os.path.exists(idirec):
                shutil.rmtree(idirec)
            os.makedirs(idirec)
            ds_list = []
            protein_init = np.random.uniform(low=0.0, high=80.0)
            glycerol_init = np.random.uniform(low=0.0, high=12.0)
            for temp in [0,10,20,30,40,50]:
                ds = sim.simple_expose({'protein':protein_init,
                                        'temperature':temp,
                                        'glycerol':glycerol_init})
                ds = ds.rename({'composition': 'design_space', 'component':'ds_dim'})
                ds['phases'] = int(ds.attrs['labels'])
                ds_list.append(ds)

            ds_init = xr.concat(ds_list, dim='sample')
            ds_running = ds_init.copy() 
            expense_over_steps = [0.0]
            pbar.update(len(ds_running.sample))
            for _ in tqdm(range(N_SAMPLES), desc="Samples", leave=False):
                if "phase_n_classes" in ds_running.dims:
                    ds_running = ds_running.drop_dims("phase_n_classes") # ensures that dynamic dims are created within the pipeline
                ds_running, ds_result = runner(ds_running, sim)
                out = expense.calculate(ds_running)
                expense_over_steps.append(out.output['expense'].values.item())

            ds_result.to_netcdf(idirec+f"/ds.nc")
            plot_sampling_densities(ds_running)
            plt.savefig(idirec+"sampling_densities.png", dpi=300)
            plt.close()
            
            np.save(idirec+"/expense.npy", expense_over_steps)