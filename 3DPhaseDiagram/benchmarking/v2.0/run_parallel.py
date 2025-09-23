#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import time
import datetime
from tqdm.auto import tqdm
import logging
os.environ["RAY_DEDUP_LOGS"] = "0" 

parser = argparse.ArgumentParser(description="")
parser.add_argument('--budget', type=int, default=10, required=True,
                    help="total number of measurements budget")
parser.add_argument('--runs', type=int, default=8, required=True,
                    help="total number of repeats runs of any pipeline")
parser.add_argument('--test', type=bool, default=False, required=False,
                    help="variable indicating whether the run is a test")
parser.add_argument('--num_cpus', type=int, default=8, required=False,
                    help="number of cpus for ray.init()")
args = parser.parse_args()

DIR = "/Users/knv2/Documents/codebase/AFL-tutorial/3DPhaseDiagram"
N_RUNS = args.runs
MAX_MEASUREMENTS = args.budget
if args.test:
    PLOT_DIR = f"./results_test/"
else:
    PLOT_DIR = f"./results_{MAX_MEASUREMENTS:d}_measurements/"

# sys.path setup for both driver and workers
folders = [DIR + "/modules/", DIR + "/dataset/", DIR + "/benchmarking/"]
for path in folders:
    if path not in sys.path:
        sys.path.append(path)

import ray
print(f"Using {args.num_cpus}/{os.cpu_count()} CPU cores")

@ray.remote
class CaseRunnerActor:
    """An Actor that holds sim, runner, and expense in memory for one case."""

    def __init__(self, case_name, case_kwargs, base_dir):
        import sys
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        folders_local = [base_dir + "/modules/", base_dir + "/dataset/", base_dir + "/benchmarking/"]
        for p in folders_local:
            if p not in sys.path:
                sys.path.append(p)

        from pipelines import get_runner, ExperimentCost
        from helpers import get_simulator

        self.case_name = case_name
        self.case_kwargs = case_kwargs
        self.base_dir = base_dir
        self.sim = get_simulator(base_dir + "/dataset/")
        self.runner = get_runner(case_name, **case_kwargs)
        self.expense = ExperimentCost(
            input_variable="design_space",
            input_dim="ds_dim",
            composition_dims_cost={"protein": 10.0, "glycerol": 5.0},
            temperature_dim_cost={"temperature": 20.0},
            output_variable="expense",
            name="ExperimentCost",
        )

        self.logger = logging.getLogger(case_name)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt=f"[{case_name}] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def run(self, run_idx, max_measurements, plot_dir):
        import numpy as np
        import xarray as xr
        import os, shutil, time, random
        import matplotlib.pyplot as plt
        from helpers import plot_sampling_densities

        case_name = self.case_name
        sim = self.sim
        runner = self.runner
        expense = self.expense

        idirec = os.path.join(plot_dir, f"{case_name}", f"{run_idx:d}")
        if os.path.exists(idirec):
            shutil.rmtree(idirec)
        os.makedirs(idirec, exist_ok=True)

        seed = int((time.time() * 1e6) % 2**31) + run_idx
        np.random.seed(seed)
        random.seed(seed)

        # build initial dataset
        ds_list = []
        protein_init = np.random.uniform(low=0.0, high=80.0)
        glycerol_init = np.random.uniform(low=0.0, high=12.0)
        for temp in [0, 10, 20, 30, 40, 50]:
            ds = sim.simple_expose({"protein": protein_init,
                                    "temperature": temp,
                                    "glycerol": glycerol_init})
            ds = ds.rename({"composition": "design_space", "component": "ds_dim"})
            ds["phases"] = int(ds.attrs["labels"])
            ds_list.append(ds)

        ds_init = xr.concat(ds_list, dim="sample")
        ds_running = ds_init.copy()
        expense_over_steps = [0.0]

        while len(ds_running.sample) < max_measurements:
            if "phase_n_classes" in ds_running.dims:
                ds_running = ds_running.drop_dims("phase_n_classes")
            ds_running, ds_result = runner(ds_running, sim)
            out = expense.calculate(ds_running)
            expense_over_steps.append(out.output["expense"].values.item())
            self.logger.info(f"[run {run_idx}], "
                         f"measurements: {len(ds_running.sample)}, "
                         f"expense: {expense_over_steps[-1]:,.2e}")
        self.logger.info(f"Finished run {run_idx} for case {self.case_name}")
        ds_result.to_netcdf(os.path.join(idirec, "ds.nc"))
        try:
            plot_sampling_densities(ds_running)
            plt.savefig(os.path.join(idirec, "sampling_densities.png"), dpi=300)
            plt.close()
        except Exception as e:
            with open(os.path.join(idirec, "plot_error.txt"), "w") as f:
                f.write(str(e))

        np.save(os.path.join(idirec, "expense.npy"), expense_over_steps)
        return {
            "case": case_name,
            "run": run_idx,
            "n_measurements": len(ds_running.sample),
            "expense_last": expense_over_steps[-1],
        }


def main():
    ray.init(num_cpus=args.num_cpus)

    case_studies = [
        ("P0", {}),
        ("P1", {}),
        # ("P2", {}),
        # ("P3", {}),
        ("P4", {"n_temperatures": 5}),
        ("P5", {"n_temperatures": 5}),
    ]

    actors = {}
    for case_name, case_kwargs in case_studies:
        actors[case_name] = CaseRunnerActor.options(name=f"{case_name}", num_cpus = 2).remote(case_name, case_kwargs, DIR)
    # Submit runs
    futures = []
    for case_name, _ in case_studies:
        actor = actors[case_name]
        for run_idx in range(N_RUNS):
            futures.append(actor.run.remote(run_idx, MAX_MEASUREMENTS, PLOT_DIR))

    start = time.time()
    results = []
    remaining = futures[:]
    pbar = tqdm(total=len(futures), desc="Runs finished")

    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1, timeout=None)
        res = ray.get(done[0])
        results.append(res)
        pbar.update(1)
        pbar.set_postfix({"last": f"{res['case']}-{res['run']}, {res['expense_last']:,.2e}"})
    pbar.close()

    ray.shutdown()
    end = time.time()
    time_str = str(datetime.timedelta(seconds=end - start))
    print(f"Total run time for all the {len(case_studies)} case studies : {time_str}")

    print("\nSummary:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
