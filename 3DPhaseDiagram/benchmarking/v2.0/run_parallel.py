#!/usr/bin/env python3
import argparse
import os
import sys
import time
import datetime
import warnings, logging, json
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["RAY_DEDUP_LOGS"] = "0" 
import psutil, gc
import ray
from tqdm.auto import tqdm
from ray.experimental.state.api import list_objects
from pathlib import Path

parser = argparse.ArgumentParser(description="")
parser.add_argument('-b', '--budget', type=int, default=10, required=True,
                    help="total number of measurements budget")
parser.add_argument('-nr', '--runs', type=int, default=8, required=True,
                    help="total number of repeats runs of any pipeline")
parser.add_argument('--test', type=bool, default=False, required=False,
                    help="variable indicating whether the run is a test")
parser.add_argument('-nc', '--num_cpus_per_actor', type=int, default=3, required=False,
                    help="number of cpus for ray.init()")
args = parser.parse_args()

DIR = "/Users/knv2/Documents/codebase/AFL-tutorial/3DPhaseDiagram"
N_RUNS = args.runs
MAX_MEASUREMENTS = args.budget
if args.test:
    PLOT_DIR = Path("./results_test").resolve()
else:
    PLOT_DIR = Path(f"./results_{MAX_MEASUREMENTS:d}_measurements/").resolve()
PLOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving results to {PLOT_DIR}")
# sys.path setup for both driver and workers
folders = [DIR + "/modules/", DIR + "/dataset/", DIR + "/benchmarking/"]
for path in folders:
    if path not in sys.path:
        sys.path.append(path)


print(f"Using {args.num_cpus_per_actor}/{os.cpu_count()} CPU cores")

def log_memory(tag=""):
    """Log both local process and Ray object store memory usage."""
    try:
        # Local process memory (RSS)
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1e9

        # Ray object store memory (global)
        objs = list_objects()
        objstore_gb = sum(o["object_size"] for o in objs) / (1024 ** 3) if objs else 0.0
        out = (
            f"[{tag}] "
            f"Process RSS: {mem_gb:6.2f} GB | "
        )
        return out

    except Exception as e:
        return f"[{tag}] Memory log failed: {e}"

def cleanup_run_artifacts(self):
    """Clean up any accumulated state after each run."""
    gc.collect()
    # If runner has internal caches, clear them
    if hasattr(self.runner, 'clear_cache'):
        self.runner.clear_cache()

@ray.remote
class CaseRunnerActor:
    """An Actor that holds sim, runner, and expense in memory for one case."""

    def __init__(self, case_name, case_kwargs, base_dir):
        import sys
        folders_local = [base_dir + "/modules/", 
                         base_dir + "/dataset/", 
                         base_dir + "/benchmarking/"
                    ]
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
            composition_dims_cost={"protein": 10.0, "glycerol": 10.0},
            temperature_dim_cost={"temperature": 20.0},
            output_variable="expense",
            name="ExperimentCost",
        )

        self.logger = logging.getLogger(case_name)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt=f"[%(asctime)s] [%(levelname)s] %(message)s"
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
        from evaluate import get_boundaries_geodesic_distance

        initial_memory = psutil.virtual_memory().percent
        self.logger.info(f"Starting run with {initial_memory:.1f}% memory usage")

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
        protein_init = 40.0
        glycerol_init = 5.0
        batch_sample_id = 0
        for temp in [0, 10, 20, 30, 40, 50]:
            ds = sim.simple_expose({"protein": protein_init,
                                    "temperature": temp,
                                    "glycerol": glycerol_init})
            ds = ds.rename({"composition": "design_space", "component": "ds_dim"})
            ds["phases"] = int(ds.attrs["labels"])
            ds['batch_sample_id'] = batch_sample_id
            ds_list.append(ds)

        ds_init = xr.concat(ds_list, dim="sample")
        ds_running = ds_init.copy()
        del ds_init, ds_list  # ← Add ds_list cleanup
        expense_over_steps = [0.0]

        iteration = 0
        while len(ds_running.sample) < max_measurements:
            iteration += 1
            batch_sample_id += 1
            
            if "phase_n_classes" in ds_running.dims:
                ds_running = ds_running.drop_dims("phase_n_classes")
            
            # Keep reference to old dataset for cleanup
            ds_old = ds_running
            ds_running, ds_result = runner(batch_sample_id, ds_old, sim)
            
            # Immediately delete old dataset
            del ds_old
            
            # Calculate expense and extract value immediately
            out = expense.calculate(ds_running)
            expense_value = out.output["expense"].values.item()
            expense_over_steps.append(expense_value)
            del out  # ← Critical: delete immediately
            
            # Calculate distances and extract values
            dists = get_boundaries_geodesic_distance(
                ds_result.design_space_grid, 
                ds_result.phase_mean
            )
            
            # Save results
            ds_result.to_netcdf(os.path.join(idirec, "ds.nc"))
            if hasattr(ds_result, 'close'):
                ds_result.close()
            del ds_result  # ← Keep this
            
            # Save expense
            np.save(os.path.join(idirec, "expense.npy"), expense_over_steps)
            
            # Logging
            self.logger.info(f"[run {run_idx}, iter {iteration}] "
                        f"measurements: {len(ds_running.sample)}/{max_measurements}, "
                        f"expense: {expense_value:,.2e}")
            self.logger.info(f"[run {run_idx}], Boundary distances : {dists}")
            
            # Save distances
            dists_serializable = {str(k): v for k, v in dists.items()}
            with open(os.path.join(idirec, "dists.json"), "a") as f:
                f.write(json.dumps(dists_serializable) + "\n")
                f.flush()
            
            # Delete dists after use
            del dists, dists_serializable
            
            # Memory logging
            self.logger.info(log_memory(f"run {run_idx}, iter {iteration}"))
            
            # Periodic aggressive cleanup
            if iteration % 5 == 0:
                gc.collect()
                
                if psutil.virtual_memory().percent > 85:
                    self.logger.warning(f"[run {run_idx}, iter {iteration}] High memory usage: {psutil.virtual_memory().percent}%")
                    gc.collect()
            
            # Emergency cleanup
            if psutil.virtual_memory().percent > 90:
                self.logger.warning(f"[run {run_idx}, iter {iteration}] CRITICAL memory usage: {psutil.virtual_memory().percent}%")
                gc.collect()
        
        self.logger.info(f"Finished run {run_idx} for case {self.case_name}")
        
        # Final measurements
        n_measurements = len(ds_running.sample)
        expense_last = expense_over_steps[-1]
        
        # Close and cleanup final dataset
        if hasattr(ds_running, 'close'):
            ds_running.close()
        del ds_running
        
        # Cleanup expense history if large
        del expense_over_steps
        
        # Final garbage collection
        gc.collect()

        return {
            "case": case_name,
            "run": run_idx,
            "n_measurements": n_measurements,
            "expense_last": expense_last,
            "dists_last": {}
        }

def main():
    from ray.job_config import JobConfig

    job_config = JobConfig(
        metadata={"description": "3DPhaseDiagram"},
        ray_namespace=f"{args.budget}_{args.runs}"
    )
    ray.init(
        job_config=job_config,
        dashboard_host="127.0.0.1",
        dashboard_port=8266
    )
    case_studies = [
        ("P0", {}),
        # ("P1", {}),
        # ("P2", {}),
        ("P3", {}),
        # ("P4", {"n_temperatures": 5}),
        ("P5", {"n_temperatures": 5}),
    ]

    num_cpus_per_actor = args.num_cpus_per_actor
    total_cpus = int(ray.available_resources().get("CPU", 1))
    max_parallel_actors = max(1, total_cpus // num_cpus_per_actor)

    print(f"Detected {total_cpus} CPUs → running up to {max_parallel_actors} actors concurrently")

    pending_cases = list(case_studies)  # queue of cases waiting for resources
    active_actors = {}                  # maps case_name → (actor, [futures])

    start = time.time()
    pbar = tqdm(total=len(case_studies) * N_RUNS, desc="Runs finished")

    # Run loop until everything completes
    while pending_cases or active_actors:
        # Launch new actors if resources available
        while pending_cases and len(active_actors) < max_parallel_actors:
            case_name, case_kwargs = pending_cases.pop(0)
            actor = CaseRunnerActor.options(
                name=f"{case_name}", 
                num_cpus=num_cpus_per_actor).remote(case_name, case_kwargs, DIR)
            futures = [
                actor.run.remote(run_idx, MAX_MEASUREMENTS, PLOT_DIR)
                for run_idx in range(N_RUNS)
            ]
            active_actors[case_name] = (actor, futures)

        # Gather all currently running futures
        all_futures = [f for _, (_, fs) in active_actors.items() for f in fs]
        if not all_futures:
            break

        # Wait for one task to finish
        done, _ = ray.wait(all_futures, num_returns=1, timeout=None)

        if done:
            completed_future = done[0]
            res = ray.get(completed_future)
            
            pbar.update(1)
            pbar.set_postfix(
                {"last": f"{res['case']}-{res['run']}, {res['expense_last']:,.2e}"}
            )
            
            # Find and cleanup the completed task
            actor_to_cleanup = None
            case_to_remove = None
            
            for cname, (actor, futures) in list(active_actors.items()):  # Use list() to avoid dict size change issues
                if completed_future in futures:
                    futures.remove(completed_future)
                    
                    # Explicitly delete the completed future
                    del completed_future
                    
                    # Also cleanup the 'done' list to free all references
                    del done
                    gc.collect()
                    
                    if not futures:
                        actor_to_cleanup = actor
                        case_to_remove = cname
                    break
            
            # Perform cleanup outside iteration
            if actor_to_cleanup is not None:
                # Kill the actor
                ray.kill(actor_to_cleanup)
                
                # Wait briefly for Ray to process the kill
                time.sleep(0.1)
                
                # Delete all references
                del actor_to_cleanup
                del active_actors[case_to_remove]
                
                # Force garbage collection
                gc.collect()

    pbar.close()
    ray.shutdown()

    end = time.time()
    time_str = str(datetime.timedelta(seconds=end - start))
    print(f"Total run time for all {len(case_studies)} case studies: {time_str}\n")

if __name__ == "__main__":
    main()
