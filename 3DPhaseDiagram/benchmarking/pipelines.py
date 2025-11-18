import xarray as xr
import numpy as np 
from typing_extensions import Self 

from cost import DesignSpaceHierarchyCost, BinaryProbabilityCost, UtilityWithCost, MarginalCost, SlicedCost
from utility import MarginalEntropyAlongDimension, MarginalEntropyOverDimension 
from query_strategy import ArgMax, FullWidthHalfMaximum1D, MinMax1DLineSampler

from AFL.double_agent.PyTorchExtrapolator import DirichletGPExtrapolator 
from AFL.double_agent.Pipeline import Pipeline 
from AFL.double_agent.Generator import CartesianGrid 
from AFL.double_agent.Preprocessor import Standardize 
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.AcquisitionFunction import MaxValueAF

gp_params_mcmc_phase = {"num_samples":150, "num_warmup":50, "method":"mcmc", "verbose":False} 
gp_params_mcmc_feasible = {"num_samples":50, "num_warmup":25, "method":"mcmc", "verbose":False} 

class FeasibilityLabeler(PipelineOp):
    def __init__(
        self,
        input_variable: str = None,
        output_variable: str = None,
        dim: str = "sample",
        name: str = "FeasibilityLabeler",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=input_variable, 
            output_variable=output_variable,
        )
        self.dim = dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Label samples as feasible or infeasible based on cost."""
        labels = dataset[self.input_variable].values
        is_feasible = np.where(labels == 0, 0, 1)  # 1 for feasible, 0 for infeasible
        self.output[self.output_variable] = xr.DataArray(is_feasible, dims=[self.dim])
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Feasibility labels based on {self.input_variable}"

        return self 
    
class ExperimentCost(PipelineOp):
    def __init__(
        self,
        input_variable: str = None,
        input_dim : str = None,
        composition_dims_cost :  dict = None,
        temperature_dim_cost: dict[str, float] = None,
        output_variable: str = None,
        name: str = "ExperimentCost",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )
        self.composition_dims_cost = composition_dims_cost
        self.temperature_dim_cost = temperature_dim_cost
        self.input_dim = input_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset` using xarray operations only."""

        data = dataset[self.input_variable]
        expense = 0.0

        # Compute cost of using sample stocks
        composition_names = [key for key in self.composition_dims_cost.keys()]
        compositions = data.sel({self.input_dim:composition_names}).values
        _, indices = np.unique(compositions, axis=0, return_index=True)
        for name, val in self.composition_dims_cost.items():
            sampled_quantity = data.sel({self.input_dim:name}).values
            expense += (sampled_quantity[indices]*val).sum()

        # Compute cost of temperature modulation
        name, val = next(iter(self.temperature_dim_cost.items()))
        sampled_temperatures = data.sel({self.input_dim:name}).values
        for i in range(sampled_temperatures.shape[0]):
            if i>=1:
                expense += val*np.abs(sampled_temperatures[i] - sampled_temperatures[i-1])
            else:
                expense += val*np.abs(sampled_temperatures[i] - 27.0)

        # Store result
        output = xr.DataArray(expense)
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = "Total cost current experiment took so far."

        return self 

def _hierarchy_feasible_single_composition_batch_temperature(name, **kwargs):
    with Pipeline(name = name) as p:
        CartesianGrid(
            output_variable="design_space_grid",
            sample_dim="ds_grid",
            component_dim ="ds_dim",
            grid_spec = {'protein':{'min': 0.0, 'max': 80.0, 'steps': 20},
                        'glycerol':{'min': 0.0, 'max': 12.0, 'steps': 20},
                        'temperature':{'min': 0.0, 'max': 50.0, 'steps': 20}}
        )    
        Standardize(
            input_variable="design_space",
            output_variable="normalized_design_space",
            dim="sample",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize",
        )
        Standardize(
            input_variable="design_space_grid",
            output_variable="normalized_design_space_grid",
            dim="ds_grid",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize",
        )
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="phases",
            output_prefix="phase",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_phase,
            name="DirichletGPExtrapolator-PhaseLabels",
        )
        FeasibilityLabeler(
            input_variable="phases",
            output_variable="feasibility_labels",
            dim="sample",
            name="FeasibilityLabeler",
        )
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="feasibility_labels",
            output_prefix="feasibility",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_feasible,
            name="DirichletGPExtrapolator-FeasibilityLabels",
        )
        MarginalEntropyOverDimension(
            input_variable= "phase_y_prob",
            coordinate_dims= ['temperature'],
            component_dim= "ds_dim",
            grid_variable= "design_space_grid",        
            output_variable="composition_utility",
            name= "UtilityMarginalEntropy",
        )
        DesignSpaceHierarchyCost(
            input_variable = "normalized_design_space_grid",
            iteration_variable = "batch_sample_id",
            grid_variable = "design_space_grid",
            grid_dim = "ds_grid",
            component_dim = "ds_dim",
            coordinates_order = ['temperature', 'protein', 'glycerol'], # from highest to lowest cost
            coordinates_offsets = [0.5, 0.0, 0.0],
            output_variable="hierarchy_cost",
            name = "DesignSpaceHierarchyCost",        
        )
        MarginalCost(
            input_variable = "hierarchy_cost",
            coordinate_dims = ['temperature'],
            component_dim = "ds_dim",
            grid_variable = "design_space_grid",
            output_variable="composition_hierarchy_cost",
            name= "MarginalCost",
        )    
        UtilityWithCost(
            input_variable = "composition_utility",
            cost_variables = ["composition_hierarchy_cost"],
            output_variable = "composition_utility_with_cost",
            name = "AcquisitonWithCost-Composition",
        )
        ArgMax(
            input_variable="composition_utility_with_cost",
            grid_variable="_".join(i for i in ['protein', 'glycerol'])+"_domain",
            output_prefix= "composition",
            name= "QueryStrategyCompositionSpace",
        )
        MarginalEntropyAlongDimension(
            input_variable= "phase_y_prob",
            conditioning_point= "composition_next",
            coordinate_dim= "temperature",
            grid_variable= "design_space_grid",        
            component_dim= "ds_dim",
            output_variable="temperature_utility",
            name= "MarginalEntropyAlongDimension",
        )
        BinaryProbabilityCost(
            input_variable="feasibility_y_prob",
            grid_variable = "design_space_grid",
            cost_label = 0,
            dim="ds_grid",
            output_variable = "feasibility_cost",
            name = "BinaryProbabilityCost"
        )
        SlicedCost(
            input_variable = "feasibility_cost",
            conditioning_point = "composition_next",
            coordinate_dim = "temperature",
            grid_variable = "design_space_grid",        
            component_dim = "ds_dim",
            output_variable = "temperature_cost",
            name = "SlicedCost",
        )    
        UtilityWithCost(
            input_variable = "temperature_utility",
            cost_variables = ["temperature_cost"],
            output_variable = "temperature_utility_with_cost",
            name = "AcquisitonWithCost-Temperature",
        )    
        FullWidthHalfMaximum1D(
            input_variable= "temperature_utility_with_cost",
            grid_variable = "temperature_domain",
            output_prefix= "temperature",
            name= "FullWidthHalfMaximum1D",
        ) 
    
    def _runner(ind, ds_running, sim):
        ds_result = p.calculate(ds_running, disable_progress_bar=True)
        ds_new = []
        for t in ds_result.temperature_next.values:
            sample = {"protein": ds_result.composition_next.sel({"protein_glycerol_d":"protein"}).values.item(),
                    "glycerol" : ds_result.composition_next.sel({"protein_glycerol_d":"glycerol"}).values.item(),
                    "temperature" : t.item()
            }  
            ds = sim.simple_expose(sample)
            ds = ds.rename({'composition': 'design_space', 'component':'ds_dim'})
            ds['phases'] = int(ds.attrs['labels'])
            ds['batch_sample_id'] = ind
            ds_new.append(ds)

        ds_new = xr.concat(ds_new, dim='sample')
        ds_running = xr.concat([ds_running, ds_new], dim='sample')

        return ds_running, ds_result

    return _runner

def _hierarchy_single_composition_single_temperature(name, **kwargs):
    """ P1
    """
    with Pipeline(name = name) as p:
        CartesianGrid(
            output_variable="design_space_grid",
            sample_dim="ds_grid",
            component_dim ="ds_dim",
            grid_spec = {'protein':{'min': 0.0, 'max': 80.0, 'steps': 20},
                        'glycerol':{'min': 0.0, 'max': 12.0, 'steps': 20},
                        'temperature':{'min': 0.0, 'max': 50.0, 'steps': 20}}
        )    
        Standardize(
            input_variable="design_space",
            output_variable="normalized_design_space",
            dim="sample",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize",
        )
        Standardize(
            input_variable="design_space_grid",
            output_variable="normalized_design_space_grid",
            dim="ds_grid",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize-Grid",
        )
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="phases",
            output_prefix="phase",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_phase,
            name="DirichletGPExtrapolator-PhaseLabels",
        )
        DesignSpaceHierarchyCost(
            input_variable = "normalized_design_space_grid",
            iteration_variable = "batch_sample_id",
            grid_variable = "design_space_grid",
            grid_dim = "ds_grid",
            component_dim = "ds_dim",
            coordinates_order = ['temperature', 'protein', 'glycerol'], # from highest to lowest cost
            coordinates_offsets = [0.5, 0.0, 0.0],
            output_variable="hierarchy_cost",
            name = "DesignSpaceHierarchyCost",        
        )
        UtilityWithCost(
            input_variable = "phase_entropy",
            cost_variables = ["hierarchy_cost"],
            output_variable = "utility_with_cost",
            name = "UtilityWithCost",
        )
        MaxValueAF(
            input_variables=['utility_with_cost'],
            grid_variable="design_space_grid",
            grid_dim="ds_grid",
            combine_coeffs=None,
            output_prefix=None,
            output_variable="next_sample",
            decision_rtol=0.05,
            excluded_comps_variables="design_space",
            excluded_comps_dim="ds_dim",
            exclusion_radius=0.01,
            count=1,
            name="MaxValueAF",
        )
    
    def _runner(ind, ds_running, sim):
        ds_result = p.calculate(ds_running, disable_progress_bar=True)
        next_sample = ds_result['next_sample'].to_pandas().to_dict(orient='records')[0]
        ds_new = sim.simple_expose(next_sample)
        ds_new = ds_new.rename({'composition': 'design_space', 'component':'ds_dim'})
        ds_new['phases'] = int(ds_new.attrs['labels'])
        ds_new['batch_sample_id'] = ind
        ds_running = xr.concat([ds_running, ds_new], dim='sample') 

        return ds_running, ds_result 
    
    return _runner
        
def _hierarchy_feasible_single_composition_single_temperature(name, **kwargs):
    """ P2
    """
    with Pipeline(name = name) as p:
        CartesianGrid(
            output_variable="design_space_grid",
            sample_dim="ds_grid",
            component_dim ="ds_dim",
            grid_spec = {'protein':{'min': 0.0, 'max': 80.0, 'steps': 20},
                        'glycerol':{'min': 0.0, 'max': 12.0, 'steps': 20},
                        'temperature':{'min': 0.0, 'max': 50.0, 'steps': 20}}
        )    
        Standardize(
            input_variable="design_space",
            output_variable="normalized_design_space",
            dim="sample",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize",
        )
        Standardize(
            input_variable="design_space_grid",
            output_variable="normalized_design_space_grid",
            dim="ds_grid",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize-Grid",
        )
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="phases",
            output_prefix="phase",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_phase,
            name="DirichletGPExtrapolator-PhaseLabels",
        )
        FeasibilityLabeler(
            input_variable="phases",
            output_variable="feasibility_labels",
            dim="sample",
            name="FeasibilityLabeler",
        )
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="feasibility_labels",
            output_prefix="feasibility",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_feasible,
            name="DirichletGPExtrapolator-FeasibilityLabels",
        )
        DesignSpaceHierarchyCost(
            input_variable = "normalized_design_space_grid",
            iteration_variable = "batch_sample_id",
            grid_variable = "design_space_grid",
            grid_dim = "ds_grid",
            component_dim = "ds_dim",
            coordinates_order = ['temperature', 'protein', 'glycerol'], # from highest to lowest cost
            coordinates_offsets = [0.5, 0.0, 0.0],
            output_variable="hierarchy_cost",
            name = "DesignSpaceHierarchyCost",        
        )
        BinaryProbabilityCost(
            input_variable="feasibility_y_prob",
            grid_variable = "design_space_grid",
            cost_label = 0,
            dim="ds_grid",
            output_variable = "feasibility_cost",
            name = "BinaryProbabilityCost"
        )
        UtilityWithCost(
            input_variable = "phase_entropy",
            cost_variables = ["hierarchy_cost", "feasibility_cost"],
            output_variable = "utility_with_cost",
            name = "UtilityWithCost",
        )
        MaxValueAF(
            input_variables=['utility_with_cost'],
            grid_variable="design_space_grid",
            grid_dim="ds_grid",
            combine_coeffs=None,
            output_prefix=None,
            output_variable="next_sample",
            decision_rtol=0.05,
            excluded_comps_variables="design_space",
            excluded_comps_dim="ds_dim",
            exclusion_radius=0.01,
            count=1,
            name="MaxValueAF",
        )
    
    def _runner(ind, ds_running, sim):
        ds_result = p.calculate(ds_running, disable_progress_bar=True)
        next_sample = ds_result['next_sample'].to_pandas().to_dict(orient='records')[0]
        ds_new = sim.simple_expose(next_sample)
        ds_new = ds_new.rename({'composition': 'design_space', 'component':'ds_dim'})
        ds_new['phases'] = int(ds_new.attrs['labels'])
        ds_new['batch_sample_id'] = ind
        ds_running = xr.concat([ds_running, ds_new], dim='sample') 

        return ds_running, ds_result 
    
    return _runner

def _hierarchy_feasible_single_composition_adaptive_batch_temperature(name, **kwargs):
    with Pipeline(name = name+"_setup") as setup:
        CartesianGrid(
            output_variable="design_space_grid",
            sample_dim="ds_grid",
            component_dim ="ds_dim",
            grid_spec = {'protein':{'min': 0.0, 'max': 80.0, 'steps': 20},
                        'glycerol':{'min': 0.0, 'max': 12.0, 'steps': 20},
                        'temperature':{'min': 0.0, 'max': 50.0, 'steps': 20}}
        )    
        Standardize(
            input_variable="design_space",
            output_variable="normalized_design_space",
            dim="sample",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize",
        )
        Standardize(
            input_variable="design_space_grid",
            output_variable="normalized_design_space_grid",
            dim="ds_grid",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize-Grid",
        )

    with Pipeline(name = name+"_select_composition") as selcomp:
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="phases",
            output_prefix="phase",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_phase,
            name="DirichletGPExtrapolator-PhaseLabels",
        ) 
        DesignSpaceHierarchyCost(
            input_variable = "normalized_design_space_grid",
            iteration_variable = "batch_sample_id",
            grid_variable = "design_space_grid",
            grid_dim = "ds_grid",
            component_dim = "ds_dim",
            coordinates_order = ['temperature', 'protein', 'glycerol'], # from highest to lowest cost
            coordinates_offsets = [0.5, 0.0, 0.0],
            output_variable="hierarchy_cost",
            name = "DesignSpaceHierarchyCost",        
        )
        MarginalCost(
            input_variable = "hierarchy_cost",
            coordinate_dims = ['temperature'],
            component_dim = "ds_dim",
            grid_variable = "design_space_grid",
            output_variable="composition_hierarchy_cost",
            name= "MarginalCost",
        )
        MarginalEntropyOverDimension(
            input_variable= "phase_y_prob",
            coordinate_dims= ['temperature'],
            component_dim= "ds_dim",
            grid_variable= "design_space_grid",        
            output_variable="composition_utility",
            name= "UtilityMarginalEntropy",
        )    
        UtilityWithCost(
            input_variable = "composition_utility",
            cost_variables = ["composition_hierarchy_cost"],
            output_variable = "composition_utility_with_cost",
            name = "UtilityWithCost-Composition",
        )
        ArgMax(
            input_variable="composition_utility_with_cost",
            grid_variable="_".join(i for i in ['protein', 'glycerol'])+"_domain",
            output_prefix= "composition",
            name= "QueryStrategy-Composition",
        )
    with Pipeline(name = name+"_select_temperature") as seltemp:
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="phases",
            output_prefix="phasetemp",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_phase,
            name="DirichletGPExtrapolator-PhaseLabels-Temp",
        )     
        FeasibilityLabeler(
            input_variable="phases",
            output_variable="feasibility_labels",
            dim="sample",
            name="FeasibilityLabeler",
        )
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="feasibility_labels",
            output_prefix="feasibility",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_feasible,
            name="DirichletGPExtrapolator-FeasibilityLabels",
        )
        BinaryProbabilityCost(
            input_variable="feasibility_y_prob",
            grid_variable = "design_space_grid",
            cost_label = 0,
            dim="ds_grid",
            output_variable = "feasibility_cost",
            name = "BinaryProbabilityCost"
        )
        MarginalEntropyAlongDimension(
            input_variable= "phasetemp_y_prob",
            conditioning_point= "composition_next",
            coordinate_dim= "temperature",
            grid_variable= "design_space_grid",        
            component_dim= "ds_dim",
            output_variable="temperature_utility",
            name= "MarginalEntropyAlongDimension",
        ) 
        SlicedCost(
            input_variable = "feasibility_cost",
            conditioning_point = "composition_next",
            coordinate_dim = "temperature",
            grid_variable = "design_space_grid",        
            component_dim = "ds_dim",
            output_variable = "temperature_cost",
            name = "SlicedCost",
        )   
        UtilityWithCost(
            input_variable = "temperature_utility",
            cost_variables = ["temperature_cost"],
            output_variable = "temperature_utility_with_cost",
            name = "AcquisitonWithCost-Temperature",
        )
        MaxValueAF(
            input_variables=['temperature_utility_with_cost'],
            grid_variable="temperature_domain",
            grid_dim="temperature_n",
            combine_coeffs=None,
            output_prefix="temperature",
            output_variable="temperature_next",
            decision_rtol=0.05,
            excluded_comps_variables=None,
            excluded_comps_dim=None,
            exclusion_radius=5.0, # Not used
            count=1,
            name="MaxValueAF",
        )

    def _get_running_dataset(ds_old, ds_new):
        ds = ds_new.rename({'composition': 'design_space', 'component':'ds_dim'})
        ds['phases'] = int(ds.attrs['labels'])
        ds = xr.concat([ds], dim='sample')

        # Only append variables that have 'sample' in their dims
        sample_vars = [var for var in ds.data_vars if 'sample' in ds[var].dims]
        non_sample_vars = [var for var in ds_old.data_vars if 'sample' not in ds_old[var].dims]

        # Get the sample and non-sample data separately
        ds_running_samples = ds_old[sample_vars]
        ds_new_samples = ds[sample_vars]
        ds_running_non_samples = ds_old[non_sample_vars]

        # Manually concatenate sample-wise data
        ds_running_samples = xr.concat([ds_running_samples, ds_new_samples], dim='sample')

        # Rebuild ds_running with updated sample vars + preserved non-sample vars
        ds_running = xr.merge([ds_running_samples, ds_running_non_samples])

        return ds_running    
    
    def _runner(ind, ds_running, sim):
        ds_result = setup.calculate(ds_running, disable_progress_bar=True)
        ds_result = selcomp.calculate(ds_result, disable_progress_bar=True)
        ds_running = ds_result.copy()

        # Measure sample at the same composition from `selcomp` but at different temperatures.
        # We update the feasibility model each time, and sample a temperature that helps us
        # decode the feasibility boundary better.
        for _ in range(kwargs.get("n_temperatures", 5)):
            ds_result = setup.calculate(ds_running, disable_progress_bar=True)
            ds_result = seltemp.calculate(ds_result, disable_progress_bar=True)
            sample = {"protein": ds_result.composition_next.sel({"protein_glycerol_d":"protein"}).values.item(),
                    "glycerol" : ds_result.composition_next.sel({"protein_glycerol_d":"glycerol"}).values.item(),
                    "temperature" : ds_result.temperature_next.values.item()
                } 
            ds = sim.simple_expose(sample)
            ds['batch_sample_id'] = ind
            ds_running = _get_running_dataset(ds_running, ds)

        return ds_running, ds_result

    return _runner        

def _hierarchy_feasible_single_composition_linesample_temperature(name, **kwargs):
    with Pipeline(name = name) as p:
        CartesianGrid(
            output_variable="design_space_grid",
            sample_dim="ds_grid",
            component_dim ="ds_dim",
            grid_spec = {'protein':{'min': 0.0, 'max': 80.0, 'steps': 20},
                        'glycerol':{'min': 0.0, 'max': 12.0, 'steps': 20},
                        'temperature':{'min': 0.0, 'max': 50.0, 'steps': 20}}
        )    
        Standardize(
            input_variable="design_space",
            output_variable="normalized_design_space",
            dim="sample",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize",
        )
        Standardize(
            input_variable="design_space_grid",
            output_variable="normalized_design_space_grid",
            dim="ds_grid",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize",
        )
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="phases",
            output_prefix="phase",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_phase,
            name="DirichletGPExtrapolator-PhaseLabels",
        )
        FeasibilityLabeler(
            input_variable="phases",
            output_variable="feasibility_labels",
            dim="sample",
            name="FeasibilityLabeler",
        )
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="feasibility_labels",
            output_prefix="feasibility",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_feasible,
            name="DirichletGPExtrapolator-FeasibilityLabels",
        )
        MarginalEntropyOverDimension(
            input_variable= "phase_y_prob",
            coordinate_dims= ['temperature'],
            component_dim= "ds_dim",
            grid_variable= "design_space_grid",        
            output_variable="composition_utility",
            name= "UtilityMarginalEntropy",
        )
        DesignSpaceHierarchyCost(
            input_variable = "normalized_design_space_grid",
            iteration_variable = "batch_sample_id",
            grid_variable = "design_space_grid",
            grid_dim = "ds_grid",
            component_dim = "ds_dim",
            coordinates_order = ['temperature', 'protein', 'glycerol'], # from highest to lowest cost
            coordinates_offsets = [0.5, 0.0, 0.0],
            output_variable="hierarchy_cost",
            name = "DesignSpaceHierarchyCost",        
        )
        MarginalCost(
            input_variable = "hierarchy_cost",
            coordinate_dims = ['temperature'],
            component_dim = "ds_dim",
            grid_variable = "design_space_grid",
            output_variable="composition_hierarchy_cost",
            name= "MarginalCost",
        )    
        UtilityWithCost(
            input_variable = "composition_utility",
            cost_variables = ["composition_hierarchy_cost"],
            output_variable = "composition_utility_with_cost",
            name = "AcquisitonWithCost-Composition",
        )
        ArgMax(
            input_variable="composition_utility_with_cost",
            grid_variable="_".join(i for i in ['protein', 'glycerol'])+"_domain",
            output_prefix= "composition",
            name= "QueryStrategyCompositionSpace",
        )
        BinaryProbabilityCost(
            input_variable="feasibility_y_prob",
            grid_variable = "design_space_grid",
            cost_label = 1,
            dim="ds_grid",
            output_variable = "feasibility_cost",
            name = "BinaryProbabilityCost"
        )  
        SlicedCost(
            input_variable = "feasibility_cost",
            conditioning_point = "composition_next",
            coordinate_dim = "temperature",
            grid_variable = "design_space_grid",        
            component_dim = "ds_dim",
            output_variable = "temperature_utility",
            name = "SlicedUtility-Temperature",
        )
        MinMax1DLineSampler(
            input_variable= "temperature_utility",
            grid_variable= "temperature_domain",
            min_value= 0.33,
            max_value= 1.0,
            n_samples = kwargs.get("n_temperatures", 5),
            output_prefix ="temperature",
            name = "TemperatureLineSampler",
            )
        
    def _runner(ind, ds_running, sim):
        ds_result = p.calculate(ds_running, disable_progress_bar=True)
        ds_new = []
        for t in ds_result.temperature_next.values:
            sample = {"protein": ds_result.composition_next.sel({"protein_glycerol_d":"protein"}).values.item(),
                    "glycerol" : ds_result.composition_next.sel({"protein_glycerol_d":"glycerol"}).values.item(),
                    "temperature" : t.item()
            }  
            ds = sim.simple_expose(sample)
            ds = ds.rename({'composition': 'design_space', 'component':'ds_dim'})
            ds['phases'] = int(ds.attrs['labels'])
            ds['batch_sample_id'] = ind
            ds_new.append(ds)

        ds_new = xr.concat(ds_new, dim='sample')
        ds_running = xr.concat([ds_running, ds_new], dim='sample')

        return ds_running, ds_result

    return _runner

def _single_composition_single_temperature(name, **kwargs):
    """ P0
    """
    with Pipeline(name = name) as p:
        CartesianGrid(
            output_variable="design_space_grid",
            sample_dim="ds_grid",
            component_dim ="ds_dim",
            grid_spec = {'protein':{'min': 0.0, 'max': 80.0, 'steps': 20},
                        'glycerol':{'min': 0.0, 'max': 12.0, 'steps': 20},
                        'temperature':{'min': 0.0, 'max': 50.0, 'steps': 20}}
        )    
        Standardize(
            input_variable="design_space",
            output_variable="normalized_design_space",
            dim="sample",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize",
        )
        Standardize(
            input_variable="design_space_grid",
            output_variable="normalized_design_space_grid",
            dim="ds_grid",
            component_dim="ds_dim",
            scale_variable=None,
            min_val={'protein': 0.0, 'glycerol': 0.0, 'temperature': 0.0},
            max_val={'protein': 80.0, 'glycerol': 12.0, 'temperature': 50.0},
            name="Standardize-Grid",
        )
        DirichletGPExtrapolator(
            feature_input_variable="normalized_design_space",
            predictor_input_variable="phases",
            output_prefix="phase",
            grid_variable="normalized_design_space_grid",
            grid_dim="ds_grid",
            sample_dim="sample",
            component_dim = "ds_dim",
            params=gp_params_mcmc_phase,
            name="DirichletGPExtrapolator-PhaseLabels",
        )
        MaxValueAF(
            input_variables=['phase_entropy'],
            grid_variable="design_space_grid",
            grid_dim="ds_grid",
            combine_coeffs=None,
            output_prefix=None,
            output_variable="next_sample",
            decision_rtol=0.05,
            excluded_comps_variables="design_space",
            excluded_comps_dim="ds_dim",
            exclusion_radius=0.01,
            count=1,
            name="MaxValueAF",
        )
    
    def _runner(ind, ds_running, sim):
        ds_result = p.calculate(ds_running, disable_progress_bar=True)
        next_sample = ds_result['next_sample'].to_pandas().to_dict(orient='records')[0]
        ds_new = sim.simple_expose(next_sample)
        ds_new = ds_new.rename({'composition': 'design_space', 'component':'ds_dim'})
        ds_new['phases'] = int(ds_new.attrs['labels'])
        ds_new['batch_sample_id'] = ind
        ds_running = xr.concat([ds_running, ds_new], dim='sample') 

        return ds_running, ds_result 
    
    return _runner

def get_runner(name, **kwargs):
    if name == "P0":
        return _single_composition_single_temperature(name, **kwargs)
    elif name == "P1":
        return _hierarchy_single_composition_single_temperature(name, **kwargs)
    elif name == "P2":
        return _hierarchy_feasible_single_composition_single_temperature(name, **kwargs)
    elif name == "P3":
        return _hierarchy_feasible_single_composition_batch_temperature(name, **kwargs)
    elif name == "P4":
        return _hierarchy_feasible_single_composition_adaptive_batch_temperature(name, **kwargs)
    elif name=="P5":
        return _hierarchy_feasible_single_composition_linesample_temperature(name, **kwargs)
    else:
        raise RuntimeError(f"No pipeline can be associated with {name}")