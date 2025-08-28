from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp

class MarginalEntropyOverDimension(PipelineOp):
    def __init__(
        self,
        input_variable: str = "probability",
        coordinate_dims :  List[str]= ['temperature'],
        component_dim: str = "ds_dim",
        grid_variable:str = "design_space_grid",
        output_variable: str = "composition_utility",
        name: str = "MarginalEntropyOverDimension",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable,
        )
        self.coordinate_dims = coordinate_dims
        self.grid_variable = grid_variable 
        self.component_dim = component_dim 


    def calculate(self, dataset: xr.Dataset) -> Self:
        grid = dataset[self.grid_variable]
        all_coordinate_dims = grid[self.component_dim].values.tolist()
        complement_dims = [d for d in all_coordinate_dims if d not in self.coordinate_dims]
        self.output_prefix = "_".join(i for i in complement_dims)

        grid_nonmarginal = grid.drop_sel({self.component_dim: self.coordinate_dims})
        unique_nonmarginal = grid_nonmarginal.to_pandas().drop_duplicates().reset_index(drop=True)

        x = unique_nonmarginal.values
        num_comps = len(unique_nonmarginal)
        ux = np.zeros(num_comps)

        Pr = dataset[self.input_variable].values 
        for i, xi in enumerate(x):
            # Find grid points for every unique non-marginal point
            squared_distances = np.sum((xi - grid_nonmarginal.values) ** 2, axis=1)
            idx = np.argwhere(squared_distances<1e-5) # indices for different coordinate_dims of x 
            Pr_xi = Pr[idx,:].squeeze() # probability at xi over different coordinate_dims
            Pr_marginal = Pr_xi.mean(axis=0) # marginalization over the coordinate_dims
            ux_i =  -np.sum(np.log(Pr_marginal) * Pr_marginal, axis=-1) # Marginal entropy as the utility
            ux[i] = ux_i.item()
  
        output = xr.DataArray(ux.squeeze(), dims=self._prefix_output("n"))
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Entropy calculated by marginalizing probability over certain dimension(s)" # type: ignore 

        domain_variable = self._prefix_output("domain")
        if not domain_variable in dataset:
            domain = xr.DataArray(x.reshape(-1, len(complement_dims)), 
                                dims=(self._prefix_output("n"), self._prefix_output("d")),
                                coords={self._prefix_output("d"): complement_dims}
                                )
            self.output[domain_variable] = domain
            self.output[domain_variable].attrs["description"] = f"Domain of the {self.output_variable} computed." # type: ignore

        return self    
    
class MarginalEntropyAlongDimension(PipelineOp):
    def __init__(
        self,
        input_variable: str = "probability",
        conditioning_point : str = "next_composition",
        coordinate_dim:str = "temperature",
        grid_variable:str = "design_space_grid",        
        component_dim: str = "ds_dim",
        output_variable:str="temperature",
        name: str = "MarginalEntropyAlongDimension",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable,
            output_prefix = f"{coordinate_dim}"
        )
        self.coordinate_dim = coordinate_dim
        self.conditioning_point = conditioning_point       
        self.grid_variable = grid_variable 
        self.component_dim = component_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        cp = dataset[self.conditioning_point]

        grid = dataset[self.grid_variable]
        all_coordinate_dims = grid[self.component_dim].values.tolist()
        complement_dims = [d for d in all_coordinate_dims if d not in self.coordinate_dim]
        grid_complement_dims = grid.sel({self.component_dim:complement_dims}).values

        squared_distances = np.sum((cp.values - grid_complement_dims) ** 2, axis=1)
        idx = np.argwhere(squared_distances<1e-5)
        T = grid.sel({self.component_dim:self.coordinate_dim}).values[idx]

        Pr = dataset[self.input_variable]

        # Extract utility at cp : u_{cp}(entropy_dim)
        Pr_cp = Pr.values[idx,:].copy() # type: ignore 
        vT =  -np.sum(np.log(Pr_cp) * Pr_cp, axis=-1) # Marginal entropy as the utility

        output = xr.DataArray(vT.squeeze(), dims = self._prefix_output("n"))
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Utility calculated along the temperature axis" # type: ignore 

        domain_variable = self._prefix_output("domain")

        if not domain_variable in dataset:
            domain = xr.DataArray(T.squeeze(), dims=self._prefix_output("n"))
            self.output[domain_variable] = domain
            self.output[domain_variable].attrs["description"] = f"Domain of the {self.output_variable} computed." # type: ignore

        return self