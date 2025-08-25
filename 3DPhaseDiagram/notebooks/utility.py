from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp
import textwrap

class MarginalEntropyOverDimension(PipelineOp):
    def __init__(
        self,
        input_variable: str = "acquisiton",
        coordinate_dims :  List[str]= ['protein', 'glycerol'],
        component_dim: str = "ds_dim",
        grid_variable:str = "design_space_grid",        
        output_variable: str = "composition_utility",
        name: str = "UtilityMarginalEntropy",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.coordinate_dims = coordinate_dims
        self.grid_variable = grid_variable 
        self.dim = component_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        grid = dataset[self.grid_variable]
        grid_nonmarginal = grid.sel({self.dim: self.coordinate_dims})
        unique_nonmarginal = grid_nonmarginal.to_pandas().drop_duplicates().reset_index(drop=True)

        x = unique_nonmarginal.values
        num_comps = len(unique_nonmarginal)
        ux = np.zeros(num_comps)

        Pr_yi_x = dataset[self.input_variable].values 
        for i, xi in enumerate(x):
            # Find grid points for every unique non-marginal point
            squared_distances = np.sum((xi - grid_nonmarginal.values) ** 2, axis=1)
            idx = np.argwhere(squared_distances<1e-5)
            Pr_yi_x_marginal = Pr_yi_x[idx,:].mean(axis=0) # marginalization over the rest of the dimensions
            ux_i =  -np.sum(np.log(Pr_yi_x_marginal) * Pr_yi_x_marginal, axis=-1) # Marginal entropy as the utility
            ux[i] = ux_i.item()
  
        coords = {"points": np.arange(num_comps)}
        for i, name in enumerate(self.coordinate_dims):
            coords[name] = ("points", x[:, i])
        output = xr.DataArray(ux, dims=("points",), coords=coords)
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Entropy calculated by marginalizing probability over certain dimension(s)" # type: ignore

        return self    
    
class MarginalEntropyAlongDimension(PipelineOp):
    def __init__(
        self,
        input_variable: str = "probability",
        conditioning_point : str = "next_composition",
        complement_coordinate_dims :  List[str]= ['protein', 'glycerol'],
        entropy_coordinate_dim:str = "temperature",
        grid_variable:str = "design_space_grid",        
        component_dim: str = "ds_dim",
        output_variable: str = "marginal_entropy_along_dim",
        name: str = "MarginalEntropyAlongDimension",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.complement_dims = complement_coordinate_dims
        self.entropy_dim = entropy_coordinate_dim
        self.conditioning_point = conditioning_point       
        self.grid_variable = grid_variable 
        self.dim = component_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        cp = dataset[self.conditioning_point]

        grid = dataset[self.grid_variable]
        grid_complement_dims = grid.sel({self.dim:self.complement_dims}).values
        grid_entropy_dim = grid.sel({self.dim:self.entropy_dim}).values

        squared_distances = np.sum((cp.values - grid_complement_dims) ** 2, axis=1)
        idx = np.argwhere(squared_distances<1e-5)

        Pr = dataset[self.input_variable]

        # Extract utility at cp : u_{cp}(entropy_dim)
        Pr_cp = Pr.values[idx,:].copy() # type: ignore 
        u =  -np.sum(np.log(Pr_cp) * Pr_cp, axis=-1) # Marginal entropy as the utility

        output = xr.DataArray(u.squeeze(), 
                              dims="entropy_dim",
                              coords={"entropy_dim": grid_entropy_dim[idx].squeeze()}
                            )
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Utility calculated along the temperature axis" # type: ignore

        return self