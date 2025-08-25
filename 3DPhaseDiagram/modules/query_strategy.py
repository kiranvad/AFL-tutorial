from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp
import textwrap
from scipy.signal import find_peaks, peak_widths

class ArgMaxOverDimension(PipelineOp):
    def __init__(
        self,
        input_variable: str = "composition_utility",
        coordinate_dims :  List[str] = ["protein", "glycerol"],
        output_variable: str = "next_sample",
        name: str = "QueryStrategyCompositionSpace",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.coordinate_dims = coordinate_dims

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Implements argmax query strategy.

        Requires that the utility passed using `input_variable` has
        coordinates that match the input `dims`
        """
        ux = dataset[self.input_variable]
        x = np.column_stack([ux.coords[name].values for name in self.coordinate_dims])
        x_opt = x[np.argmax(ux.values).item(),:] # x^* = argmax_{x} u(x)

        output = xr.DataArray(x_opt.reshape(-1, len(self.coordinate_dims)),
                              dims=("n_counts", "comp_dim"),
                              coords={"comp_dim": self.coordinate_dims}
                            )
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = textwrap.dedent("""
        Optimal composition to be added to the dataset
        that maximizes the utility.
        """).strip()
        return self

class FullWidthHalfMaximum1D(PipelineOp):
    def __init__(
        self,
        input_variable: str = "utility",
        coordinate_dim: str = "entropy_dim",
        output_variable: str = "next_sample",
        name: str = "FullWidthHalfMaximum1D",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.coordinate_dim = coordinate_dim
    
    def calculate(self, dataset: xr.Dataset) -> Self:
        u = dataset[self.input_variable]
        x = u.coords[self.coordinate_dim]

        x_opt = self.optimize(x, u.values)
        
        output = xr.DataArray(np.asarray(x_opt).reshape(-1),
                              dims=("n_next"),
                            )
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = textwrap.dedent("""
        Optimal 1D locations that are the full-width-half-maximum points of the
        peaks in the utility.
        """).strip()
        return self

    def optimize(self, x, f, **kwargs):
        """
        Find peaks in f(x) and compute their full width at half maximum (FWHM).

        If no clear peaks are found, returns the location of the global maximum.

        Parameters
        ----------
        x : array-like
            1D array of x values.
        f : array-like
            1D array of function values f(x).
        **kwargs : dict
            Extra keyword arguments passed to scipy.signal.find_peaks,
            e.g. height=..., distance=..., prominence=...

        Returns
        -------
        xb : ndarray
            Sorted array of x values. For each peak, three values are returned:
            [left_half_max, peak_max, right_half_max].
            If no peaks are found, returns [x[argmax(f)]].
        """
        x = np.asarray(x)
        f = np.asarray(f).squeeze()

        # Find peak indices
        peaks, _ = find_peaks(f, **kwargs)

        if len(peaks) > 0:
            # Compute widths at half maximum
            results_half = peak_widths(f, peaks, rel_height=0.5)

            # Left and right half-max positions 
            # (fractional indices, interpolate to x)
            left_ips = results_half[2]
            right_ips = results_half[3]

            xb = []
            for i, peak_idx in enumerate(peaks):
                left_x = np.interp(left_ips[i], np.arange(len(x)), x)
                peak_x = x[peak_idx]
                right_x = np.interp(right_ips[i], np.arange(len(x)), x)
                xb.extend([left_x, peak_x, right_x])

            xb = np.sort(np.array(xb))
        else:
            xb = np.array([x[np.argmax(f)]])  # fallback to global max

        return xb