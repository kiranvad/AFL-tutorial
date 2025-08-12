import numpy as np
import torch 
from torch.utils.data import Dataset
from sasmodels.core import load_model
from sasmodels.direct_model import call_kernel, call_Fq

class SASDataset(Dataset):
    def __init__(self, generator, **params):
        curves, _ = generator.generate_dataset(**params)
        # Prepare data
        all_x, all_y = [], []
        for q, intensity in curves:
            all_x.append(q)
            all_y.append(intensity)
        # Combine all data points
        self.x = torch.cat(all_x).unsqueeze(-1)
        self.y = torch.cat(all_y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class SASModelSampler:
    """Generate synthetic SAS data by sampling parameters from a sasmodels model."""
    
    def __init__(self, model, q_min=0.001, q_max=0.5, n_domain=100):
        """
        model : sasmodels model object (from load_model)
        q_min, q_max : min/max q values
        n_domain : number of q-points
        """
        if isinstance(model, str):
            model = load_model(model)

        self.model = model
        self.model = model
        self.q_min = q_min
        self.q_max = q_max
        self.n_domain = n_domain

        q = np.logspace(np.log10(q_min), np.log10(q_max), n_domain)
        self.kernel = self.model.make_kernel([q])
        # Extract default parameters directly from model.info
        self.default_pars = {p.name: p.default for p in self.model.info.parameters.call_parameters}

    def _random_params(self, param_ranges=None):
        """
        Sample random parameters for the model.
        param_ranges : dict of {param_name: (min, max)}
        If None, uses +/-50% of default value for non-fixed parameters.
        """
        pars = self.default_pars.copy()
        for name, val in pars.items():
            # Skip scale and background unless ranges given explicitly
            if name in ("scale", "background") and not (param_ranges and name in param_ranges):
                continue

            if param_ranges and name in param_ranges:
                lo, hi = param_ranges[name]
            else:
                lo, hi = (
                    0.5 * val if val != 0 else 0.1,
                    1.5 * val if val != 0 else 1.0
                )
            pars[name] = np.random.uniform(lo, hi)
        return pars

    def generate_curve(self, param_ranges=None, noise_level=0.05):
        """
        Generate one SAS curve with noise.
        param_ranges : optional dict for parameter sampling bounds
        noise_level : relative Gaussian noise
        """
        q = np.logspace(np.log10(self.q_min), np.log10(self.q_max), self.n_domain)

        # Sample parameters
        pars = self._random_params(param_ranges)

        # Compute intensity from sasmodels
        intensity = call_kernel(self.kernel, pars)

        return torch.tensor(q, dtype=torch.float32), torch.tensor(intensity, dtype=torch.float32), pars

    def generate_dataset(self, n_curves, param_ranges=None, noise_level=0.05):
        """
        Generate multiple SAS curves from the model.
        """
        curves = []
        params = []
        for _ in range(n_curves):
            q, intensity, pars = self.generate_curve(param_ranges, noise_level)
            curves.append((q, intensity))
            params.append(pars)
        return curves, params

