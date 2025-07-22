import torch
import gpytorch
from torch.autograd import grad
from torch.autograd.functional import hessian
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class GPHessianComputer:
    """
    Compute Hessian of log-likelihood for GP models w.r.t. parameters.
    
    Supports kernel parameters, mean function parameters, likelihood parameters,
    and inducing points (for sparse GPs).
    """
    
    def __init__(self, model: gpytorch.models.GP, likelihood: gpytorch.likelihoods.Likelihood):
        """
        Initialize Hessian computer.
        
        Args:
            model: GPyTorch GP model
            likelihood: GPyTorch likelihood
        """
        self.model = model
        self.likelihood = likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
    def get_parameter_dict(self) -> Dict[str, torch.Tensor]:
        """Get all trainable parameters as a dictionary."""
        param_dict = {}
        
        # Model parameters (kernel, mean function, etc.)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_dict[f"model.{name}"] = param
                
        # Likelihood parameters
        for name, param in self.likelihood.named_parameters():
            if param.requires_grad:
                param_dict[f"likelihood.{name}"] = param
                
        return param_dict
    
    def get_parameter_vector(self) -> torch.Tensor:
        """Get all trainable parameters as a single vector."""
        param_dict = self.get_parameter_dict()
        return torch.cat([p.flatten() for p in param_dict.values()])
    
    def set_parameter_vector(self, param_vector: torch.Tensor) -> None:
        """Set parameters from a vector."""
        param_dict = self.get_parameter_dict()
        idx = 0
        
        for name, param in param_dict.items():
            param_size = param.numel()
            param.data = param_vector[idx:idx + param_size].reshape(param.shape)
            idx += param_size
    
    def log_likelihood_function(self, param_vector: torch.Tensor, 
                              train_x: torch.Tensor, train_y: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood as a function of parameter vector.
        
        Args:
            param_vector: Flattened parameter vector
            train_x: Training inputs
            train_y: Training targets
            
        Returns:
            Log-likelihood value
        """
        # Set parameters
        original_params = self.get_parameter_vector().clone()
        self.set_parameter_vector(param_vector)
        
        try:
            # Compute log-likelihood
            self.model.train()
            self.likelihood.train()
            
            output = self.model(train_x)
            loss = self.mll(output, train_y)
            return loss
        finally:
            # Restore original parameters
            self.set_parameter_vector(original_params)
    
    def compute_hessian_autograd(self, train_x: torch.Tensor, 
                                train_y: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian using PyTorch's automatic differentiation.
        
        Args:
            train_x: Training inputs [n x d]
            train_y: Training targets [n]
            
        Returns:
            Hessian matrix [p x p] where p is number of parameters
        """
        param_vector = self.get_parameter_vector()
        param_vector.requires_grad_(True)
        
        # Use torch.autograd.functional.hessian
        def ll_fn(params):
            return self.log_likelihood_function(params, train_x, train_y)
        
        H = hessian(ll_fn, param_vector)
        return H
    
    def compute_hessian_manual(self, train_x: torch.Tensor, 
                              train_y: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian manually using second-order gradients.
        More memory efficient for large parameter spaces.
        
        Args:
            train_x: Training inputs [n x d]
            train_y: Training targets [n]
            
        Returns:
            Hessian matrix [p x p]
        """
        param_vector = self.get_parameter_vector()
        param_vector.requires_grad_(True)
        n_params = param_vector.shape[0]
        
        # Compute log-likelihood
        ll = self.log_likelihood_function(param_vector, train_x, train_y)
        
        # Compute first-order gradients
        first_grads = grad(ll, param_vector, create_graph=True)[0]
        
        # Compute Hessian row by row
        hessian_matrix = torch.zeros(n_params, n_params)
        
        for i in range(n_params):
            # Compute second derivatives w.r.t. i-th parameter
            second_grads = grad(first_grads[i], param_vector, 
                              retain_graph=(i < n_params - 1))[0]
            hessian_matrix[i] = second_grads
            
        return hessian_matrix
    
    def compute_hessian_finite_diff(self, train_x: torch.Tensor, 
                                   train_y: torch.Tensor, 
                                   eps: float = 1e-5) -> torch.Tensor:
        """
        Compute Hessian using finite differences (for verification).
        
        Args:
            train_x: Training inputs
            train_y: Training targets
            eps: Finite difference step size
            
        Returns:
            Hessian matrix
        """
        param_vector = self.get_parameter_vector()
        n_params = param_vector.shape[0]
        hessian_matrix = torch.zeros(n_params, n_params)
        
        for i in range(n_params):
            for j in range(n_params):
                # Compute f(x + ei*eps + ej*eps)
                params_pp = param_vector.clone()
                params_pp[i] += eps
                params_pp[j] += eps
                ll_pp = self.log_likelihood_function(params_pp, train_x, train_y)
                
                # Compute f(x + ei*eps - ej*eps)
                params_pm = param_vector.clone()
                params_pm[i] += eps
                params_pm[j] -= eps
                ll_pm = self.log_likelihood_function(params_pm, train_x, train_y)
                
                # Compute f(x - ei*eps + ej*eps)
                params_mp = param_vector.clone()
                params_mp[i] -= eps
                params_mp[j] += eps
                ll_mp = self.log_likelihood_function(params_mp, train_x, train_y)
                
                # Compute f(x - ei*eps - ej*eps)
                params_mm = param_vector.clone()
                params_mm[i] -= eps
                params_mm[j] -= eps
                ll_mm = self.log_likelihood_function(params_mm, train_x, train_y)
                
                # Central difference formula for mixed derivatives
                hessian_matrix[i, j] = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * eps**2)
                
        return hessian_matrix
    
    def compute_parameter_uncertainty(self, train_x: torch.Tensor, 
                                    train_y: torch.Tensor, 
                                    method: str = 'autograd') -> Dict[str, torch.Tensor]:
        """
        Compute parameter uncertainties from Hessian (Laplace approximation).
        
        Args:
            train_x: Training inputs
            train_y: Training targets
            method: Method to compute Hessian ('autograd', 'manual', 'finite_diff')
            
        Returns:
            Dictionary with parameter standard deviations
        """
        # Compute Hessian
        if method == 'autograd':
            H = self.compute_hessian_autograd(train_x, train_y)
        elif method == 'manual':
            H = self.compute_hessian_manual(train_x, train_y)
        elif method == 'finite_diff':
            H = self.compute_hessian_finite_diff(train_x, train_y)
        else:
            raise ValueError("Method must be 'autograd', 'manual', or 'finite_diff'")
        
        # Compute covariance matrix (inverse of negative Hessian)
        try:
            cov_matrix = torch.linalg.inv(-H)
            param_std = torch.sqrt(torch.diag(cov_matrix))
        except:
            # If Hessian is not invertible, use pseudo-inverse
            cov_matrix = torch.linalg.pinv(-H)
            param_std = torch.sqrt(torch.diag(cov_matrix))
        
        # Map back to parameter names
        param_dict = self.get_parameter_dict()
        param_names = list(param_dict.keys())
        
        uncertainties = {}
        idx = 0
        for name, param in param_dict.items():
            param_size = param.numel()
            param_std_reshaped = param_std[idx:idx + param_size].reshape(param.shape)
            uncertainties[name] = param_std_reshaped
            idx += param_size
            
        return uncertainties, cov_matrix


# Example usage and testing
def example_usage():
    """Example of how to use the GPHessianComputer."""
    
    # Generate synthetic data
    torch.manual_seed(42)
    train_x = torch.linspace(0, 1, 50).unsqueeze(-1)
    train_y = torch.sin(2 * torch.pi * train_x.squeeze()) + 0.1 * torch.randn(50)
    
    # Define GP model
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
            
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # Initialize model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    
    # Create Hessian computer
    hessian_computer = GPHessianComputer(model, likelihood)
    
    # Print parameter information
    param_dict = hessian_computer.get_parameter_dict()
    print("Trainable parameters:")
    for name, param in param_dict.items():
        print(f"  {name}: {param.shape}")
    
    # Compute Hessian using different methods
    print("\nComputing Hessian using autograd...")
    H_autograd = hessian_computer.compute_hessian_autograd(train_x, train_y)
    print(f"Hessian shape: {H_autograd.shape}")
    print(f"Hessian condition number: {torch.linalg.cond(H_autograd).item():.2e}")

    # Compute Hessian using different methods
    print("\nComputing Hessian using finite diff...")
    H_finite_diff = hessian_computer.compute_hessian_finite_diff(train_x, train_y)
    print(f"Hessian shape: {H_finite_diff.shape}")
    print(f"Hessian condition number: {torch.linalg.cond(H_finite_diff).item():.2e}")

    # Compute parameter uncertainties
    print("\nComputing parameter uncertainties...")
    uncertainties, cov_matrix = hessian_computer.compute_parameter_uncertainty(
        train_x, train_y, method='autograd'
    )
    
    print("Parameter uncertainties (standard deviations):")
    for name, std in uncertainties.items():
        print(f"  {name}: {std.item():.4f}")
    
    return hessian_computer, H_autograd, uncertainties


if __name__ == "__main__":
    # Run example
    hessian_computer, H, uncertainties = example_usage()
    
    # Additional analysis
    print(f"\nHessian eigenvalues: {torch.linalg.eigvals(H).real}")
    print(f"Hessian determinant: {torch.linalg.det(H).item():.2e}")