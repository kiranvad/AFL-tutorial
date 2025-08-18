import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


class NeuralNetworkKernel(Kernel):
    """
    Neural network covariance function with a single parameter for the distance
    measure. The covariance function is parameterized as:
    
    k(x,z) = sf2 * asin(x'*P*z / sqrt[(1+x'*P*x)*(1+z'*P*z)])
    
    where the x and z vectors on the right hand side have an added extra bias
    entry with unit value. P is ell^-2 times the unit matrix and sf2 controls the
    signal variance.
    
    Args:
        lengthscale_prior: Prior for the lengthscale parameter
        outputscale_prior: Prior for the output scale parameter
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension
        active_dims: Set this if you want to compute the covariance of only a few input dimensions
    """
    
    has_lengthscale = True
    
    def __init__(
        self,
        lengthscale_prior=None,
        outputscale_prior=None,
        ard_num_dims=None,
        active_dims=None,
        **kwargs
    ):
        super(NeuralNetworkKernel, self).__init__(
            ard_num_dims=ard_num_dims, active_dims=active_dims, **kwargs
        )
        
        # Determine the number of lengthscale parameters
        if self.ard_num_dims is None:
            lengthscale_num_dims = 1
        else:
            lengthscale_num_dims = self.ard_num_dims
        
        # Register the lengthscale parameter
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
        )
        
        # Register the output scale parameter
        self.register_parameter(
            name="raw_outputscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )
        
        # Set up constraints
        self.register_constraint("raw_lengthscale", Positive())
        self.register_constraint("raw_outputscale", Positive())
        
        # Set priors
        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v),
            )
            
        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior", 
                outputscale_prior,
                lambda m: m.outputscale,
                lambda m, v: m._set_outputscale(v),
            )
    
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
    
    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)
        
    def _set_lengthscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))
    
    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)
    
    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)
        
    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the neural network covariance between x1 and x2.
        
        Args:
            x1: First set of data points of shape (..., n1, d)
            x2: Second set of data points of shape (..., n2, d) 
            diag: Whether to return only the diagonal elements
            
        Returns:
            Covariance matrix of shape (..., n1, n2) or diagonal of shape (..., n1) if diag=True
        """
        
        # Get parameters
        lengthscale = self.lengthscale
        outputscale = self.outputscale
        
        # Add bias term (unit value) to inputs
        # In the original MATLAB code, this is implicit through the 1 + x*x' computation
        x1_with_bias = torch.cat([x1, torch.ones(*x1.shape[:-1], 1, device=x1.device, dtype=x1.dtype)], dim=-1)
        x2_with_bias = torch.cat([x2, torch.ones(*x2.shape[:-1], 1, device=x2.device, dtype=x2.dtype)], dim=-1)
        
        # Scale by lengthscale (P = ell^-2 * I in the original formulation)
        # Handle both ARD and non-ARD cases
        if self.ard_num_dims is None:
            # Single lengthscale for all dimensions including bias
            lengthscale_expanded = lengthscale.expand(*lengthscale.shape[:-1], x1_with_bias.shape[-1])
        else:
            # ARD case - need to handle bias term
            bias_lengthscale = torch.ones(*lengthscale.shape[:-1], 1, device=lengthscale.device, dtype=lengthscale.dtype)
            lengthscale_expanded = torch.cat([lengthscale, bias_lengthscale], dim=-1)
        
        x1_scaled = x1_with_bias / lengthscale_expanded.sqrt()
        x2_scaled = x2_with_bias / lengthscale_expanded.sqrt()
        
        if diag:
            # Diagonal case: k(x_i, x_i)
            # sx = 1 + sum(x.*x, 2) in scaled coordinates becomes norm squared
            sx = torch.sum(x1_scaled.pow(2), dim=-1, keepdim=True)  # Shape: (..., n1, 1)
            A = sx / (sx + 1.0)  # Note: ell2 becomes 1 after scaling
            K_diag = outputscale * torch.asin(A.squeeze(-1))
            return K_diag
        else:
            # Full covariance matrix case
            # Compute norms
            sx = torch.sum(x1_scaled.pow(2), dim=-1, keepdim=True)  # Shape: (..., n1, 1)
            sz = torch.sum(x2_scaled.pow(2), dim=-1, keepdim=True)  # Shape: (..., n2, 1)
            
            # Compute inner products
            S = torch.matmul(x1_scaled, x2_scaled.transpose(-2, -1))  # Shape: (..., n1, n2)
            
            # Compute the argument to arcsin: S / sqrt[(1+sx)*(1+sz)]
            # After scaling, this becomes S / sqrt[(ell2+sx)*(ell2+sz)] with ell2=1
            denominator = torch.sqrt((1.0 + sx) * (1.0 + sz.transpose(-2, -1)))
            A = S / denominator
            
            # Clamp A to [-1, 1] to ensure numerical stability for arcsin
            A = torch.clamp(A, -1.0 + 1e-7, 1.0 - 1e-7)
            
            # Compute covariance
            K = outputscale * torch.asin(A)
            
            return K

def sample_from_gp_prior(mean_module, covar_module, x_test, num_samples=10):
    """
    Sample functions from a GP prior with robust numerical handling.
    """
    with torch.no_grad():
        # Create GP distribution
        mean = mean_module(x_test)
        covar = covar_module(x_test)
        
        # Convert to tensor if it's a lazy tensor
        if hasattr(covar, 'to_dense'):
            covar_dense = covar.to_dense()
        else:
            covar_dense = covar
        
        # Add substantial jitter for numerical stability
        jitter = 1e-4
        covar_dense = covar_dense + jitter * torch.eye(covar_dense.shape[-1])
        
        # Try Cholesky decomposition with increasing jitter if needed
        max_tries = 3
        for i in range(max_tries):
            try:
                # Test if matrix is positive definite by attempting Cholesky
                L = torch.linalg.cholesky(covar_dense)
                break
            except torch._C._LinAlgError:
                if i == max_tries - 1:
                    # If Cholesky still fails, use eigendecomposition
                    print(f"Warning: Using eigendecomposition fallback due to numerical issues")
                    eigenvals, eigenvecs = torch.linalg.eigh(covar_dense)
                    eigenvals = torch.clamp(eigenvals, min=jitter)  # Ensure positive eigenvalues
                    covar_dense = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
                    L = torch.linalg.cholesky(covar_dense)
                    break
                else:
                    # Increase jitter and try again
                    jitter *= 10
                    covar_dense = covar_dense + jitter * torch.eye(covar_dense.shape[-1])
                    print(f"Increasing jitter to {jitter:.2e}")
        
        # Sample using Cholesky decomposition
        z = torch.randn(num_samples, x_test.shape[0])
        samples = mean.unsqueeze(0) + torch.matmul(z, L.T)
        
        return samples, covar_dense


def main():
    """
    Demonstrate Neural Network kernel with GPyTorch prior model.
    """
    import numpy as np 
    import matplotlib.pyplot as plt 
    # Set random seed
    torch.manual_seed(42)
    
    # Create 1D test points
    x_test = torch.linspace(-3, 3, 80).unsqueeze(-1)
    
    # Create GP prior components
    mean_module = gpytorch.means.ZeroMean()
    covar_module = NeuralNetworkKernel()
    
    # Set kernel hyperparameters
    covar_module.lengthscale = 1.5
    covar_module.outputscale = 2.0
    
    print(f"Kernel hyperparameters:")
    print(f"  Lengthscale: {covar_module.lengthscale.item():.3f}")
    print(f"  Output scale: {covar_module.outputscale.item():.3f}")
    
    # Sample from GP prior
    samples, K = sample_from_gp_prior(mean_module, covar_module, x_test, num_samples=10)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: GP Prior samples
    ax1 = axes[0, 0]
    x_np = x_test.squeeze().numpy()
    
    for i in range(samples.shape[0]):
        ax1.plot(x_np, samples[i].numpy(), alpha=0.7, linewidth=1.5, label=f'Sample {i+1}' if i < 3 else '')
    
    ax1.set_title('GP Prior Samples (Neural Network Kernel)', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Covariance matrix heatmap
    ax2 = axes[0, 1]
    im = ax2.imshow(K.numpy(), cmap='viridis', aspect='auto', extent=[-3, 3, -3, 3])
    ax2.set_title('Covariance Matrix K(X, X)', fontsize=14)
    ax2.set_xlabel('x index')
    ax2.set_ylabel('x index')
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Kernel function k(x, 0)
    ax3 = axes[1, 0]
    x_origin = torch.zeros(1, 1)
    k_origin = covar_module(x_test, x_origin).squeeze(-1).detach().numpy()
    
    ax3.plot(x_np, k_origin, 'b-', linewidth=3, label='k(x, 0)')
    ax3.set_title('Kernel Function k(x, 0)', fontsize=14)
    ax3.set_xlabel('x')
    ax3.set_ylabel('Covariance')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Sample mean and variance
    ax4 = axes[1, 1]
    sample_mean = samples.mean(dim=0).numpy()
    sample_std = samples.std(dim=0).numpy()
    
    ax4.plot(x_np, sample_mean, 'r-', linewidth=2, label='Sample Mean')
    ax4.fill_between(x_np, sample_mean - 2*sample_std, sample_mean + 2*sample_std, 
                     alpha=0.3, color='red', label='±2 Sample Std')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='True Mean (0)')
    ax4.set_title('Sample Statistics', fontsize=14)
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nCovariance matrix statistics:")
    print(f"  Shape: {K.shape}")
    print(f"  Max value: {K.max().item():.4f}")
    print(f"  Min value: {K.min().item():.4f}")
    print(f"  Diagonal (variance): {torch.diag(K)[:5].numpy()}")
    
    # Show kernel behavior at different points
    print(f"\nKernel evaluation examples:")
    test_pairs = [
        (torch.tensor([[0.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[0.0]]), torch.tensor([[1.0]])),
        (torch.tensor([[0.0]]), torch.tensor([[2.0]])),
        (torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
    ]
    
    for x1, x2 in test_pairs:
        k_val = covar_module(x1, x2).item()
        print(f"  k({x1.item():.1f}, {x2.item():.1f}) = {k_val:.4f}")
    
    # Compare with different hyperparameters
    print(f"\n" + "="*50)
    print("HYPERPARAMETER COMPARISON")
    print("="*50)
    
    configs = [
        {"lengthscale": 0.5, "outputscale": 1.0, "name": "Short length, low scale"},
        {"lengthscale": 2.0, "outputscale": 3.0, "name": "Long length, high scale"},
    ]
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, config in enumerate(configs):
        # Set new hyperparameters
        covar_module.lengthscale = config["lengthscale"]
        covar_module.outputscale = config["outputscale"]
        
        # Sample new functions
        samples_new, _ = sample_from_gp_prior(mean_module, covar_module, x_test, num_samples=5)
        
        # Plot samples
        ax = axes2[i, 0]
        for j in range(samples_new.shape[0]):
            ax.plot(x_np, samples_new[j].numpy(), alpha=0.8, linewidth=1.5)
        ax.set_title(f'{config["name"]}\n(ell={config["lengthscale"]}, sf²={config["outputscale"]})')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.3)
        
        # Plot kernel function
        ax = axes2[i, 1]
        k_origin_new = covar_module(x_test, torch.zeros(1, 1)).squeeze().numpy()
        ax.plot(x_np, k_origin_new, 'b-', linewidth=2)
        ax.set_title(f'Kernel k(x, 0)')
        ax.set_xlabel('x')
        ax.set_ylabel('Covariance')
        ax.grid(True, alpha=0.3)
        
        print(f"\n{config['name']}:")
        print(f"  Max covariance: {k_origin_new.max():.4f}")
        print(f"  Effective range: ~{np.abs(x_np[k_origin_new > k_origin_new.max()/2]).max():.2f}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, fig2

if __name__ == "__main__":
    main()