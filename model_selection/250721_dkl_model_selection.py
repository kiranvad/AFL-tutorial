import torch
import torch.nn as nn
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SASDataGenerator:
    """Generate synthetic Small Angle Scattering data for spherical particles"""
    
    def __init__(self, q_min=0.001, q_max=0.5, n_domain=100):
        self.q_min = q_min
        self.q_max = q_max
        self.n_domain = n_domain
        
    def sphere_form_factor(self, q, radius, sld_contrast=1.0):
        """
        Compute the form factor for a sphere
        P(q) = [3(sin(qr) - qr*cos(qr))/(qr)³]²
        """
        qr = q * radius
        # Avoid division by zero
        qr = torch.where(qr == 0, torch.tensor(1e-10), qr)
        
        # Spherical Bessel function of first kind, order 1
        j1 = (torch.sin(qr) - qr * torch.cos(qr)) / (qr**2)
        form_factor = (3 * j1 / qr)**2
        
        return sld_contrast**2 * form_factor
    
    def generate_curve(self, radius, sld_contrast=1.0, noise_level=0.05):
        """Generate a single SAS curve"""
        q = torch.logspace(np.log10(self.q_min), np.log10(self.q_max), self.n_domain)
        intensity = self.sphere_form_factor(q, radius, sld_contrast)
        
        # Add noise
        noise = torch.randn_like(intensity) * noise_level * intensity
        intensity_noisy = intensity + noise
        
        return q, intensity_noisy, intensity
    
    def generate_dataset(self, n_curves, radius_range=(10, 100), sld_range=(0.5, 2.0)):
        """Generate multiple SAS curves with different parameters"""
        curves = []
        params = []
        
        for i in range(n_curves):
            radius = torch.rand(1).item() * (radius_range[1] - radius_range[0]) + radius_range[0]
            sld_contrast = torch.rand(1).item() * (sld_range[1] - sld_range[0]) + sld_range[0]
            
            q, intensity, _ = self.generate_curve(radius, sld_contrast)
            curves.append((q, intensity))
            params.append((radius, sld_contrast))
            
        return curves, params

class DeepKernelNetwork(nn.Module):
    """Neural network feature extractor for DKL"""
    
    def __init__(self, input_dim=1, hidden_dims=[64, 64, 2]):
        super().__init__()
        layers = []
        prev_dim = input_dim 
        self.dim = hidden_dims[-1]
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Final layer (no activation)
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DKLModel(gpytorch.models.ExactGP):
    """Deep Kernel Learning model combining neural network with GP"""
    
    def __init__(self, train_x, train_y, likelihood, phi):
        super().__init__(train_x, train_y, likelihood)
        self.phi = phi
        
        # GP components
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Deep kernel: neural network features + Matern kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.phi.dim)
        )
        
    def forward(self, x):
        # Extract features using neural network
        projected_x = self.phi(x)
        
        # Apply GP
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DKLTrainer:
    """Trainer for the DKL model"""
    
    def __init__(self, model, likelihood, lr=0.01):
        self.model = model
        self.likelihood = likelihood
        params = [{'params': model.phi.parameters()},
                  {'params': model.covar_module.parameters()},
                  {'params': model.mean_module.parameters()},
                  {'params': likelihood.parameters()}
                ]
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
    def train(self, train_x, train_y, epochs=100, verbose=True):
        """Train the DKL model"""
        self.model.train()
        self.likelihood.train()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                     milestones=[0.5 * epochs], 
                                                     gamma=0.1)
        losses = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            losses.append(loss.item())
            
            if verbose and (epoch + 1) % 1 == 0:
                print(f'Epoch {epoch+1:3d}/{epochs:3d} - Loss: {loss.item():.4f}')
        
        return losses

def compute_predictive_probability(model, likelihood, x_test, y_test):
    """
    Compute p(y|x, M, θ) - the predictive probability
    """
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        pred_dist = model(x_test)
        pred_dist_with_noise = likelihood(pred_dist)
        log_prob = pred_dist_with_noise.log_prob(y_test)
        
        return log_prob, pred_dist_with_noise

def plot_training_results(losses, train_data, model, likelihood, data_generator):
    """Create comprehensive plots of training results"""
    fig, axes = plt.subplots(2, 2, figsize=(4*2, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    
    # Plot 1A: Training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Negative Log Likelihood')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    
    # Plot 1B: Feature space visualization
    model.eval()
    with torch.no_grad():
        train_x, train_y = train_data
        features = model.phi(train_x).numpy()
        
    scatter = axes[0, 1].scatter(features[:, 0], 
                                 features[:, 1], 
                                 c=train_y.numpy(), 
                                 cmap='viridis', 
                                 alpha=0.6
                            )
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    axes[0, 1].set_title('Learned Feature Space')
    plt.colorbar(scatter, ax=axes[0, 1], label='Intensity')

    # Plot 1C: Kernel visualization
    ax = axes[1,0]
    model.eval()
    with torch.no_grad():
        # Sample some points in feature space
        test_points = torch.logspace(np.log10(0.001), np.log10(0.5), 50)
        X, Y = torch.meshgrid(test_points, test_points, indexing='ij')
        test_features = torch.stack([X.flatten(), Y.flatten()], dim=0)
        
        # Compute kernel matrix for a reference point
        ref_point = torch.zeros(2, 1)
        import pdb 
        pdb.set_trace()
        kernel_values = model.covar_module(model.phi(ref_point), model.phi(test_features)).evaluate()

        kernel_matrix = kernel_values.reshape(50, 50)
    
    im = ax.imshow(kernel_matrix.numpy(), extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('DKL Kernel Function')
    plt.colorbar(im, ax=ax)

    # Plot 1D: Prior samples from trained model
    ax = axes[1,1]
    model.eval()
    with torch.no_grad():
        # Create test points
        test_x = torch.logspace(np.log10(0.001), np.log10(0.5), 100).unsqueeze(-1)
        # Sample from prior
        with gpytorch.settings.prior_mode(True):
            prior_dist = model(test_x)
            prior_samples = prior_dist.sample(sample_shape=torch.Size([32]))
    
    for i in range(32):
        pred_y = prior_samples[i].numpy()
        ax.plot(test_x.numpy(), pred_y, alpha=0.7)
    ax.set_xlabel('q-value')
    ax.set_ylabel('Intensity')
    ax.set_title('Prior Samples from Trained DKL Model')
    ax.set_xscale('log')
    ax.grid(True)
    plt.show()

    # Plot 2: Predictions on single curve in GP style
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(wspace=0.5)
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        q, Iq,_ = data_generator.generate_curve(10.0, sld_contrast=1.0, noise_level=0.05)
        print("Input sas data shapes: ", q.shape, Iq.shape)
        test_x = torch.tensor(q.numpy(), dtype=torch.float32)
        pred_dist = likelihood(model(test_x))
        pred_mean = pred_dist.mean.numpy()
        pred_std = pred_dist.stddev.numpy()
    
    ax.plot(q.numpy(), Iq.numpy(), 'ko', alpha=0.5, label='Training Data')
    ax.plot(q.numpy(), pred_mean.flatten(), 'b-', label='Mean Prediction')
    ax.fill_between(q.numpy().flatten(), 
                    (pred_mean - 2*pred_std).flatten(),
                    (pred_mean + 2*pred_std).flatten(),
                    alpha=0.3, 
                    color='blue', 
                    label='95% Confidence'
                )
    ax.set_xlabel('q-value')
    ax.set_ylabel('Intensity')
    ax.set_title('Training Data Fit')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function"""
    print("Generating synthetic SAS data...")
    
    # Generate synthetic data
    generator = SASDataGenerator(n_domain=100)
    curves, _ = generator.generate_dataset(n_curves=200, radius_range=(10, 100))
    
    # Prepare data
    all_x, all_y = [], []
    for q, intensity in curves:
        all_x.append(q)
        all_y.append(intensity)
    
    # Combine all data points
    train_x = torch.cat(all_x).unsqueeze(-1)
    train_y = torch.cat(all_y)
    
    print(f"Training data shape: {train_x.shape}, {train_y.shape}")
    
    # Initialize model components first
    phi = DeepKernelNetwork(input_dim=1, hidden_dims=[128, 128, 8])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # Create DKL model with training data
    model = DKLModel(train_x, train_y, likelihood, phi)
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    print("Training DKL model...")
    trainer = DKLTrainer(model, likelihood, lr=0.01)
    losses = trainer.train(train_x, train_y, epochs=2, verbose=True)
    
    # Create plots
    print("Creating plots...")
    fig = plot_training_results(losses, 
                                (train_x, train_y), 
                                model, 
                                likelihood, 
                                generator,
                            )
    
    plt.show()
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Feature extractor: {sum(p.numel() for p in phi.parameters())} parameters")
    print(f"GP kernel lengthscale: {model.covar_module.base_kernel.lengthscale}")
    print(f"GP output scale: {model.covar_module.outputscale.item():.4f}")
    print(f"Likelihood noise: {likelihood.noise.item():.4f}")
    
    return model, likelihood, fig

if __name__ == "__main__":
    model, likelihood, fig = main()