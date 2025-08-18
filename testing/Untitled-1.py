# %%
import math
import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.priors import UniformPrior
import pyro
from pyro.infer.mcmc import NUTS, MCMC
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel 


def gen_data(num_data, seed = 2019):
    torch.random.manual_seed(seed)

    x = torch.randn(num_data,1)
    y = torch.randn(num_data,1)

    u = torch.rand(1)
    data_fn = lambda x, y: 1 * torch.sin(0.15 * u * 3.1415 * (x + y)) + 1
    latent_fn = data_fn(x, y)
    z = torch.round(latent_fn).long().squeeze()
    return torch.cat((x,y),dim=1), z, data_fn

train_x, train_y, genfn = gen_data(500)

test_d1 = np.linspace(-3, 3, 20)
test_d2 = np.linspace(-3, 3, 20)

test_x_mat, test_y_mat = np.meshgrid(test_d1, test_d2)
test_x_mat, test_y_mat = torch.Tensor(test_x_mat), torch.Tensor(test_y_mat)

test_x = torch.cat((test_x_mat.view(-1,1), test_y_mat.view(-1,1)),dim=1)
test_labels = torch.round(genfn(test_x_mat, test_y_mat))
test_y = test_labels.view(-1)

print(train_x.shape, train_y.shape, test_x.shape)

class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# %%
likelihood = DirichletClassificationLikelihood(
    train_y, learn_additional_noise=True
    )
model = DirichletGPModel(train_x, 
                         likelihood.transformed_targets, 
                         likelihood, 
                         num_classes=likelihood.num_classes
                    )

# %%
# Add priors for sampling 
model.mean_module.register_prior(
    "mean_prior", UniformPrior(-1, 1), "constant"
    )
model.covar_module.base_kernel.register_prior(
    "lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale"
    )
model.covar_module.register_prior(
    "outputscale_prior", UniformPrior(1, 2), "outputscale"
    )
likelihood.register_prior(
    "noise_prior", UniformPrior(0.01, 0.5), "second_noise"
    )

# %%
num_samples = 10
warmup_steps = 10

def pyro_model(x, y):
    with gpytorch.settings.fast_computations(False, False, False):
        sampled_model = model.pyro_sample_from_prior()
        output = sampled_model.likelihood(sampled_model(x))
        pyro.sample("obs", output, obs=y)
    return y

nuts = NUTS(pyro_model)
mcmc = MCMC(
    nuts, 
    num_samples=num_samples, 
    warmup_steps=warmup_steps, 
    disable_progbar=False
    )
mcmc.run(train_x, likelihood.transformed_targets)

# %%
# SOLUTION: Use the pyro_model directly for predictions
def make_predictions_with_pyro(mcmc_samples, train_x, train_y, test_x, model, likelihood):
    """
    Make predictions by sampling from the posterior using the pyro_model.
    This avoids all the batch dimension issues with pyro_load_from_samples.
    """
    all_predictions = []
    
    # Set model and likelihood to eval mode
    model.eval()
    likelihood.eval()
    
    mcmc_sample_keys = list(mcmc_samples.keys())
    num_mcmc_samples = mcmc_samples[mcmc_sample_keys[0]].shape[0]
    
    print(f"Using {num_mcmc_samples} MCMC samples for predictions")
    
    for i in range(num_mcmc_samples):
        print(f"Processing MCMC sample {i+1}/{num_mcmc_samples}")
        
        # Extract the i-th sample from each parameter
        current_sample = {key: val[i] for key, val in mcmc_samples.items()}
        
        # Create a new model instance for this sample (clean slate)
        sample_likelihood = DirichletClassificationLikelihood(
            train_y, learn_additional_noise=True
        )
        sample_model = DirichletGPModel(
            train_x, 
            sample_likelihood.transformed_targets, 
            sample_likelihood, 
            num_classes=sample_likelihood.num_classes
        )
        
        # Set the parameters manually (only the actual model parameters, not priors)
        sample_model.eval()
        sample_likelihood.eval()
        
        with torch.no_grad():
            # Set model parameters
            if 'mean_module.constant' in current_sample:
                sample_model.mean_module.constant.data = current_sample['mean_module.constant']
            
            if 'covar_module.base_kernel.lengthscale' in current_sample:
                sample_model.covar_module.base_kernel.lengthscale = current_sample['covar_module.base_kernel.lengthscale']
                
            if 'covar_module.outputscale' in current_sample:
                sample_model.covar_module.outputscale = current_sample['covar_module.outputscale']
                
            if 'likelihood.second_noise' in current_sample:
                sample_likelihood.second_noise = current_sample['likelihood.second_noise']
            
            # Make prediction with this sample
            pred_dist = sample_model(test_x)
            all_predictions.append(pred_dist.mean)  # Shape: [num_classes, num_test_points]
    
    # Stack all predictions: [num_mcmc_samples, num_classes, num_test_points]
    return torch.stack(all_predictions, dim=0)

# Get predictions
mcmc_samples = mcmc.get_samples()
print("Available MCMC sample keys:")
for key, val in mcmc_samples.items():
    print(f"  {key}: {val.shape}")

pred_means = make_predictions_with_pyro(
    mcmc_samples, train_x, train_y, test_x, model, likelihood
)
print(f"Predictions shape: {pred_means.shape}")

# %%
# Plot mean predictions across all MCMC samples
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Mean predictions
mean_pred = pred_means.mean(0)  # Average over MCMC samples: [num_classes, num_test_points]
for i in range(3):
    im = ax[0, i].contourf(
        test_x_mat.numpy(), 
        test_y_mat.numpy(), 
        mean_pred[i].numpy().reshape((20, 20))
    )
    fig.colorbar(im, ax=ax[0, i])
    ax[0, i].set_title(f"Mean Logits: Class {i}", fontsize=16)

# Standard deviation across MCMC samples
std_pred = pred_means.std(0)  # Std over MCMC samples: [num_classes, num_test_points]
for i in range(3):
    im = ax[1, i].contourf(
        test_x_mat.numpy(), 
        test_y_mat.numpy(), 
        std_pred[i].numpy().reshape((20, 20))
    )
    fig.colorbar(im, ax=ax[1, i])
    ax[1, i].set_title(f"Std Logits: Class {i}", fontsize=16)

plt.tight_layout()
plt.show()

# %%
# Convert to probabilities
# pred_means shape: [num_mcmc_samples, num_classes, num_test_points]
# Rearrange for softmax: [num_mcmc_samples, num_test_points, num_classes]
logits_transposed = pred_means.permute(0, 2, 1)
probabilities = torch.softmax(logits_transposed, dim=-1)  # Softmax over classes

# Average probabilities over MCMC samples: [num_test_points, num_classes]
prob_mean = probabilities.mean(0)
# Back to [num_classes, num_test_points] for plotting
prob_mean = prob_mean.permute(1, 0)

print(f"Probabilities shape: {prob_mean.shape}")

# Plot probabilities
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
levels = np.linspace(0, 1.0, 20)
for i in range(3):
    im = ax[i].contourf(
        test_x_mat.numpy(), 
        test_y_mat.numpy(), 
        prob_mean[i].numpy().reshape((20, 20)), 
        levels=levels
    )
    fig.colorbar(im, ax=ax[i])
    ax[i].set_title(f"Mean Probabilities: Class {i}", fontsize=16)

plt.tight_layout()
plt.show()

# %%
# Compute prediction uncertainty in probability space
prob_std = probabilities.std(0).permute(1, 0)  # [num_classes, num_test_points]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    im = ax[i].contourf(
        test_x_mat.numpy(), 
        test_y_mat.numpy(), 
        prob_std[i].numpy().reshape((20, 20))
    )
    fig.colorbar(im, ax=ax[i])
    ax[i].set_title(f"Probability Uncertainty: Class {i}", fontsize=16)

plt.tight_layout()
plt.show()


# Make predictions on training data to check model fit
train_pred_means = make_predictions_with_pyro(
    mcmc_samples, train_x, train_y, train_x, model, likelihood
)

# Convert to probabilities and get predicted classes
train_logits_transposed = train_pred_means.permute(0, 2, 1)
train_probabilities = torch.softmax(train_logits_transposed, dim=-1)
train_prob_mean = train_probabilities.mean(0)  # [num_train_points, num_classes]

# Get predicted classes
predicted_classes = train_prob_mean.argmax(dim=1)
accuracy = (predicted_classes == train_y).float().mean()
print(f"Training accuracy: {accuracy:.3f}")

# Plot training predictions vs true labels
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(train_x[:, 0], train_x[:, 1], c=train_y, s=20, alpha=0.7)
ax[0].set_title("True Training Labels")
ax[1].scatter(train_x[:, 0], train_x[:, 1], c=predicted_classes, s=20, alpha=0.7)
ax[1].set_title(f"Predicted Labels (Acc: {accuracy:.3f})")
plt.show()