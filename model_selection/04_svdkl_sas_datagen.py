import matplotlib.pyplot as plt 
import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

from sasdata import SASModelSampler, SASDataset 

generator = SASModelSampler(model="cylinder")
dataset = SASDataset(generator, n_curves=100)
# Define split ratios (e.g., 80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = int(len(dataset) - train_size)
# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

fig, ax = plt.subplots()
ax.scatter(dataset.x.numpy(),
        dataset.y.numpy(), 
        alpha=0.1,
        color = "k"
    )
ax.set_title("Training data from SAS Generator")
ax.set_xlabel("q")
ax.set_ylabel("I(q)")
ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig("./input.png")
plt.close()

# 1. Define a feature extractor for input-output pairs {(q, Iq)}_{i=1}^{N}
class FeatureExtractor1D(nn.Module):
    """1D convolutional feature extractor for DKL"""
    
    def __init__(self, input_dim=1, hidden_dims=[64, 64, 2], kernel_size=3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        self.dim = hidden_dims[-1]
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Conv1d(prev_dim, 
                          hidden_dim, 
                          kernel_size=kernel_size, 
                          padding=kernel_size // 2
                        ),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final conv layer without activation
        layers.append(
            nn.Conv1d(prev_dim, 
                      hidden_dims[-1], 
                      kernel_size=kernel_size, 
                      padding=kernel_size // 2
                    )
        )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (batch, length, channels) -> convert to (batch, channels, length)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # add channel dim if missing
        elif x.ndim == 3 and x.shape[1] != self.network[0].in_channels:
            x = x.permute(0, 2, 1)  # swap channel and length
        return self.network(x).squeeze(-1)

    
# 2. Define a GP model that takes the feature extractor `phi` and creates a GP
class DKLModel(ApproximateGP):
    """Deep Kernel Learning model combining neural network with GP"""
    
    def __init__(self, inducing_points, phi):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, 
                                                   inducing_points, 
                                                   variational_distribution, 
                                                   learn_inducing_locations=True
                                                )
        super().__init__(variational_strategy)
        self.phi = phi
        
        # GP components
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Deep kernel: neural network features + Matern kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.phi.dim)
        )
        
    def forward(self, x):
        projected_x = self.phi(x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# 3. Define train and test module 
inducing_points = torch.linspace(generator.q_min, generator.q_max, 16).unsqueeze(-1)
phi = FeatureExtractor1D()
model = DKLModel(inducing_points=inducing_points, phi=phi)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

n_epochs = 100
lr = 0.1
optimizer = Adam([
    {'params': model.phi.parameters(), 'weight_decay': 1e-4},
    {'params': model.mean_module.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.covar_module.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))

def train(epoch):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())

def test():
    model.eval()
    likelihood.eval()
    error = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = likelihood(model(data)) 
            pred = output.loc
            error += ((pred-target.view_as(pred))**2).cpu().sum()
    print(f'Test MSE Error: {error/len(test_loader):.2f}')

for epoch in range(1, n_epochs + 1):
    with gpytorch.settings.use_toeplitz(False):
        train(epoch)
        test()
    scheduler.step()

# 4. Plot data generator from GP with the learned kernel.
model.eval()
likelihood.eval()
test_x = torch.linspace(generator.q_min, generator.q_max, 100)
with torch.no_grad():
    dist = model(test_x)
    samples = dist.sample(torch.Size([5]))

fig, ax = plt.subplots()
for i in range(samples.size(0)):
    ax.plot(test_x.numpy(), 
             samples[i].numpy(), 
             lw=1.5, 
             alpha=0.8,
             color = "k"
             )
ax.set_title("Samples from DKL-GP")
ax.set_xlabel("q")
ax.set_ylabel("I(q)")
ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig("./samples.png")
plt.close()