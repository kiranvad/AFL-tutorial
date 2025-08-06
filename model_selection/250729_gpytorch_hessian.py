import torch
import gpytorch
from torch.autograd.functional import hessian
from torch.nn.utils.stateless import _reparametrize_module

# Training data
train_x = torch.linspace(-1, 1, 10)
train_y = torch.sin(train_x * (2 * torch.pi)) + 0.2 * torch.randn(train_x.size())

# GP Model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(train_x, train_y, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter = 250
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=[0.5 * training_iter], 
                                                    gamma=0.1)
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if i%50==0:
        print(f"({i}/{training_iter})Log-likelihood loss {loss:.2f}")

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item()}")
    else:
        print(f"{name}: no gradient!")

# Collect and flatten model parameters
orig_params = dict(model.named_parameters())
shapes = {k: v.shape for k, v in orig_params.items()}
numels = {k: v.numel() for k, v in orig_params.items()}

def flatten(params_dict):
    return torch.cat([params_dict[k].reshape(-1) for k in orig_params])

def unflatten(flat_tensor):
    out = {}
    pointer = 0
    for k in orig_params:
        n = numels[k]
        out[k] = flat_tensor[pointer:pointer + n].reshape(shapes[k])
        pointer += n
    return out

# Define function that returns log-likelihood given theta
def log_likelihood_fn(theta):
    param_dict = unflatten(theta)
    with _reparametrize_module(model, param_dict):
        output = model(train_x)  # populates H[1, :], H[:, 1]
        normal = likelihood(output)  # populates H[0, :], H[:, 0]
        log_prob = normal.log_prob(train_y)  # populates H[2:, :], H[:, 2:]
    return log_prob

# Compute Hessian
flat_params = flatten(orig_params).detach().clone().requires_grad_(True)
log_lik_fn = lambda theta: log_likelihood_fn(theta)
H_v1 = hessian(log_lik_fn, flat_params)

print("Hessian shape:", H_v1.shape)
print("Hessian matrix:\n", H_v1)

# using autograd
def compute_hessian_autograd(model, likelihood, x, y):
    """Compute Hessian using pure autograd - more stable approach"""
    
    params = list(model.parameters())
    params_flat = torch.cat([p.flatten() for p in params])
    n_params = len(params_flat)
    output = model(x)
    log_lik = likelihood(output).log_prob(y)

    # Compute first derivatives
    first_grads = torch.autograd.grad(log_lik, params, create_graph=True, retain_graph=True)
    first_grads_flat = torch.cat([g.flatten() for g in first_grads])
        
    # Compute Hessian row by row
    hessian_rows = []
    for i in range(n_params):
        grad_outputs = torch.zeros_like(first_grads_flat)
        grad_outputs[i] = 1.0
        
        second_grads = torch.autograd.grad(
            outputs=first_grads_flat,
            inputs=params,
            grad_outputs=grad_outputs,
            retain_graph=True,
            allow_unused=True
        )
        
        # Handle None gradients (parameters not involved in computation)
        second_grads_flat = torch.cat([
            g.flatten() if g is not None else torch.zeros_like(p.flatten())
            for g, p in zip(second_grads, params)
        ])
        
        hessian_rows.append(second_grads_flat)
    
    hessian_matrix = torch.stack(hessian_rows)

    return hessian_matrix

H_v2 = compute_hessian_autograd(model, likelihood, train_x, train_y)
print("Hessian computed through plain autograd:\n", H_v2) 
