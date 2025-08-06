import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import hessian
from torch.nn.utils.stateless import _reparametrize_module

from kernels import NeuralNetworkKernel 

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

def compute_hessian_autograd(model, likelihood, x, y):
    """Compute Hessian using pure autograd - more stable approach"""
    
    params = list(model.parameters())
    params_flat = torch.cat([p.flatten() for p in params])
    n_params = len(params_flat)
    output = model(x)
    
    # Compute log likelihood with create_graph=True for second derivatives
    model.zero_grad()
    log_lik = likelihood(output).log_prob(y)

    # Compute first derivatives
    first_grads = torch.autograd.grad(log_lik, params, create_graph=True, retain_graph=True)
    first_grads_flat = torch.cat([g.flatten() for g in first_grads])
        
    # Compute Hessian row by row
    hessian_rows = []
    for i in range(n_params):
        # Compute gradient of the i-th component of the gradient
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

    return hessian_matrix, log_lik

def get_hessian(model, likelihood, x, y):
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
            output = model(x)  # populates H[1, :], H[:, 1]
            normal = likelihood(output)  # populates H[0, :], H[:, 0]
            log_prob = normal.log_prob(y)  # populates H[2:, :], H[:, 2:]
        return log_prob

    # Compute Hessian
    flat_params = flatten(orig_params).detach().clone().requires_grad_(True)
    log_lik_fn = lambda theta: log_likelihood_fn(theta)
    H = hessian(log_lik_fn, flat_params)

    return H, log_lik_fn(flat_params)

def project_to_psd(H):
    """Projects a symmetric matrix H to the nearest PSD matrix.
    
    This function makes the stability modification of 
    Hessian based on arXiv:2403.09215
    """
    H = (H + H.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(H)
    eigvals_clipped = torch.clamp(eigvals, min=2*torch.pi)
    H_psd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T

    return H_psd

def laplace_log_evidence(model, likelihood, train_x, train_y):
    model.eval()
    likelihood.eval()
    # H, log_lik = compute_hessian_autograd(model, likelihood, train_x, train_y)
    H, log_lik = get_hessian(model, likelihood, train_x, train_y)
    H = project_to_psd(H)
    d = H.shape[0]
    Sigma_inv = -H

    sign, logdet = torch.linalg.slogdet(Sigma_inv)

    log_evidence = (
        log_lik
        - 0.5 * sign*logdet
        + 0.5 * d * torch.log(torch.tensor(2 * np.pi))
    ).item()

    print("Model evidence terms:" , log_lik, 0.5 * sign*logdet, 0.5 * d * torch.log(torch.tensor(2 * np.pi)))
    return log_evidence

def train_and_plot(kernel, name, train_x, train_y, ax):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, kernel)

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 250
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[0.5 * training_iter], 
                                                     gamma=0.1)
    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Compute evidence
    log_evidence = laplace_log_evidence(model, likelihood, train_x, train_y)

    # Predictive posterior
    model.eval(); likelihood.eval()
    test_x = torch.linspace(-3.0, 3.0, 200)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))

    mean = pred.mean.numpy()
    lower, upper = pred.confidence_region()
    lower = lower.numpy()
    upper = upper.numpy()

    # Plot
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*', label="Train")
    ax.plot(test_x.numpy(), mean, 'b', label="Predictive Mean")
    ax.fill_between(test_x.numpy(), lower, upper, alpha=0.3, label="±2 std")
    ax.set_title(f"{name} kernel\nlog p(y|x, M) ≈ {log_evidence:.2f}")
    ax.legend()

    return model, likelihood, log_evidence

def sample_from_gp_prior(kernel, n=30, noise_std=0.05):
    x_min, x_max = -3.0, 3.0
    X = (x_max - x_min) * torch.rand(n, 1) + x_min  # shape: (n, 1)

    # Mean and kernel (GP prior)
    mean = gpytorch.means.ZeroMean()
    cov = kernel(X)

    # Construct multivariate normal
    mvn = gpytorch.distributions.MultivariateNormal(mean(X), cov)
    Y = mvn.sample()

    # Add observation noise
    Y_noisy = Y + noise_std * torch.randn_like(Y)

    return X, Y_noisy 

def get_model_posteriors(log_model_evidences):
    log_post_unnorm = torch.tensor(log_model_evidences) 

    # For numerical stability, use log-sum-exp trick
    log_post_norm_const = torch.logsumexp(log_post_unnorm, dim=0)

    log_posteriors = log_post_unnorm - log_post_norm_const
    model_posteriors = torch.exp(log_posteriors)  # p(M_i | D), sum to 1 

    return model_posteriors


if __name__ == "__main__":
    # torch.manual_seed(1234)
    # kernel_ground_truth = gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.PeriodicKernel()
    #     )
    # kernel_ground_truth = gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.PeriodicKernel()
    #     )

    kernel_ground_truth = gpytorch.kernels.ScaleKernel(
        NeuralNetworkKernel()
        )
    
    X, Y = sample_from_gp_prior(kernel_ground_truth, n=50)

    kernels = {
        "Periodic": gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()),
        "Linear": gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()),
        "Matern 3/2": gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)),
        "Ground Truth": kernel_ground_truth,
    }

    fig, axs = plt.subplots(1, len(kernels), figsize=(4*len(kernels), 4))
    log_evidences = []
    for i, (name, kernel) in enumerate(kernels.items()):
        _,_,log_model_evidence = train_and_plot(kernel, name, X, Y.squeeze(), axs[i])
        log_evidences.append(log_model_evidence)
        print(f"Log-Model evidence p(y|X, M) for {name} is {log_model_evidence:.3f}")

    model_posteriors = get_model_posteriors(log_evidences)
    for i, (name, _) in enumerate(kernels.items()):
        print(f"p(M|D) for {name} = {model_posteriors[i]:.3f}")
    plt.tight_layout()
    plt.show()
