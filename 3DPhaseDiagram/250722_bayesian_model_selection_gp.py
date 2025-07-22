import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import hessian

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

def flatten_params(model):
    return torch.cat([p.flatten() for p in model.parameters()])

def set_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        new_data = flat_params[pointer:pointer + numel].view(p.shape)
        p.data = new_data
        pointer += numel

def finite_difference_hessian(f, x, epsilon=5e-3):
    """
    Compute the Hessian matrix of a scalar function f at input x using central finite differences.

    Args:
        f (callable): Function that takes a tensor x and returns a scalar tensor.
        x (torch.Tensor): Input tensor of shape (d,) with requires_grad=False.
        epsilon (float): Perturbation step size.

    Returns:
        hessian (torch.Tensor): Hessian matrix of shape (d, d).
    """
    x = x.detach().clone()
    d = x.numel()
    hessian = torch.zeros(d, d, dtype=x.dtype)

    for i in range(d):
        for j in range(d):
            x_ijp = x.clone()
            x_ijm = x.clone()
            x_imp_jp = x.clone()
            x_imp_jm = x.clone()

            x_ijp[i] += epsilon
            x_ijp[j] += epsilon

            x_ijm[i] += epsilon
            x_ijm[j] -= epsilon

            x_imp_jp[i] -= epsilon
            x_imp_jp[j] += epsilon

            x_imp_jm[i] -= epsilon
            x_imp_jm[j] -= epsilon

            f_pp = f(x_ijp)
            f_pm = f(x_ijm)
            f_mp = f(x_imp_jp)
            f_mm = f(x_imp_jm)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon ** 2)

    return hessian

def project_to_psd(H, tol=1e-8):
    """Projects a symmetric matrix H to the nearest PSD matrix."""
    # Symmetrize first
    H = (H + H.T) / 2
    # Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(H)
    # Clip negative eigenvalues
    eigvals_clipped = torch.clamp(eigvals, min=tol)
    # Reconstruct
    H_psd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T

    return H_psd

def laplace_log_evidence(model, likelihood, train_x, train_y, jitter=1e-8):
    theta_hat = flatten_params(model).requires_grad_(True)

    def log_lik_fn(params):
        set_params(model, params)
        output = model(train_x)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        return mll(output, train_y)
    
    H = finite_difference_hessian(log_lik_fn, theta_hat)
    H = 0.5 * (H + H.T) # make it symmetric
    d = H.shape[0]
    Sigma_inv = project_to_psd(-H, tol=jitter)

    sign, logdet = torch.slogdet(Sigma_inv)
    # assert sign > 0, "Hessian not PSD"
    # print(Sigma_inv)

    log_evidence = (
        log_lik_fn(theta_hat)
        - 0.5 * logdet
        + 0.5 * d * torch.log(torch.tensor(2 * np.pi))
    ).item()

    print(log_lik_fn(theta_hat), 0.5 * logdet, 0.5 * d * torch.log(torch.tensor(2 * np.pi)))
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

if __name__ == "__main__":
    torch.manual_seed(0)
    kernel_ground_truth = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.LinearKernel()
        )
    X, Y = sample_from_gp_prior(kernel_ground_truth, n=50)

    kernels = {
        "Periodic": gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()),
        "Linear": gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()),
        "Ground Truth": kernel_ground_truth,
    }

    fig, axs = plt.subplots(1, len(kernels), figsize=(15, 4))
    evidences = []
    for i, (name, kernel) in enumerate(kernels.items()):
        _,_,log_model_evidence = train_and_plot(kernel, name, X, Y.squeeze(), axs[i])
        evidences.append(log_model_evidence)
    model_posteriors = np.asarray(evidences)/sum(evidences)
    for i, (name, _) in enumerate(kernels.items()):
        print(f"p(M|D) for {name} = {model_posteriors[i]:.3f}")
    plt.tight_layout()
    plt.show()
