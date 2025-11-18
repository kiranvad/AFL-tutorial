from typing import Optional, Tuple

import numpy as np

import torch
import geoopt

from funcshape.functions import Function
from torch.nn import Module
from botorch.models.transforms import Normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.posteriors.torch import TorchPosterior
from botorch.optim.initializers import initialize_q_batch_nonneg

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)

def make_grid_point_cloud(bounds_xy, bounds_t, nx=10, ny=10, nz=10):
    xs = torch.linspace(bounds_xy[0,0], bounds_xy[1,0], steps=nx, device=device)
    ys = torch.linspace(bounds_xy[0,1], bounds_xy[1,1], steps=ny, device=device)
    ft = torch.linspace(bounds_t[0], bounds_t[1], steps=nz, device=device)

    grid = torch.meshgrid(xs, ys, indexing="ij")
    points_xy = torch.stack(grid, dim=-1).reshape(-1, 2)
    grid_3d = torch.meshgrid(xs, ys, ft, indexing="ij")
    points_xyz = torch.stack(grid_3d, dim=-1).reshape(-1, 3)
    return points_xy, ft, points_xyz

def generate_random_samples(sim, n_samples, bounds_xy, n_temps, bounds_t):
    random_xy = draw_sobol_samples(bounds=bounds_xy, n=1, q=n_samples).squeeze(0)
    ft = torch.linspace(bounds_t[0], bounds_t[1], steps=n_temps).reshape(-1, 1)
    train_X, train_Y = [],[]
    for xy in random_xy:
        train_Y.append(sim(xy, ft))
        xy = xy.view(-1, 1, 2).expand(-1, len(ft), -1).squeeze(0)
        train_X.append(torch.cat([xy, ft], dim=1))

    train_X = torch.stack(train_X).reshape(-1, 3)
    train_Y = torch.stack(train_Y).reshape(-1, 1)

    return train_X, train_Y

class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, transform : Module = Normalize(d=3)):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )
        self.transform = transform

    def forward(self, x):
        x = self.transform(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def fit_dirchlet_gp(train_x, model, likelihood, n_iterations=200):
    model.train()
    likelihood.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    pbar = tqdm(range(n_iterations), desc="Fitting GP Model")
    for _ in pbar:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()

        pbar.set_postfix(
            loss=loss.item(), 
            lengthscale = model.covar_module.base_kernel.lengthscale.mean().item(),
            noise = model.likelihood.second_noise_covar.noise.mean().item()
        )
        optimizer.step()
    
    return model, likelihood

def PosteriorModel(X, model):
    model.eval()
    dist = model(X)
    return TorchPosterior(dist)

class SRSFHilbertSphere(geoopt.Manifold):
    """
    Hilbert Sphere manifold for functional data using SRSF representation.
    
    The Square Root Slope Function (SRSF) transformation maps functions to a
    Hilbert sphere, enabling efficient elastic shape analysis of functional data.
    
    For a function f: [0,1] -> R^d, the SRSF is defined as:
        q(t) = sign(f'(t)) * sqrt(|f'(t)|)
    
    The manifold structure is that of the unit sphere in L^2([0,1], R^d).
    
    Parameters
    ----------
    n_sampling_points : int
        Number of discrete sampling points for the functions
    ambient_dim : int, optional
        Dimension of the codomain (default: 1 for scalar functions)
    
    Attributes
    ----------
    name : str
        Name of the manifold
    reversible : bool
        Whether the exponential map is reversible
    ndim : int
        Number of dimensions
    """
    
    name = "SRSF Hilbert Sphere"
    reversible = False
    ndim = 1
    
    def __init__(self, n_sampling_points: int, ambient_dim: int = 1):
        super().__init__()
        self.n_sampling_points = n_sampling_points
        self.ambient_dim = ambient_dim
        self.dt = 1.0 / (n_sampling_points - 1)  # Time step for [0,1] discretization
        
    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol: float = 1e-5, rtol: float = 1e-5
    ) -> Tuple[bool, Optional[str]]:
        """Check if point lies on the Hilbert sphere (has unit L2 norm)."""
        try:
            self._check_shape(x, "x")
        except ValueError as e:
            return False, str(e)
        
        # Compute L2 norm using trapezoidal rule
        norm_sq = self._l2_inner(x, x)
        norm = torch.sqrt(norm_sq)
        
        ok = torch.allclose(norm, torch.ones_like(norm), atol=atol, rtol=rtol)
        if not ok:
            reason = f"L2 norm is {norm.mean().item():.6f}, expected 1.0"
            return False, reason
        return True, None
    
    def _check_vector_on_tangent(
        self, 
        x: torch.Tensor, 
        u: torch.Tensor, 
        *, 
        atol: float = 1e-5, 
        rtol: float = 1e-5
    ) -> Tuple[bool, Optional[str]]:
        """Check if vector u is tangent to the sphere at point x."""
        self._check_shape(x, "x")
        self._check_shape(u, "u")
        
        # Tangent vectors must be orthogonal to the point: <x, u> = 0
        inner_prod = self._l2_inner(x, u)
        
        ok = torch.allclose(
            inner_prod, torch.zeros_like(inner_prod), atol=atol, rtol=rtol
        )
        if not ok:
            reason = f"Inner product <x,u> = {inner_prod.mean().item():.6f}, expected 0.0"
            return False, reason
        return True, None
    
    def _l2_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 inner product using trapezoidal rule integration.
        
        Returns scalar for each batch element.
        """
        # Pointwise product
        pointwise = (x * y).sum(dim=-1)  # Sum over ambient dimensions
        
        # Trapezoidal rule: integral ≈ dt * (0.5*f[0] + f[1] + ... + f[n-2] + 0.5*f[n-1])
        integral = torch.zeros(pointwise.shape[:-1], device=x.device, dtype=x.dtype)
        integral = integral + 0.5 * pointwise[..., 0]
        integral = integral + pointwise[..., 1:-1].sum(dim=-1)
        integral = integral + 0.5 * pointwise[..., -1]
        integral = integral * self.dt
        
        return integral
    
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point onto the manifold (normalize to unit L2 norm).
        
        Parameters
        ----------
        x : torch.Tensor
            Point to project, shape (..., n_sampling_points, ambient_dim)
            
        Returns
        -------
        torch.Tensor
            Projected point on the Hilbert sphere
        """
        norm_sq = self._l2_inner(x, x)
        norm = torch.sqrt(norm_sq)
        
        # Add dimensions for broadcasting
        while norm.ndim < x.ndim:
            norm = norm.unsqueeze(-1)
        
        return x / (norm + 1e-8)
    
    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Project vector u onto the tangent space at point x.
        
        The tangent space at x consists of all vectors orthogonal to x.
        Projection: u_tangent = u - <u,x> * x
        
        Parameters
        ----------
        x : torch.Tensor
            Point on manifold, shape (..., n_sampling_points, ambient_dim)
        u : torch.Tensor
            Vector to project, shape (..., n_sampling_points, ambient_dim)
            
        Returns
        -------
        torch.Tensor
            Projected tangent vector
        """
        inner = self._l2_inner(u, x)
        
        # Add dimensions for broadcasting
        while inner.ndim < x.ndim:
            inner = inner.unsqueeze(-1)
        
        return u - inner * x
    
    def inner(
        self, 
        x: torch.Tensor, 
        u: torch.Tensor, 
        v: Optional[torch.Tensor] = None, 
        *, 
        keepdim: bool = False
    ) -> torch.Tensor:
        """
        Compute inner product in the tangent space at x.
        
        On the Hilbert sphere, the Riemannian metric is inherited from
        the ambient L2 space.
        
        Parameters
        ----------
        x : torch.Tensor
            Point on manifold
        u : torch.Tensor
            First tangent vector
        v : torch.Tensor, optional
            Second tangent vector (if None, compute norm of u)
        keepdim : bool
            Keep reduced dimensions
            
        Returns
        -------
        torch.Tensor
            Inner product value
        """
        if v is None:
            v = u
        
        result = self._l2_inner(u, v)
        
        if keepdim:
            while result.ndim < x.ndim:
                result = result.unsqueeze(-1)
        
        return result
    
    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, norm_u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Exponential map: move along geodesic from x in direction u.
        
        On the sphere, geodesics are great circles:
            exp_x(u) = cos(||u||) * x + sin(||u||) / ||u|| * u
        
        Parameters
        ----------
        x : torch.Tensor
            Base point on manifold
        u : torch.Tensor
            Tangent vector at x
        norm_u : torch.Tensor, optional
            Pre-computed norm of u
            
        Returns
        -------
        torch.Tensor
            Point on manifold reached by exponential map
        """
        if norm_u is None:
            norm_u = torch.sqrt(self._l2_inner(u, u))
        
        # Add dimensions for broadcasting
        norm_u_expanded = norm_u
        while norm_u_expanded.ndim < x.ndim:
            norm_u_expanded = norm_u_expanded.unsqueeze(-1)
        
        # Handle zero tangent vector
        mask = (norm_u > 1e-8).float()
        while mask.ndim < x.ndim:
            mask = mask.unsqueeze(-1)
        
        cos_norm = torch.cos(norm_u_expanded)
        sin_norm = torch.sin(norm_u_expanded)
        
        # When norm is small, sin(norm)/norm ≈ 1
        sinc = torch.where(
            norm_u_expanded > 1e-8,
            sin_norm / norm_u_expanded,
            torch.ones_like(norm_u_expanded)
        )
        
        return self.projx(cos_norm * x + sinc * u)
    
    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Retraction: fast approximate exponential map.
        
        Uses projection-based retraction: retr_x(u) = proj(x + u)
        
        Parameters
        ----------
        x : torch.Tensor
            Base point on manifold
        u : torch.Tensor
            Tangent vector at x
            
        Returns
        -------
        torch.Tensor
            Point on manifold
        """
        return self.projx(x + u)
    
    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map: compute tangent vector pointing from x to y.
        
        On the sphere:
            log_x(y) = (y - <x,y> * x) * arccos(<x,y>) / ||y - <x,y> * x||
        
        Parameters
        ----------
        x : torch.Tensor
            Base point on manifold
        y : torch.Tensor
            Target point on manifold
            
        Returns
        -------
        torch.Tensor
            Tangent vector at x pointing toward y
        """
        inner_xy = self._l2_inner(x, y)
        
        # Add dimensions for broadcasting
        inner_xy_expanded = inner_xy
        while inner_xy_expanded.ndim < x.ndim:
            inner_xy_expanded = inner_xy_expanded.unsqueeze(-1)
        
        # Project y onto tangent space at x
        u = y - inner_xy_expanded * x
        
        # Compute norm of projection
        norm_u = torch.sqrt(self._l2_inner(u, u))
        norm_u_expanded = norm_u
        while norm_u_expanded.ndim < x.ndim:
            norm_u_expanded = norm_u_expanded.unsqueeze(-1)
        
        # Compute angle
        angle = torch.acos(torch.clamp(inner_xy, -1.0 + 1e-7, 1.0 - 1e-7))
        angle_expanded = angle
        while angle_expanded.ndim < x.ndim:
            angle_expanded = angle_expanded.unsqueeze(-1)
        
        # Scale by angle/norm
        scale = torch.where(
            norm_u_expanded > 1e-8,
            angle_expanded / norm_u_expanded,
            torch.ones_like(norm_u_expanded)
        )
        
        return u * scale
    
    def transp(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport of vector v from tangent space at x to tangent space at y.
        
        Uses the projection-based parallel transport.
        
        Parameters
        ----------
        x : torch.Tensor
            Starting point
        y : torch.Tensor
            Ending point
        v : torch.Tensor
            Tangent vector at x
            
        Returns
        -------
        torch.Tensor
            Transported tangent vector at y
        """
        return self.proju(y, v)
    
    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Convert Euclidean gradient to Riemannian gradient.
        
        The Riemannian gradient is the projection of the Euclidean gradient
        onto the tangent space at x.
        
        Parameters
        ----------
        x : torch.Tensor
            Point on manifold
        u : torch.Tensor
            Euclidean gradient at x
            
        Returns
        -------
        torch.Tensor
            Riemannian gradient (tangent vector at x)
        """
        return self.proju(x, u)
        
    def origin(
        self, 
        *size, 
        dtype: Optional[torch.dtype] = None, 
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate the origin point (constant function with appropriate norm).
        
        Parameters
        ----------
        size : tuple
            Desired shape (excluding the manifold dimensions)
        dtype : torch.dtype, optional
            Data type
        device : torch.device, optional
            Device
            
        Returns
        -------
        torch.Tensor
            Origin point on the manifold
        """
        if dtype is None:
            dtype = torch.get_default_dtype()
        
        shape = size + (self.n_sampling_points, self.ambient_dim)
        x = torch.ones(shape, dtype=dtype, device=device)
        return self.projx(x)
    
    def function_to_srsf(self, f: torch.Tensor) -> torch.Tensor:
        """
        Convert function to its SRSF representation.
        
        SRSF(f)(t) = f'(t) / sqrt(|f'(t)|)
        
        Parameters
        ----------
        f : torch.Tensor
            Function values, shape (..., n_sampling_points, ambient_dim)
            
        Returns
        -------
        torch.Tensor
            SRSF representation (on the Hilbert sphere)
        """
        # Compute derivative using central differences
        df = torch.zeros_like(f)
        df[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * self.dt)
        df[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / self.dt
        df[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / self.dt
        
        # Compute SRSF
        df_norm = torch.sqrt((df ** 2).sum(dim=-1, keepdim=True))
        srsf = df / torch.sqrt(df_norm + 1e-8)
        
        # Project onto sphere
        return self.projx(srsf)
    
    def srsf_to_function(
        self, q: torch.Tensor, f0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert SRSF representation back to function (approximate).
        
        Integrates: f(t) = f(0) + integral(q(s) * |q(s)|, s=0..t)
        
        Parameters
        ----------
        q : torch.Tensor
            SRSF representation
        f0 : torch.Tensor, optional
            Initial value f(0), shape (..., ambient_dim)
            
        Returns
        -------
        torch.Tensor
            Reconstructed function
        """
        if f0 is None:
            f0 = torch.zeros(
                q.shape[:-2] + (self.ambient_dim,), 
                dtype=q.dtype, 
                device=q.device
            )
        # Now recover φ(t) = normalized ∫ ψ(s)^2 ds
        dphi = q**2                                         # (N,T,1)
        phi = torch.cumsum(dphi, dim=1) * self.dt                  # integrate
        phi = phi / phi[..., -1:, :]                          # enforce φ(1)=1
        phi[..., 0, :] = 0.0

        return phi
    
    def random(
        self,
        n_samples: int,
        n_basis: int = 10,
        sigma: float = 0.35,
        beta: float = 2.0,
        device=None,
        dtype=None,
    ):
        """
        Sample random monotone warpings φ(t) using Riemannian normal sampling
        on the SRD (Hilbert sphere) at the identity warp.

        Steps:
        1. Construct cosine basis e_k in T_p S^∞ (orthogonal to constant function).
        2. Sample Gaussian coefficients with spectrum λ_k ~ k^{-2β}.
        3. Form tangent vector v and map to sphere via expmap.
        4. Recover φ(t) = ∫ ψ(s)^2 ds normalized to φ(1)=1.

        Returns
        -------
        t : (T,)
        phi : (N, T, 1)
        psi : (N, T, 1)
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.get_default_dtype()

        T = self.n_sampling_points

        # Base point (identity warp SRD psi ≡ 1)
        p = self.origin(n_samples, dtype=dtype, device=device)  # (N,T,1)

        # Build cosine basis: e_k(t) = sqrt(2) cos(k pi t)
        t = torch.linspace(0., 1., T, device=device, dtype=dtype)
        k = torch.arange(1, n_basis + 1, device=device, dtype=dtype).view(-1, 1)
        e = np.sqrt(2.) * torch.cos(np.pi * k * t)        # (K,T)
        e = e.unsqueeze(-1)                                   # (K,T,1)

        # Normalize each basis vector w.r.t. L2
        # and ensure tangency: <e_k, p> = 0
        for i in range(n_basis):
            inner = self._l2_inner(p[0], e[i])  # scalar
            e[i] = e[i] - inner * p[0]          # enforce tangency
            norm = torch.sqrt(self._l2_inner(e[i], e[i]))
            e[i] = e[i] / (norm + 1e-8)

        # Spectral decay: λ_k ~ k^{-2β}
        lam = (sigma**2) / (k.squeeze() ** (2.0 * beta))      # (K,)

        # Sample Gaussian tangent vectors
        xi = torch.randn(n_samples, n_basis, device=device, dtype=dtype)
        coeffs = xi * torch.sqrt(lam).unsqueeze(0)            # (N,K)

        # Form v = sum coeffs_k * e_k
        v = torch.einsum("nk,ktd->ntd", coeffs, e)            # (N,T,1)

        # Ensure v is tangent numerically
        v = self.proju(p, v)

        # Exponential map → SRD ψ
        psi = self.expmap(p, v)                              # (N,T,1)

        return psi

def batch_acqusition_mc(xy, q, model, manifold, bounds_t):
    fx = manifold.srsf_to_function(q)
    t_values = ((fx*(bounds_t[1] - bounds_t[0])) + bounds_t[0])
    xy_expand = xy.view(-1, 1, 2).expand(t_values.shape[0], t_values.shape[1], -1)
    X = torch.cat([xy_expand, t_values], dim=-1).reshape(-1, 3)
    posterior = PosteriorModel(X, model)
    pred_samples = posterior.rsample(torch.Size((256,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
    probabilities = probabilities.reshape(-1, t_values.shape[0], t_values.shape[1])
    entropy = torch.special.entr(probabilities).sum(dim=0)

    return entropy

def batch_acqusition_xy(xy, model, manifold, bounds_t, n_samples = 32, **kwargs):
    q = manifold.random(n_samples)
    entropy = batch_acqusition_mc(xy, q, model, manifold, bounds_t, **kwargs)

    return entropy.mean()

def optimize_xy(bounds, acqf, n_iterations=200, n_restarts=16, device="cpu"):
    Xraw = draw_sobol_samples(bounds=bounds, n=n_restarts, q=1).squeeze(0).to(device)
    Yraw = torch.stack([acqf(xy) for xy in Xraw])
    X = initialize_q_batch_nonneg(Xraw, Yraw, 1)[0].squeeze(0)
    X.requires_grad_(True)
    optimizer = torch.optim.Adam([X], lr=0.1)
    X_traj = []
    pbar = tqdm(range(n_iterations), desc="Optimizing XY")
    for _ in pbar:
        X_traj.append(X.clone().detach())
        optimizer.zero_grad()
        loss = -acqf(X)
        loss.backward() 
        optimizer.step()

        # clamp values to the feasible set
        for i, (lb, ub) in enumerate(zip(*bounds)):
            X.data[..., i].clamp_(lb, ub) 
            
        grad_mean = (X.grad.mean().item() if X.grad is not None else float('nan'))
        pbar.set_postfix(loss=loss.item(), grad_mean=grad_mean)
    
    return X.clone().detach(), torch.stack(X_traj, dim=1).squeeze()

def optimize_f(model, manifold, xy_star, bounds_t, n_restarts=16, n_iterations=50, **kwargs):
    q_init = manifold.random(n_restarts)
    q = geoopt.ManifoldParameter(q_init, manifold=manifold)
    opt = geoopt.optim.RiemannianAdam([q], lr=0.1)
    traj = [q.clone().detach()]
    pbar = tqdm(range(n_iterations), desc="Optimizing q")
    loss_traj = []
    for _ in pbar:
        opt.zero_grad()
        acqv = -batch_acqusition_mc(xy_star, q, model, manifold, bounds_t, **kwargs)
        loss_traj.append(acqv.clone().detach())
        loss = acqv.mean(dim=1).sum() # mean over different t-values and sum over different batches
        loss.backward()
        opt.step()
        grad_mean = (q.grad.mean().item() if q.grad is not None else float('nan'))
        pbar.set_postfix(loss=loss.item(), grad_mean=grad_mean)
        traj.append(q.clone().detach())

    ind_star = acqv.mean(dim=1).argmin()    
    q_star = q[ind_star,...].clone().detach()
    fx_star = manifold.srsf_to_function(q_star.reshape(1,-1,1))

    f_star = Function(
        torch.linspace(0,1, steps=manifold.n_sampling_points),
        fx_star
    )

    return q_star, f_star, torch.stack(traj, dim=1).squeeze(), torch.stack(loss_traj, dim=1).squeeze(), ind_star

def get_temperature_profile(f_star, bounds_t, dt=0.05, dT=3.0):
    tt = torch.linspace(0.0, 1.0, int(np.floor(1/dt)))
    temps = f_star(tt)*(bounds_t[1]-bounds_t[0])+bounds_t[0]  
    rounded = (temps / dT).round() * dT
    t_values, indx = np.unique(rounded.squeeze().detach().numpy(), return_index=True)
    profile = Function(
            tt[indx],
            torch.from_numpy(t_values).clamp(bounds_t[0], bounds_t[1]).reshape(-1,1)
        )
    
    return profile