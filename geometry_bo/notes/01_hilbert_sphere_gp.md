# Gaussian Processes on the Hilbert Sphere of Warping Functions — theory

## 1. Problem setting

Let $\Gamma$ be the set of warping functions

$$
\gamma : [0,1]\to[0,1], \qquad \gamma(0)=0,\ \gamma(1)=1,\ \gamma'\ge 0.
$$

Define the **square-root slope function (SRSF)** transformation

$$
\psi_\gamma(t) = \sqrt{\gamma'(t)}.
$$

Because $\int_0^1 \gamma'(t)\,dt = \gamma(1)-\gamma(0)=1$, the SRSF satisfies

$$
\lVert \psi_\gamma\rVert_{L^2}^2 = \int_0^1 \psi_\gamma(t)^2 \, dt = \int_0^1 \gamma'(t)\,dt = 1,
$$

so $\psi_\gamma$ lies on the unit sphere of $L^2([0,1])$, which we denote $\mathbb{S}_\infty$ (an infinite-dimensional Hilbert sphere).

We wish to model a scalar-valued function $f$ defined on warpings,

$$
y = f(\gamma) + \varepsilon,\quad \varepsilon\sim\mathcal{N}(0,\sigma^2_n),
$$

using a Gaussian process prior on $f$. After the SRSF transform we may write the GP prior as

$$
f(\gamma) \sim \mathcal{GP}\big(\mu(\psi_\gamma),\,k(\psi_\gamma,\psi_{\gamma'})\big),
$$

where inputs $\psi$ belong to the Hilbert sphere.

## 2. Why not use direct spherical harmonics in infinite dimensions?

On finite-dimensional spheres $\mathbb{S}^{d} \subset \mathbb{R}^{d+1}$ the Laplace–Beltrami operator has a discrete spectrum and corresponding eigenfunctions — the **spherical harmonics**. Isotropic positive-definite kernels on finite spheres admit expansions in that basis (Gegenbauer / spherical-harmonic expansions); Schoenberg-type results provide necessary and sufficient conditions for positive-definiteness.

However, for the **Hilbert sphere** (unit sphere in an infinite-dimensional separable Hilbert space such as $L^2([0,1])$):

* One does not have a simple finite-dimensional Lie group and associated compact elliptic operator with a countable, practically usable set of eigenfunctions analogous to spherical harmonics.
* While abstract spectral constructions exist in infinite-dimensional functional analysis, there is **no practical closed-form spherical-harmonic basis** to compute with for the Hilbert sphere.
* Conclusion: one cannot directly build a Matérn kernel via spherical harmonics on the Hilbert sphere in the same way as on $\mathbb{S}^d$. Instead, we **approximate** by projection to a finite-dimensional subspace and use finite-sphere techniques there.

## 3. Finite-dimensional projection approximation

Let $\{b_m\}_{m=1}^\infty$ be an orthonormal basis of $L^2([0,1])$ (e.g., Fourier sines/cosines, wavelets, or PCA basis). For computation we truncate at $D$ terms. For a discretized SRSF $\psi(t)$ we compute coefficients

$$
c_m = \langle \psi, b_m\rangle_{L^2},\qquad m=1,\dots,D,
$$

forming $c\in\mathbb{R}^D$. Normalize:

$$
z = \frac{c}{\lVert c\rVert_2} \in \mathbb{S}^{D-1}.
$$

Thus each SRSF is approximated by a point $z$ on the finite sphere $\mathbb{S}^{D-1}$.

As $D\to\infty$ and for dense basis, this projection converges (in $L^2$) to the original SRSF; the sphere approximation converges to the Hilbert sphere in the sense of coordinates.

### The coefficients live on a finite-dimensional hypersphere

We work in the Hilbert space $L^2([0,1])$ with inner product

$$
\langle f,g\rangle := \int_0^1 f(t)\,g(t)\,dt,\qquad \|f\|_{L^2}^2=\langle f,f\rangle.
$$

Let $q\in L^2([0,1])$ be an SRSF (so $\|q\|_{L^2}=1$). Let $\{\phi_1,\dots,\phi_d\}\subset L^2([0,1])$ be an orthonormal system:

$$
\langle \phi_i,\phi_j\rangle = \delta_{ij}\quad (1\le i,j\le d).
$$

Define the projection $P_d: L^2\to \operatorname{span}\{\phi_1,\dots,\phi_d\}$ by

$$
P_d q = \sum_{i=1}^d c_i \phi_i,\qquad c_i := \langle q,\phi_i\rangle.
$$

We want to show that the coefficient vector $c=(c_1,\dots,c_d)\in\mathbb{R}^d$ satisfies $\|c\|_2 = \|P_d q\|_{L^2}$, and in particular if $q$ is **entirely in the span** so that $q=P_d q$, then $\|c\|_2=\|q\|_{L^2}$. Thus if $\|q\|_{L^2}=1$ then $\|c\|_2=1$: the coefficients lie on the unit Euclidean sphere $S^{d-1}$.

### Step 1 — Parseval / Pythagoras for orthonormal systems

Because the $\phi_i$ are orthonormal,

$$
\|P_d q\|_{L^2}^2
= \Big\langle \sum_{i=1}^d c_i\phi_i,\ \sum_{j=1}^d c_j\phi_j\Big\rangle
= \sum_{i=1}^d\sum_{j=1}^d c_i c_j \langle\phi_i,\phi_j\rangle
= \sum_{i=1}^d c_i^2.
$$

So

$$
\|P_d q\|_{L^2} = \|c\|_2.
$$

This is the finite-dimensional version of Parseval’s identity (or Pythagoras in orthogonal decomposition).

### Step 2 — exact equality when $q$ lies in the span

If $q\in\operatorname{span}\{\phi_1,\dots,\phi_d\}$ then $q = P_d q$. Hence

$$
\|q\|_{L^2} = \|P_d q\|_{L^2} = \|c\|_2.
$$

Therefore, if $\|q\|_{L^2}=1$ (the SRSF condition), the coefficient vector satisfies $\|c\|_2=1$; equivalently the coefficients lie on the finite-dimensional unit sphere $S^{d-1}\subset\mathbb{R}^d$.

### Step 3 — general case (approximation)

If $q$ is not exactly in the span, decompose it into orthogonal parts

$$
q = P_d q + r,\quad\text{with } r \perp \operatorname{span}\{\phi_i\}.
$$

Then by orthogonality,

$$
\|q\|_{L^2}^2 = \|P_d q\|_{L^2}^2 + \|r\|_{L^2}^2 = \|c\|_2^2 + \|r\|_{L^2}^2.
$$

So unless $r=0$, the projected coefficient norm $\|c\|_2$ is strictly less than $\|q\|_{L^2}$. If $\|q\|_{L^2}=1$ but we only use a truncated basis, then $\|c\|_2<1$; you can normalize the coefficient vector to unit length to place it on the finite sphere, but that normalization changes the represented function (you are then using the *normalized projection* $ \widetilde q = (P_d q)/\|P_d q\|_{L^2}$), which is a different function on the sphere. Whether to normalize depends on whether you want to enforce the unit-norm constraint exactly in the finite approximation (project-then-normalize) or keep the true projected energy (project-only). Both choices are used in practice — the former enforces the manifold constraint strictly in the finite model, the latter preserves approximation fidelity.


## 4. Spectral Matérn-like kernel on the finite sphere

On $\mathbb{S}^{D-1}$ a Matérn-like kernel consistent with the sphere's spectral properties can be constructed via a Mercer expansion. Given orthonormal eigenfunctions $\{\Phi_\ell\}$ on the sphere and nonnegative weights $w_\ell$, define

$$
k(z,z') = \sum_{\ell=0}^{L-1} w_\ell \,\Phi_\ell(z)\,\Phi_\ell(z').
$$

This is positive semidefinite by construction. In practice we do the following constructive variant:

* Use the *ambient basis coefficients* (the projected and normalized coefficients $z$), and use the coordinate functions $z_m$ themselves as the basis functions on the sphere (i.e., $\Phi_m(z) = z_m$). Then any kernel of the form

$$
k(z,z') = \sum_{m=1}^{L} w_m\, z_m z'_m,
$$

with $w_m\ge 0$, is PSD (finite Mercer sum). Choosing weights with decay yields Matérn-like behavior.

* Choose weights based on Laplacian eigenvalues (for the ambient basis). For a Fourier basis with eigenvalues $\lambda_m\sim (2\pi m)^2$, a Matérn-like spectral weight is

$$
w_m = \biggl(1 + \frac{\lambda_m}{\kappa^2}\biggr)^{-(\nu + \alpha)},
$$

where $\kappa$ is a lengthscale-like parameter, $\nu$ a smoothness parameter, and $\alpha$ an offset to adjust decay. This produces polynomial spectral decay analogous to Matérn kernels.

## 5. Practical algorithm

1. Discretize domain $[0,1]$ on grid $t_j$, $j=1,\dots,N$.
2. For each warp $\gamma$, compute discrete SRSF $\psi_j \approx \sqrt{\gamma'(t_j)}$ and normalize to unit $L^2$ norm.
3. Choose an orthonormal basis $B\in\mathbb{R}^{N\times D}$ (Fourier or PCA). Compute coefficients $c = B^\top \psi$.
4. Normalize coefficients $z = c / \|c\|$ to obtain a point on $\mathbb{S}^{D-1}$.
5. Define kernel

   $$
   k(\psi,\psi') = \sum_{m=1}^{L} w_m z_m z'_m,\quad w_m\ge 0,
   $$

   where $w_m$ chosen per the Matérn-like spectral formula.
6. Use this kernel in a standard GP framework (GPyTorch), learn hyperparameters (e.g., $\log\kappa, \nu$ if desired) by maximizing marginal likelihood.





