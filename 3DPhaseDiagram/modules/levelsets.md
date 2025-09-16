## Problem formulation 

* $\Omega\subset\mathbb R^3$ is a smooth bounded domain.
* $f,g\in C^\infty(\Omega)$ and $|\nabla f|,|\nabla g|>0$ on the regions of interest (no critical points on those level sets).
* For any scalar $s$ in the range of $f$, denote the level surface

  $$
  \Sigma_f(s):=\{x\in\Omega:\; f(x)=s\}
  $$

  and similarly $\Sigma_g(t)$ for $g$.
* For a point $x\in\Sigma_f(s)$ write the unit normal and projector

  $$
  n_f(x)=\frac{\nabla f(x)}{\|\nabla f(x)\|},\qquad P_f(x)=I-n_f(x)\otimes n_f(x).
  $$
* Let $S_f(x)$ denote the shape operator (Weingarten map) of $\Sigma_f(s)$ at $x$. The mean curvature $H_f=\operatorname{tr}S_f$. Use $|\cdot|_F$ for Frobenius norm on linear maps.


## Correspondence between level sets

To compare level sets we need a *correspondence*:

1. a monotone mapping $\tau$ from level-values of $f$ to level-values of $g$: $\tau:\operatorname{Range}(f)\to\operatorname{Range}(g)$, with $\tau'\!>0$ (this enforces reparametrization invariance), and
2. for each level $s$, a smooth bijection (diffeomorphism) $\Psi_s:\Sigma_f(s)\to\Sigma_g(\tau(s))$ giving pointwise correspondences between surfaces.

(When no canonical $\Psi_s$ exists one can take the infimum over all reasonable correspondences; see the final variational formula.)

Define the **signed normal displacement** $u_s(x)$ as the signed distance along the normal from $x\in\Sigma_f(s)$ to the point $\Psi_s(x)\in\Sigma_g(\tau(s))$. If $\Psi_s(x)$ is reached by moving a signed distance $u_s(x)$ in the $n_f(x)$ direction, then locally

$$
\Psi_s(x) \approx x + u_s(x)\,n_f(x) \quad\text{(first order)}.
$$



## Per-level energy (intuitive form)

For each level $s$ define the **per-level energy**

$$
\boxed{%
E_s(\Psi_s;\,f,g,\tau)\;=\;\alpha\int_{\Sigma_f(s)} u_s(x)^2\,dA(x)
\;+\;\beta\int_{\Sigma_f(s)}\big\|S_g(\Psi_s(x)) - S_f(x)\big\|_F^2\,dA(x),
}
\tag{1}
$$

where $\alpha,\beta\ge0$ are weights:

* the first term penalizes **location mismatch** (normal shift) of corresponding level points;
* the second penalizes **geometric mismatch** (shape-operator / curvature differences).

Remarks:

* Using mean-curvature instead of full shape operator is also possible (replace the second integrand by $(H_g(\Psi_s(x))-H_f(x))^2$).
* No explicit $|\nabla f|,\;|\nabla g|$ appear in (1) because we are integrating on the surfaces directly (surface area element $dA$).


## Total energy via coarea

To combine contributions from *all* levels we integrate $E_s$ over the level parameter $s$. By the coarea formula one can pass between integrals over $\Omega$ and integrals over level surfaces; working level-by-level is natural for level-set geometry. Define the total energy of a correspondence $(\tau,\{\Psi_s\})$ by

$$
\boxed{%
E(\tau,\Psi_\cdot;f,g)
\;=\;\int_{s\in\operatorname{Range}(f)} E_s(\Psi_s;\,f,g,\tau)\;ds.
}
\tag{2}
$$

Equivalently, this is the surface-wise integral

$$
E(\tau,\Psi_\cdot;f,g)
=\alpha\int_{s}\!\!\int_{\Sigma_f(s)} u_s(x)^2\,dA\,ds
+\beta\int_{s}\!\!\int_{\Sigma_f(s)}\|S_g(\Psi_s(x))-S_f(x)\|_F^2\,dA\,ds.
$$

(If one prefers a formulation that integrates over $\Omega$ rather than levels, insert the factor $\delta(f(x)-s)$ or use coarea identity $\int_\Omega \phi(x)\,dx=\int_s\int_{\Sigma_f(s)} \phi(x)/\|\nabla f(x)\|\,dA\,ds$; the surface formulation above is most direct for level-set geometry.)


## Riemannian distance — variational (infimum over correspondences)

The distance between the *level-set geometries* of $f$ and $g$ (i.e., their equivalence classes under monotone reparametrization) is the minimal square-root energy over all admissible correspondences:

$$
\boxed{%
d([f],[g]) \;=\; \inf_{\substack{\tau:\,\text{monotone}\\ \Psi_s:\,\Sigma_f(s)\to\Sigma_g(\tau(s))}} \;\sqrt{\,E(\tau,\Psi_\cdot;f,g)\, }.
}
\tag{3}
$$

This is the natural generalization of the sphere computation: (i) we allow any monotone relabeling $\tau$, making the distance invariant to $f\mapsto h\circ f$; (ii) for each level we compare displacements and shape differences, then integrate across levels.

* If there exists $\tau$ and $\Psi_s$ such that $u_s\equiv0$ and $S_g\circ\Psi_s\equiv S_f$ for all $s$, then $E=0$ and the distance is zero (i.e., identical level-set geometry).
* If level surfaces are displaced (nonzero $u_s$) or geometrically different (nonzero shape-operator difference) the energy and thus the distance are > 0.


## Special cases / computational remarks

1. **Sphere example**. Take $f=r-1$ and $g=r-2$. Choose $\tau(s)=s-1$ (so level 0 maps to 0) and $\Psi_s(x)=\big(1+\Delta r\big)\,x$ with $\Delta r\equiv1$. Then $u_s\equiv1$ on every level $s$ (only the 0-level mattered in the single-surface example), and $S$-difference for spheres of radii $r$ and $r+1$ reduces to a constant — integrating over $s$ restricted to the relevant levels recovers the previous closed-form $E=4\pi(\alpha+\beta)$.

2. **Non-concentric surfaces**. Then $u_s(x)$ varies over $\Sigma_f(s)$ and the optimal $\Psi_s$ typically is the closest-point map (under normal projection) or the map minimizing the local $u_s^2$ plus curvature mismatch integrand. Numerically one can discretize surfaces and solve for $\Psi_s$ (e.g., by optimal transport or closest-point projection).

3. **Choice of curvature penalty**. You can replace the shape-operator Frobenius norm by any other local geometric penalty (mean curvature difference squared, principal curvature differences, or a spectral-norm penalty) depending on whether you care about full second fundamental form or only mean curvature.

4. **Normalization**. If you want a scale-invariant or per-level normalized distance, divide each per-level surface integral by the surface area $|\Sigma_f(s)|$ or weight by a measure $w(s)$, then integrate.


A practical, general Riemannian distance between level-set geometries of $f$ and $g$ is

$$
\boxed{%
d([f],[g])
\;=\;
\inf_{\substack{\tau:\operatorname{Range}(f)\to\operatorname{Range}(g)\\ \Psi_s:\Sigma_f(s)\to\Sigma_g(\tau(s))}}
\left\{\;\int_{s}\Big[\alpha\!\int_{\Sigma_f(s)} u_s(x)^2\,dA
+\beta\!\int_{\Sigma_f(s)}\|S_g(\Psi_s(x))-S_f(x)\|_F^2\,dA\Big]\,ds\right\}^{1/2},
}
$$

where $u_s(x)$ is the signed normal displacement from $x\in\Sigma_f(s)$ to $\Psi_s(x)\in\Sigma_g(\tau(s))$, $S_f,S_g$ are the shape operators, and $\alpha,\beta\ge0$ are weights.
