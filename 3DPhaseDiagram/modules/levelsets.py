"""
levelset_fmap.py

Compute Riemannian distance between level-set geometries of two scalar fields (or meshes)
using a Functional Map pipeline to establish correspondences Ψ.

Dependencies:
  - numpy
  - scipy
  - scikit-image (optional, for marching_cubes if supplying voxel fields)
  - (tests use pytest)

Main classes / functions:
  - Mesh: lightweight container + utilities (areas, normals, cotangent Laplacian)
  - spectral_basis(mesh, k): compute first k eigenpairs of Laplace-Beltrami
  - compute_HKS(eigvals, eigvecs, times): compute HKS descriptors
  - solve_functional_map(A_src, A_tgt, Phi_src, Phi_tgt, reg=1e-6): solve for C
  - recover_map_pointwise(C, Phi_src, Phi_tgt, verts_tgt, method='nn'): pointwise correspondence
  - LevelSetFunctionalMapDistance: top-level class that takes voxel fields or meshes, builds maps, and returns energy & distance

Usage example:
    from levelset_fmap import LevelSetFunctionalMapDistance
    L = LevelSetFunctionalMapDistance(f_grid, g_grid, c=0.0)
    L.prepare()           # extracts meshes, computes bases & descriptors
    out = L.compute()     # returns mapping, energies, distance

Author: ChatGPT (assistant)
"""

from typing import Tuple, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Optional imports
try:
    from skimage import measure as skmeasure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# -----------------------------
# Basic mesh utilities / container
# -----------------------------
class Mesh:
    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        """
        verts: (n,3)
        faces: (m,3) integer indices
        """
        self.verts = np.asarray(verts, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        self.nv = self.verts.shape[0]
        self.nf = self.faces.shape[0]
        # cached values
        self._face_areas = None
        self._vertex_areas = None
        self._vertex_normals = None
        self._L = None  # cotangent Laplacian (sparse)
        self._M = None  # mass matrix (vertex-area diagonal)
        self._mean_curvature = None

    def face_areas_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._face_areas is not None:
            return self._face_areas, self.face_normals
        v0 = self.verts[self.faces[:, 0], :]
        v1 = self.verts[self.faces[:, 1], :]
        v2 = self.verts[self.faces[:, 2], :]
        cross = np.cross(v1 - v0, v2 - v0)
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        face_normals = cross / (np.linalg.norm(cross, axis=1)[:, None] + 1e-20)
        self._face_areas = face_areas
        self.face_normals = face_normals
        return face_areas, face_normals

    def vertex_areas(self) -> np.ndarray:
        if self._vertex_areas is not None:
            return self._vertex_areas
        face_areas, _ = self.face_areas_normals()
        v_area = np.zeros(self.nv, dtype=float)
        for i, f in enumerate(self.faces):
            a_third = face_areas[i] / 3.0
            v_area[f[0]] += a_third
            v_area[f[1]] += a_third
            v_area[f[2]] += a_third
        self._vertex_areas = v_area
        return v_area

    def vertex_normals(self) -> np.ndarray:
        if self._vertex_normals is not None:
            return self._vertex_normals
        _, f_normals = self.face_areas_normals()
        vnorm = np.zeros((self.nv, 3), dtype=float)
        for i, f in enumerate(self.faces):
            for vi in f:
                vnorm[vi] += f_normals[i]
        lens = np.linalg.norm(vnorm, axis=1)
        nonzero = lens > 1e-12
        vnorm[nonzero] /= lens[nonzero][:, None]
        vnorm[~nonzero] = np.array([0.0, 0.0, 1.0])
        self._vertex_normals = vnorm
        return vnorm

    def cotangent_laplacian(self) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Returns L (symmetric cotangent Laplacian) and M (mass matrix diagonal).
        L is constructed so that L * f approximates -div(grad f).
        """
        if self._L is not None and self._M is not None:
            return self._L, self._M
        nv = self.nv
        I = []
        J = []
        V = []
        # adjacency weights
        W = {}
        for tri in self.faces:
            i, j, k = tri
            vi = self.verts[i]; vj = self.verts[j]; vk = self.verts[k]
            # angles cotangents
            def cot(a, b, c):
                ba = b - a
                ca = c - a
                cross = np.cross(ba, ca)
                denom = np.linalg.norm(cross) + 1e-20
                return np.dot(ba, ca) / denom
            cot_i = cot(vj, vi, vk)
            cot_j = cot(vk, vj, vi)
            cot_k = cot(vi, vk, vj)
            # add weights
            W.setdefault((j, k), 0.0)
            W[(j, k)] += cot_i
            W.setdefault((k, j), 0.0)
            W[(k, j)] += cot_i
            W.setdefault((i, k), 0.0)
            W[(i, k)] += cot_j
            W.setdefault((k, i), 0.0)
            W[(k, i)] += cot_j
            W.setdefault((i, j), 0.0)
            W[(i, j)] += cot_k
            W.setdefault((j, i), 0.0)
            W[(j, i)] += cot_k
        rows = []
        cols = []
        vals = []
        diag = np.zeros(nv, dtype=float)
        for (i, j), w in W.items():
            rows.append(i); cols.append(j); vals.append(-0.5 * w)  # symmetric weight
            diag[i] += 0.5 * w
        # build sparse
        L = sp.csr_matrix((vals, (rows, cols)), shape=(nv, nv))
        # add diagonal
        L = L + sp.diags(diag)
        # mass matrix
        M = sp.diags(self.vertex_areas())
        self._L = L
        self._M = M
        return L, M

    def mean_curvature(self) -> np.ndarray:
        if self._mean_curvature is not None:
            return self._mean_curvature
        L, M = self.cotangent_laplacian()
        # mean curvature vector approximate: Hvec = M^{-1} L x
        # then scalar H = 2 * <Hvec, normal> (convention dependent). We'll use H = <Hvec, normal>
        # Solve Hvec = M^{-1} L V (vector per vertex)
        vpos = self.verts
        # apply L to each coordinate
        Hvec = np.zeros_like(vpos)
        for dim in range(3):
            Hvec[:, dim] = L.dot(vpos[:, dim]) / (self.vertex_areas() + 1e-20)
        normals = self.vertex_normals()
        H_scalar = np.einsum('ij,ij->i', Hvec, normals)
        self._mean_curvature = H_scalar
        return H_scalar


# -----------------------------
# Spectral basis and descriptors
# -----------------------------
def spectral_basis(mesh: Mesh, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute first k eigenpairs of generalized eigenproblem L phi = lambda M phi.
    Returns (evals (k,), evecs (nv, k))
    """
    L, M = mesh.cotangent_laplacian()
    # Build symmetric problem: M^{-1/2} L M^{-1/2} y = lambda y
    # but we can use eigsh on (L, M)
    # convert to sparse csr
    L = L.tocsr()
    M = M.tocsr()
    # use shift-invert? For stability use scipy.sparse.linalg.eigsh with generalized problem
    try:
        vals, vecs = spla.eigsh(L, k=k, M=M, sigma=0.0, which='LM')
    except Exception:
        # fallback: small dense solve (nv small)
        nv = mesh.nv
        Ld = L.toarray()
        Md = M.toarray()
        vals_all, vecs_all = np.linalg.eigh(np.linalg.pinv(Md).dot(Ld))
        idx = np.argsort(vals_all)[:k]
        vals = vals_all[idx]
        vecs = vecs_all[:, idx]
    # eigsh may return negative small values due to numerical; ensure non-negative
    vals = np.real(vals)
    vecs = np.real(vecs)
    # sort ascending
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs


def compute_HKS(evals: np.ndarray, evecs: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Heat Kernel Signature (HKS) descriptor:
      HKS(t, x) = sum_k exp(-t * lambda_k) * phi_k(x)^2
    Returns descriptor matrix (nv, len(times))
    """
    # ensure evects squared
    phi2 = evecs ** 2  # (nv,k)
    # weights exp(-t lambda)
    T = len(times)
    nv, k = phi2.shape
    desc = np.zeros((nv, T), dtype=float)
    for ti, t in enumerate(times):
        weights = np.exp(-t * evals)  # (k,)
        desc[:, ti] = phi2.dot(weights)
    return desc


# -----------------------------
# Functional map routines
# -----------------------------
def solve_functional_map(descriptors_src: np.ndarray, descriptors_tgt: np.ndarray,
                         Phi_src: np.ndarray, Phi_tgt: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Solve for C (k_t x k_s) minimizing ||C A_s - A_t||_F^2 + reg * ||C||_F^2
    where A_s = Phi_src^T * descriptors_src (k_s x d) and similar for A_t.
    descriptors_*: (nv, d)
    Phi_src: (nv, k_s)
    Phi_tgt: (nv_t, k_t)
    Returns C (k_t, k_s)
    """
    # Project descriptors into spectral bases
    A_s = Phi_src.T.dot(descriptors_src)  # (k_s, d)
    A_t = Phi_tgt.T.dot(descriptors_tgt)  # (k_t, d)
    # Solve for C in least squares: C @ A_s = A_t  -> vectorize over columns
    # closed-form: C = A_t @ A_s^T @ (A_s @ A_s^T + reg I)^{-1}
    AsAsT = A_s.dot(A_s.T)
    k_s = AsAsT.shape[0]
    reg_matrix = reg * np.eye(k_s)
    inv = np.linalg.inv(AsAsT + reg_matrix)
    C = A_t.dot(A_s.T).dot(inv)
    return C


def recover_pointwise_map_nn(C: np.ndarray, Phi_src: np.ndarray, Phi_tgt: np.ndarray, verts_tgt: np.ndarray) -> np.ndarray:
    """
    Recover pointwise map by mapping basis functions: map f_src = Phi_src * a  -> coefficients to target: Phi_tgt * (C a)
    One way: embed each vertex to k-dim descriptors (Phi_src @ diag?) Simpler: compute spectral embedding and nearest neighbor.

    We'll compute spectral coordinates for each vertex:
      coords_src = Phi_src * S_src  (k_s dims)
      coords_tgt = Phi_tgt * S_tgt  (k_t dims)
    For matching we map coords_src -> C^T coords_tgt? Simpler robust approach:
      - Map delta functions (indicator basis) at each src vertex to target spectral coefficients: c = C @ (Phi_src^T e_i) = C @ (Phi_src^T[:, i]) => vector length k_t
      - Reconstruct mapped function on target: f_mapped = Phi_tgt @ c  (values on target vertices)
      - For each source vertex i, we take f_mapped (vector on target verts) and find index of max value as mapped point.
    This is robust for indicator mapping.
    """
    nv_src = Phi_src.shape[0]
    nv_tgt = Phi_tgt.shape[0]
    # Precompute Phi_src^T (k_s, nv_src)
    PhisT = Phi_src.T  # (k_s, nv_src)
    C_mat = C  # (k_t, k_s)
    mapped_idxs = np.zeros(nv_src, dtype=int)
    # For efficiency process in blocks
    block = 256
    for i0 in range(0, nv_src, block):
        i1 = min(nv_src, i0 + block)
        # extract columns of PhisT corresponding to vertices
        # for each source vertex j in block, a = PhisT[:, j] (k_s,)
        A_block = PhisT[:, i0:i1]  # (k_s, b)
        # mapped coefficients on target for each source vertex: C @ A_block  -> (k_t, b)
        Ccoeff = C_mat.dot(A_block)
        # reconstruct on target: Phi_tgt @ Ccoeff  -> (nv_tgt, b)
        mapped_vals = Phi_tgt.dot(Ccoeff)  # potentially large; handle columnwise
        # for each column take argmax
        for local_j in range(i1 - i0):
            col = mapped_vals[:, local_j]
            mapped_idxs[i0 + local_j] = int(np.argmax(col))
    return mapped_idxs


# -----------------------------
# Top-level pipeline class
# -----------------------------
class LevelSetFunctionalMapDistance:
    def __init__(self, f_grid: Optional[np.ndarray] = None, g_grid: Optional[np.ndarray] = None,
                 c: float = 0.0,
                 meshes_f: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 meshes_g: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 k_eig: int = 30,
                 n_hks_times: int = 8,
                 alpha: float = 1.0,
                 beta: float = 1.0):
        """
        Provide either (f_grid, g_grid) plus scikit-image for marching_cubes or provide meshes_f and meshes_g
          meshes_f/g: lists or single tuple (verts, faces)
        c: level value
        spacing: voxel spacing for marchng_cubes
        k_eig: number of spectral basis functions
        n_hks_times: number of HKS time samples/descriptors
        """
        self.f_grid = f_grid
        self.g_grid = g_grid
        self.meshes_f = meshes_f
        self.meshes_g = meshes_g
        self.c = float(c)
        self.spacing = spacing
        self.k = k_eig
        self.n_hks = n_hks_times
        self.alpha = float(alpha)
        self.beta = float(beta)
        # placeholders to be filled in prepare()
        self.mesh_f = None
        self.mesh_g = None
        self.evals_f = None; self.evecs_f = None
        self.evals_g = None; self.evecs_g = None
        self.Hf = None; self.Hg = None
        self.des_f = None; self.des_g = None
        self.C = None
        self.map_idxs = None

    def extract_meshes(self):
        if self.meshes_f is not None and self.meshes_g is not None:
            verts_f, faces_f = self.meshes_f
            verts_g, faces_g = self.meshes_g
        else:
            if not _HAS_SKIMAGE:
                raise RuntimeError("scikit-image required to extract isosurfaces from volumes.")
            # use marching_cubes with spacing
            verts_f, faces_f, _, _ = skmeasure.marching_cubes(self.f_grid, level=self.c, spacing=self.spacing)
            verts_g, faces_g, _, _ = skmeasure.marching_cubes(self.g_grid, level=self.c, spacing=self.spacing)
        self.mesh_f = Mesh(verts_f, faces_f)
        self.mesh_g = Mesh(verts_g, faces_g)
        return self.mesh_f, self.mesh_g

    def prepare(self):
        # extract meshes then spectral bases and descriptors
        self.extract_meshes()
        # spectral bases
        k = self.k
        ev_f, evec_f = spectral_basis(self.mesh_f, k=k)
        ev_g, evec_g = spectral_basis(self.mesh_g, k=k)
        self.evals_f = ev_f; self.evecs_f = evec_f
        self.evals_g = ev_g; self.evecs_g = evec_g
        # descriptors: HKS times
        # select times from spectrum: t_i = 4 * logspace between max_eig and min_eig? Simple heuristic:
        lam_max = max(np.max(ev_f[1:]) if len(ev_f) > 1 else 1.0, np.max(ev_g[1:]) if len(ev_g) > 1 else 1.0)
        times = np.logspace(np.log10(1e-3), np.log10(1.0), self.n_hks)
        desc_f = compute_HKS(ev_f, evec_f, times)
        desc_g = compute_HKS(ev_g, evec_g, times)
        # optionally normalize descriptors per-vertex (L2)
        desc_f = desc_f / (np.linalg.norm(desc_f, axis=1)[:, None] + 1e-20)
        desc_g = desc_g / (np.linalg.norm(desc_g, axis=1)[:, None] + 1e-20)
        self.des_f = desc_f
        self.des_g = desc_g
        # mean curvature (mesh based)
        self.Hf = self.mesh_f.mean_curvature()
        self.Hg = self.mesh_g.mean_curvature()
        return True

    def compute_functional_map(self, reg=1e-6):
        # Project descriptors into spectral bases and solve for C
        Phi_f = self.evecs_f  # (nv_f, k)
        Phi_g = self.evecs_g  # (nv_g, k)
        # Solve C in k x k (we use same k for both)
        C = solve_functional_map(self.des_f, self.des_g, Phi_f, Phi_g, reg=reg)
        self.C = C
        return C

    def recover_pointwise_map(self):
        # recover using indicator mapping argmax approach
        C = self.C
        Phi_f = self.evecs_f
        Phi_g = self.evecs_g
        idxs = recover_pointwise_map_nn(C, Phi_f, Phi_g, self.mesh_g.verts)
        self.map_idxs = idxs
        return idxs

    def evaluate_energy(self):
        if self.map_idxs is None:
            raise RuntimeError("Recover pointwise map first.")
        idxs = self.map_idxs
        verts_f = self.mesh_f.verts
        verts_g = self.mesh_g.verts[idxs]
        normals_f = self.mesh_f.vertex_normals()
        # compute signed displacement along normals
        disp = verts_g - verts_f
        u = np.einsum('ij,ij->i', disp, normals_f)
        v_areas = self.mesh_f.vertex_areas()
        E_disp = self.alpha * np.sum((u ** 2) * v_areas)
        Hg_matched = self.Hg[idxs]
        E_shape = self.beta * np.sum(((Hg_matched - self.Hf) ** 2) * v_areas)
        E_total = float(E_disp + E_shape)
        return {"E_disp": float(E_disp), "E_shape": float(E_shape), "E_total": E_total, "distance": float(np.sqrt(E_total)),
                "u_mean": float(np.mean(u)), "area_ref": float(np.sum(v_areas))}

    def compute(self, reg=1e-6):
        self.compute_functional_map(reg=reg)
        self.recover_pointwise_map()
        return self.evaluate_energy()


# -----------------------------
# Example CLI-style usage (not executed on import)
# -----------------------------
def _demo_spheres():
    # create grids sampling spheres r-1 and r-2
    nx = 80
    x = np.linspace(-3.0, 3.0, nx)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    f = R - 1.0
    g = R - 2.0
    L = LevelSetFunctionalMapDistance(f_grid=f, g_grid=g, c=0.0, k_eig=20, n_hks_times=6, alpha=1.0, beta=1.0)
    L.prepare()
    out = L.compute()
    print("Sphere demo energy:", out)
    return out


if __name__ == "__main__":
    if _HAS_SKIMAGE:
        _demo_spheres()
    else:
        print("skimage not available; run tests with meshes instead.")
