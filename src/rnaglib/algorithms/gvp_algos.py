import math
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import coalesce, to_undirected

# Small epsilon value added to distances to avoid division by zero
DISTANCE_EPS = 0.001

def internal_coords(
    X: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    return_masks: bool = False,
    distance_eps: float = DISTANCE_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal coordinates layer for RNA.

    This layer computes internal coordinates (ICs) from a batch of RNA
    backbones. To make the ICs differentiable everywhere, this layer replaces
    distance calculations of the form `sqrt(sum_sq)` with smooth, non-cusped
    approximation `sqrt(sum_sq + eps)`.
    
    Adapted from Chroma. In our case, num_batch == num_conformations, so we
    could almost directly repurpose their batched featurisation code in torch.

    Args:
        distance_eps (float, optional): Small parameter to add to squared
            distances to make gradients smooth near 0.

    Inputs:
        X (Tensor): Backbone coordinates with shape
            `(num_residues, num_atom_types, 3)`.
        C (Tensor): Chain map tensor with shape
            `(num_residues)`.

    Outputs:
        dihedrals (Tensor): Backbone dihedral angles with shape
            `(num_batch, num_residues, 4)`
        angles (Tensor): Backbone bond angles with shape
            `(num_batch, num_residues, 4)`
        lengths (Tensor): Backbone bond lengths with shape
            `(num_batch, num_residues, 4)`
    """
    mask = (C > 0).float()
    X_chain = X[:, :2, :]
    num_residues, _, _ = X_chain.shape
    X_chain = X_chain.reshape(2 * num_residues, 3)

    # This function historically returns the angle complement
    _lengths = lambda Xi, Xj: lengths(Xi, Xj, distance_eps=distance_eps)
    _angles = lambda Xi, Xj, Xk: np.pi - angles(
        Xi, Xj, Xk, distance_eps=distance_eps
    )
    _dihedrals = lambda Xi, Xj, Xk, Xl: dihedrals(
        Xi, Xj, Xk, Xl, distance_eps=distance_eps
    )

    # Compute internal coordinates associated with -[P]-[C4']-
    PC4p_L = _lengths(X_chain[1:, :], X_chain[:-1, :])
    PC4p_A = _angles(X_chain[:-2, :], X_chain[1:-1, :], X_chain[2:, :])
    PC4p_D = _dihedrals(
        X_chain[:-3, :],
        X_chain[1:-2, :],
        X_chain[2:-1, :],
        X_chain[3:, :],
    )

    # Compute internal coordinates associated with [C4']-[N]
    X_P, X_C4p, X_N = X.unbind(dim=1)
    X_P_next = X[1:, 0, :]
    N_L = _lengths(X_C4p, X_N)
    N_A = _angles(X_P, X_C4p, X_N)
    N_D = _dihedrals(X_P_next, X_N[:-1, :], X_C4p[:-1, :], X_P[:-1, :])

    if C is None:
        C = torch.zeros_like(mask)

    # Mask nonphysical bonds and angles
    # Note: this could probably also be expressed as a Conv, unclear
    # which is faster and this probably not rate-limiting.
    C = C * (mask.type(torch.long))
    ii = torch.stack(2 * [C], dim=-1).view([-1])
    L0, L1 = ii[:-1], ii[1:]
    A0, A1, A2 = ii[:-2], ii[1:-1], ii[2:]
    D0, D1, D2, D3 = ii[:-3], ii[1:-2], ii[2:-1], ii[3:]

    # Mask for linear backbone
    mask_L = torch.eq(L0, L1)
    mask_A = torch.eq(A0, A1) * torch.eq(A0, A2)
    mask_D = torch.eq(D0, D1) * torch.eq(D0, D2) * torch.eq(D0, D3)
    mask_L = mask_L.type(torch.float32)
    mask_A = mask_A.type(torch.float32)
    mask_D = mask_D.type(torch.float32)

    # Masks for branched nitrogen
    mask_N_D = torch.eq(C[:-1], C[1:])
    mask_N_D = mask_N_D.type(torch.float32)
    mask_N_A = mask
    mask_N_L = mask

    def _pad_pack(D, A, L, N_D, N_A, N_L):
        # Pad and pack together the components
        D = F.pad(D, (1, 2))
        A = F.pad(A, (1, 1))
        L = F.pad(L, (0, 1))
        N_D = F.pad(N_D, (0, 1))
        D, A, L = [x.reshape(num_residues, 2) for x in [D, A, L]]
        _pack = lambda a, b: torch.cat([a, b.unsqueeze(-1)], dim=-1)
        L = _pack(L, N_L)
        A = _pack(A, N_A)
        D = _pack(D, N_D)
        return D, A, L

    D, A, L = _pad_pack(PC4p_D, PC4p_A, PC4p_L, N_D, N_A, N_L)
    mask_D, mask_A, mask_L = _pad_pack(
        mask_D, mask_A, mask_L, mask_N_D, mask_N_A, mask_N_L
    )
    mask_expand = mask.unsqueeze(-1)
    mask_D = mask_expand * mask_D
    mask_A = mask_expand * mask_A
    mask_L = mask_expand * mask_L

    D = mask_D * D
    A = mask_A * A
    L = mask_L * L

    if not return_masks:
        return D, A, L
    else:
        return D, A, L, mask_D, mask_A, mask_L
    

def normed_vec(V: torch.Tensor, distance_eps: float = DISTANCE_EPS) -> torch.Tensor:
    """Normalized vectors with distance smoothing.

    This normalization is computed as `U = V / sqrt(|V|^2 + eps)` to avoid cusps
    and gradient discontinuities.

    Args:
        V (Tensor): Batch of vectors with shape `(..., num_dims)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        U (Tensor): Batch of normalized vectors with shape `(..., num_dims)`.
    """
    # Unit vector from i to j
    mag_sq = (V ** 2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + distance_eps)
    U = V / mag
    return U

def normed_cross(
    V1: torch.Tensor, V2: torch.Tensor, distance_eps: float = DISTANCE_EPS
) -> torch.Tensor:
    """Normalized cross product between vectors.

    This normalization is computed as `U = V / sqrt(|V|^2 + eps)` to avoid cusps
    and gradient discontinuities.

    Args:
        V1 (Tensor): Batch of vectors with shape `(..., 3)`.
        V2 (Tensor): Batch of vectors with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        C (Tensor): Batch of cross products `v_1 x v_2` with shape `(..., 3)`.
    """
    C = normed_vec(torch.cross(V1, V2, dim=-1), distance_eps=distance_eps)
    return C

def lengths(
    atom_i: torch.Tensor, atom_j: torch.Tensor, distance_eps: float = DISTANCE_EPS
) -> torch.Tensor:
    """Batched bond lengths given batches of atom i and j.

    Args:
        atom_i (Tensor): Atom `i` coordinates with shape `(..., 3)`.
        atom_j (Tensor): Atom `j` coordinates with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        L (Tensor): Elementwise bond lengths `||x_i - x_j||` with shape `(...)`.
    """
    # Bond length of i-j
    dX = atom_j - atom_i
    L = torch.sqrt((dX ** 2).sum(dim=-1) + distance_eps)
    return L


def angles(
    atom_i: torch.Tensor,
    atom_j: torch.Tensor,
    atom_k: torch.Tensor,
    distance_eps: float = DISTANCE_EPS,
    degrees: bool = False,
) -> torch.Tensor:
    """Batched bond angles given atoms `i-j-k`.

    Args:
        atom_i (Tensor): Atom `i` coordinates with shape `(..., 3)`.
        atom_j (Tensor): Atom `j` coordinates with shape `(..., 3)`.
        atom_k (Tensor): Atom `k` coordinates with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.
        degrees (bool, optional): If True, convert to degrees. Default: False.

    Returns:
        A (Tensor): Elementwise bond angles with shape `(...)`.
    """
    # Bond angle of i-j-k
    U_ji = normed_vec(atom_i - atom_j, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    inner_prod = torch.einsum("ix,ix->i", U_ji, U_jk)
    inner_prod = torch.clamp(inner_prod, -1, 1)
    A = torch.acos(inner_prod)
    if degrees:
        A = A * 180.0 / np.pi
    return A


def dihedrals(
    atom_i: torch.Tensor,
    atom_j: torch.Tensor,
    atom_k: torch.Tensor,
    atom_l: torch.Tensor,
    distance_eps: float = DISTANCE_EPS,
    degrees: bool = False,
) -> torch.Tensor:
    """Batched bond dihedrals given atoms `i-j-k-l`.

    Args:
        atom_i (Tensor): Atom `i` coordinates with shape `(..., 3)`.
        atom_j (Tensor): Atom `j` coordinates with shape `(..., 3)`.
        atom_k (Tensor): Atom `k` coordinates with shape `(..., 3)`.
        atom_l (Tensor): Atom `l` coordinates with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.
        degrees (bool, optional): If True, convert to degrees. Default: False.

    Returns:
        D (Tensor): Elementwise bond dihedrals with shape `(...)`.
    """
    U_ij = normed_vec(atom_j - atom_i, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    U_kl = normed_vec(atom_l - atom_k, distance_eps=distance_eps)
    normal_ijk = normed_cross(U_ij, U_jk, distance_eps=distance_eps)
    normal_jkl = normed_cross(U_jk, U_kl, distance_eps=distance_eps)
    # _inner_product = lambda a, b: torch.einsum("bix,bix->bi", a, b)
    _inner_product = lambda a, b: (a * b).sum(-1)
    cos_dihedrals = _inner_product(normal_ijk, normal_jkl)
    angle_sign = _inner_product(U_ij, normal_jkl)
    cos_dihedrals = torch.clamp(cos_dihedrals, -1, 1)
    D = torch.sign(angle_sign) * torch.acos(cos_dihedrals)
    if degrees:
        D = D * 180.0 / np.pi
    return D


def rbf_expansion(
        h: torch.Tensor,
        value_min: float = 0.0,
        value_max: float = 30.0,
        num_rbf: int = 32,
    ):
    rbf_centers = torch.linspace(value_min, value_max, num_rbf)
    std = (rbf_centers[1] - rbf_centers[0]).item()
    h = torch.exp(-(((h.unsqueeze(-1) - rbf_centers) / std) ** 2))
    return h


def positional_encoding(inputs, num_posenc=32, period_range=(1.0, 1000.0)):
    
    num_frequencies = num_posenc // 2
    log_bounds = np.log10(period_range)
    p = torch.logspace(log_bounds[0], log_bounds[1], num_frequencies, base=10.0)
    w = 2 * math.pi / p
    
    batch_dims = list(inputs.shape)[:-1]
    # (..., 1, num_out) * (..., num_in, 1)
    w = w.reshape(len(batch_dims) * [1] + [1, -1])
    h = w * inputs[..., None]
    h = torch.cat([h.cos(), h.sin()], -1).reshape(batch_dims + [-1])
    return h


def forward_reverse_vecs(X_c, no_forward, no_backward):
    # Relative displacement vectors along backbone
    # X : num_res x num_bb_atoms x 3
    forward = F.pad(X_c[1:] - X_c[:-1], [0, 0, 0, 1])
    forward[no_forward] = torch.zeros_like(forward[no_forward])
    backward = F.pad(X_c[:-1] - X_c[1:], [0, 0, 1, 0])
    backward[no_backward] = torch.zeros_like(backward[no_backward])
    return torch.cat([
        normed_vec(forward).unsqueeze_(-2), 
        normed_vec(backward).unsqueeze_(-2),
    ], dim=-2)

def internal_vecs(X):
    # Relative displacement vectors along backbone
    # X : num_res x num_bb_atoms x 3
    p, c4p, n = X[:, 0], X[:, 1], X[:, 2]
    n, p = n - c4p, p - c4p
    return torch.cat([
        normed_vec(p).unsqueeze_(-2), 
        normed_vec(n).unsqueeze_(-2), 
    ], dim=-2)

def normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.linalg.norm(tensor, dim=dim, keepdim=True)))

def get_backbone_coords(graph, node_map, pyrimidine_bb_indices, purine_bb_indices, fill_value=1e-5):
    """Extract coordinates of the selected backbone atom beads """
    all_bb_atom_coords = []
    all_mask_coords = []

    for purine_atom, pyrimidine_atom in zip(purine_bb_indices, pyrimidine_bb_indices):
        atom_coords_list = []
        atom_mask_coords = []
        for n in node_map.keys():
            if  graph.nodes()[n]['nt_code'] in ["A","G","a","g"]:
                atom = purine_atom
            else:
                atom = pyrimidine_atom
            coords = graph.nodes()[n][f'xyz_{atom}']
            if coords is not None:
                atom_coords_list.append(torch.as_tensor(coords))
                atom_mask_coords.append(torch.tensor(0))
            else:
                atom_coords_list.append(fill_value*torch.ones(3))
                atom_mask_coords.append(torch.tensor(1))

        atom_coords = torch.stack(atom_coords_list)
        all_bb_atom_coords.append(atom_coords)
        mask_coords = torch.stack(atom_mask_coords)
        all_mask_coords.append(mask_coords)
        
    mask_valid_coords = torch.stack(all_mask_coords, dim = 1).sum(axis=1)==0

    return torch.stack(all_bb_atom_coords, dim = 1), mask_valid_coords