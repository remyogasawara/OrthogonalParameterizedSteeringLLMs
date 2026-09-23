import torch
from typing import Tuple, Dict, Optional


def project_onto_span(
    x: torch.Tensor,
    basis: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Project x onto the column span of basis.

    x: (..., d) or (d,)
    basis: (d, k), columns are basis vectors
    returns: same shape as x
    """
    if device is None:
        device = x.device

    orig_was_vector = (x.dim() == 1)
    x_batch = x.unsqueeze(0) if orig_was_vector else x

    x_f = x_batch.to(device=device, dtype=dtype)
    B_f = basis.to(device=device, dtype=dtype)

    # General projection onto span(B): B (B^T B)^+ B^T x
    gram = B_f.T @ B_f                          # (k, k)
    gram_inv = torch.linalg.pinv(gram)          # (k, k)
    coeffs = (x_f @ B_f) @ gram_inv             # (..., k)
    proj = coeffs @ B_f.T                       # (..., d)

    if orig_was_vector:
        proj = proj.squeeze(0)
    return proj

import torch

def precompute_orthogonalization_matrices(
    steering_dirs: dict[str, torch.Tensor],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
    use_pinv: bool = True,
):
    """
    Returns:
      U_prime (d, k) : orthonormal basis (float32)
      V       (k, k) : coordinates of original steering vectors in U' basis (float32)
      V_inv   (k, k) : inverse (or pseudoinverse) of V (float32)
      indexes : dict mapping name -> column index
    Notes:
      - All matrices returned in float32 for numerical stability. Cast to `dtype` only when using.
    """
    names = list(steering_dirs.keys())
    indexes = {name: i for i, name in enumerate(names)}

    vectors = []
    for name in names:
        vec = steering_dirs[name][LAYER]["answer_token"]["sample_diff_of_means"]
        vec = torch.as_tensor(vec, device=device, dtype=torch.float32)
        vectors.append(vec)

    concept_matrix = torch.stack(vectors, dim=1)   # (d, k)
    d, k = concept_matrix.shape
    assert k < d, "Need k < d for efficient orthogonalization."

    # Step 2: orthonormal basis for span(u_1, ..., u_k)
    U_prime, _ = torch.linalg.qr(concept_matrix, mode="reduced")  # (d, k)

    # Step 3-4: coordinates of each steering vector in U_prime basis
    # V[:, j] are the coordinates of u_j
    V = U_prime.T @ concept_matrix                 # (k, k)
    V_inv = torch.linalg.pinv(V)                   # robust inverse

    return U_prime, V, V_inv, indexes


def fU(
    x: torch.Tensor,
    U_prime: torch.Tensor,     # (d, k), columns are orthonormal
    V_inv: torch.Tensor,       # (k, k)
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Steps 1, 5, 6.

    Returns:
      x_null: component orthogonal to span(U_prime), shape (..., d)
      x_perp: efficient coordinates, shape (..., k)

    Row-vector convention:
      x_{U'} = x @ U_prime
      x_perp = x_{U'} @ V_inv.T
      x_recon = x_null + x_perp @ V.T @ U_prime.T
    """
    orig_was_vector = (x.dim() == 1)
    x_batch = x.unsqueeze(0) if orig_was_vector else x

    x_f = x_batch.to(device=device, dtype=torch.float32)
    U_f = U_prime.to(device=device, dtype=torch.float32)
    V_inv_f = V_inv.to(device=device, dtype=torch.float32)

    # Step 1: project onto span(U_prime)
    x_U = project_onto_span(x_f, U_f, dtype=torch.float32, device=device)  # (..., d)
    x_null = x_f - x_U                                                     # (..., d)

    # Step 5: coordinates in the U' basis
    x_U_coords = x_U @ U_f                                                 # (..., k)

    # Step 6: efficient orthogonal coordinates
    x_perp = x_U_coords @ V_inv_f.T                                        # (..., k)

    x_null = x_null.to(dtype=dtype)
    x_perp = x_perp.to(dtype=dtype)

    if orig_was_vector:
        return x_null.squeeze(0), x_perp.squeeze(0)
    return x_null, x_perp


def reconstruct_from_orthogonal(
    x_null: torch.Tensor,
    x_perp: torch.Tensor,
    U_prime: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    Step 8:
      x' = x_null + x_perp @ V.T @ U_prime.T
    """
    return x_null + x_perp @ V.T @ U_prime.T


def orthogonalize_vectors_op(
    x: torch.Tensor,          # [n, d_model] or [d_model]
    target_class: str,
    U: torch.Tensor,          # unused here, kept for API compatibility
    U_prime: torch.Tensor,    # (d_model, k)
    V: torch.Tensor,          # (k, k)
    V_prime: torch.Tensor,    # unused here, kept for API compatibility
    V_inv: torch.Tensor,      # (k, k)
    V_prime_inv: torch.Tensor,# unused here, kept for API compatibility
    indexes: dict[str, int],
    alpha: float,
    steering_method: str = None,
    steering_dirs: dict[str, torch.Tensor] = None,
    mean_As: dict[str, torch.Tensor] = None,   # unused here
    mean_Bs: dict[str, torch.Tensor] = None,    # unused here
) -> torch.Tensor:
    device = x.device
    orig_dtype = x.dtype

    if not steering_method:
        return x

    if steering_dirs is None:
        raise ValueError("steering_dirs must be provided.")


    # Work in float32
    x_f = x.to(device=device, dtype=torch.float32)
    U_prime = U_prime.to(device=device, dtype=torch.float32)
    V = V.to(device=device, dtype=torch.float32)
    V_inv = V_inv.to(device=device, dtype=torch.float32)

    steering_np = steering_dirs[target_class][LAYER]["answer_token"]["sample_diff_of_means"]
    steering_vector = torch.as_tensor(steering_np, device=device, dtype=torch.float32)

    # Direct original-space steering
    if steering_method == "alpha-iterative":
        out = x_f + alpha * steering_vector
        return out.to(dtype=orig_dtype)

    # Step 1, 5, 6 for x
    x_null, x_perp = fU(x_f, U_prime, V_inv, dtype=torch.float32, device=device)

    if steering_method == "orthogonal-alpha-iterative":
        # Decompose steering vector too, so the orthogonal version matches the original update
        s_null, s_perp = fU(steering_vector, U_prime, V_inv, dtype=torch.float32, device=device)

        x_null_steered = x_null + alpha * s_null
        x_perp_steered = x_perp + alpha * s_perp

    elif steering_method == "orthogonal-parameterized":
        # Placeholder: replace with your own parameterization rule in orthogonal space.
        # Example: modify x_perp_steered[..., j] by some learned / hand-crafted rule.
        if indexes is None:
            raise ValueError("indexes must be provided for orthogonal-parameterized.")
        j = indexes[target_class]
        x_null_steered = x_null
        x_perp_steered = x_perp.clone()
        x_perp_steered[..., j] += alpha * torch.norm(steering_vector)

    else:
        raise ValueError(f"Unknown steering_method: {steering_method}")

    # Step 8: back to original representation
    x_steered = reconstruct_from_orthogonal(x_null_steered, x_perp_steered, U_prime, V)

    return x_steered.to(dtype=orig_dtype)