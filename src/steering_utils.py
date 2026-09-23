import torch
import numpy as np

# hook functions are expected to take in activation and hook
# they output the new activation where they are applied
def single_direction_hook(
    activation: torch.Tensor,
    hook, # needed for transformer_lens interface
    steering_dir: dict[str, torch.Tensor],
    target_class: str,
    alpha: float = 1,
    normalize = False
) -> torch.Tensor:
    """
    Add a scaled steering direction to the activation.
    """
    if target_class not in steering_dir:
        raise ValueError(
            f"No steering direction found for class '{target_class}'.")

    direction = steering_dir[target_class]

    if isinstance(direction, np.ndarray):
        direction = torch.as_tensor(direction, device=activation.device, dtype=activation.dtype)
    else:
        direction = direction.to(device=activation.device, dtype=activation.dtype)
    
    if normalize:
        norm = direction.norm(p=2)
        if norm == 0:
            return activation
        direction = direction / norm

    return activation + alpha * direction

def normalize_steering_dir(steering_dir: dict[str, torch.Tensor | np.ndarray]):
    normalized_steering_dir = {}
    for name, vec in steering_dir.items():
        if name == "no_steer" or vec is None:
            normalized_steering_dir[name] = vec
            continue
        # Convert numpy arrays to torch tensors for consistent handling
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec)

        # Ensure it's a float tensor (to avoid integer division)
        vec = vec.float()

        # Normalize using L2 norm
        norm = torch.norm(vec, p=2)
        if norm > 0:
            normalized_vec = vec / norm
        else:
            normalized_vec = vec  # avoid divide-by-zero case

        # Convert back to numpy if original was numpy
        if isinstance(steering_dir[name], np.ndarray):
            normalized_vec = normalized_vec.numpy()

        normalized_steering_dir[name] = normalized_vec

    return normalized_steering_dir
