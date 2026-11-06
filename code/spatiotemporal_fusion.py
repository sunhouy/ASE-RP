import torch
import torch.nn.functional as F


def region_marginal_correction(
    poi_logits: torch.Tensor,
    region_probabilities: torch.Tensor,
    poi_region_index: torch.Tensor,
    weight: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if poi_logits.ndim != 2 or region_probabilities.ndim != 2:
        raise ValueError("POI logits and region probabilities must be matrices")
    if poi_logits.size(0) != region_probabilities.size(0):
        raise ValueError("POI and region rows must match")
    lookup = poi_region_index.to(device=poi_logits.device, dtype=torch.long)
    if lookup.ndim != 1 or lookup.numel() != poi_logits.size(1):
        raise ValueError("poi_region_index must contain one region per POI")
    if bool(((lookup < 0) | (lookup >= region_probabilities.size(1))).any()):
        raise ValueError("poi_region_index contains an invalid region")
    compute_dtype = (
        torch.float32
        if poi_logits.dtype in (torch.float16, torch.bfloat16)
        else poi_logits.dtype
    )
    poi_probabilities = F.softmax(poi_logits.to(compute_dtype), dim=-1)
    region_probabilities = region_probabilities.to(compute_dtype)
    poi_region_marginal = torch.zeros_like(region_probabilities)
    indices = lookup.unsqueeze(0).expand(poi_logits.size(0), -1)
    poi_region_marginal.scatter_add_(1, indices, poi_probabilities)
    correction = (
        region_probabilities.clamp_min(eps).log()
        - poi_region_marginal.clamp_min(eps).log()
    )
    return poi_logits.to(compute_dtype) + float(weight) * correction[:, lookup]


def user_time_poi_priors(
    time_probabilities: torch.Tensor,
    user_indices,
    user_time_poi,
    eps: float = 1e-8,
) -> torch.Tensor:
    if time_probabilities.ndim != 2:
        raise ValueError("time_probabilities must have shape [N, T]")
    table = torch.as_tensor(
        user_time_poi,
        device=time_probabilities.device,
        dtype=time_probabilities.dtype,
    )
    users = torch.as_tensor(
        user_indices,
        device=time_probabilities.device,
        dtype=torch.long,
    )
    if table.ndim != 3:
        raise ValueError("user_time_poi must have shape [U, T, P]")
    if users.ndim != 1 or users.numel() != time_probabilities.size(0):
        raise ValueError("user_indices must contain one user per row")
    if time_probabilities.size(1) != table.size(1):
        raise ValueError("time dimensions do not match")
    if users.numel() and bool(((users < 0) | (users >= table.size(0))).any()):
        raise ValueError("user index is outside user_time_poi")
    priors = torch.bmm(
        time_probabilities.unsqueeze(1), table[users]
    ).squeeze(1)
    mass = priors.sum(dim=-1, keepdim=True)
    normalized = priors / mass.clamp_min(eps)
    uniform = torch.full_like(normalized, 1.0 / max(table.size(2), 1))
    return torch.where(mass > 0, normalized, uniform)


def continuous_spatiotemporal_fusion(
    poi_logits: torch.Tensor,
    region_logits: torch.Tensor,
    time_logits: torch.Tensor,
    batch_users,
    sequence_lengths,
    poi_region_index: torch.Tensor,
    user_time_poi,
    region_weight: float = 0.10,
    time_weight: float = 0.10,
    smoothing: float = 0.05,
) -> torch.Tensor:
    if poi_logits.ndim != 3:
        raise ValueError("poi_logits must have shape [B, L, P]")
    if region_logits.ndim != 3 or time_logits.ndim != 3:
        raise ValueError("region_logits and time_logits must have rank 3")
    batch_size, max_length, num_pois = poi_logits.shape
    if region_logits.shape[:2] != (batch_size, max_length):
        raise ValueError("region logits do not match POI dimensions")
    if time_logits.shape[:2] != (batch_size, max_length):
        raise ValueError("time logits do not match POI dimensions")
    if len(batch_users) != batch_size or len(sequence_lengths) != batch_size:
        raise ValueError("users and sequence lengths must match batch size")
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("smoothing must be in [0, 1]")
    lengths = torch.as_tensor(
        sequence_lengths, device=poi_logits.device, dtype=torch.long
    )
    if bool(((lengths < 1) | (lengths > max_length)).any()):
        raise ValueError("sequence length is outside the padded dimension")
    positions = torch.arange(max_length, device=poi_logits.device).unsqueeze(0)
    row_index, step_index = torch.nonzero(
        positions < lengths.unsqueeze(1), as_tuple=True
    )
    current = poi_logits[row_index, step_index, :]
    if region_weight > 0:
        region_probabilities = F.softmax(
            region_logits[row_index, step_index, :].float(), dim=-1
        )
        current = region_marginal_correction(
            current,
            region_probabilities,
            poi_region_index,
            weight=region_weight,
        )
    if time_weight > 0:
        time_probabilities = F.softmax(
            time_logits[row_index, step_index, :].float(), dim=-1
        )
        users = torch.as_tensor(
            batch_users, device=poi_logits.device, dtype=torch.long
        )[row_index]
        time_prior = user_time_poi_priors(
            time_probabilities, users, user_time_poi
        )
        time_prior = (
            (1.0 - float(smoothing)) * time_prior
            + float(smoothing) / max(num_pois, 1)
        )
        current = current + float(time_weight) * time_prior.clamp_min(1e-8).log()
    fused = poi_logits.clone()
    fused[row_index, step_index, :] = current.to(dtype=fused.dtype)
    return fused
