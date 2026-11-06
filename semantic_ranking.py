import pandas as pd
import torch
import torch.nn.functional as F


def build_poi_category_mapping(
    dataframe: pd.DataFrame,
    num_pois: int,
    poi_id_to_index: dict,
) -> dict[int, str]:
    required = {"POI_id", "POI_catname"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"trajectory data missing columns: {missing}")
    grouped = dataframe.groupby("POI_id")["POI_catname"]
    inconsistent = grouped.nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if len(inconsistent):
        raise ValueError(
            f"{len(inconsistent)} POIs have inconsistent category metadata"
        )
    category_by_poi = grouped.first()
    mapping = {}
    for raw_id, category in category_by_poi.items():
        index = poi_id_to_index.get(raw_id)
        if index is None:
            index = poi_id_to_index.get(str(raw_id))
        if index is not None and 0 <= int(index) < int(num_pois):
            mapping[int(index)] = str(category)
    if not mapping:
        raise ValueError("POI category mapping is empty")
    return mapping


def _normalize(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    compute_dtype = (
        torch.float32
        if probabilities.dtype in (torch.float16, torch.bfloat16)
        else probabilities.dtype
    )
    probabilities = probabilities.to(compute_dtype).clamp_min(eps)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)


def semantic_residual(
    llm_distribution: torch.Tensor,
    empirical_distribution: torch.Tensor,
    clip: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if llm_distribution.shape != empirical_distribution.shape:
        raise ValueError("LLM and empirical distributions must have the same shape")
    if clip <= 0:
        raise ValueError("clip must be positive")
    llm = _normalize(llm_distribution, eps=eps)
    empirical = _normalize(empirical_distribution, eps=eps)
    return (
        llm.clamp_min(eps).log() - empirical.clamp_min(eps).log()
    ).clamp(-float(clip), float(clip))


def reversal_pairs(
    reference_scores: torch.Tensor,
    enhanced_scores: torch.Tensor,
    candidate_categories: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    if not (
        reference_scores.shape
        == enhanced_scores.shape
        == candidate_categories.shape
    ):
        raise ValueError("candidate tensors must have the same shape")
    if reference_scores.dim() != 1:
        raise ValueError("candidate tensors must have shape [K]")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    reference = reference_scores.float()
    enhanced = enhanced_scores.float()
    categories = candidate_categories.to(
        device=reference.device, dtype=torch.long
    )
    enhanced_gap = enhanced[:, None] - enhanced[None, :]
    reference_gap = reference[:, None] - reference[None, :]
    valid = (
        (categories[:, None] != categories[None, :])
        & (reference_gap <= 0.0)
        & (enhanced_gap >= float(threshold))
    )
    valid.fill_diagonal_(False)
    return valid.nonzero(as_tuple=False).to(dtype=torch.long)


def ranking_loss(
    candidate_logits: torch.Tensor,
    pair_sets: list[torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    if candidate_logits.dim() != 2:
        raise ValueError("candidate logits must have shape [B_rank, K]")
    if len(pair_sets) != candidate_logits.size(0):
        raise ValueError("pair_sets must contain one tensor per row")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    total = candidate_logits.new_zeros((), dtype=torch.float32)
    for logits, pairs in zip(candidate_logits, pair_sets):
        pairs = pairs.to(device=logits.device, dtype=torch.long)
        if pairs.numel() == 0:
            continue
        if pairs.dim() != 2 or pairs.size(1) != 2:
            raise ValueError("each pair tensor must have shape [M, 2]")
        score_gap = logits.float()[pairs[:, 0]] - logits.float()[pairs[:, 1]]
        total = total + F.softplus(-score_gap).mean()
    return total / float(batch_size)


def category_marginal_projection(
    poi_logits: torch.Tensor,
    category_logits: torch.Tensor,
    poi_category_index: torch.Tensor,
    strength: float = 0.02,
    clip: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if clip <= 0:
        raise ValueError("clip must be positive")
    if poi_logits.shape[:-1] != category_logits.shape[:-1]:
        raise ValueError("POI and category logits must share leading dimensions")
    category_index = poi_category_index.to(
        device=poi_logits.device, dtype=torch.long
    )
    if category_index.dim() != 1 or category_index.numel() != poi_logits.size(-1):
        raise ValueError("poi_category_index must contain one category per POI")
    num_categories = category_logits.size(-1)
    valid_pois = (category_index >= 0) & (category_index < num_categories)
    if not bool(valid_pois.any()) or strength == 0:
        return poi_logits.clone()
    compute_dtype = (
        torch.float32
        if poi_logits.dtype in (torch.float16, torch.bfloat16)
        else poi_logits.dtype
    )
    poi_probabilities = F.softmax(poi_logits.to(compute_dtype), dim=-1)
    poi_marginal = torch.zeros(
        (*poi_probabilities.shape[:-1], num_categories),
        device=poi_probabilities.device,
        dtype=poi_probabilities.dtype,
    )
    scatter_index = category_index[valid_pois].view(
        *((1,) * (poi_probabilities.dim() - 1)), -1
    ).expand(*poi_probabilities.shape[:-1], int(valid_pois.sum()))
    poi_marginal.scatter_add_(
        -1, scatter_index, poi_probabilities[..., valid_pois]
    )
    poi_marginal = _normalize(poi_marginal, eps=eps)
    category_probabilities = _normalize(
        F.softmax(category_logits.to(compute_dtype), dim=-1), eps=eps
    )
    offsets = (
        category_probabilities.clamp_min(eps).log()
        - poi_marginal.clamp_min(eps).log()
    ).clamp(-float(clip), float(clip))
    correction = torch.zeros_like(poi_logits, dtype=compute_dtype)
    correction[..., valid_pois] = offsets[..., category_index[valid_pois]]
    projected = poi_logits.to(compute_dtype) + float(strength) * correction
    return projected.to(dtype=poi_logits.dtype)
