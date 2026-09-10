"""Spatial Granular Matrix Block Swapping (GMBS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .granular import GranuleTuple as Granule, granule_bounds, img2graph
GranuleConstructor = Callable[..., Sequence[Granule]]


@dataclass(frozen=True)
class GMBSResult:
    """Outputs and detached structural information for one GMBS operation."""

    mixed_a: Tensor
    mixed_b: Tensor
    swap_mask: Tensor
    disagreement: Tensor
    granules: tuple[tuple[Granule, ...], ...]
    high_value_indices: tuple[tuple[int, ...], ...]
    high_value_thresholds: tuple[float, ...]


def _validate_features(feature_a: Tensor, feature_b: Tensor) -> tuple[int, int]:
    if feature_a.ndim != 4:
        raise ValueError("feature tensors must have shape [B, C, H, W]")
    if feature_a.shape != feature_b.shape:
        raise ValueError("feature_a and feature_b must have identical shapes")
    if feature_a.device != feature_b.device:
        raise ValueError("feature_a and feature_b must be on the same device")
    if not feature_a.is_floating_point() or not feature_b.is_floating_point():
        raise TypeError("feature tensors must use a floating-point dtype")
    return int(feature_a.shape[-2]), int(feature_a.shape[-1])


def _as_probability_tensor(probability: Tensor, batch_size: int) -> Tensor:
    """Return a probability tensor with shape [B, K, H, W]."""

    if probability.ndim == 3:
        probability = probability.unsqueeze(1)
    if probability.ndim != 4:
        raise ValueError("probability tensors must have shape [B, H, W] or [B, K, H, W]")
    if probability.shape[0] != batch_size:
        raise ValueError("probability and feature batch sizes must match")
    if not probability.is_floating_point():
        raise TypeError("probability tensors must use a floating-point dtype")
    if not torch.isfinite(probability).all():
        raise ValueError("probability tensors must contain only finite values")
    if torch.any(probability < 0) or torch.any(probability > 1):
        raise ValueError("probability values must lie in [0, 1]")
    return probability


def _aligned_disagreement(
    probability_a: Tensor,
    probability_b: Tensor,
    feature_hw: tuple[int, int],
    batch_size: int,
) -> Tensor:
    """Compute detached mean absolute class disagreement on the feature lattice."""

    probability_a = _as_probability_tensor(probability_a, batch_size)
    probability_b = _as_probability_tensor(probability_b, batch_size)
    if probability_a.shape[:2] != probability_b.shape[:2]:
        raise ValueError("peer probability tensors must have the same batch and class dimensions")

    aligned_a = F.interpolate(
        probability_a,
        size=feature_hw,
        mode="bilinear",
        align_corners=False,
    )
    aligned_b = F.interpolate(
        probability_b,
        size=feature_hw,
        mode="bilinear",
        align_corners=False,
    )
    return torch.abs(aligned_a - aligned_b).mean(dim=1).detach()


def _quantize_unit_interval(array: np.ndarray) -> np.ndarray:
    """Linearly quantize a finite [0, 1] array to uint8."""

    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _granule_bounds(granule: Granule, height: int, width: int) -> tuple[int, int, int, int]:
    return granule_bounds(granule, height, width)


def build_granular_swap_mask(
    disagreement: Tensor,
    *,
    purity: float = 0.85,
    intensity_threshold: int = 15,
    variance_threshold: float = 40.0,
    minimum_radius: int = 0,
    high_value_sigma: float = 0.5,
    constructor: GranuleConstructor = img2graph,
) -> tuple[Tensor, tuple[tuple[Granule, ...], ...], tuple[tuple[int, ...], ...], tuple[float, ...]]:
    """Construct a non-differentiated granular swap mask.

    Args:
        disagreement: Detached or differentiable tensor with shape [B, H, W]
            and values in [0, 1]. It is detached internally before CPU granular
            construction.
        constructor: Granular constructor returning ``((y, x), Rx, Ry)`` tuples.

    Returns:
        Boolean mask [B, 1, H, W], all granules, selected granule indices, and
        the per-sample high-value thresholds.
    """

    if disagreement.ndim != 3:
        raise ValueError("disagreement must have shape [B, H, W]")
    if not disagreement.is_floating_point():
        raise TypeError("disagreement must use a floating-point dtype")
    if not torch.isfinite(disagreement).all():
        raise ValueError("disagreement must contain only finite values")
    if torch.any(disagreement < 0) or torch.any(disagreement > 1):
        raise ValueError("disagreement values must lie in [0, 1]")
    if not 0 <= purity <= 1:
        raise ValueError("purity must lie in [0, 1]")
    if intensity_threshold < 0 or variance_threshold < 0 or minimum_radius < 0:
        raise ValueError("granular thresholds and minimum_radius must be non-negative")

    batch_size, height, width = disagreement.shape
    mask = torch.zeros(
        (batch_size, 1, height, width),
        dtype=torch.bool,
        device=disagreement.device,
    )
    all_granules: list[tuple[Granule, ...]] = []
    all_high_value_indices: list[tuple[int, ...]] = []
    high_value_thresholds: list[float] = []

    disagreement_cpu = disagreement.detach().to(dtype=torch.float32, device="cpu").numpy()
    for batch_index, disagreement_map in enumerate(disagreement_cpu):
        minimum = float(disagreement_map.min())
        dynamic_range = float(disagreement_map.max()) - minimum
        normalized_for_structure = (
            (disagreement_map - minimum) / dynamic_range
            if dynamic_range > 1e-8
            else np.zeros_like(disagreement_map)
        )
        # Granule construction should depend on the spatial disagreement
        # pattern, not on whether two well-initialized peers differ by 0.01 or
        # 0.5 in absolute calibration. Selection still uses original values.
        quantized = _quantize_unit_interval(normalized_for_structure)
        generated = constructor(
            quantized,
            purity=purity,
            threshold=intensity_threshold,
            var_threshold=variance_threshold,
            min_size=minimum_radius,
        )
        granules = tuple(
            ((int(center[0]), int(center[1])), int(radius_x), int(radius_y))
            for center, radius_x, radius_y in generated
        )

        high_value_threshold = float(
            disagreement_map.mean() + high_value_sigma * disagreement_map.std()
        )
        high_value_thresholds.append(high_value_threshold)
        selected: list[int] = []

        for granule_index, granule in enumerate(granules):
            upper, lower, left, right = _granule_bounds(granule, height, width)
            if upper > lower or left > right:
                continue
            region_mean = float(disagreement_map[upper : lower + 1, left : right + 1].mean())
            if region_mean > high_value_threshold:
                mask[batch_index, 0, upper : lower + 1, left : right + 1] = True
                selected.append(granule_index)

        all_granules.append(granules)
        all_high_value_indices.append(tuple(selected))

    return (
        mask,
        tuple(all_granules),
        tuple(all_high_value_indices),
        tuple(high_value_thresholds),
    )


def granular_matrix_block_swap(
    feature_a: Tensor,
    feature_b: Tensor,
    probability_a: Tensor,
    probability_b: Tensor,
    *,
    purity: float = 0.85,
    intensity_threshold: int = 15,
    variance_threshold: float = 40.0,
    minimum_radius: int = 0,
    high_value_sigma: float = 0.5,
    constructor: GranuleConstructor = img2graph,
) -> GMBSResult:
    """Swap peer feature values inside high-disagreement rectangular supports."""

    feature_hw = _validate_features(feature_a, feature_b)
    disagreement = _aligned_disagreement(
        probability_a,
        probability_b,
        feature_hw,
        batch_size=int(feature_a.shape[0]),
    )
    swap_mask, granules, selected, thresholds = build_granular_swap_mask(
        disagreement,
        purity=purity,
        intensity_threshold=intensity_threshold,
        variance_threshold=variance_threshold,
        minimum_radius=minimum_radius,
        high_value_sigma=high_value_sigma,
        constructor=constructor,
    )

    mask_for_features = swap_mask.to(dtype=feature_a.dtype)
    mixed_a = (1.0 - mask_for_features) * feature_a + mask_for_features * feature_b
    mixed_b = (1.0 - mask_for_features) * feature_b + mask_for_features * feature_a

    return GMBSResult(
        mixed_a=mixed_a,
        mixed_b=mixed_b,
        swap_mask=swap_mask,
        disagreement=disagreement,
        granules=granules,
        high_value_indices=selected,
        high_value_thresholds=thresholds,
    )
