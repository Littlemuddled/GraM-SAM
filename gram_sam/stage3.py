"""Memory-efficient dual-branch SAM2 training primitives for GraM-SAM Phase 3."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from gram_sam.datasets import PromptGroup


class FeatureAdapter(nn.Module):
    """Small residual adapter that gives two branches distinct feature views."""

    def __init__(self, channels: int = 256, bottleneck: int = 64) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1)
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.up(F.gelu(self.down(values)))


class Stage3Branch(nn.Module):
    """A trainable prompt/decoder branch over one shared frozen encoder."""

    def __init__(self, prompt_encoder: nn.Module, mask_decoder: nn.Module) -> None:
        super().__init__()
        self.prompt_encoder = copy.deepcopy(prompt_encoder)
        self.mask_decoder = copy.deepcopy(mask_decoder)
        self.feature_adapter = FeatureAdapter()
        # These projections were already applied by the frozen encoder/cache.
        for name, parameter in self.mask_decoder.named_parameters():
            if name.startswith("conv_s0.") or name.startswith("conv_s1."):
                parameter.requires_grad_(False)

    def adapted_features(self, image_embedding: Tensor) -> Tensor:
        return self.feature_adapter(image_embedding)


@dataclass(frozen=True)
class DecodedMasks:
    logits: Tensor
    quality: Tensor
    low_resolution_logits: Tensor


@dataclass(frozen=True)
class SpatialTransform:
    """Shape-preserving geometric transform shared by both peer views."""

    horizontal_flip: bool = False
    vertical_flip: bool = False


def sample_spatial_transform(generator: np.random.Generator) -> SpatialTransform:
    """Sample identity, horizontal/vertical flip, or 180-degree rotation."""

    return SpatialTransform(
        horizontal_flip=bool(generator.integers(0, 2)),
        vertical_flip=bool(generator.integers(0, 2)),
    )


def transform_spatial_tensor(values: Tensor, transform: SpatialTransform) -> Tensor:
    """Apply a shape-preserving transform over the last two tensor axes."""

    if values.ndim < 2:
        raise ValueError("spatial tensors must have at least two dimensions")
    dimensions = []
    if transform.vertical_flip:
        dimensions.append(-2)
    if transform.horizontal_flip:
        dimensions.append(-1)
    return torch.flip(values, dimensions) if dimensions else values


def transform_prompt_groups(
    groups: Sequence[PromptGroup],
    original_hw: tuple[int, int],
    transform: SpatialTransform,
) -> tuple[PromptGroup, ...]:
    """Transform Phase-2 point/box prompts with the peer-view geometry."""

    height, width = (int(original_hw[0]), int(original_hw[1]))
    if height < 1 or width < 1:
        raise ValueError("original_hw must contain positive dimensions")
    transformed_groups = []
    for group in groups:
        coordinates = np.asarray(group.coordinates_xy, dtype=np.float32).copy()
        if transform.horizontal_flip:
            coordinates[:, 0] = width - 1 - coordinates[:, 0]
        if transform.vertical_flip:
            coordinates[:, 1] = height - 1 - coordinates[:, 1]
        coordinates[:, 0] = np.clip(coordinates[:, 0], 0, width - 1)
        coordinates[:, 1] = np.clip(coordinates[:, 1], 0, height - 1)

        box = None
        if group.box_xyxy is not None:
            x0, y0, x1, y1 = np.asarray(group.box_xyxy, dtype=np.float32)
            if transform.horizontal_flip:
                x0, x1 = width - 1 - x1, width - 1 - x0
            if transform.vertical_flip:
                y0, y1 = height - 1 - y1, height - 1 - y0
            box = np.asarray([x0, y0, x1, y1], dtype=np.float32)

        transformed_groups.append(
            replace(group, coordinates_xy=coordinates, box_xyxy=box)
        )
    return tuple(transformed_groups)


def weak_feature_view(
    features: tuple[Tensor, Sequence[Tensor], tuple[int, int]],
    transform: SpatialTransform,
    *,
    generator: np.random.Generator,
    gain_range: float = 0.05,
    bias_range: float = 0.01,
) -> tuple[Tensor, list[Tensor], tuple[int, int]]:
    """Create one weak feature view while preserving spatial correspondence.

    Geometry is supplied separately so two peer views can share it.  The
    independently sampled affine feature perturbation supplies weak appearance
    diversity without re-running the frozen image encoder.
    """

    if gain_range < 0 or bias_range < 0:
        raise ValueError("weak-view perturbation ranges must be non-negative")
    image_embedding, high_resolution_features, original_hw = features
    gain = float(generator.uniform(1.0 - gain_range, 1.0 + gain_range))
    bias_factor = float(generator.uniform(-bias_range, bias_range))

    def augment(values: Tensor) -> Tensor:
        geometrically_transformed = transform_spatial_tensor(values, transform)
        scale = geometrically_transformed.detach().float().std(unbiased=False)
        bias = bias_factor * scale.to(
            device=geometrically_transformed.device,
            dtype=geometrically_transformed.dtype,
        )
        return geometrically_transformed * gain + bias

    return (
        augment(image_embedding),
        [augment(value) for value in high_resolution_features],
        original_hw,
    )


def perturb_trainable_parameters(module: nn.Module, scale: float, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            noise = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            ).to(device=parameter.device, dtype=parameter.dtype)
            parameter.add_(noise * scale)


def pack_prompt_groups(
    groups: Sequence[PromptGroup],
    *,
    max_positive_points: int = 1,
    max_negative_points: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if max_positive_points < 1 or max_negative_points < 0:
        raise ValueError("prompt limits must be positive/non-negative")
    selected: list[tuple[np.ndarray, np.ndarray]] = []
    class_ids: list[int] = []
    for group in groups:
        positive = np.flatnonzero(group.point_labels == 1)[:max_positive_points]
        negative = np.flatnonzero(group.point_labels == 0)[:max_negative_points]
        indices = np.concatenate([positive, negative])
        if not len(positive):
            raise ValueError(f"class {group.class_id} has no positive prompt")
        selected.append((group.coordinates_xy[indices], group.point_labels[indices]))
        class_ids.append(int(group.class_id))
    if not selected:
        raise ValueError("at least one prompt group is required")
    width = max(len(item[0]) for item in selected)
    coordinates = np.zeros((len(selected), width, 2), dtype=np.float32)
    labels = np.full((len(selected), width), -1, dtype=np.int32)
    for index, (item_coordinates, item_labels) in enumerate(selected):
        coordinates[index, : len(item_coordinates)] = item_coordinates
        labels[index, : len(item_labels)] = item_labels
    return coordinates, labels, tuple(class_ids)


def mask_interior_prompt_variants(
    mask: np.ndarray,
    class_ids: Sequence[int],
    *,
    number_of_variants: int,
    generator: np.random.Generator,
    interior_quantile: float = 0.75,
) -> tuple[tuple[PromptGroup, ...], ...]:
    """Sample deterministic training-only positive prompts deep inside targets."""

    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if number_of_variants < 0:
        raise ValueError("number_of_variants must be non-negative")
    if not 0.0 <= interior_quantile < 1.0:
        raise ValueError("interior_quantile must lie in [0, 1)")
    candidate_sets: list[np.ndarray] = []
    for class_id in class_ids:
        foreground = mask == int(class_id)
        if not foreground.any():
            raise ValueError(f"class {class_id} is absent from the training mask")
        distances = distance_transform_edt(foreground)
        positive_distances = distances[foreground]
        cutoff = float(np.quantile(positive_distances, interior_quantile))
        candidates_yx = np.argwhere(foreground & (distances >= cutoff))
        if not len(candidates_yx):
            candidates_yx = np.argwhere(foreground)
        candidate_sets.append(candidates_yx)

    selected_by_class = []
    for candidates in candidate_sets:
        indices = generator.choice(
            len(candidates),
            size=number_of_variants,
            replace=len(candidates) < number_of_variants,
        )
        selected_by_class.append(candidates[indices])

    variants = []
    for variant_index in range(number_of_variants):
        groups = []
        for class_id, selected in zip(class_ids, selected_by_class, strict=True):
            y, x = selected[variant_index]
            groups.append(
                PromptGroup(
                    int(class_id),
                    np.asarray([[x, y]], dtype=np.float32),
                    np.ones(1, dtype=np.int32),
                    np.ones(1, dtype=np.float32),
                    None,
                )
            )
        variants.append(tuple(groups))
    return tuple(variants)


def decode_branch(
    branch: Stage3Branch,
    predictor,
    image_embedding: Tensor,
    high_resolution_features: Sequence[Tensor],
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    original_hw: tuple[int, int],
    image_embeddings_are_adapted: bool = False,
    multimask_output: bool = True,
) -> DecodedMasks:
    predictor._orig_hw = [tuple(int(value) for value in original_hw)]
    _, transformed_coordinates, transformed_labels, _ = predictor._prep_prompts(
        coordinates,
        labels,
        box=None,
        mask_logits=None,
        normalize_coords=True,
    )
    sparse, dense = branch.prompt_encoder(
        points=(transformed_coordinates, transformed_labels), boxes=None, masks=None
    )
    adapted = (
        image_embedding
        if image_embeddings_are_adapted
        else branch.adapted_features(image_embedding)
    )
    low_resolution, quality, _, _ = branch.mask_decoder(
        image_embeddings=adapted,
        image_pe=branch.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=multimask_output,
        repeat_image=(coordinates.shape[0] > 1),
        high_res_features=list(high_resolution_features),
    )
    logits = predictor._transforms.postprocess_masks(low_resolution, original_hw)
    return DecodedMasks(logits=logits, quality=quality, low_resolution_logits=low_resolution)


def candidate_segmentation_losses(
    logits: Tensor,
    target: Tensor,
    *,
    loss_mode: str = "bce_dice",
    focal_gamma: float = 2.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> Tensor:
    """Return a loss for every [instance, mask-token] candidate.

    ``bce_dice`` provides a conventional alternative. The paper configuration
    uses ``focal_tversky`` to target hard pixels and class imbalance; beta
    greater than alpha penalizes false negatives more strongly.
    """

    if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0]:
        raise ValueError("logits/target must have shapes [K,M,H,W] and [K,H,W]")
    expanded_target = target[:, None].expand_as(logits).to(logits.dtype)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * expanded_target).sum(dim=(-2, -1))
    if loss_mode == "bce_dice":
        bce = F.binary_cross_entropy_with_logits(
            logits, expanded_target, reduction="none"
        ).mean(dim=(-2, -1))
        denominator = probabilities.sum(dim=(-2, -1)) + expanded_target.sum(
            dim=(-2, -1)
        )
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        return bce + dice
    if loss_mode != "focal_tversky":
        raise ValueError(f"Unsupported supervised loss mode: {loss_mode}")
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative")
    if tversky_alpha < 0.0 or tversky_beta < 0.0:
        raise ValueError("Tversky weights must be non-negative")

    pixel_bce = F.binary_cross_entropy_with_logits(
        logits, expanded_target, reduction="none"
    )
    probability_true = (
        probabilities * expanded_target
        + (1.0 - probabilities) * (1.0 - expanded_target)
    )
    focal = ((1.0 - probability_true).pow(focal_gamma) * pixel_bce).mean(
        dim=(-2, -1)
    )
    false_positive = (probabilities * (1.0 - expanded_target)).sum(dim=(-2, -1))
    false_negative = ((1.0 - probabilities) * expanded_target).sum(dim=(-2, -1))
    tversky = 1.0 - (intersection + 1.0) / (
        intersection
        + tversky_alpha * false_positive
        + tversky_beta * false_negative
        + 1.0
    )
    return focal + tversky


def candidate_iou_targets(logits: Tensor, target: Tensor) -> Tensor:
    expanded_target = target[:, None].expand_as(logits).to(logits.dtype)
    prediction = (torch.sigmoid(logits) >= 0.5).to(logits.dtype)
    intersection = (prediction * expanded_target).sum(dim=(-2, -1))
    union = prediction.sum(dim=(-2, -1)) + expanded_target.sum(dim=(-2, -1)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).detach()


def supervised_branch_loss(
    decoded: DecodedMasks,
    target: Tensor,
    *,
    quality_weight: float = 0.1,
    loss_mode: str = "bce_dice",
    focal_gamma: float = 2.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> tuple[Tensor, Tensor]:
    losses = candidate_segmentation_losses(
        decoded.logits,
        target,
        loss_mode=loss_mode,
        focal_gamma=focal_gamma,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
    )
    best_indices = losses.detach().argmin(dim=1)
    selected = losses.gather(1, best_indices[:, None]).mean()
    quality_target = candidate_iou_targets(decoded.logits, target)
    quality_loss = F.mse_loss(decoded.quality.float(), quality_target.float())
    return selected + quality_weight * quality_loss, best_indices


def select_quality_candidates(decoded: DecodedMasks) -> tuple[Tensor, Tensor, Tensor]:
    indices = decoded.quality.detach().argmax(dim=1)
    row = torch.arange(len(indices), device=indices.device)
    logits = decoded.logits[row, indices]
    quality = decoded.quality[row, indices]
    return logits, quality, indices


InferenceFusion = Literal[
    "independent_best",
    "joint_best",
    "best_quality",
    "quality_weighted",
]


def fuse_candidate_probabilities(
    decoded_branches: Sequence[DecodedMasks],
    *,
    mode: InferenceFusion = "independent_best",
    temperature: float = 0.1,
) -> Tensor:
    """Fuse mask-token probabilities from multiple independently trained branches.

    All choices depend only on the model-predicted mask quality.  This keeps
    inference target-free while allowing calibration on the selected training
    masks instead of silently tuning a rule on validation targets.
    """

    if not decoded_branches:
        raise ValueError("at least one decoded branch is required")
    logits = torch.stack([decoded.logits.float() for decoded in decoded_branches])
    quality = torch.stack([decoded.quality.float() for decoded in decoded_branches])
    if logits.ndim != 5 or quality.ndim != 3 or logits.shape[:3] != quality.shape:
        raise ValueError("decoded branches must share [instances, candidates] shapes")
    probabilities = torch.sigmoid(logits)
    branch_count, instance_count, candidate_count = quality.shape

    if mode == "independent_best":
        indices = quality.argmax(dim=2)
        branch = torch.arange(branch_count, device=quality.device)[:, None]
        instance = torch.arange(instance_count, device=quality.device)[None, :]
        return probabilities[branch, instance, indices].mean(dim=0)

    if mode == "joint_best":
        indices = quality.mean(dim=0).argmax(dim=1)
        instance = torch.arange(instance_count, device=quality.device)
        selected = [probabilities[index, instance, indices] for index in range(branch_count)]
        return torch.stack(selected).mean(dim=0)

    flattened_quality = quality.permute(1, 0, 2).reshape(instance_count, -1)
    flattened_probability = probabilities.permute(1, 0, 2, 3, 4).reshape(
        instance_count,
        branch_count * candidate_count,
        probabilities.shape[-2],
        probabilities.shape[-1],
    )
    if mode == "best_quality":
        indices = flattened_quality.argmax(dim=1)
        instance = torch.arange(instance_count, device=quality.device)
        return flattened_probability[instance, indices]
    if mode == "quality_weighted":
        if temperature <= 0:
            raise ValueError("quality-fusion temperature must be positive")
        weights = torch.softmax(flattened_quality / temperature, dim=1)
        return (flattened_probability * weights[:, :, None, None]).sum(dim=1)
    raise ValueError(f"unknown inference fusion mode: {mode}")


def confidence_weighted_cps_loss(
    student_logits: Tensor,
    peer_probabilities: Tensor,
    peer_quality: Tensor,
    *,
    quality_threshold: float = 0.0,
    pixel_confidence_threshold: float = 0.0,
    minimum_foreground_fraction: float = 0.0,
    maximum_foreground_fraction: float = 1.0,
    instance_validity: Tensor | None = None,
) -> tuple[Tensor, int]:
    if student_logits.shape != peer_probabilities.shape or student_logits.ndim != 3:
        raise ValueError("student and peer tensors must have matching [K,H,W] shapes")
    pseudo = (peer_probabilities.detach() >= 0.5).to(student_logits.dtype)
    confidence = (2.0 * torch.abs(peer_probabilities.detach() - 0.5)).clamp(0.0, 1.0)
    foreground_fraction = pseudo.mean(dim=(-2, -1))
    valid = (
        (peer_quality.detach().float() >= quality_threshold)
        & (foreground_fraction >= minimum_foreground_fraction)
        & (foreground_fraction <= maximum_foreground_fraction)
    )
    if instance_validity is not None:
        if instance_validity.shape != valid.shape:
            raise ValueError("instance_validity must have shape [K]")
        valid = valid & instance_validity.to(device=valid.device, dtype=torch.bool)
    if not bool(valid.any()):
        return student_logits.sum() * 0.0, 0
    pixel_weight = (confidence >= pixel_confidence_threshold).to(student_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(student_logits, pseudo, reduction="none")
    bce = (bce * pixel_weight).sum(dim=(-2, -1)) / pixel_weight.sum(
        dim=(-2, -1)
    ).clamp_min(1.0)
    probability = torch.sigmoid(student_logits)
    intersection = (probability * pseudo).sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum(dim=(-2, -1)) + pseudo.sum(dim=(-2, -1)) + 1.0
    )
    return (bce[valid] + dice[valid]).mean(), int(valid.sum())


def gaussian_rampup(epoch: int, rampup_epochs: int) -> float:
    if rampup_epochs <= 0:
        return 1.0
    phase = 1.0 - np.clip(float(epoch), 0.0, float(rampup_epochs)) / rampup_epochs
    return float(np.exp(-5.0 * phase * phase))


__all__ = [
    "DecodedMasks",
    "FeatureAdapter",
    "SpatialTransform",
    "Stage3Branch",
    "candidate_segmentation_losses",
    "confidence_weighted_cps_loss",
    "decode_branch",
    "fuse_candidate_probabilities",
    "gaussian_rampup",
    "pack_prompt_groups",
    "perturb_trainable_parameters",
    "sample_spatial_transform",
    "select_quality_candidates",
    "supervised_branch_loss",
    "transform_prompt_groups",
    "transform_spatial_tensor",
    "weak_feature_view",
]
