"""Segmentation metrics shared by GraM-SAM experiments.

HD95 is reported in pixels unless an explicit ``(row, column)`` spacing is
provided.  Empty predictions receive the image-diagonal distance instead of a
dataset-specific magic constant.  This keeps the failure policy finite,
deterministic and explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label as label_components


@dataclass(frozen=True)
class BinarySegmentationMetrics:
    dice: float
    iou: float
    hd95: float
    precision: float
    recall: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _binary_mask(value: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional mask")
    return mask.astype(bool, copy=False)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))


def prompt_connected_component(
    prediction: np.ndarray,
    point_xy: Sequence[float],
) -> np.ndarray:
    """Keep the predicted 8-connected component containing/nearest a prompt.

    This target-free post-processing removes disconnected islands while
    retaining the component justified by the positive point prompt.
    """

    prediction = _binary_mask(prediction, "prediction")
    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("point_xy must be one finite [x, y] coordinate")
    component_map, count = label_components(
        prediction, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if count <= 1:
        return prediction.copy()
    x = int(np.clip(np.rint(point[0]), 0, prediction.shape[1] - 1))
    y = int(np.clip(np.rint(point[1]), 0, prediction.shape[0] - 1))
    selected = int(component_map[y, x])
    if selected == 0:
        best_distance = np.inf
        for component_id in range(1, count + 1):
            rows, columns = np.nonzero(component_map == component_id)
            distance = float(np.min((rows - y) ** 2 + (columns - x) ** 2))
            if distance < best_distance:
                best_distance = distance
                selected = component_id
    return component_map == selected


def hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> float:
    """Return the symmetric 95th-percentile surface Hausdorff distance."""

    prediction = _binary_mask(prediction, "prediction")
    target = _binary_mask(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share a shape")
    if len(spacing) != 2 or any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("spacing must contain two positive finite values")
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        height, width = prediction.shape
        return float(hypot(max(0, height - 1) * spacing[0], max(0, width - 1) * spacing[1]))

    prediction_surface = _surface(prediction)
    target_surface = _surface(target)
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing)
    distances = np.concatenate(
        [distance_to_target[prediction_surface], distance_to_prediction[target_surface]]
    )
    return float(np.percentile(distances, 95))


def binary_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
    epsilon: float = 1e-8,
) -> BinarySegmentationMetrics:
    prediction = _binary_mask(prediction, "prediction")
    target = _binary_mask(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share a shape")

    true_positive = float(np.logical_and(prediction, target).sum())
    prediction_count = float(prediction.sum())
    target_count = float(target.sum())
    union = prediction_count + target_count - true_positive
    dice = (2.0 * true_positive + epsilon) / (
        prediction_count + target_count + epsilon
    )
    iou = (true_positive + epsilon) / (union + epsilon)
    precision = (true_positive + epsilon) / (prediction_count + epsilon)
    recall = (true_positive + epsilon) / (target_count + epsilon)
    return BinarySegmentationMetrics(
        dice=float(dice),
        iou=float(iou),
        hd95=hd95(prediction, target, spacing=spacing),
        precision=float(precision),
        recall=float(recall),
    )


def summarize_metrics(
    metrics: Sequence[BinarySegmentationMetrics] | Iterable[BinarySegmentationMetrics],
) -> dict[str, float | int]:
    values = list(metrics)
    if not values:
        raise ValueError("at least one metric record is required")
    return {
        "count": len(values),
        "dice": float(np.mean([item.dice for item in values])),
        "iou": float(np.mean([item.iou for item in values])),
        "hd95": float(np.mean([item.hd95 for item in values])),
        "precision": float(np.mean([item.precision for item in values])),
        "recall": float(np.mean([item.recall for item in values])),
    }
