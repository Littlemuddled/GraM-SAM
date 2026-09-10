"""Granular Matrix Graph prompting (GraM-Prompt, Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from .granular import img2graph


DATASET_CLASS_VALUES: dict[str, tuple[int, ...]] = {
    "spine": tuple(range(12)),
    "Promise12": (0, 1),
    "ISIC2016": (0, 1),
    "BUSD": (0, 255),
}


@dataclass(frozen=True)
class GraphBuildConfig:
    image_size: int = 256
    purity: float = 0.85
    intensity_threshold: int = 15
    variance_threshold: float = 40.0
    minimum_radius: int = 0
    feature_knn: int = 5


@dataclass(frozen=True)
class PromptCandidate:
    class_index: int
    class_value: int
    coordinates_xy: np.ndarray
    point_labels: np.ndarray
    scores: np.ndarray
    box_xyxy: np.ndarray


def _rectangle_bounds(
    granules: Sequence[tuple[tuple[int, int], int, int]], height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers_y = np.asarray([item[0][0] for item in granules], dtype=np.int64)
    centers_x = np.asarray([item[0][1] for item in granules], dtype=np.int64)
    radii_x = np.asarray([item[1] for item in granules], dtype=np.int64)
    radii_y = np.asarray([item[2] for item in granules], dtype=np.int64)
    top = np.maximum(0, centers_y - radii_y)
    bottom = np.minimum(height - 1, centers_y + radii_y)
    left = np.maximum(0, centers_x - radii_x)
    right = np.minimum(width - 1, centers_x + radii_x)
    return top, bottom, left, right


def _integral_statistics(
    image: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    top, bottom, left, right = bounds
    values = np.asarray(image, dtype=np.float64)
    integral = cv2.integral(values, sdepth=cv2.CV_64F)
    squared = cv2.integral(values * values, sdepth=cv2.CV_64F)

    def rectangle_sum(table: np.ndarray) -> np.ndarray:
        return (
            table[bottom + 1, right + 1]
            - table[top, right + 1]
            - table[bottom + 1, left]
            + table[top, left]
        )

    area = (bottom - top + 1) * (right - left + 1)
    mean = rectangle_sum(integral) / area
    variance = np.maximum(0.0, rectangle_sum(squared) / area - mean * mean)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def granular_node_features(
    rgb_image: np.ndarray,
    granules: Sequence[tuple[tuple[int, int], int, int]],
) -> np.ndarray:
    """Return 21 geometry, appearance, and boundary features per granule."""

    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb_image must be a uint8 [H, W, 3] array")
    if not granules:
        raise ValueError("at least one granule is required")
    height, width = image.shape[:2]
    bounds = _rectangle_bounds(granules, height, width)
    top, bottom, left, right = bounds
    centers_y = np.asarray([item[0][0] for item in granules], dtype=np.int64)
    centers_x = np.asarray([item[0][1] for item in granules], dtype=np.int64)
    radii_x = np.asarray([item[1] for item in granules], dtype=np.float32)
    radii_y = np.asarray([item[2] for item in granules], dtype=np.float32)
    area = ((bottom - top + 1) * (right - left + 1)).astype(np.float32)
    rectangle_height = (bottom - top + 1).astype(np.float32)
    rectangle_width = (right - left + 1).astype(np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradient_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    maximum = float(gradient.max())
    gradient = gradient / maximum if maximum > 0 else np.zeros_like(gradient)

    statistics: list[np.ndarray] = []
    for channel in range(3):
        mean, standard_deviation = _integral_statistics(image[..., channel] / 255.0, bounds)
        statistics.extend([mean, standard_deviation])
    gray_mean, gray_standard_deviation = _integral_statistics(gray / 255.0, bounds)
    gradient_mean, gradient_standard_deviation = _integral_statistics(gradient, bounds)
    center_rgb = image[centers_y, centers_x].astype(np.float32) / 255.0
    center_gray = gray[centers_y, centers_x].astype(np.float32)[:, None] / 255.0
    center_gradient = gradient[centers_y, centers_x].astype(np.float32)[:, None]
    geometry = np.column_stack(
        [
            centers_y / max(1, height - 1),
            centers_x / max(1, width - 1),
            radii_y / height,
            radii_x / width,
            area / (height * width),
            rectangle_width / np.maximum(rectangle_height, 1.0),
        ]
    ).astype(np.float32)
    features = np.column_stack(
        [geometry, *statistics, gray_mean, gray_standard_deviation,
         gradient_mean, gradient_standard_deviation, center_rgb,
         center_gray, center_gradient]
    ).astype(np.float32)
    if features.shape[1] != 21 or not np.isfinite(features).all():
        raise RuntimeError(f"invalid node-feature matrix {features.shape}")
    return features


def _knn_edge_set(values: np.ndarray, number_of_neighbors: int) -> set[tuple[int, int]]:
    number_of_nodes = len(values)
    if number_of_nodes < 2 or number_of_neighbors <= 0:
        return set()
    neighbors = min(number_of_neighbors, number_of_nodes - 1)
    squared_distances = np.sum(
        (values[:, None, :] - values[None, :, :]) ** 2, axis=2
    )
    indices = np.arange(number_of_nodes)
    edges: set[tuple[int, int]] = set()
    for source in range(number_of_nodes):
        order = np.lexsort((indices, squared_distances[source]))
        targets = [index for index in order if index != source][:neighbors]
        for target in targets:
            target = int(target)
            if source != target:
                edges.add((min(source, target), max(source, target)))
    return edges


def efficient_graph_edges(
    granules: Sequence[tuple[tuple[int, int], int, int]],
    features: np.ndarray,
    *,
    feature_knn: int = 5,
) -> np.ndarray:
    if len(granules) != len(features):
        raise ValueError("granules and features must contain the same number of nodes")
    feature_scale = np.maximum(features.std(axis=0, keepdims=True), 1e-6)
    standardized_features = (features - features.mean(axis=0, keepdims=True)) / feature_scale
    top, bottom, left, right = _rectangle_bounds(granules, 10**9, 10**9)
    adjacency: set[tuple[int, int]] = set()
    for first in range(len(granules)):
        for second in range(first + 1, len(granules)):
            row_overlap = max(top[first], top[second]) <= min(bottom[first], bottom[second])
            column_overlap = max(left[first], left[second]) <= min(right[first], right[second])
            overlap = row_overlap and column_overlap
            horizontal_contact = row_overlap and (
                right[first] + 1 == left[second] or right[second] + 1 == left[first]
            )
            vertical_contact = column_overlap and (
                bottom[first] + 1 == top[second] or bottom[second] + 1 == top[first]
            )
            if overlap or horizontal_contact or vertical_contact:
                adjacency.add((first, second))
    edges = adjacency | _knn_edge_set(standardized_features, feature_knn)
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    directed = sorted(
        [(source, target) for source, target in edges]
        + [(target, source) for source, target in edges]
    )
    return np.asarray(directed, dtype=np.int64).T


def majority_class_indices(
    mask: np.ndarray,
    class_values: Sequence[int],
    granules: Sequence[tuple[tuple[int, int], int, int]],
) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    unknown = set(np.unique(mask).tolist()) - {int(value) for value in class_values}
    if unknown:
        raise ValueError(f"mask contains unknown class values: {sorted(unknown)}")
    bounds = _rectangle_bounds(granules, *mask.shape)
    counts = []
    for value in class_values:
        mean, _ = _integral_statistics((mask == value).astype(np.float32), bounds)
        counts.append(mean)
    # np.argmax deterministically chooses the lower class index on ties.
    return np.argmax(np.column_stack(counts), axis=1).astype(np.int64)


def build_granular_graph(
    rgb_image: np.ndarray,
    *,
    config: GraphBuildConfig = GraphBuildConfig(),
    mask: np.ndarray | None = None,
    class_values: Sequence[int] | None = None,
) -> Data:
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb_image must have shape [H, W, 3]")
    original_height, original_width = image.shape[:2]
    resized = cv2.resize(
        image, (config.image_size, config.image_size), interpolation=cv2.INTER_AREA
    )
    scalar = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    granules = img2graph(
        scalar,
        purity=config.purity,
        threshold=config.intensity_threshold,
        var_threshold=config.variance_threshold,
        min_size=config.minimum_radius,
    )
    features = granular_node_features(resized, granules)
    edge_index = efficient_graph_edges(
        granules,
        features,
        feature_knn=config.feature_knn,
    )
    centers = np.asarray(
        [[item[0][0], item[0][1]] for item in granules], dtype=np.float32
    )
    data = Data(
        x=torch.from_numpy(features),
        edge_index=torch.from_numpy(edge_index),
        centers_yx=torch.from_numpy(centers),
        bounds_tblr=torch.from_numpy(np.column_stack(_rectangle_bounds(
            granules, config.image_size, config.image_size
        )).astype(np.float32)),
    )
    data.original_hw = torch.tensor([original_height, original_width], dtype=torch.long)
    data.graph_image_size = int(config.image_size)
    if mask is not None:
        if class_values is None:
            raise ValueError("class_values are required when a mask is supplied")
        resized_mask = cv2.resize(
            np.asarray(mask),
            (config.image_size, config.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        data.y = torch.from_numpy(
            majority_class_indices(resized_mask, class_values, granules)
        )
    return data


class Phase2GraphSAGE(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        number_of_classes: int,
        *,
        hidden_dimension: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
        )
        self.convolutions = nn.ModuleList(
            [SAGEConv(hidden_dimension, hidden_dimension) for _ in range(3)]
        )
        self.normalizations = nn.ModuleList(
            [nn.LayerNorm(hidden_dimension) for _ in range(3)]
        )
        self.dropout = float(dropout)
        self.classifier = nn.Linear(hidden_dimension, number_of_classes)

    def forward(self, data: Data) -> Tensor:
        values = self.input_projection(data.x)
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            residual = values
            values = convolution(values, data.edge_index)
            values = normalization(values)
            values = F.gelu(values) + residual
            values = F.dropout(values, p=self.dropout, training=self.training)
        return self.classifier(values)


def select_spatially_diverse_prompts(
    graph: Data,
    probabilities: np.ndarray,
    class_values: Sequence[int],
    *,
    max_points_per_class: int = 1,
    minimum_separation_fraction: float = 0.08,
    negative_points_per_class: int = 0,
    negative_confidence_threshold: float = 0.70,
    negative_minimum_distance_fraction: float = 0.06,
    component_radius_fraction: float = 0.0,
    positive_interior_weight: float = 0.0,
    box_component_radius_fraction: float = 0.10,
    box_expansion_fraction: float = 0.05,
) -> tuple[PromptCandidate, ...]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    centers_yx = graph.centers_yx.detach().cpu().numpy()
    if probabilities.shape != (len(centers_yx), len(class_values)):
        raise ValueError("probabilities must have shape [nodes, classes]")
    if max_points_per_class < 1:
        raise ValueError("max_points_per_class must be positive")
    if negative_points_per_class < 0:
        raise ValueError("negative_points_per_class must be non-negative")
    if not 0.0 <= negative_confidence_threshold <= 1.0:
        raise ValueError("negative_confidence_threshold must lie in [0, 1]")
    if component_radius_fraction < 0.0 or positive_interior_weight < 0.0:
        raise ValueError("component radius and interior weight must be non-negative")
    if box_component_radius_fraction <= 0.0 or box_expansion_fraction < 0.0:
        raise ValueError("box component radius must be positive and expansion non-negative")
    original_height, original_width = [int(value) for value in graph.original_hw]
    scale_y = original_height / float(graph.graph_image_size)
    scale_x = original_width / float(graph.graph_image_size)
    coordinates_xy = np.column_stack(
        [
            (centers_yx[:, 1] + 0.5) * scale_x,
            (centers_yx[:, 0] + 0.5) * scale_y,
        ]
    )
    minimum_distance = minimum_separation_fraction * float(
        np.hypot(original_height, original_width)
    )
    candidates: list[PromptCandidate] = []
    for class_index, class_value in enumerate(class_values):
        if class_index == 0:
            continue
        scores = probabilities[:, class_index]
        predicted = probabilities.argmax(axis=1) == class_index
        eligible = np.flatnonzero(predicted)
        if len(eligible) == 0:
            eligible = np.arange(len(scores))
        elif len(eligible) > 1 and component_radius_fraction > 0.0:
            radius = component_radius_fraction * float(
                np.hypot(original_height, original_width)
            )
            component_labels = DBSCAN(eps=radius, min_samples=1).fit_predict(
                coordinates_xy[eligible]
            )
            component_scores = [
                float(scores[eligible[component_labels == label]].sum())
                for label in range(int(component_labels.max()) + 1)
            ]
            best_component = int(np.argmax(component_scores))
            eligible = eligible[component_labels == best_component]
        ranking_scores = scores.copy()
        non_class = np.flatnonzero(~predicted)
        if positive_interior_weight > 0.0 and len(non_class) and len(eligible):
            distance_to_non_class = NearestNeighbors(n_neighbors=1).fit(
                coordinates_xy[non_class]
            ).kneighbors(coordinates_xy[eligible], return_distance=True)[0][:, 0]
            interior_scale = max(
                1.0, 0.15 * float(np.hypot(original_height, original_width))
            )
            ranking_scores[eligible] += positive_interior_weight * np.clip(
                distance_to_non_class / interior_scale, 0.0, 1.0
            )
        order = eligible[np.argsort(-ranking_scores[eligible], kind="stable")]
        chosen: list[int] = []
        for index in order:
            if all(
                np.linalg.norm(coordinates_xy[index] - coordinates_xy[previous])
                >= minimum_distance
                for previous in chosen
            ):
                chosen.append(int(index))
            if len(chosen) == max_points_per_class:
                break
        if not chosen:
            chosen = [int(np.argmax(scores))]
        negative_chosen: list[int] = []
        if negative_points_per_class:
            predicted_class = probabilities.argmax(axis=1)
            negative_confidence = 1.0 - scores
            negative_distance = np.min(
                np.linalg.norm(
                    coordinates_xy[:, None, :] - coordinates_xy[chosen][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            negative_minimum_distance = negative_minimum_distance_fraction * float(
                np.hypot(original_height, original_width)
            )
            eligible_negative = np.flatnonzero(
                (predicted_class != class_index)
                & (negative_confidence >= negative_confidence_threshold)
                & (negative_distance >= negative_minimum_distance)
            )
            # Nearby, confident non-class nodes suppress the most plausible
            # false-positive continuation around an automatically found object.
            negative_order = eligible_negative[
                np.lexsort(
                    (
                        -negative_confidence[eligible_negative],
                        negative_distance[eligible_negative],
                    )
                )
            ]
            for index in negative_order:
                if all(
                    np.linalg.norm(coordinates_xy[index] - coordinates_xy[previous])
                    >= minimum_distance
                    for previous in negative_chosen
                ):
                    negative_chosen.append(int(index))
                if len(negative_chosen) == negative_points_per_class:
                    break
        selected = chosen + negative_chosen
        box_components = DBSCAN(
            eps=box_component_radius_fraction
            * float(np.hypot(original_height, original_width)),
            min_samples=1,
        ).fit_predict(coordinates_xy[eligible])
        anchor_position = int(np.flatnonzero(eligible == chosen[0])[0])
        box_nodes = eligible[box_components == box_components[anchor_position]]
        bounds_tblr = graph.bounds_tblr.detach().cpu().numpy()[box_nodes]
        x0 = float(bounds_tblr[:, 2].min() * scale_x)
        y0 = float(bounds_tblr[:, 0].min() * scale_y)
        x1 = float((bounds_tblr[:, 3].max() + 1.0) * scale_x)
        y1 = float((bounds_tblr[:, 1].max() + 1.0) * scale_y)
        expansion_x = box_expansion_fraction * max(1.0, x1 - x0)
        expansion_y = box_expansion_fraction * max(1.0, y1 - y0)
        box_xyxy = np.asarray(
            [
                max(0.0, x0 - expansion_x),
                max(0.0, y0 - expansion_y),
                min(float(original_width - 1), x1 + expansion_x),
                min(float(original_height - 1), y1 + expansion_y),
            ],
            dtype=np.float32,
        )
        candidates.append(
            PromptCandidate(
                class_index=class_index,
                class_value=int(class_value),
                coordinates_xy=coordinates_xy[selected].astype(np.float32),
                point_labels=np.asarray(
                    [1] * len(chosen) + [0] * len(negative_chosen), dtype=np.int32
                ),
                scores=np.concatenate(
                    [scores[chosen], negative_confidence[negative_chosen]]
                    if negative_chosen
                    else [scores[chosen]]
                ).astype(np.float32),
                box_xyxy=box_xyxy,
            )
        )
    return tuple(candidates)


def class_balanced_weights(labels: Iterable[Tensor], number_of_classes: int) -> Tensor:
    counts = torch.zeros(number_of_classes, dtype=torch.float64)
    for values in labels:
        counts += torch.bincount(values.cpu(), minlength=number_of_classes)
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (number_of_classes * counts)
    weights = torch.clamp(weights, min=0.25, max=12.0)
    return (weights / weights.mean()).float()


def foreground_soft_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Node-level soft Dice over foreground classes, complementing weighted CE."""

    if logits.ndim != 2 or target.ndim != 1 or len(logits) != len(target):
        raise ValueError("logits and target must have shapes [N, C] and [N]")
    if logits.shape[1] < 2:
        raise ValueError("at least background and one foreground class are required")
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).to(probabilities.dtype)
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum(dim=0)
    denominator = probabilities[:, 1:].sum(dim=0) + one_hot[:, 1:].sum(dim=0)
    present = one_hot[:, 1:].sum(dim=0) > 0
    if not bool(present.any()):
        return logits.sum() * 0.0
    dice = (2.0 * intersection[present] + 1e-6) / (denominator[present] + 1e-6)
    return 1.0 - dice.mean()


__all__ = [
    "DATASET_CLASS_VALUES",
    "GraphBuildConfig",
    "Phase2GraphSAGE",
    "PromptCandidate",
    "build_granular_graph",
    "class_balanced_weights",
    "efficient_graph_edges",
    "foreground_soft_dice_loss",
    "granular_node_features",
    "majority_class_indices",
    "select_spatially_diverse_prompts",
]
