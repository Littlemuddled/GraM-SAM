"""Granular challenge and representativeness scores for GraM-Select."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .granular import GranuleTuple as Granule, granule_bounds


@dataclass(frozen=True)
class HVGMetrics:
    density: float
    anatomical_complexity: float
    high_value_indices: tuple[int, ...]
    high_value_threshold: float


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    medoid_index: int | None
    challenge_scores: tuple[float, ...]
    combined_scores_at_selection: tuple[float, ...]


def bernoulli_variance_from_tta(aligned_probabilities: np.ndarray) -> np.ndarray:
    """Compute Eq. 3 after every TTA prediction has been inverse-transformed.

    Args:
        aligned_probabilities: Array shaped ``[K_TTA, H, W]`` in the original
            image coordinates, with values in ``[0, 1]``.
    """

    probabilities = np.asarray(aligned_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] == 0:
        raise ValueError("aligned_probabilities must have shape [K_TTA, H, W]")
    if not np.isfinite(probabilities).all():
        raise ValueError("aligned probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("aligned probabilities must lie in [0, 1]")
    mean_probability = probabilities.mean(axis=0)
    return mean_probability * (1.0 - mean_probability)


def _validate_uncertainty_map(uncertainty_map: np.ndarray) -> np.ndarray:
    uncertainty_map = np.asarray(uncertainty_map, dtype=np.float64)
    if uncertainty_map.ndim != 2:
        raise ValueError("uncertainty_map must be two-dimensional")
    if uncertainty_map.size == 0:
        raise ValueError("uncertainty_map must not be empty")
    if not np.isfinite(uncertainty_map).all():
        raise ValueError("uncertainty_map must contain only finite values")
    return uncertainty_map


def _granule_areas_and_means(
    granules: Sequence[Granule], uncertainty_map: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    height, width = uncertainty_map.shape
    areas: list[int] = []
    means: list[float] = []
    bounds: list[tuple[int, int, int, int]] = []
    for granule in granules:
        upper, lower, left, right = granule_bounds(granule, height, width)
        if upper > lower or left > right:
            raise ValueError("granule does not intersect the uncertainty map")
        region = uncertainty_map[upper : lower + 1, left : right + 1]
        areas.append(int(region.size))
        means.append(float(region.mean()))
        bounds.append((upper, lower, left, right))
    return np.asarray(areas), np.asarray(means), bounds


def compute_hvg_metrics(
    granules: Sequence[Granule],
    uncertainty_map: np.ndarray,
    *,
    high_value_sigma: float = 0.5,
    epsilon: float = 1e-8,
) -> HVGMetrics:
    """Compute the paper's union-area HVG fraction and area coefficient of variation."""

    uncertainty_map = _validate_uncertainty_map(uncertainty_map)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if len(granules) == 0:
        return HVGMetrics(0.0, 0.0, (), float(uncertainty_map.mean()))

    areas, granule_means, bounds = _granule_areas_and_means(granules, uncertainty_map)
    high_value_threshold = float(
        uncertainty_map.mean() + high_value_sigma * uncertainty_map.std()
    )
    selected_array = np.flatnonzero(granule_means > high_value_threshold)
    selected = tuple(int(index) for index in selected_array)

    complexity = float(areas.std() / (areas.mean() + epsilon))
    coverage = np.zeros(uncertainty_map.shape, dtype=bool)
    for index in selected:
        upper, lower, left, right = bounds[index]
        coverage[upper : lower + 1, left : right + 1] = True
    density = float(coverage.mean())

    return HVGMetrics(density, complexity, selected, high_value_threshold)


def compute_challenge_score(
    granules: Sequence[Granule],
    uncertainty_map: np.ndarray,
    *,
    high_value_sigma: float = 0.5,
) -> tuple[float, HVGMetrics]:
    uncertainty_map = _validate_uncertainty_map(uncertainty_map)
    metrics = compute_hvg_metrics(
        granules,
        uncertainty_map,
        high_value_sigma=high_value_sigma,
    )
    score = (
        metrics.density
        * float(uncertainty_map.mean())
        * float(np.log1p(metrics.anatomical_complexity))
    )
    return score, metrics


def cosine_distance_to_nearest(feature: np.ndarray, anchors: np.ndarray) -> float:
    feature = np.asarray(feature, dtype=np.float64).reshape(-1)
    anchors = np.asarray(anchors, dtype=np.float64)
    if anchors.ndim == 1:
        anchors = anchors.reshape(1, -1)
    if anchors.ndim != 2 or anchors.shape[1] != feature.size:
        raise ValueError("anchors must have shape [N, feature_dimension]")
    if anchors.shape[0] == 0:
        raise ValueError("at least one anchor is required")
    feature_norm = np.linalg.norm(feature)
    anchor_norms = np.linalg.norm(anchors, axis=1)
    similarities = (anchors @ feature) / np.maximum(anchor_norms * feature_norm, 1e-8)
    return float(np.min(1.0 - similarities))


def pool_medoid_index(features: np.ndarray) -> int:
    """Return the deterministic cosine-distance medoid, breaking ties by index."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape [N, D] with N > 0")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-8)
    distances = 1.0 - normalized @ normalized.T
    return int(np.argmin(distances.mean(axis=1)))


def greedy_select(
    challenge_scores: Sequence[float],
    features: np.ndarray,
    number_to_select: int,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    initial_anchor_features: np.ndarray | None = None,
    normalize_terms: bool = False,
) -> SelectionResult:
    """Greedily combine challenge and nearest-anchor cosine distance.

    At cold start the pool medoid is used as an anchor. It is not automatically
    counted as an annotated sample.
    """

    scores = np.asarray(challenge_scores, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or scores.ndim != 1 or len(scores) != len(features):
        raise ValueError("challenge_scores and features must describe the same candidates")
    if not np.isfinite(scores).all() or not np.isfinite(features).all():
        raise ValueError("scores and features must be finite")
    if not 0 <= number_to_select <= len(scores):
        raise ValueError("number_to_select must lie between zero and the pool size")

    medoid_index: int | None = None
    if initial_anchor_features is None or np.asarray(initial_anchor_features).size == 0:
        medoid_index = pool_medoid_index(features)
        anchors = features[[medoid_index]].copy()
    else:
        anchors = np.asarray(initial_anchor_features, dtype=np.float64)
        if anchors.ndim == 1:
            anchors = anchors.reshape(1, -1)
        if anchors.ndim != 2 or anchors.shape[1] != features.shape[1]:
            raise ValueError("initial anchors and candidate features must share a dimension")

    challenge = scores.copy()
    if normalize_terms and len(challenge) > 0:
        span = challenge.max() - challenge.min()
        challenge = (challenge - challenge.min()) / span if span > 0 else np.zeros_like(challenge)

    remaining = np.ones(len(scores), dtype=bool)
    selected: list[int] = []
    selected_combined_scores: list[float] = []
    for _ in range(number_to_select):
        representativeness = np.asarray(
            [cosine_distance_to_nearest(feature, anchors) for feature in features]
        )
        if normalize_terms:
            span = representativeness[remaining].max() - representativeness[remaining].min()
            if span > 0:
                minimum = representativeness[remaining].min()
                representativeness = (representativeness - minimum) / span
            else:
                representativeness = np.zeros_like(representativeness)
        combined = alpha * challenge + beta * representativeness
        combined[~remaining] = -np.inf
        best_index = int(np.argmax(combined))
        selected.append(best_index)
        selected_combined_scores.append(float(combined[best_index]))
        remaining[best_index] = False
        anchors = np.vstack([anchors, features[best_index]])

    return SelectionResult(
        selected_indices=tuple(selected),
        medoid_index=medoid_index,
        challenge_scores=tuple(float(value) for value in scores),
        combined_scores_at_selection=tuple(selected_combined_scores),
    )
