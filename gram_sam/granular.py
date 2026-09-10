"""Adaptive Granular Matrix construction used by all three GraM-SAM phases.

The implementation follows Algorithm 1 in the paper: scalar maps are
min--max normalized and quantized before this function is called, a 3x3
Gaussian filter is applied, seeds are visited by stable ascending gradient,
and horizontal/vertical candidate growth is accepted only while both purity
and regional-variance criteria hold.  Covered pixels cannot become new seeds;
rectangular supports may overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Granule:
    """One coordinate-aligned rectangular granule."""

    center_y: int
    center_x: int
    radius_x: int
    radius_y: int

    def as_tuple(self) -> tuple[tuple[int, int], int, int]:
        return ((self.center_y, self.center_x), self.radius_x, self.radius_y)


GranuleTuple = tuple[tuple[int, int], int, int]


def normalize_scalar_map(values: np.ndarray) -> np.ndarray:
    """Min--max normalize a finite 2-D scalar map to uint8 [0, 255]."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("values must be a non-empty two-dimensional scalar map")
    if not np.isfinite(array).all():
        raise ValueError("values must contain only finite values")
    minimum = float(array.min())
    span = float(array.max()) - minimum
    if span <= 1e-12:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.rint((array - minimum) * (255.0 / span)).astype(np.uint8)


def granule_bounds(
    granule: GranuleTuple, height: int, width: int
) -> tuple[int, int, int, int]:
    """Return inclusive top, bottom, left, right bounds."""

    (center_y, center_x), radius_x, radius_y = granule
    if radius_x < 0 or radius_y < 0:
        raise ValueError("granule radii must be non-negative")
    return (
        max(0, int(center_y) - int(radius_y)),
        min(height - 1, int(center_y) + int(radius_y)),
        max(0, int(center_x) - int(radius_x)),
        min(width - 1, int(center_x) + int(radius_x)),
    )


def _criterion(
    image: np.ndarray,
    integral: np.ndarray,
    squared_integral: np.ndarray,
    center_y: int,
    center_x: int,
    radius_x: int,
    radius_y: int,
    purity_threshold: float,
    intensity_threshold: int,
    variance_threshold: float,
) -> bool:
    top, bottom, left, right = granule_bounds(
        ((center_y, center_x), radius_x, radius_y), *image.shape
    )
    region = image[top : bottom + 1, left : right + 1].astype(np.int16)
    center_value = int(image[center_y, center_x])
    purity = float(np.mean(np.abs(region - center_value) <= intensity_threshold))
    area = (bottom - top + 1) * (right - left + 1)
    total = (
        integral[bottom + 1, right + 1]
        - integral[top, right + 1]
        - integral[bottom + 1, left]
        + integral[top, left]
    )
    squared_total = (
        squared_integral[bottom + 1, right + 1]
        - squared_integral[top, right + 1]
        - squared_integral[bottom + 1, left]
        + squared_integral[top, left]
    )
    variance = max(0.0, float(squared_total / area - (total / area) ** 2))
    return purity > purity_threshold and variance < variance_threshold


def construct_granular_matrix(
    scalar_map: np.ndarray,
    *,
    purity_threshold: float = 0.85,
    intensity_threshold: int = 15,
    variance_threshold: float = 40.0,
) -> list[GranuleTuple]:
    """Construct the complete coordinate-aligned rectangular cover."""

    image = np.asarray(scalar_map)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("scalar_map must be a non-empty two-dimensional array")
    if image.dtype != np.uint8:
        raise TypeError("scalar_map must be uint8 after min--max normalization")
    if not 0.0 <= purity_threshold <= 1.0:
        raise ValueError("purity_threshold must lie in [0, 1]")
    if intensity_threshold < 0 or variance_threshold < 0:
        raise ValueError("thresholds must be non-negative")

    smoothed = cv2.GaussianBlur(np.ascontiguousarray(image), (3, 3), 0)
    dx = cv2.Scharr(smoothed, cv2.CV_64F, 1, 0)
    dy = cv2.Scharr(smoothed, cv2.CV_64F, 0, 1)
    gradient = cv2.magnitude(dx, dy)
    seed_order = np.argsort(gradient.ravel(), kind="stable")
    values = smoothed.astype(np.float64)
    integral = cv2.integral(values, sdepth=cv2.CV_64F)
    squared_integral = cv2.integral(values * values, sdepth=cv2.CV_64F)
    height, width = smoothed.shape
    coverage = np.zeros((height, width), dtype=bool)
    granules: list[GranuleTuple] = []

    for flat_index in seed_order:
        center_y, center_x = divmod(int(flat_index), width)
        if coverage[center_y, center_x]:
            continue
        radius_x = radius_y = 0
        while True:
            grew_x = False
            grew_y = False
            current = granule_bounds(
                ((center_y, center_x), radius_x, radius_y), height, width
            )
            candidate_x = granule_bounds(
                ((center_y, center_x), radius_x + 1, radius_y), height, width
            )
            if candidate_x != current and _criterion(
                smoothed, integral, squared_integral, center_y, center_x,
                radius_x + 1, radius_y, purity_threshold,
                intensity_threshold, variance_threshold,
            ):
                radius_x += 1
                grew_x = True

            current = granule_bounds(
                ((center_y, center_x), radius_x, radius_y), height, width
            )
            candidate_y = granule_bounds(
                ((center_y, center_x), radius_x, radius_y + 1), height, width
            )
            if candidate_y != current and _criterion(
                smoothed, integral, squared_integral, center_y, center_x,
                radius_x, radius_y + 1, purity_threshold,
                intensity_threshold, variance_threshold,
            ):
                radius_y += 1
                grew_y = True
            if not grew_x and not grew_y:
                break

        granule = ((center_y, center_x), radius_x, radius_y)
        top, bottom, left, right = granule_bounds(granule, height, width)
        coverage[top : bottom + 1, left : right + 1] = True
        granules.append(granule)

    if not coverage.all():
        raise RuntimeError("granular construction failed to cover the full lattice")
    return granules


def img2graph(
    img: np.ndarray,
    purity: float = 0.85,
    threshold: int = 15,
    var_threshold: float = 40.0,
    min_size: int = 0,
) -> list[GranuleTuple]:
    """Compatibility wrapper around :func:`construct_granular_matrix`.

    ``min_size`` is accepted for callers from the earlier code layout.  The
    paper's Algorithm 1 has no forced minimum radius, so only zero is valid.
    """

    if min_size != 0:
        raise ValueError("Algorithm 1 does not use a forced minimum radius")
    return construct_granular_matrix(
        img,
        purity_threshold=purity,
        intensity_threshold=threshold,
        variance_threshold=var_threshold,
    )


__all__ = [
    "Granule",
    "GranuleTuple",
    "construct_granular_matrix",
    "granule_bounds",
    "img2graph",
    "normalize_scalar_map",
]
