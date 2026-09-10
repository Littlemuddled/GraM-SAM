"""GraM-SAM: granular-matrix-driven active semi-supervised segmentation."""

from .granular import construct_granular_matrix, normalize_scalar_map
from .gmbs import granular_matrix_block_swap
from .selection import bernoulli_variance_from_tta, compute_challenge_score, greedy_select

__all__ = [
    "bernoulli_variance_from_tta",
    "compute_challenge_score",
    "construct_granular_matrix",
    "granular_matrix_block_swap",
    "greedy_select",
    "normalize_scalar_map",
]
