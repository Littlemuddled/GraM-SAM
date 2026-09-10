"""Leakage-resistant active split and prompt-manifest helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def canonical_image_name(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name


def load_selected_image_names(csv_path: str | Path) -> tuple[str, ...]:
    table = pd.read_csv(csv_path)
    if "image_name" not in table.columns:
        raise ValueError("active-selection CSV must contain an image_name column")
    names = tuple(canonical_image_name(value) for value in table["image_name"])
    if len(names) == 0 or any(not name for name in names):
        raise ValueError("active-selection CSV must contain non-empty image names")
    if len(names) != len(set(names)):
        raise ValueError("active-selection CSV contains duplicate image names")
    return names


def resolve_selected_image_names(
    selected_image_names: Sequence[str],
    image_directory: str | Path,
    mask_directory: str | Path | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve a saved active selection against the materialized training pool.

    Saved selections can contain slices removed during a foreground-only
    dataset export. Preserve the CSV as evidence, but return the usable
    intersection and unavailable entries separately so callers can record the
    discrepancy instead of silently substituting samples.
    """

    suffixes = {".png", ".jpg", ".jpeg"}
    image_directory = Path(image_directory)
    image_stems = {
        path.stem
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }
    mask_stems = None
    if mask_directory is not None:
        mask_directory = Path(mask_directory)
        mask_stems = {
            path.stem
            for path in mask_directory.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        }

    resolved: list[str] = []
    unavailable: list[str] = []
    for value in selected_image_names:
        name = canonical_image_name(value)
        stem = Path(name).stem
        exists = stem in image_stems and (mask_stems is None or stem in mask_stems)
        (resolved if exists else unavailable).append(name)
    if not resolved:
        raise ValueError("none of the selected images exist in the materialized training pool")
    return tuple(resolved), tuple(unavailable)


def partition_active_pool(
    labeled_prompt_entries: Sequence[Mapping],
    unlabeled_prompt_entries: Sequence[Mapping],
    selected_image_names: Sequence[str],
) -> tuple[list[Mapping], list[Mapping]]:
    """Use the active-selection CSV, never a random prefix, to define the split."""

    selected = {canonical_image_name(name) for name in selected_image_names}
    labeled_by_name = {
        canonical_image_name(entry["image_name"]): entry
        for entry in labeled_prompt_entries
    }
    unlabeled_by_name = {
        canonical_image_name(entry["image_name"]): entry
        for entry in unlabeled_prompt_entries
    }
    missing_labeled = sorted(selected - set(labeled_by_name))
    if missing_labeled:
        raise ValueError(
            "selected images are missing from the labeled prompt manifest: "
            + ", ".join(missing_labeled[:5])
        )

    pool_names = set(labeled_by_name)
    expected_unlabeled = pool_names - selected
    missing_unlabeled = sorted(expected_unlabeled - set(unlabeled_by_name))
    if missing_unlabeled:
        raise ValueError(
            "unselected images are missing from the GCN prompt manifest: "
            + ", ".join(missing_unlabeled[:5])
        )

    labeled = [labeled_by_name[name] for name in sorted(selected)]
    unlabeled = [unlabeled_by_name[name] for name in sorted(expected_unlabeled)]
    return labeled, unlabeled
