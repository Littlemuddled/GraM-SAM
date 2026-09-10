"""Read-only dataset contracts for the four GraM-SAM datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import cv2
import numpy as np


DATASET_NAMES = ("spine", "Promise12", "ISIC2016", "BUSD")


@dataclass(frozen=True)
class PromptGroup:
    class_id: int
    coordinates_xy: np.ndarray
    point_labels: np.ndarray
    point_scores: np.ndarray | None = None
    box_xyxy: np.ndarray | None = None


@dataclass(frozen=True)
class PromptedSample:
    dataset: str
    split: str
    image_name: str
    image_path: Path
    mask_path: Path
    prompt_groups: tuple[PromptGroup, ...]


@dataclass(frozen=True)
class ImagePromptSample:
    dataset: str
    split: str
    image_name: str
    image_path: Path
    prompt_groups: tuple[PromptGroup, ...]


def _find_image(directory: Path, stem: str) -> Path:
    candidates = [directory / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg")]
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one image for {stem!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def _group_points(points: Sequence[Mapping]) -> tuple[PromptGroup, ...]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for point in points:
        class_id = int(point["class"])
        coordinate = np.asarray(point["coord"], dtype=np.float32)
        if coordinate.shape != (2,) or not np.isfinite(coordinate).all():
            raise ValueError("point coordinates must be finite [x, y] pairs")
        grouped.setdefault(class_id, []).append((float(coordinate[0]), float(coordinate[1])))
    return tuple(
        PromptGroup(
            class_id,
            np.asarray(coordinates, dtype=np.float32),
            np.ones(len(coordinates), dtype=np.int32),
            np.ones(len(coordinates), dtype=np.float32),
            None,
        )
        for class_id, coordinates in sorted(grouped.items())
    )


def _prompt_groups_from_record(record: Mapping) -> tuple[PromptGroup, ...]:
    if "prompt_groups" not in record:
        return _group_points(record.get("points", ()))
    groups = []
    seen: set[int] = set()
    for item in record["prompt_groups"]:
        class_id = int(item["class"])
        if class_id in seen:
            raise ValueError(f"duplicate prompt group for class {class_id}")
        seen.add(class_id)
        coordinates = np.asarray(item["coords"], dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
            raise ValueError("prompt-group coordinates must have shape [N, 2]")
        if not np.isfinite(coordinates).all():
            raise ValueError("prompt-group coordinates must be finite")
        labels = np.asarray(item.get("labels", np.ones(len(coordinates))), dtype=np.int32)
        if labels.shape != (len(coordinates),) or not np.isin(labels, (0, 1)).all():
            raise ValueError("prompt-group labels must be a binary vector matching coords")
        if not np.any(labels == 1):
            raise ValueError("every prompt group must contain at least one positive point")
        scores = np.asarray(item.get("scores", np.ones(len(coordinates))), dtype=np.float32)
        if scores.shape != (len(coordinates),) or not np.isfinite(scores).all():
            raise ValueError("prompt-group scores must be a finite vector matching coords")
        box = item.get("box")
        box_xyxy = None if box is None else np.asarray(box, dtype=np.float32)
        if box_xyxy is not None:
            if box_xyxy.shape != (4,) or not np.isfinite(box_xyxy).all():
                raise ValueError("prompt-group box must be a finite [x0, y0, x1, y1]")
            if box_xyxy[2] <= box_xyxy[0] or box_xyxy[3] <= box_xyxy[1]:
                raise ValueError("prompt-group box must have positive width and height")
        groups.append(PromptGroup(class_id, coordinates, labels, scores, box_xyxy))
    return tuple(sorted(groups, key=lambda group: group.class_id))


def iter_prompted_samples(
    data_base: str | Path,
    dataset: str,
    *,
    split: str = "val",
) -> Iterator[PromptedSample]:
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unknown dataset {dataset!r}")
    root = Path(data_base) / dataset
    prompt_path = root / split / "mask_points.json"
    records = json.loads(prompt_path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for record in records:
        stem = str(record["image"])
        if stem in seen:
            raise ValueError(f"duplicate prompt record for {stem}")
        seen.add(stem)
        image_path = _find_image(root / split / "images", stem)
        mask_path = root / split / "masks" / f"{stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        prompt_groups = _prompt_groups_from_record(record)
        if not prompt_groups:
            raise ValueError(f"sample {stem} has no prompts")
        yield PromptedSample(
            dataset=dataset,
            split=split,
            image_name=image_path.name,
            image_path=image_path,
            mask_path=mask_path,
            prompt_groups=prompt_groups,
        )


def iter_external_prompted_samples(
    data_base: str | Path,
    dataset: str,
    prompt_manifest: str | Path,
    *,
    split: str = "val",
) -> Iterator[PromptedSample]:
    """Load an image-only-generated prompt manifest against read-only targets."""

    if dataset not in DATASET_NAMES:
        raise ValueError(f"unknown dataset {dataset!r}")
    root = Path(data_base) / dataset
    records = json.loads(Path(prompt_manifest).read_text(encoding="utf-8"))
    seen: set[str] = set()
    for record in records:
        stem = str(record["image"])
        if stem in seen:
            raise ValueError(f"duplicate prompt record for {stem}")
        seen.add(stem)
        image_path = _find_image(root / split / "images", stem)
        mask_path = root / split / "masks" / f"{stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        prompt_groups = _prompt_groups_from_record(record)
        if not prompt_groups:
            raise ValueError(f"sample {stem} has no prompts")
        yield PromptedSample(
            dataset=dataset,
            split=split,
            image_name=image_path.name,
            image_path=image_path,
            mask_path=mask_path,
            prompt_groups=prompt_groups,
        )


def iter_image_only_prompt_samples(
    data_base: str | Path,
    dataset: str,
    prompt_manifest: str | Path,
    *,
    split: str = "train",
) -> Iterator[ImagePromptSample]:
    """Load image-only prompts without constructing or checking any mask path."""

    if dataset not in DATASET_NAMES:
        raise ValueError(f"unknown dataset {dataset!r}")
    root = Path(data_base) / dataset
    records = json.loads(Path(prompt_manifest).read_text(encoding="utf-8"))
    seen: set[str] = set()
    for record in records:
        stem = str(record["image"])
        if stem in seen:
            raise ValueError(f"duplicate prompt record for {stem}")
        seen.add(stem)
        image_path = _find_image(root / split / "images", stem)
        prompt_groups = _prompt_groups_from_record(record)
        if not prompt_groups:
            raise ValueError(f"sample {stem} has no prompts")
        yield ImagePromptSample(
            dataset=dataset,
            split=split,
            image_name=image_path.name,
            image_path=image_path,
            prompt_groups=prompt_groups,
        )


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"unable to read mask: {path}")
    return mask
