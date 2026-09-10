"""Run GraM-Select from precomputed TTA uncertainty maps and SAM 2 features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gram_sam.granular import construct_granular_matrix, normalize_scalar_map
from gram_sam.selection import compute_challenge_score, greedy_select


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GraM-Select at cumulative 5%, 10%, and 20% labeled-data budgets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase1"))
    parser.add_argument("--budgets", type=int, nargs="+", default=(5, 10, 20))
    return parser.parse_args()


def _load_feature(path: Path) -> np.ndarray:
    feature = np.asarray(np.load(path), dtype=np.float64)
    if feature.ndim >= 3:
        feature = feature.mean(axis=tuple(range(feature.ndim - 2, feature.ndim)))
    return feature.reshape(-1)


def main() -> None:
    args = parse_args()
    budgets = tuple(sorted(set(args.budgets)))
    if not budgets or budgets[0] <= 0 or budgets[-1] > 100:
        raise ValueError("budgets must be unique percentages in (0, 100]")

    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_name", "uncertainty_path", "feature_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain columns: {sorted(required)}")

    base = args.manifest.parent
    names: list[str] = []
    features: list[np.ndarray] = []
    challenge_scores: list[float] = []
    for row in rows:
        uncertainty_path = (base / row["uncertainty_path"]).resolve()
        feature_path = (base / row["feature_path"]).resolve()
        uncertainty = np.asarray(np.load(uncertainty_path), dtype=np.float64)
        granules = construct_granular_matrix(normalize_scalar_map(uncertainty))
        challenge, _ = compute_challenge_score(granules, uncertainty)
        names.append(row["image_name"])
        features.append(_load_feature(feature_path))
        challenge_scores.append(challenge)

    feature_matrix = np.stack(features)
    # Each acquisition round adds five percentage points of the full training
    # set. The greedy order is cumulative; only requested budgets are reported.
    maximum_count = int(np.floor(len(rows) * budgets[-1] / 100.0))
    result = greedy_select(challenge_scores, feature_matrix, maximum_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for budget in budgets:
        count = int(np.floor(len(rows) * budget / 100.0))
        selected = result.selected_indices[:count]
        output = args.output_dir / f"labeled_names_{budget}_select.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("selection_order", "image_name", "challenge_score"),
            )
            writer.writeheader()
            for order, index in enumerate(selected, start=1):
                writer.writerow(
                    {
                        "selection_order": order,
                        "image_name": names[index],
                        "challenge_score": f"{challenge_scores[index]:.12g}",
                    }
                )
        print(f"{budget}%: {len(selected)} samples -> {output}")


if __name__ == "__main__":
    main()
