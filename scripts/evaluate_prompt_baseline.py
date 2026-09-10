"""Evaluate official SAM2 with the supplied per-class point prompts.

This is an oracle-point diagnostic because the supplied validation points were
derived from ground-truth masks.  It verifies the image/prompt/metric pipeline;
it is not the final automatic GraM-SAM result.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gram_sam.datasets import (  # noqa: E402
    DATASET_NAMES,
    iter_external_prompted_samples,
    iter_prompted_samples,
    read_mask,
    read_rgb,
)
from gram_sam.evaluation import (  # noqa: E402
    BinarySegmentationMetrics,
    binary_segmentation_metrics,
    prompt_connected_component,
    summarize_metrics,
)
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", default="configs/sam2.1_hiera_s.yaml")
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES)
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--max-points-per-class", type=int)
    parser.add_argument("--max-negative-points-per-class", type=int)
    parser.add_argument(
        "--independent-positive-candidates",
        action="store_true",
        help=(
            "Decode every retained positive point independently and select the "
            "mask with the highest model-predicted quality, without target access."
        ),
    )
    parser.add_argument("--use-box-prompts", action="store_true")
    parser.add_argument("--prompt-component-postprocess", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("runs/prompt_baseline"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_prompt_groups(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    prompt_groups,
    *,
    max_points_per_class: int | None = None,
    max_negative_points_per_class: int | None = None,
    use_box_prompts: bool = False,
    independent_positive_candidates: bool = False,
) -> dict[int, np.ndarray]:
    predictor.set_image(image)
    predictions: dict[int, np.ndarray] = {}
    for group in prompt_groups:
        coordinates = group.coordinates_xy
        labels = group.point_labels
        if max_points_per_class is not None:
            positive = np.flatnonzero(labels == 1)[:max_points_per_class]
        else:
            positive = np.flatnonzero(labels == 1)
        if max_negative_points_per_class is not None:
            negative = np.flatnonzero(labels == 0)[:max_negative_points_per_class]
        else:
            negative = np.flatnonzero(labels == 0)
        if not len(positive):
            raise ValueError(f"class {group.class_id} has no positive prompt")
        candidate_positive_sets = (
            [np.asarray([index], dtype=np.int64) for index in positive]
            if independent_positive_candidates
            else [positive]
        )
        best_mask = None
        best_score = -np.inf
        for candidate_positive in candidate_positive_sets:
            selected = np.concatenate([candidate_positive, negative])
            masks, scores, _ = predictor.predict(
                point_coords=coordinates[selected],
                point_labels=labels[selected],
                box=group.box_xyxy if use_box_prompts else None,
                multimask_output=True,
                return_logits=False,
            )
            best = int(np.argmax(scores))
            if float(scores[best]) > best_score:
                best_score = float(scores[best])
                best_mask = np.asarray(masks[best], dtype=bool)
        if best_mask is None:
            raise RuntimeError(f"no prediction candidate for class {group.class_id}")
        predictions[group.class_id] = best_mask
    return predictions


def evaluate_dataset(
    predictor: SAM2ImagePredictor,
    args: argparse.Namespace,
    dataset: str,
    run_root: Path,
) -> dict:
    if args.prompt_manifest is None:
        sample_iterator = iter_prompted_samples(args.data_base, dataset, split=args.split)
    else:
        sample_iterator = iter_external_prompted_samples(
            args.data_base, dataset, args.prompt_manifest, split=args.split
        )
    samples = list(sample_iterator)
    if args.limit is not None:
        samples = samples[: args.limit]
    dataset_dir = run_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    image_metrics: list[BinarySegmentationMetrics] = []
    class_metrics: list[BinarySegmentationMetrics] = []
    started = time.perf_counter()

    for sample in tqdm(samples, desc=f"SAM2 oracle points: {dataset}"):
        image = read_rgb(sample.image_path)
        target = read_mask(sample.mask_path)
        predictions = predict_prompt_groups(
            predictor,
            image,
            sample.prompt_groups,
            max_points_per_class=args.max_points_per_class,
            max_negative_points_per_class=args.max_negative_points_per_class,
            use_box_prompts=args.use_box_prompts,
            independent_positive_candidates=args.independent_positive_candidates,
        )
        per_image: list[BinarySegmentationMetrics] = []
        for group in sample.prompt_groups:
            class_target = target == group.class_id
            prediction = predictions[group.class_id]
            if args.prompt_component_postprocess:
                positive = np.flatnonzero(group.point_labels == 1)
                prediction = prompt_connected_component(
                    prediction, group.coordinates_xy[positive[0]]
                )
            metrics = binary_segmentation_metrics(
                prediction, class_target
            )
            per_image.append(metrics)
            class_metrics.append(metrics)
            rows.append(
                {
                    "dataset": dataset,
                    "image_name": sample.image_name,
                    "class_id": group.class_id,
                    "point_count": min(
                        int(np.count_nonzero(group.point_labels == 1)),
                        args.max_points_per_class
                        or int(np.count_nonzero(group.point_labels == 1)),
                    ),
                    "negative_point_count": min(
                        int(np.count_nonzero(group.point_labels == 0)),
                        args.max_negative_points_per_class
                        if args.max_negative_points_per_class is not None
                        else int(np.count_nonzero(group.point_labels == 0)),
                    ),
                    "box_prompt": bool(args.use_box_prompts and group.box_xyxy is not None),
                    "independent_positive_candidates": bool(
                        args.independent_positive_candidates
                    ),
                    "prompt_component_postprocess": bool(
                        args.prompt_component_postprocess
                    ),
                    **metrics.as_dict(),
                }
            )
        mean = summarize_metrics(per_image)
        image_metrics.append(
            BinarySegmentationMetrics(
                dice=float(mean["dice"]),
                iou=float(mean["iou"]),
                hd95=float(mean["hd95"]),
                precision=float(mean["precision"]),
                recall=float(mean["recall"]),
            )
        )

    with (dataset_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": dataset,
        "split": args.split,
        "evaluation_protocol": (
            "ground_truth_point_prompt_baseline"
            if args.prompt_manifest is None
            else "automatic_image_only_prompt_manifest"
        ),
        "hd95_unit": "pixel",
        "empty_prediction_policy": "image_diagonal",
        "image_macro": summarize_metrics(image_metrics),
        "class_instance_macro": summarize_metrics(class_metrics),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (dataset_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.max_points_per_class is not None and args.max_points_per_class < 1:
        raise ValueError("max_points_per_class must be positive")
    if (
        args.max_negative_points_per_class is not None
        and args.max_negative_points_per_class < 0
    ):
        raise ValueError("max_negative_points_per_class must be non-negative")
    if args.prompt_manifest is not None and len(args.datasets) != 1:
        raise ValueError("an external prompt manifest requires exactly one dataset")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    model = build_sam2(args.model_config, str(args.checkpoint), device=args.device)
    predictor = SAM2ImagePredictor(model)
    predictor.model.eval()
    summaries = [
        evaluate_dataset(predictor, args, dataset, run_root)
        for dataset in args.datasets
    ]
    metadata = {
        "run_id": run_id,
        "command_arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_sha256": sha256(args.checkpoint),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if str(args.device).startswith("cuda") else None,
        "summaries": summaries,
    }
    (run_root / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(run_root.resolve()), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
