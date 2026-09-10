"""Calibrate target-free Stage-3 inference on selected training masks only.

The prompt manifest must be generated from images without opening masks.  Mask
access in this script is restricted to the explicitly selected labeled training
set; validation masks are never loaded.  The resulting fusion rule and global
probability threshold can then be fixed for a separate validation run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gram_sam.data_contract import load_selected_image_names  # noqa: E402
from gram_sam.datasets import (  # noqa: E402
    DATASET_NAMES,
    iter_external_prompted_samples,
    read_mask,
)
from gram_sam.evaluation import binary_segmentation_metrics, summarize_metrics  # noqa: E402
from gram_sam.stage3 import (  # noqa: E402
    Stage3Branch,
    fuse_candidate_probabilities,
)
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from scripts.train_stage3 import (  # noqa: E402
    FrozenFeatureCache,
    decode_sample,
    sha256,
)


FUSION_MODES = (
    "independent_best",
    "joint_best",
    "best_quality",
    "quality_weighted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--labeled-ratio", type=int, choices=(5, 10, 20), default=10)
    parser.add_argument("--phase2-labeled-prompts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--model-config", default="configs/sam2.1_hiera_s.yaml")
    parser.add_argument(
        "--feature-cache-root", type=Path, default=Path("runs/feature_cache")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/stage3_inference_calibration")
    )
    parser.add_argument("--max-positive-points", type=int, default=1)
    parser.add_argument("--max-negative-points", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
    )
    parser.add_argument(
        "--fusion-modes", nargs="+", choices=FUSION_MODES, default=FUSION_MODES
    )
    parser.add_argument(
        "--quality-temperatures", type=float, nargs="+", default=(0.03, 0.1, 0.3)
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_selected_calibration_samples(args: argparse.Namespace):
    selected_csv = (
        args.data_base
        / args.dataset
        / f"labeled_names_{args.labeled_ratio}_select.csv"
    )
    selected_stems = {
        Path(name).stem for name in load_selected_image_names(selected_csv)
    }
    samples = [
        sample
        for sample in iter_external_prompted_samples(
            args.data_base,
            args.dataset,
            args.phase2_labeled_prompts,
            split="train",
        )
        if Path(sample.image_name).stem in selected_stems
    ]
    actual = {Path(sample.image_name).stem for sample in samples}
    if actual != selected_stems or len(samples) != len(selected_stems):
        missing = sorted(selected_stems - actual)
        extras = sorted(actual - selected_stems)
        raise ValueError(
            "calibration prompts must cover each selected image exactly once; "
            f"missing={missing[:5]}, extras={extras[:5]}"
        )
    return selected_csv, samples


def fusion_configs(args: argparse.Namespace) -> list[tuple[str, float]]:
    configs = []
    for mode in args.fusion_modes:
        temperatures = (
            args.quality_temperatures if mode == "quality_weighted" else (0.1,)
        )
        configs.extend((mode, float(value)) for value in temperatures)
    return configs


def confusion_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    true_positive = float(np.logical_and(prediction, target).sum())
    prediction_count = float(prediction.sum())
    target_count = float(target.sum())
    union = prediction_count + target_count - true_positive
    return {
        "dice": (2.0 * true_positive + 1e-6)
        / (prediction_count + target_count + 1e-6),
        "iou": (true_positive + 1e-6) / (union + 1e-6),
        "precision": (true_positive + 1e-6) / (prediction_count + 1e-6),
        "recall": (true_positive + 1e-6) / (target_count + 1e-6),
    }


@torch.inference_mode()
def decode_probabilities(branches, predictor, cache, sample, decode_args, configs):
    features = cache.get(sample)
    decoded = []
    class_ids = None
    amp_enabled = str(decode_args.device).startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        for branch in branches:
            branch_decoded, class_ids = decode_sample(
                branch, predictor, features, sample, decode_args
            )
            decoded.append(branch_decoded)
        probabilities = {
            (mode, temperature): fuse_candidate_probabilities(
                decoded, mode=mode, temperature=temperature
            )
            .cpu()
            .numpy()
            for mode, temperature in configs
        }
    return probabilities, class_ids


def main() -> None:
    args = parse_args()
    required = (args.phase2_labeled_prompts, args.checkpoint, args.resume)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.thresholds or not all(0.0 < value < 1.0 for value in args.thresholds):
        raise ValueError("all thresholds must lie in (0, 1)")
    if not args.quality_temperatures or not all(
        value > 0.0 for value in args.quality_temperatures
    ):
        raise ValueError("all quality temperatures must be positive")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    selected_csv, samples = load_selected_calibration_samples(args)
    base_model = build_sam2(args.model_config, str(args.checkpoint), device=args.device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    predictor = SAM2ImagePredictor(base_model)
    branches = (
        Stage3Branch(base_model.sam_prompt_encoder, base_model.sam_mask_decoder).to(
            args.device
        ),
        Stage3Branch(base_model.sam_prompt_encoder, base_model.sam_mask_decoder).to(
            args.device
        ),
    )
    resume = torch.load(args.resume, map_location=args.device, weights_only=False)
    branches[0].load_state_dict(resume["branch_a"])
    branches[1].load_state_dict(resume["branch_b"])
    for branch in branches:
        branch.eval()
    cache = FrozenFeatureCache(
        predictor, args.feature_cache_root, sha256(args.checkpoint), args.device
    )
    decode_args = SimpleNamespace(
        max_positive_points=args.max_positive_points,
        max_negative_points=args.max_negative_points,
        device=args.device,
    )
    configs = fusion_configs(args)
    thresholds = tuple(float(value) for value in args.thresholds)
    score_lists = {
        (mode, temperature, threshold): {
            name: [] for name in ("dice", "iou", "precision", "recall")
        }
        for mode, temperature in configs
        for threshold in thresholds
    }
    started = time.perf_counter()
    for sample in tqdm(samples, desc="Calibrate selected-train inference"):
        probabilities, class_ids = decode_probabilities(
            branches, predictor, cache, sample, decode_args, configs
        )
        target = read_mask(sample.mask_path)
        per_image = {
            key: {name: [] for name in values}
            for key, values in score_lists.items()
        }
        for index, class_id in enumerate(class_ids):
            class_target = target == class_id
            for mode, temperature in configs:
                probability = probabilities[(mode, temperature)][index]
                for threshold in thresholds:
                    key = (mode, temperature, threshold)
                    metrics = confusion_metrics(probability >= threshold, class_target)
                    for name, value in metrics.items():
                        per_image[key][name].append(value)
        for key, values in per_image.items():
            for name, instances in values.items():
                score_lists[key][name].append(float(np.mean(instances)))

    candidates = []
    for (mode, temperature, threshold), values in score_lists.items():
        candidates.append(
            {
                "inference_fusion": mode,
                "quality_fusion_temperature": temperature,
                "prediction_threshold": threshold,
                **{name: float(np.mean(items)) for name, items in values.items()},
            }
        )
    candidates.sort(
        key=lambda item: (item["dice"], item["iou"], -abs(item["precision"] - item["recall"])),
        reverse=True,
    )
    selected = candidates[0]

    selected_metrics = []
    selected_config = [
        (
            str(selected["inference_fusion"]),
            float(selected["quality_fusion_temperature"]),
        )
    ]
    for sample in tqdm(samples, desc="Verify selected-train calibration", leave=False):
        probabilities, class_ids = decode_probabilities(
            branches, predictor, cache, sample, decode_args, selected_config
        )
        probability = probabilities[selected_config[0]]
        target = read_mask(sample.mask_path)
        per_image = [
            binary_segmentation_metrics(
                probability[index] >= float(selected["prediction_threshold"]),
                target == class_id,
            )
            for index, class_id in enumerate(class_ids)
        ]
        selected_metrics.extend(per_image)

    result = {
        "dataset": args.dataset,
        "labeled_ratio": args.labeled_ratio,
        "calibration_scope": "selected_training_masks_with_image_only_phase2_prompts",
        "validation_masks_opened": False,
        "selected_csv": str(selected_csv.resolve()),
        "selected_csv_sha256": sha256(selected_csv),
        "phase2_labeled_prompts": str(args.phase2_labeled_prompts.resolve()),
        "phase2_labeled_prompts_sha256": sha256(args.phase2_labeled_prompts),
        "stage3_checkpoint": str(args.resume.resolve()),
        "stage3_checkpoint_sha256": sha256(args.resume),
        "calibration_images": len(samples),
        "selected": {
            **selected,
            "class_instance_macro_with_hd95": summarize_metrics(selected_metrics),
        },
        "candidates": candidates,
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"{args.dataset}_{args.labeled_ratio}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "calibration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir.resolve()), **result}, indent=2))


if __name__ == "__main__":
    main()
