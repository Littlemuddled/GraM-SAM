"""Leakage-resistant, resource-efficient GraM-SAM Phase-3 trainer.

One frozen SAM2 image encoder is shared by two trainable prompt/decoder
branches.  Image features are cached inside the experiment workspace; the
unlabeled loader never constructs or opens a mask path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gram_sam.data_contract import (  # noqa: E402
    load_selected_image_names,
    resolve_selected_image_names,
)
from gram_sam.datasets import (  # noqa: E402
    DATASET_NAMES,
    iter_external_prompted_samples,
    iter_image_only_prompt_samples,
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
from gram_sam.gmbs import granular_matrix_block_swap  # noqa: E402
from gram_sam.stage3 import (  # noqa: E402
    Stage3Branch,
    confidence_weighted_cps_loss,
    decode_branch,
    fuse_candidate_probabilities,
    gaussian_rampup,
    mask_interior_prompt_variants,
    pack_prompt_groups,
    perturb_trainable_parameters,
    sample_spatial_transform,
    select_quality_candidates,
    supervised_branch_loss,
    transform_prompt_groups,
    transform_spatial_tensor,
    weak_feature_view,
)
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--labeled-ratio", type=int, choices=(5, 10, 20), default=10)
    parser.add_argument("--phase2-train-prompts", type=Path, required=True)
    parser.add_argument(
        "--phase2-labeled-prompts",
        type=Path,
        help=(
            "Optional image-only Phase-2 prompts for the selected labeled images. "
            "Targets remain restricted to the selected-mask contract."
        ),
    )
    parser.add_argument(
        "--mix-labeled-prompt-sources",
        action="store_true",
        help=(
            "Train on both target-derived interior prompts and image-only Phase-2 "
            "prompts for each selected image. Requires --phase2-labeled-prompts."
        ),
    )
    parser.add_argument("--phase2-validation-prompts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", default="configs/sam2.1_hiera_s.yaml")
    parser.add_argument("--output-root", type=Path, default=Path("runs/stage3"))
    parser.add_argument("--feature-cache-root", type=Path, default=Path("runs/feature_cache"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation used for the paper's effective batch size of 4.",
    )
    parser.add_argument("--quality-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--supervised-loss",
        choices=("bce_dice", "focal_tversky"),
        default="focal_tversky",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument(
        "--mask-derived-prompt-variants",
        type=int,
        default=0,
        help=(
            "Extra positive-point variants sampled only from selected training "
            "masks; validation masks are never used for prompt generation."
        ),
    )
    parser.add_argument("--cps-weight", type=float, default=0.1)
    parser.add_argument("--cps-rampup-epochs", type=int, default=5)
    parser.add_argument("--pseudo-quality-threshold", type=float, default=0.0)
    parser.add_argument("--pixel-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--phase2-confidence-quantile", type=float, default=0.0)
    parser.add_argument("--max-positive-points", type=int, default=1)
    parser.add_argument("--max-negative-points", type=int, default=0)
    parser.add_argument("--disable-gmbs", action="store_true")
    parser.add_argument("--gmbs-high-value-sigma", type=float, default=0.5)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--branch-perturbation", type=float, default=1e-4)
    parser.add_argument(
        "--disable-weak-view-augmentation",
        action="store_true",
        help="Disable spatially corresponding weak feature-view augmentation.",
    )
    parser.add_argument(
        "--weak-view-gain-range",
        type=float,
        default=0.05,
        help="Independent multiplicative feature jitter range for each peer view.",
    )
    parser.add_argument(
        "--weak-view-bias-range",
        type=float,
        default=0.01,
        help="Independent additive jitter range in feature-standard-deviation units.",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--ensemble-resume",
        type=Path,
        action="append",
        default=[],
        help="Additional Phase-3 checkpoints for evaluation-only probability ensembling.",
    )
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument(
        "--inference-fusion",
        choices=(
            "independent_best",
            "joint_best",
            "best_quality",
            "quality_weighted",
        ),
        default="independent_best",
    )
    parser.add_argument("--quality-fusion-temperature", type=float, default=0.1)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument(
        "--prompt-component-postprocess",
        action="store_true",
        help="Keep only the predicted component containing/nearest the positive prompt.",
    )
    parser.add_argument(
        "--prediction-export-dir",
        type=Path,
        help=(
            "Optional directory for exporting predictions from a "
            "prespecified image-name list. Exports never alter validation metrics."
        ),
    )
    parser.add_argument(
        "--prediction-export-names",
        type=Path,
        help=(
            "UTF-8 text file containing one exact validation image name or stem per "
            "line. Requires --prediction-export-dir."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenFeatureCache:
    def __init__(
        self,
        predictor: SAM2ImagePredictor,
        root: Path,
        checkpoint_hash: str,
        device: str,
    ) -> None:
        self.predictor = predictor
        self.root = root / checkpoint_hash[:16]
        self.checkpoint_hash = checkpoint_hash
        self.device = device

    def get(self, sample) -> tuple[torch.Tensor, list[torch.Tensor], tuple[int, int]]:
        source = sample.image_path
        stat = source.stat()
        cache_path = self.root / sample.dataset / sample.split / f"{source.stem}.pt"
        payload = None
        if cache_path.is_file():
            candidate = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (
                candidate.get("source_size") == stat.st_size
                and candidate.get("source_mtime_ns") == stat.st_mtime_ns
                and candidate.get("checkpoint_sha256") == self.checkpoint_hash
            ):
                payload = candidate
        if payload is None:
            image = read_rgb(source)
            self.predictor.set_image(image)
            payload = {
                "image_embedding": self.predictor._features["image_embed"].detach().cpu().half(),
                "high_resolution_features": [
                    value.detach().cpu().half()
                    for value in self.predictor._features["high_res_feats"]
                ],
                "original_hw": tuple(int(value) for value in image.shape[:2]),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "checkpoint_sha256": self.checkpoint_hash,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(f".tmp_{os.getpid()}.pt")
            torch.save(payload, temporary)
            os.replace(temporary, cache_path)
        image_embedding = payload["image_embedding"].to(
            self.device, non_blocking=True
        ).clone()
        high_resolution = [
            value.to(self.device, non_blocking=True).clone()
            for value in payload["high_resolution_features"]
        ]
        original_hw = tuple(int(value) for value in payload["original_hw"])
        return image_embedding, high_resolution, original_hw


def load_contracts(args: argparse.Namespace):
    dataset_root = args.data_base / args.dataset
    selected_csv = dataset_root / f"labeled_names_{args.labeled_ratio}_select.csv"
    selected_csv_names = load_selected_image_names(selected_csv)
    selected_names, unavailable_selected_names = resolve_selected_image_names(
        selected_csv_names,
        dataset_root / "train" / "images",
        dataset_root / "train" / "masks",
    )
    selected_stems = {Path(name).stem for name in selected_names}
    oracle_labeled = [
        sample
        for sample in iter_prompted_samples(args.data_base, args.dataset, split="train")
        if Path(sample.image_name).stem in selected_stems
    ]
    derived_labeled = []
    if args.mask_derived_prompt_variants:
        prompt_generator = np.random.default_rng(args.seed + 3103)
        for sample in oracle_labeled:
            variants = mask_interior_prompt_variants(
                read_mask(sample.mask_path),
                [group.class_id for group in sample.prompt_groups],
                number_of_variants=args.mask_derived_prompt_variants,
                generator=prompt_generator,
            )
            derived_labeled.extend(
                replace(sample, prompt_groups=groups) for groups in variants
            )
    if args.phase2_labeled_prompts is None:
        labeled = oracle_labeled
    else:
        automatic_labeled = [
            sample
            for sample in iter_external_prompted_samples(
            args.data_base,
            args.dataset,
            args.phase2_labeled_prompts,
            split="train",
            )
            if Path(sample.image_name).stem in selected_stems
        ]
        automatic_stems = {
            Path(sample.image_name).stem for sample in automatic_labeled
        }
        if automatic_stems != selected_stems:
            missing = sorted(selected_stems - automatic_stems)
            raise ValueError(
                f"automatic labeled-prompt contract mismatch; missing={missing[:5]}"
            )
        labeled = (
            oracle_labeled + automatic_labeled
            if args.mix_labeled_prompt_sources
            else automatic_labeled
        )
    labeled = labeled + derived_labeled
    unlabeled = list(
        iter_image_only_prompt_samples(
            args.data_base,
            args.dataset,
            args.phase2_train_prompts,
            split="train",
        )
    )
    validation = list(
        iter_external_prompted_samples(
            args.data_base,
            args.dataset,
            args.phase2_validation_prompts,
            split="val",
        )
    )
    labeled_stems = {Path(sample.image_name).stem for sample in labeled}
    unlabeled_stems = {Path(sample.image_name).stem for sample in unlabeled}
    if labeled_stems != selected_stems:
        missing = sorted(selected_stems - labeled_stems)
        raise ValueError(f"selected labeled contract mismatch; missing={missing[:5]}")
    if labeled_stems & unlabeled_stems:
        raise ValueError("labeled and unlabeled Phase-3 pools overlap")
    all_train_stems = {
        path.stem
        for path in (dataset_root / "train" / "images").iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    if labeled_stems | unlabeled_stems != all_train_stems:
        raise ValueError("labeled plus unlabeled prompts do not exactly cover the training images")
    if args.validation_limit is not None:
        validation = validation[: args.validation_limit]
    return (
        selected_csv,
        labeled,
        unlabeled,
        validation,
        selected_csv_names,
        unavailable_selected_names,
    )


def target_tensor(sample, class_ids, original_hw, device: str) -> torch.Tensor:
    target = read_mask(sample.mask_path)
    if tuple(target.shape) != tuple(original_hw):
        raise ValueError(f"image/mask shape mismatch for {sample.image_name}")
    values = np.stack([(target == class_id) for class_id in class_ids]).astype(np.float32)
    return torch.from_numpy(values).to(device)


def spatially_corresponding_weak_views(features, sample, args, generator):
    """Build two weak views with common geometry and aligned Phase-2 prompts."""

    if args.disable_weak_view_augmentation:
        return features, features, sample, None
    transform = sample_spatial_transform(generator)
    view_a = weak_feature_view(
        features,
        transform,
        generator=generator,
        gain_range=args.weak_view_gain_range,
        bias_range=args.weak_view_bias_range,
    )
    view_b = weak_feature_view(
        features,
        transform,
        generator=generator,
        gain_range=args.weak_view_gain_range,
        bias_range=args.weak_view_bias_range,
    )
    transformed_sample = replace(
        sample,
        prompt_groups=transform_prompt_groups(
            sample.prompt_groups, features[2], transform
        ),
    )
    return view_a, view_b, transformed_sample, transform


def prompt_group_confidences(sample) -> np.ndarray:
    values = []
    for group in sample.prompt_groups:
        positive = np.flatnonzero(group.point_labels == 1)
        if not len(positive):
            raise ValueError(f"class {group.class_id} has no positive prompt")
        values.append(
            float(group.point_scores[positive[0]])
            if group.point_scores is not None
            else 1.0
        )
    return np.asarray(values, dtype=np.float32)


def decode_sample(branch, predictor, features, sample, args, *, adapted=None):
    image_embedding, high_resolution, original_hw = features
    coordinates, labels, class_ids = pack_prompt_groups(
        sample.prompt_groups,
        max_positive_points=args.max_positive_points,
        max_negative_points=args.max_negative_points,
    )
    decoded = decode_branch(
        branch,
        predictor,
        adapted if adapted is not None else image_embedding,
        high_resolution,
        coordinates,
        labels,
        original_hw=original_hw,
        image_embeddings_are_adapted=adapted is not None,
        multimask_output=True,
    )
    return decoded, class_ids


@torch.inference_mode()
def evaluate(
    branches,
    predictor,
    cache,
    samples,
    args,
    run_dir: Path,
    epoch: int,
) -> dict:
    for branch in branches:
        branch.eval()
    rows = []
    image_metrics = []
    class_metrics = []
    export_records = []
    export_names = None
    export_dir = None
    if args.prediction_export_names is not None:
        export_names = {
            line.strip()
            for line in args.prediction_export_names.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        export_dir = (
            args.prediction_export_dir
            / f"{args.dataset}_{args.labeled_ratio}"
            / run_dir.name
            / f"epoch_{epoch:04d}"
        )
        export_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    amp_enabled = str(args.device).startswith("cuda")
    for sample in tqdm(samples, desc=f"Validate epoch {epoch}", leave=False):
        features = cache.get(sample)
        decoded_branches = []
        class_ids = None
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            for branch in branches:
                decoded, branch_class_ids = decode_sample(
                    branch, predictor, features, sample, args
                )
                decoded_branches.append(decoded)
                class_ids = branch_class_ids
        ensemble = (
            fuse_candidate_probabilities(
                decoded_branches,
                mode=args.inference_fusion,
                temperature=args.quality_fusion_temperature,
            )
            .cpu()
            .numpy()
            >= args.prediction_threshold
        )
        target = read_mask(sample.mask_path)
        per_image = []
        exported_predictions = []
        for index, class_id in enumerate(class_ids):
            prediction = ensemble[index]
            if args.prompt_component_postprocess:
                group = next(
                    group
                    for group in sample.prompt_groups
                    if int(group.class_id) == int(class_id)
                )
                positive = np.flatnonzero(group.point_labels == 1)
                prediction = prompt_connected_component(
                    prediction, group.coordinates_xy[positive[0]]
                )
            exported_predictions.append(np.asarray(prediction, dtype=np.uint8))
            metrics = binary_segmentation_metrics(prediction, target == class_id)
            per_image.append(metrics)
            class_metrics.append(metrics)
            rows.append(
                {
                    "dataset": args.dataset,
                    "image_name": sample.image_name,
                    "class_id": class_id,
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
        if export_names is not None and (
            sample.image_name in export_names
            or Path(sample.image_name).stem in export_names
        ):
            export_path = export_dir / f"{Path(sample.image_name).stem}.npz"
            np.savez_compressed(
                export_path,
                prediction=np.stack(exported_predictions).astype(np.uint8),
                target=np.asarray(target),
                class_ids=np.asarray(class_ids, dtype=np.int64),
            )
            export_records.append(
                {
                    "dataset": args.dataset,
                    "image_name": sample.image_name,
                    "image_path": str(Path(sample.image_path).resolve()),
                    "image_sha256": sha256(Path(sample.image_path)),
                    "mask_path": str(Path(sample.mask_path).resolve()),
                    "mask_sha256": sha256(Path(sample.mask_path)),
                    "prediction_npz": str(export_path.resolve()),
                    "class_ids": [int(value) for value in class_ids],
                }
            )
    summary = {
        "epoch": epoch,
        "dataset": args.dataset,
        "prompt_scope": "phase2_image_only_automatic_validation_prompts",
        "inference_fusion": args.inference_fusion,
        "quality_fusion_temperature": args.quality_fusion_temperature,
        "prediction_threshold": args.prediction_threshold,
        "hd95_unit": "pixel",
        "image_macro": summarize_metrics(image_metrics),
        "class_instance_macro": summarize_metrics(class_metrics),
        "elapsed_seconds": time.perf_counter() - started,
    }
    validation_dir = run_dir / "validation" / f"epoch_{epoch:04d}"
    validation_dir.mkdir(parents=True, exist_ok=False)
    with (validation_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (validation_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if export_dir is not None:
        requested_missing = sorted(
            export_names
            - {
                record["image_name"] for record in export_records
            }
            - {
                Path(record["image_name"]).stem for record in export_records
            }
        )
        (export_dir / "export_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "labeled_ratio_percent": args.labeled_ratio,
                    "source_validation_run": str(run_dir.resolve()),
                    "source_epoch": epoch,
                    "prediction_threshold": args.prediction_threshold,
                    "inference_fusion": args.inference_fusion,
                    "quality_fusion_temperature": args.quality_fusion_temperature,
                    "requested_names_file": str(
                        args.prediction_export_names.resolve()
                    ),
                    "requested_count": len(export_names),
                    "exported_count": len(export_records),
                    "requested_missing": requested_missing,
                    "records": export_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return summary


def checkpoint_payload(epoch, branches, optimizer, scheduler, scaler, history, args):
    return {
        "epoch": epoch,
        "branch_a": branches[0].state_dict(),
        "branch_b": branches[1].state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
        "arguments": vars(args),
    }


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.accumulation_steps < 1 or args.epochs < 1:
        raise ValueError("epochs and accumulation steps must be positive")
    if args.mix_labeled_prompt_sources and args.phase2_labeled_prompts is None:
        raise ValueError(
            "--mix-labeled-prompt-sources requires --phase2-labeled-prompts"
        )
    if args.evaluation_only and args.resume is None:
        raise ValueError("--evaluation-only requires --resume")
    if args.ensemble_resume and not args.evaluation_only:
        raise ValueError("--ensemble-resume is restricted to --evaluation-only")
    if (args.prediction_export_dir is None) != (
        args.prediction_export_names is None
    ):
        raise ValueError(
            "--prediction-export-dir and --prediction-export-names must be provided together"
        )
    if not 0.0 < args.prediction_threshold < 1.0:
        raise ValueError("prediction_threshold must lie in (0, 1)")
    if args.quality_fusion_temperature <= 0.0:
        raise ValueError("quality_fusion_temperature must be positive")
    if args.focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative")
    if args.tversky_alpha < 0.0 or args.tversky_beta < 0.0:
        raise ValueError("Tversky weights must be non-negative")
    if args.tversky_alpha + args.tversky_beta <= 0.0:
        raise ValueError("at least one Tversky weight must be positive")
    if args.mask_derived_prompt_variants < 0:
        raise ValueError("mask_derived_prompt_variants must be non-negative")
    if args.weak_view_gain_range < 0.0 or args.weak_view_gain_range >= 1.0:
        raise ValueError("weak_view_gain_range must lie in [0, 1)")
    if args.weak_view_bias_range < 0.0:
        raise ValueError("weak_view_bias_range must be non-negative")
    required_paths = [
        args.checkpoint,
        args.phase2_train_prompts,
        args.phase2_validation_prompts,
    ]
    if args.phase2_labeled_prompts is not None:
        required_paths.append(args.phase2_labeled_prompts)
    if args.prediction_export_names is not None:
        required_paths.append(args.prediction_export_names)
    required_paths.extend(args.ensemble_resume)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    (
        selected_csv,
        labeled,
        unlabeled,
        validation,
        selected_csv_names,
        unavailable_selected_names,
    ) = load_contracts(args)
    if not 0.0 <= args.phase2_confidence_quantile < 1.0:
        raise ValueError("phase2_confidence_quantile must lie in [0, 1)")
    all_phase2_confidences = np.concatenate(
        [prompt_group_confidences(sample) for sample in unlabeled]
    )
    phase2_confidence_threshold = float(
        np.quantile(all_phase2_confidences, args.phase2_confidence_quantile)
    )
    checkpoint_hash = sha256(args.checkpoint)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"{args.dataset}_{args.labeled_ratio}" / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)

    base_model = build_sam2(args.model_config, str(args.checkpoint), device=args.device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    predictor = SAM2ImagePredictor(base_model)
    branch_a = Stage3Branch(
        base_model.sam_prompt_encoder, base_model.sam_mask_decoder
    ).to(args.device)
    branch_b = Stage3Branch(
        base_model.sam_prompt_encoder, base_model.sam_mask_decoder
    ).to(args.device)
    perturb_trainable_parameters(branch_b, args.branch_perturbation, args.seed + 1)
    branches = (branch_a, branch_b)
    parameters = [
        parameter
        for branch in branches
        for parameter in branch.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = args.steps_per_epoch or (
        len(unlabeled) if args.cps_weight > 0 else len(labeled)
    )
    optimizer_steps = args.epochs * math.ceil(steps_per_epoch / args.accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, optimizer_steps), eta_min=args.learning_rate * 0.05
    )
    amp_enabled = str(args.device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    cache = FrozenFeatureCache(
        predictor, args.feature_cache_root, checkpoint_hash, args.device
    )
    start_epoch = 1
    history = []
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=args.device, weights_only=False)
        branch_a.load_state_dict(resume["branch_a"])
        branch_b.load_state_dict(resume["branch_b"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        history = list(resume.get("history", []))
        start_epoch = int(resume["epoch"]) + 1
    if args.ensemble_resume:
        ensemble_branches = list(branches)
        for ensemble_checkpoint in args.ensemble_resume:
            payload = torch.load(
                ensemble_checkpoint, map_location=args.device, weights_only=False
            )
            extra_a = Stage3Branch(
                base_model.sam_prompt_encoder, base_model.sam_mask_decoder
            ).to(args.device)
            extra_b = Stage3Branch(
                base_model.sam_prompt_encoder, base_model.sam_mask_decoder
            ).to(args.device)
            extra_a.load_state_dict(payload["branch_a"])
            extra_b.load_state_dict(payload["branch_b"])
            ensemble_branches.extend((extra_a, extra_b))
        branches = tuple(ensemble_branches)

    run_metadata = {
        "run_id": run_id,
        "dataset": args.dataset,
        "pipeline": "GraM-SAM three-phase training",
        "selected_csv": str(selected_csv.resolve()),
        "selected_csv_sha256": sha256(selected_csv),
        "selected_csv_entries": len(selected_csv_names),
        "selected_unavailable_images": list(unavailable_selected_names),
        "selection_resolution": "selected_csv_intersection_with_available_image_mask_pool",
        "phase2_train_prompts": str(args.phase2_train_prompts.resolve()),
        "phase2_train_prompts_sha256": sha256(args.phase2_train_prompts),
        "phase2_labeled_prompts": (
            str(args.phase2_labeled_prompts.resolve())
            if args.phase2_labeled_prompts is not None
            else None
        ),
        "phase2_labeled_prompts_sha256": (
            sha256(args.phase2_labeled_prompts)
            if args.phase2_labeled_prompts is not None
            else None
        ),
        "phase2_validation_prompts": str(args.phase2_validation_prompts.resolve()),
        "phase2_validation_prompts_sha256": sha256(args.phase2_validation_prompts),
        "sam2_checkpoint_sha256": checkpoint_hash,
        "labeled_images": len({Path(sample.image_name).stem for sample in labeled}),
        "labeled_prompt_records": len(labeled),
        "labeled_prompt_variants_per_image": len(labeled) / len(
            {Path(sample.image_name).stem for sample in labeled}
        ),
        "unlabeled_images": len(unlabeled),
        "validation_images": len(validation),
        "shared_frozen_encoder": True,
        "trainable_dual_prompt_decoder_branches": True,
        "inference_branch_count": len(branches),
        "phase2_prompt_confidence_threshold": phase2_confidence_threshold,
        "arguments": {
            key: (
                [str(item) if isinstance(item, Path) else item for item in value]
                if isinstance(value, list)
                else str(value) if isinstance(value, Path) else value
            )
            for key, value in vars(args).items()
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if amp_enabled else None,
    }
    (run_dir / "run.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    metrics_path = run_dir / "training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(
            [
                "epoch", "train_loss", "supervised_loss", "cps_loss", "cps_weight",
                "accepted_pseudo_instances", "gmbs_mask_fraction", "learning_rate",
                "val_dice", "val_iou", "val_hd95",
            ]
        )

    if args.evaluation_only:
        evaluated_epoch = start_epoch - 1
        validation_summary = evaluate(
            branches,
            predictor,
            cache,
            validation,
            args,
            run_dir,
            evaluated_epoch,
        )
        final = {
            **run_metadata,
            "best_validation_dice": float(
                validation_summary["image_macro"]["dice"]
            ),
            "elapsed_seconds": validation_summary["elapsed_seconds"],
            "epochs_completed": 0,
            "evaluation_only_checkpoint_epoch": evaluated_epoch,
            "validation": validation_summary,
        }
        (run_dir / "final.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8"
        )
        print(json.dumps({"run_dir": str(run_dir.resolve()), **final}, indent=2))
        return

    best_dice = -1.0
    stale = 0
    generator = np.random.default_rng(args.seed)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        for branch in branches:
            branch.train()
        optimizer.zero_grad(set_to_none=True)
        labeled_order = generator.permutation(len(labeled))
        unlabeled_order = generator.permutation(len(unlabeled))
        losses = []
        supervised_losses = []
        cps_losses = []
        accepted_total = 0
        swap_fractions = []
        current_cps_weight = args.cps_weight * gaussian_rampup(
            epoch - 1, args.cps_rampup_epochs
        )
        progress = tqdm(range(steps_per_epoch), desc=f"Stage3 epoch {epoch}/{args.epochs}")
        for step in progress:
            labeled_sample = labeled[int(labeled_order[step % len(labeled_order)])]
            labeled_features = cache.get(labeled_sample)
            (
                labeled_view_a,
                labeled_view_b,
                labeled_view_sample,
                labeled_transform,
            ) = spatially_corresponding_weak_views(
                labeled_features, labeled_sample, args, generator
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                decoded_a, class_ids = decode_sample(
                    branch_a, predictor, labeled_view_a, labeled_view_sample, args
                )
                decoded_b, _ = decode_sample(
                    branch_b, predictor, labeled_view_b, labeled_view_sample, args
                )
                target = target_tensor(
                    labeled_sample, class_ids, labeled_features[2], args.device
                )
                if labeled_transform is not None:
                    target = transform_spatial_tensor(target, labeled_transform)
                supervised_a, _ = supervised_branch_loss(
                    decoded_a,
                    target,
                    quality_weight=args.quality_loss_weight,
                    loss_mode=args.supervised_loss,
                    focal_gamma=args.focal_gamma,
                    tversky_alpha=args.tversky_alpha,
                    tversky_beta=args.tversky_beta,
                )
                supervised_b, _ = supervised_branch_loss(
                    decoded_b,
                    target,
                    quality_weight=args.quality_loss_weight,
                    loss_mode=args.supervised_loss,
                    focal_gamma=args.focal_gamma,
                    tversky_alpha=args.tversky_alpha,
                    tversky_beta=args.tversky_beta,
                )
                supervised = supervised_a + supervised_b
                cps = supervised.sum() * 0.0
                accepted = 0

                if current_cps_weight > 0.0:
                    unlabeled_sample = unlabeled[
                        int(unlabeled_order[step % len(unlabeled_order)])
                    ]
                    unlabeled_features = cache.get(unlabeled_sample)
                    (
                        unlabeled_view_a,
                        unlabeled_view_b,
                        unlabeled_view_sample,
                        _,
                    ) = spatially_corresponding_weak_views(
                        unlabeled_features, unlabeled_sample, args, generator
                    )
                    image_embedding_a, _, _ = unlabeled_view_a
                    image_embedding_b, _, _ = unlabeled_view_b
                    adapted_a = branch_a.adapted_features(image_embedding_a)
                    adapted_b = branch_b.adapted_features(image_embedding_b)
                    clean_a, _ = decode_sample(
                        branch_a,
                        predictor,
                        unlabeled_view_a,
                        unlabeled_view_sample,
                        args,
                        adapted=adapted_a,
                    )
                    clean_b, _ = decode_sample(
                        branch_b,
                        predictor,
                        unlabeled_view_b,
                        unlabeled_view_sample,
                        args,
                        adapted=adapted_b,
                    )
                    clean_logits_a, clean_quality_a, _ = select_quality_candidates(clean_a)
                    clean_logits_b, clean_quality_b, _ = select_quality_candidates(clean_b)
                    probability_a = torch.sigmoid(clean_logits_a.float())
                    probability_b = torch.sigmoid(clean_logits_b.float())
                    phase2_validity = torch.from_numpy(
                        prompt_group_confidences(unlabeled_view_sample)
                        >= phase2_confidence_threshold
                    ).to(args.device)
                    if not args.disable_gmbs:
                        swapped = granular_matrix_block_swap(
                            adapted_a,
                            adapted_b,
                            probability_a.unsqueeze(0),
                            probability_b.unsqueeze(0),
                            high_value_sigma=args.gmbs_high_value_sigma,
                        )
                        swap_fractions.append(float(swapped.swap_mask.float().mean()))
                        mixed_a, _ = decode_sample(
                            branch_a,
                            predictor,
                            unlabeled_view_a,
                            unlabeled_view_sample,
                            args,
                            adapted=swapped.mixed_a,
                        )
                        mixed_b, _ = decode_sample(
                            branch_b,
                            predictor,
                            unlabeled_view_b,
                            unlabeled_view_sample,
                            args,
                            adapted=swapped.mixed_b,
                        )
                        student_logits_a, _, _ = select_quality_candidates(mixed_a)
                        student_logits_b, _, _ = select_quality_candidates(mixed_b)
                    else:
                        student_logits_a, student_logits_b = clean_logits_a, clean_logits_b
                    cps_a, accepted_a = confidence_weighted_cps_loss(
                        student_logits_a,
                        probability_b,
                        clean_quality_b,
                        quality_threshold=args.pseudo_quality_threshold,
                        pixel_confidence_threshold=args.pixel_confidence_threshold,
                        instance_validity=phase2_validity,
                    )
                    cps_b, accepted_b = confidence_weighted_cps_loss(
                        student_logits_b,
                        probability_a,
                        clean_quality_a,
                        quality_threshold=args.pseudo_quality_threshold,
                        pixel_confidence_threshold=args.pixel_confidence_threshold,
                        instance_validity=phase2_validity,
                    )
                    cps = cps_a + cps_b
                    accepted = accepted_a + accepted_b
                loss = supervised + current_cps_weight * cps

            scaler.scale(loss / args.accumulation_steps).backward()
            is_update = (
                (step + 1) % args.accumulation_steps == 0
                or step + 1 == steps_per_epoch
            )
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            cps_losses.append(float(cps.detach()))
            accepted_total += accepted
            progress.set_postfix(loss=float(np.mean(losses)), accepted=accepted_total)

        validation_summary = None
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            validation_summary = evaluate(
                branches, predictor, cache, validation, args, run_dir, epoch
            )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "supervised_loss": float(np.mean(supervised_losses)),
            "cps_loss": float(np.mean(cps_losses)),
            "cps_weight": current_cps_weight,
            "accepted_pseudo_instances": accepted_total,
            "gmbs_mask_fraction": float(np.mean(swap_fractions)) if swap_fractions else 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation_summary,
        }
        history.append(record)
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            macro = validation_summary["image_macro"] if validation_summary else {}
            csv.writer(stream).writerow(
                [
                    epoch, record["train_loss"], record["supervised_loss"],
                    record["cps_loss"], current_cps_weight, accepted_total,
                    record["gmbs_mask_fraction"], record["learning_rate"],
                    macro.get("dice"), macro.get("iou"), macro.get("hd95"),
                ]
            )
        checkpoint_path = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
        torch.save(
            checkpoint_payload(
                epoch, branches, optimizer, scheduler, scaler, history, args
            ),
            checkpoint_path,
        )
        print(json.dumps(record), flush=True)
        if validation_summary is not None:
            dice = float(validation_summary["image_macro"]["dice"])
            if dice > best_dice + 1e-5:
                best_dice = dice
                stale = 0
                (run_dir / "best_checkpoint.json").write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "checkpoint": str(checkpoint_path.resolve()),
                            "validation": validation_summary,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            else:
                stale += 1
            if stale >= args.early_stopping_patience:
                break

    final = {
        **run_metadata,
        "best_validation_dice": best_dice,
        "elapsed_seconds": time.perf_counter() - started,
        "epochs_completed": len(history),
    }
    (run_dir / "final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir.resolve()), **final}, indent=2))


if __name__ == "__main__":
    main()
