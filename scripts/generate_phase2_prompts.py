"""Regenerate image-only Phase-2 prompts from a trained GraphSAGE checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gram_sam.data_contract import resolve_selected_image_names  # noqa: E402
from gram_sam.datasets import DATASET_NAMES  # noqa: E402
from gram_sam.phase2 import GraphBuildConfig, Phase2GraphSAGE  # noqa: E402
from gram_sam.sam_features import FrozenSAMNodeFeatureExtractor  # noqa: E402
from scripts.train_phase2_prompts import (  # noqa: E402
    generate_manifest,
    image_files,
    read_selected_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/phase2_inference"))
    parser.add_argument(
        "--split",
        choices=("val", "train-unlabeled", "train-selected", "both", "all"),
        default="val",
    )
    parser.add_argument("--max-points-per-class", type=int, default=1)
    parser.add_argument("--minimum-separation-fraction", type=float, default=0.08)
    parser.add_argument("--negative-points-per-class", type=int, default=0)
    parser.add_argument("--negative-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--negative-minimum-distance-fraction", type=float, default=0.06)
    parser.add_argument("--component-radius-fraction", type=float, default=0.0)
    parser.add_argument("--positive-interior-weight", type=float, default=0.0)
    parser.add_argument("--box-component-radius-fraction", type=float, default=0.10)
    parser.add_argument("--box-expansion-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--limit-inference", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stored_arguments = payload.get("arguments", {})
    if payload.get("class_values") is None or payload.get("graph_config") is None:
        raise ValueError("checkpoint lacks Phase-2 class or graph metadata")
    checkpoint_dataset = stored_arguments.get("dataset")
    if checkpoint_dataset is not None and checkpoint_dataset != args.dataset:
        raise ValueError(
            f"checkpoint dataset {checkpoint_dataset!r} does not match {args.dataset!r}"
        )
    class_values = tuple(int(value) for value in payload["class_values"])
    feature_source = payload.get("feature_source", {"kind": "handcrafted_only"})
    feature_extractor = None
    if feature_source.get("kind") == "sam2_multiscale":
        if args.sam_checkpoint is None:
            raise ValueError("this checkpoint requires --sam-checkpoint for feature extraction")
        if sha256(args.sam_checkpoint) != feature_source.get("checkpoint_sha256"):
            raise ValueError("SAM2 checkpoint hash differs from the Phase-2 training source")
        feature_extractor = FrozenSAMNodeFeatureExtractor(
            args.sam_checkpoint,
            model_config=feature_source.get("model_config", "configs/sam2.1_hiera_s.yaml"),
            device=args.device,
            include_high_resolution=bool(feature_source.get("include_high_resolution", True)),
        )
    input_dimension = 21 + (feature_extractor.output_dimension if feature_extractor else 0)
    model = Phase2GraphSAGE(
        input_dimension=input_dimension,
        number_of_classes=len(class_values),
        hidden_dimension=int(stored_arguments.get("hidden_dimension", 128)),
        dropout=float(stored_arguments.get("dropout", 0.25)),
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    config = GraphBuildConfig(**payload["graph_config"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"{args.dataset}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset_root = args.data_base / args.dataset
    started = time.perf_counter()
    outputs: dict[str, int] = {}
    selection_resolution: dict = {}

    if args.split in {"val", "both", "all"}:
        paths = image_files(dataset_root / "val" / "images")
        manifest = generate_manifest(
            model, paths, class_values, config, args, "Regenerate validation prompts",
            feature_extractor
        )
        (run_dir / "validation_prompts.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        outputs["validation_prompt_records"] = len(manifest)

    if args.split in {"train-unlabeled", "both", "all"}:
        ratio = int(stored_arguments.get("labeled_ratio", 10))
        selected_path = dataset_root / f"labeled_names_{ratio}_select.csv"
        selected_csv_names = read_selected_names(selected_path)
        resolved, unavailable = resolve_selected_image_names(
            selected_csv_names,
            dataset_root / "train" / "images",
            dataset_root / "train" / "masks",
        )
        selected = {Path(name).stem for name in resolved}
        selection_resolution = {
            "selected_csv_entries": len(selected_csv_names),
            "selected_images": len(resolved),
            "selected_unavailable_images": list(unavailable),
            "selection_resolution": "selected_csv_intersection_with_available_image_mask_pool",
        }
        paths = [
            path
            for path in image_files(dataset_root / "train" / "images")
            if path.stem not in selected
        ]
        manifest = generate_manifest(
            model, paths, class_values, config, args, "Regenerate unlabeled prompts",
            feature_extractor
        )
        (run_dir / "train_unlabeled_prompts.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        outputs["unlabeled_prompt_records"] = len(manifest)

    if args.split in {"train-selected", "all"}:
        ratio = int(stored_arguments.get("labeled_ratio", 10))
        selected_path = dataset_root / f"labeled_names_{ratio}_select.csv"
        selected_csv_names = read_selected_names(selected_path)
        resolved, unavailable = resolve_selected_image_names(
            selected_csv_names,
            dataset_root / "train" / "images",
            dataset_root / "train" / "masks",
        )
        selected = {Path(name).stem for name in resolved}
        selection_resolution = {
            "selected_csv_entries": len(selected_csv_names),
            "selected_images": len(resolved),
            "selected_unavailable_images": list(unavailable),
            "selection_resolution": "selected_csv_intersection_with_available_image_mask_pool",
        }
        paths = [
            path
            for path in image_files(dataset_root / "train" / "images")
            if path.stem in selected
        ]
        if {path.stem for path in paths} != selected:
            missing = sorted(selected - {path.stem for path in paths})
            raise ValueError(f"resolved selected image contract mismatch; missing={missing[:5]}")
        manifest = generate_manifest(
            model, paths, class_values, config, args, "Regenerate selected-train prompts",
            feature_extractor
        )
        (run_dir / "train_selected_prompts.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        outputs["selected_prompt_records"] = len(manifest)

    metadata = {
        "run_id": run_id,
        "dataset": args.dataset,
        "pipeline": "GraM-Prompt checkpoint inference without target-mask access",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **selection_resolution,
        **outputs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir.resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
