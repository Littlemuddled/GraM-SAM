"""Train Phase-2 GraphSAGE only on actively selected masks and emit prompts.

Unselected training images and validation images are converted to graphs from
their images alone.  Their masks are never opened while prompts are generated.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gram_sam.data_contract import resolve_selected_image_names  # noqa: E402
from gram_sam.datasets import DATASET_NAMES, read_mask, read_rgb  # noqa: E402
from gram_sam.phase2 import (  # noqa: E402
    DATASET_CLASS_VALUES,
    GraphBuildConfig,
    Phase2GraphSAGE,
    build_granular_graph,
    class_balanced_weights,
    foreground_soft_dice_loss,
    select_spatially_diverse_prompts,
)
from gram_sam.sam_features import FrozenSAMNodeFeatureExtractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--labeled-ratio", type=int, choices=(5, 10, 20), default=10)
    parser.add_argument("--output-root", type=Path, default=Path("runs/phase2"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-points-per-class", type=int, default=1)
    parser.add_argument("--minimum-separation-fraction", type=float, default=0.08)
    parser.add_argument("--negative-points-per-class", type=int, default=0)
    parser.add_argument("--negative-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--negative-minimum-distance-fraction", type=float, default=0.06)
    parser.add_argument("--component-radius-fraction", type=float, default=0.0)
    parser.add_argument("--positive-interior-weight", type=float, default=0.0)
    parser.add_argument("--box-component-radius-fraction", type=float, default=0.10)
    parser.add_argument("--box-expansion-fraction", type=float, default=0.05)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model-config", default="configs/sam2.1_hiera_s.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--limit-selected", type=int)
    parser.add_argument("--limit-inference", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_image(directory: Path, image_name: str) -> Path:
    direct = directory / image_name
    if direct.is_file():
        return direct
    stem = Path(image_name).stem
    matches = [
        directory / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg")
        if (directory / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"unable to resolve {image_name!r} in {directory}")
    return matches[0]


def image_files(directory: Path) -> list[Path]:
    return sorted(
        [
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=lambda path: path.name,
    )


def read_selected_names(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "image_name" not in reader.fieldnames:
            raise ValueError("selected CSV must contain image_name")
        names = [str(row["image_name"]) for row in reader]
    if not names or len(names) != len(set(names)):
        raise ValueError("selected CSV must be non-empty and duplicate-free")
    return names


def build_labeled_graphs(
    dataset_root: Path,
    selected_names: list[str],
    class_values: tuple[int, ...],
    config: GraphBuildConfig,
    feature_extractor=None,
) -> list:
    graphs = []
    for name in tqdm(selected_names, desc="Build selected labeled graphs"):
        image_path = find_image(dataset_root / "train" / "images", name)
        mask_path = dataset_root / "train" / "masks" / f"{image_path.stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        image = read_rgb(image_path)
        graph = build_granular_graph(
            image,
            config=config,
            mask=read_mask(mask_path),
            class_values=class_values,
        )
        if feature_extractor is not None:
            graph = feature_extractor(image, graph)
        graph.image_stem = image_path.stem
        graphs.append(graph)
    return graphs


def evaluate_nodes(model, loader, device: str, number_of_classes: int) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            labels.append(batch.y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    target = np.concatenate(labels)
    predicted = np.concatenate(predictions)
    classes = list(range(number_of_classes))
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "macro_f1": float(f1_score(target, predicted, labels=classes, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(target, predicted, labels=classes, average="macro", zero_division=0)),
    }


def train_model(
    train_graphs,
    validation_graphs,
    args: argparse.Namespace,
    number_of_classes: int,
) -> tuple[Phase2GraphSAGE, list[dict], int, dict]:
    model = Phase2GraphSAGE(
        train_graphs[0].x.shape[1],
        number_of_classes,
        hidden_dimension=args.hidden_dimension,
        dropout=args.dropout,
    ).to(args.device)
    weights = class_balanced_weights(
        (graph.y for graph in train_graphs), number_of_classes
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_graphs, batch_size=args.batch_size, shuffle=False
    )
    history: list[dict] = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.cross_entropy(
                logits, batch.y, weight=weights, label_smoothing=0.02
            ) + 0.5 * foreground_soft_dice_loss(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_nodes(
            model, validation_loader, args.device, number_of_classes
        )
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        score = validation["macro_f1"]
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training failed to produce a model")
    model.load_state_dict(best_state)
    best_metrics = evaluate_nodes(
        model, validation_loader, args.device, number_of_classes
    )
    return model, history, best_epoch, best_metrics


def refit_all_selected(
    graphs,
    args: argparse.Namespace,
    number_of_classes: int,
    epochs: int,
) -> Phase2GraphSAGE:
    torch.manual_seed(args.seed + 1)
    model = Phase2GraphSAGE(
        graphs[0].x.shape[1],
        number_of_classes,
        hidden_dimension=args.hidden_dimension,
        dropout=args.dropout,
    ).to(args.device)
    weights = class_balanced_weights((graph.y for graph in graphs), number_of_classes).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.cross_entropy(
                logits, batch.y, weight=weights, label_smoothing=0.02
            ) + 0.5 * foreground_soft_dice_loss(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(json.dumps({"refit_epoch": epoch, "loss": float(np.mean(losses))}), flush=True)
    return model


def prompt_record(model, graph, image_stem: str, class_values, args) -> dict:
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(graph.to(args.device)), dim=1).cpu().numpy()
    graph = graph.cpu()
    candidates = select_spatially_diverse_prompts(
        graph,
        probabilities,
        class_values,
        max_points_per_class=args.max_points_per_class,
        minimum_separation_fraction=args.minimum_separation_fraction,
        negative_points_per_class=args.negative_points_per_class,
        negative_confidence_threshold=args.negative_confidence_threshold,
        negative_minimum_distance_fraction=args.negative_minimum_distance_fraction,
        component_radius_fraction=args.component_radius_fraction,
        positive_interior_weight=args.positive_interior_weight,
        box_component_radius_fraction=args.box_component_radius_fraction,
        box_expansion_fraction=args.box_expansion_fraction,
    )
    groups = [
        {
            "class": candidate.class_value,
            "coords": candidate.coordinates_xy.tolist(),
            "labels": candidate.point_labels.tolist(),
            "scores": candidate.scores.tolist(),
            "box": candidate.box_xyxy.tolist(),
        }
        for candidate in candidates
    ]
    # The first point per class preserves compatibility with the initial
    # Stage-3 flat manifest.  The repaired trainer consumes prompt_groups.
    points = [
        {"class": group["class"], "coord": group["coords"][0]}
        for group in groups
    ]
    return {"image": image_stem, "prompt_groups": groups, "points": points}


def generate_manifest(
    model,
    paths: list[Path],
    class_values,
    config,
    args,
    description: str,
    feature_extractor=None,
) -> list[dict]:
    records = []
    if args.limit_inference is not None:
        paths = paths[: args.limit_inference]
    for path in tqdm(paths, desc=description):
        # Deliberately image-only: no mask path is constructed or opened here.
        image = read_rgb(path)
        graph = build_granular_graph(image, config=config)
        if feature_extractor is not None:
            graph = feature_extractor(image, graph)
        records.append(prompt_record(model, graph, path.stem, class_values, args))
    return records


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_root = args.data_base / args.dataset
    selected_csv = dataset_root / f"labeled_names_{args.labeled_ratio}_select.csv"
    selected_csv_names = read_selected_names(selected_csv)
    selected_names, unavailable_selected_names = resolve_selected_image_names(
        selected_csv_names,
        dataset_root / "train" / "images",
        dataset_root / "train" / "masks",
    )
    selected_names = list(selected_names)
    if unavailable_selected_names:
        print(
            json.dumps(
                {
                    "selection_contract_warning": (
                        "selected entries absent from materialized training pool"
                    ),
                    "unavailable_count": len(unavailable_selected_names),
                    "unavailable_images": list(unavailable_selected_names),
                }
            ),
            flush=True,
        )
    if args.limit_selected is not None:
        selected_names = selected_names[: args.limit_selected]
    class_values = DATASET_CLASS_VALUES[args.dataset]
    config = GraphBuildConfig()
    feature_extractor = FrozenSAMNodeFeatureExtractor(
        args.sam_checkpoint,
        model_config=args.sam_model_config,
        device=args.device,
        include_high_resolution=True,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"{args.dataset}_{args.labeled_ratio}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    graphs = build_labeled_graphs(
        dataset_root, selected_names, class_values, config, feature_extractor
    )
    if len(graphs) < 5:
        raise ValueError("at least five selected images are required")
    order = np.random.default_rng(args.seed).permutation(len(graphs))
    validation_count = max(1, int(round(len(graphs) * args.validation_fraction)))
    validation_indices = set(int(value) for value in order[:validation_count])
    training_graphs = [graph for index, graph in enumerate(graphs) if index not in validation_indices]
    validation_graphs = [graph for index, graph in enumerate(graphs) if index in validation_indices]
    candidate, history, best_epoch, holdout_metrics = train_model(
        training_graphs, validation_graphs, args, len(class_values)
    )
    model = refit_all_selected(graphs, args, len(class_values), best_epoch)
    torch.save(
        {
            "model": model.state_dict(),
            "class_values": class_values,
            "graph_config": config.__dict__,
            "arguments": vars(args),
            "holdout_best_epoch": best_epoch,
            "feature_source": {
                "kind": "sam2_multiscale",
                "model_config": args.sam_model_config,
                "include_high_resolution": True,
                "checkpoint_sha256": sha256(args.sam_checkpoint),
            },
        },
        run_dir / "phase2_graphsage.pt",
    )

    selected_stems = {Path(name).stem for name in selected_names}
    all_training = image_files(dataset_root / "train" / "images")
    unlabeled_training = [path for path in all_training if path.stem not in selected_stems]
    validation = image_files(dataset_root / "val" / "images")
    train_manifest = generate_manifest(
        model, unlabeled_training, class_values, config, args,
        "Generate unlabeled train prompts", feature_extractor
    )
    validation_manifest = generate_manifest(
        model, validation, class_values, config, args,
        "Generate validation prompts", feature_extractor
    )
    (run_dir / "train_unlabeled_prompts.json").write_text(
        json.dumps(train_manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "validation_prompts.json").write_text(
        json.dumps(validation_manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "dataset": args.dataset,
        "labeled_ratio_percent": args.labeled_ratio,
        "pipeline": "GraM-Prompt training and image-only prompt generation",
        "selected_csv": str(selected_csv.resolve()),
        "selected_csv_sha256": sha256(selected_csv),
        "selected_csv_entries": len(selected_csv_names),
        "selected_images": len(selected_names),
        "selected_unavailable_images": list(unavailable_selected_names),
        "selection_resolution": "selected_csv_intersection_with_available_image_mask_pool",
        "train_holdout_images": len(validation_graphs),
        "best_epoch": best_epoch,
        "holdout_node_metrics": holdout_metrics,
        "unlabeled_prompt_records": len(train_manifest),
        "validation_prompt_records": len(validation_manifest),
        "class_values": class_values,
        "graph_config": config.__dict__,
        "feature_source": {
            "kind": "sam2_multiscale",
            "dimension": feature_extractor.output_dimension,
            "checkpoint_sha256": sha256(args.sam_checkpoint),
        },
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if str(args.device).startswith("cuda") else None,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir.resolve()), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
