"""Frozen SAM2 multi-scale features sampled at granular-graph nodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class FrozenSAMNodeFeatureExtractor:
    """Attach normalized SAM2 features without exposing or training the encoder."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model_config: str = "configs/sam2.1_hiera_s.yaml",
        device: str = "cuda",
        include_high_resolution: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.model_config = model_config
        self.device = device
        self.include_high_resolution = bool(include_high_resolution)
        model = build_sam2(model_config, str(self.checkpoint), device=device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.predictor = SAM2ImagePredictor(model)

    @property
    def output_dimension(self) -> int:
        return 352 if self.include_high_resolution else 256

    @torch.inference_mode()
    def __call__(self, rgb_image: np.ndarray, graph):
        self.predictor.set_image(rgb_image)
        feature_maps = [self.predictor._features["image_embed"]]
        if self.include_high_resolution:
            feature_maps.extend(self.predictor._features["high_res_feats"])
        centers_yx = graph.centers_yx.to(self.device, dtype=torch.float32)
        size = float(graph.graph_image_size)
        grid = torch.stack(
            [
                2.0 * (centers_yx[:, 1] + 0.5) / size - 1.0,
                2.0 * (centers_yx[:, 0] + 0.5) / size - 1.0,
            ],
            dim=1,
        ).view(1, -1, 1, 2)
        sampled_levels = []
        for feature_map in feature_maps:
            sampled = F.grid_sample(
                feature_map,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )[0, :, :, 0].T
            sampled_levels.append(F.layer_norm(sampled.float(), (sampled.shape[1],)))
        dense_features = torch.cat(sampled_levels, dim=1).cpu()
        if dense_features.shape != (graph.num_nodes, self.output_dimension):
            raise RuntimeError(f"unexpected SAM2 node features {dense_features.shape}")
        graph.x = torch.cat([graph.x, dense_features], dim=1)
        return graph


__all__ = ["FrozenSAMNodeFeatureExtractor"]
