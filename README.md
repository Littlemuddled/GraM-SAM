# GraM-SAM

Official PyTorch implementation of **GraM-SAM: Granular Matrix-Driven Active Semi-Supervised Learning for Medical Image Segmentation with SAM 2**.

The framework reuses a unified adaptive, coordinate-aligned granular representation across three core phases:

- **GraM-Select**: Selects high-value labeled samples via granular challenge and frozen-encoder representativeness.
- **GraM-Prompt**: Classifies granular graph nodes using a shared GraphSAGE model, generating one positive point prompt per predicted foreground class.
- **GMBS**: Symmetrically exchanges high-disagreement feature blocks between dual SAM 2 branches during CPS training (training-only).

### Repository Layout

```
gram_sam/
  granular.py        # Algorithm 1: core granular constructor
  selection.py       # GraM-Select scoring & greedy active acquisition
  phase2.py          # Granular graph, GraphSAGE classifier & prompt generation
  gmbs.py            # Disagreement masking & symmetric feature swapping
  stage3.py          # SAM 2 branches, loss functions & CPS training
scripts/             # End-to-end execution pipelines for each phase
configs/             # Configuration files (configs/method.yaml)
```

### Installation

Tested with **Python 3.10+**, **PyTorch 2.4.1**, and **TorchVision 0.19.1**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Install the official [Meta SAM 2](https://github.com/facebookresearch/segment-anything-2) repository and place `sam2.1_hiera_small.pt` in the `checkpoints/` directory.

### Quick Start

Supported dataset identifiers: `spine`, `Promise12`, `ISIC2016`, and `BUSD`.

**1. Phase 1: Active Sample Selection (GraM-Select)**

After computing Bernoulli variance maps and SAM 2 features across 10 TTA views, run:

```bash
python scripts/select_active_set.py \
  --manifest data/Spine/phase1_manifest.csv \
  --output-dir data/Spine
```

**2. Phase 2: Prompt Classifier Training (GraM-Prompt)**

```base
python scripts/train_phase2_prompts.py \
  --data-base data \
  --dataset Spine \
  --labeled-ratio 10 \
  --sam-checkpoint checkpoints/sam2.1_hiera_small.pt
```

*Note: Use `generate_phase2_prompts.py` after training to extract point prompts for the unlabeled pool.*

**3. Phase 3: GMBS Semi-Supervised Consistency Training**

```base
python scripts/train_stage3.py \
  --data-base data \
  --dataset Spine \
  --labeled-ratio 10 \
  --phase2-train-prompts PATH/TO/train_unlabeled_prompts.json \
  --phase2-validation-prompts PATH/TO/validation_prompts.json \
  --checkpoint checkpoints/sam2.1_hiera_small.pt
```

 
