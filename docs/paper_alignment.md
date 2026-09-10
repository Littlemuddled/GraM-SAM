# Paper-to-code alignment

| Paper component | Public implementation |
|---|---|
| Algorithm 1: stable low-to-high gradient seeds, purity/variance growth, complete coverage | `gram_sam/granular.py` |
| Eq. 3: inverse-aligned TTA Bernoulli variance input contract | `docs/data_layout.md`, `scripts/select_active_set.py` |
| Eqs. 4--5: HVG union-area fraction, area coefficient of variation, Challenge Score | `gram_sam/selection.py` |
| Eqs. 5b--6: nearest-anchor cosine distance, pool-medoid cold start, unit-weight greedy selection | `gram_sam/selection.py` |
| Five-percentage-point acquisition rounds; 5%, 10%, 20% reports | `scripts/select_active_set.py` |
| 21-D granule descriptor and frozen SAM 2 multi-scale node features | `gram_sam/phase2.py::granular_node_features`, `gram_sam/sam_features.py` |
| Touch/overlap plus standardized feature 5-NN edges | `gram_sam/phase2.py::efficient_graph_edges` |
| Shared three-block, 128-wide GraphSAGE; dropout 0.25 | `gram_sam/phase2.py::Phase2GraphSAGE` |
| Majority node labels and one positive point per detected foreground class | `gram_sam/phase2.py`, `scripts/train_phase2_prompts.py` |
| Eq. 8: detached, class-averaged peer disagreement aligned to the feature lattice | `gram_sam/gmbs.py` |
| Eq. 9: union-mask symmetric feature exchange | `gram_sam/gmbs.py::granular_matrix_block_swap` |
| Eqs. 10--11: dual-branch supervised loss and cross-pseudo supervision with Gaussian ramp-up | `gram_sam/stage3.py`, `scripts/train_stage3.py` |
| Training-only GMBS; target-free inference fusion | `gram_sam/stage3.py`, `scripts/calibrate_stage3_inference.py` |

All dataset class maps are fixed at dataset level. Graph message passing is
confined to one image, and validation masks are not used to generate automatic
prompts.

The GraphSAGE input concatenates the 21-D granular descriptor with 352-D
frozen SAM 2 multi-scale features. This gives 147,842 trainable parameters for
binary datasets and 149,132 for the 12-class Spine configuration, matching the
Supplementary Material.
