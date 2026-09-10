# ReMemLOCO

Core research-code snapshot for logical-anomaly detection on MVTec LOCO AD.

## Scope

- Evaluation subset: `test/good` and `test/logical_anomalies` only.
- Structural anomalies are excluded.
- Backbone: DINOv2 ViT-S/14.
- Input resolution: 672 pixels on the shorter edge.
- Features use complete patch grids; no PCA or foreground mask is applied.
- `data/`, `features/`, and `results/` are intentionally excluded and must never be uploaded.

## Current pipeline

```text
GB-MSSM initialization
        ↓
DINOv2 dual-branch full-grid features
        ↓
TLFME structured dynamic memory
        ↓
TL-IterMSSM V1 scoring and memory update
        ↓
Image-level logical-anomaly AUROC
```

### GB-MSSM initialization

MSSM-G and MSSM-B scores are converted to per-object empirical percentile ranks and fused as:

```text
S_init = 0.90 R_G + 0.10 R_B
```

The three lowest-scoring images initialize memory. Historical G+C+B fusion is not used or accepted by the current ranking protocol.

### Dual-branch features

- Texture branch: DINOv2 Blocks 7 and 10 (code indices 6 and 9), fused with MLMP at scales 1 and 5.
- Logical branch: complete Block-12 patch grid (code index 11).
- Both branches are L2-normalized per patch.

### TLFME structured memory

- Stores at most 256 deterministic spatial-feature k-center texture prototypes per selected image.
- Stores the complete Block-12 logical grid for every selected image.
- Writes memory only; scoring and sample selection are handled externally.

### TL-IterMSSM V1

- Texture score: distance at the `ceil(10% * N)` similarity rank.
- Logical score: appearance-only global bidirectional cosine matching against each stored full grid.
- Logical template selection: minimum template Top-1% patch mean.
- Patch fusion: `M(i) = max(M_T(i), M_L(i))`.
- Image score: `MeanTop1%(M)`.
- No position constraint is used in the iterative logical matcher.

The loop starts with three references, admits up to three lowest-score unused images per round, and stops at `floor(N * tau)`. The default memory fraction is `tau = 0.10`.

## Evaluation

The current default evaluates image-level AUROC for `good` versus `logical_anomalies` from raw TL-IterMSSM image scores. It does not generate anomaly-map TIFF files or compute pixel-level/AUC-sPRO metrics.

## Included source

```text
ReRem_run.py
config.py
evaluate_mvtec_loco.py
fuse_loco_mssm_gb_scores.py
generate_loco_mssm_b_scores.py
generate_loco_mssm_scores.py
generate_mssm_scores.py
validate_mpdd.py
validate_mvtec_loco.py
src/ReRem_detection_test.py
src/backbones.py
src/dataset_info.py
src/detection_per_object_test.py
src/dpfe.py
src/dual_branch_features.py
src/loco_mssm_b.py
src/post_eval.py
src/tl_iter_mssm.py
src/utils.py
tests/test_dpfe.py
tests/test_mvtec_loco_support.py
tests/test_tl_iter_mssm.py
```

The shared `generate_mssm_scores.py`, `validate_mpdd.py`, `src/post_eval.py`, and `src/utils.py` files remain because current imports require them.

The current GB-MSSM ranking and metadata are included at:

```text
json/MVTecLOCO/dinov2_vits14/batch-0-shot/logical_only/mssm_gb/weights=9-1/
```

## Validation

The active TLFME and TL-IterMSSM core tests passed:

```bash
python -m pytest -q tests/test_dpfe.py tests/test_tl_iter_mssm.py
# 15 passed
```

The repository does not include datasets, generated feature caches, experiment result directories, model weights, or the non-default vendored pixel-level evaluator. Update the machine-specific dataset path in `config.py` when using another workstation.
