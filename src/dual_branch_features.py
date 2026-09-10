"""Shared two-branch feature projection for memory and query images.

This stateless projector is intentionally separate from TLFME. Iterative MSSM
may use it to encode queries, while TLFME alone decides when an externally
selected image is appended to memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


DUAL_BRANCH_CACHE_FORMAT_VERSION = 1
DEFAULT_TEXTURE_LAYER_INDICES = (6, 9)  # zero-based DINO blocks 7 and 10
DEFAULT_LOGICAL_BLOCK = 12  # one-based DINO block number


def _as_patch_matrix(features: Any, *, name: str) -> np.ndarray:
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 2 or not len(features) or not features.shape[1]:
        raise ValueError(
            f"{name} must be a non-empty [patches, channels] matrix, "
            f"got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(features, dtype=np.float32)


def normalise_patch_features(features: Any, *, name: str) -> np.ndarray:
    """Return strictly validated, row-wise L2-normalised patch features."""

    matrix = _as_patch_matrix(features, name=name)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero-norm patch feature")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _validate_grid_size(grid_size: Sequence[int]) -> tuple[int, int]:
    if len(grid_size) != 2:
        raise ValueError(f"grid_size must have two dimensions, got {grid_size}")
    height, width = map(int, grid_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"grid_size must be positive, got {(height, width)}")
    return height, width


@dataclass(frozen=True)
class DualBranchFeatures:
    """Two aligned, full-grid feature views for one image."""

    texture: np.ndarray
    logical: np.ndarray
    grid_size: tuple[int, int]

    def __post_init__(self) -> None:
        grid_size = _validate_grid_size(self.grid_size)
        texture = normalise_patch_features(
            self.texture, name="texture features"
        )
        logical = normalise_patch_features(
            self.logical, name="logical features"
        )
        expected_patches = grid_size[0] * grid_size[1]
        if len(texture) != expected_patches:
            raise ValueError(
                "texture patch count does not match grid_size: "
                f"patches={len(texture)}, grid={grid_size}"
            )
        if len(logical) != expected_patches:
            raise ValueError(
                "logical patch count does not match grid_size: "
                f"patches={len(logical)}, grid={grid_size}"
            )
        object.__setattr__(self, "texture", texture)
        object.__setattr__(self, "logical", logical)
        object.__setattr__(self, "grid_size", grid_size)

    @property
    def logical_grid(self) -> np.ndarray:
        height, width = self.grid_size
        return self.logical.reshape(height, width, self.logical.shape[1])


def extract_dual_branch_features(
    model: Any,
    image_tensor: Any,
    grid_size: Sequence[int],
    *,
    texture_layer_indices: Iterable[int] = DEFAULT_TEXTURE_LAYER_INDICES,
    logical_block: int = DEFAULT_LOGICAL_BLOCK,
    scales: Iterable[int] = (1, 5),
) -> DualBranchFeatures:
    """Project Block 7+10 with MLMP and retain the Block 12 logic grid.

    ``texture_layer_indices`` follows the existing zero-based detector API.
    ``logical_block`` is intentionally one-based to match the MSSM CLI and
    paper-facing terminology.
    """

    grid_size = _validate_grid_size(grid_size)
    texture_layer_indices = tuple(int(index) for index in texture_layer_indices)
    scales = tuple(int(scale) for scale in scales)
    if not texture_layer_indices:
        raise ValueError("at least one texture layer is required")
    if len(texture_layer_indices) != len(set(texture_layer_indices)):
        raise ValueError("texture layer indices must be unique")
    if any(index < 0 for index in texture_layer_indices):
        raise ValueError("texture layer indices must be non-negative")
    if logical_block <= 0:
        raise ValueError("logical block must be positive and one-based")
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("scales must contain positive integers")

    logical_layer_index = int(logical_block) - 1
    requested_indices = tuple(
        dict.fromkeys((*texture_layer_indices, logical_layer_index))
    )
    backbone = getattr(model, "model", None)
    blocks = getattr(backbone, "blocks", None)
    if blocks is not None and max(requested_indices) >= len(blocks):
        raise ValueError(
            f"requested DINO block {max(requested_indices) + 1}, "
            f"but the model has only {len(blocks)} blocks"
        )

    outputs = model.extract_features(
        image_tensor,
        feature_list=list(requested_indices),
    )
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if len(outputs) != len(requested_indices):
        raise RuntimeError(
            "DINO returned a different number of dual-branch layers than "
            f"requested: requested={list(requested_indices)}, "
            f"returned={len(outputs)}"
        )
    by_index = dict(zip(requested_indices, outputs))

    texture_inputs = [by_index[index] for index in texture_layer_indices]
    texture = model.MLMP(
        texture_inputs,
        grid_size=grid_size,
        scales=list(scales),
    )
    logical = by_index[logical_layer_index]
    return DualBranchFeatures(
        texture=texture,
        logical=logical,
        grid_size=grid_size,
    )
