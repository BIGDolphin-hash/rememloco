"""TLFME structured memory update for externally selected images.

TLFME is deliberately limited to images that have already been selected by an
external ranking module. It stores a deterministic per-image texture coreset
and the complete logical grid without anomaly scoring or sample selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.dual_branch_features import DualBranchFeatures


DEFAULT_TLFME_TEXTURE_CORESET_SIZE = 256
DEFAULT_TLFME_TEXTURE_SPATIAL_WEIGHT = 0.25


def _grid_coordinates(grid_size: tuple[int, int]) -> np.ndarray:
    height, width = grid_size
    rows, columns = np.indices((height, width), dtype=np.int32)
    return np.stack((rows.reshape(-1), columns.reshape(-1)), axis=1)


def _normalised_grid_coordinates(grid_size: tuple[int, int]) -> np.ndarray:
    coordinates = _grid_coordinates(grid_size).astype(np.float32)
    height, width = grid_size
    if height > 1:
        coordinates[:, 0] = 2.0 * coordinates[:, 0] / (height - 1) - 1.0
    else:
        coordinates[:, 0] = 0.0
    if width > 1:
        coordinates[:, 1] = 2.0 * coordinates[:, 1] / (width - 1) - 1.0
    else:
        coordinates[:, 1] = 0.0
    return coordinates


def _select_texture_coreset(
    features: np.ndarray,
    grid_size: tuple[int, int],
    *,
    coreset_size: int,
    spatial_weight: float,
) -> np.ndarray:
    """Select deterministic feature-diverse patches with spatial coverage."""

    patch_count = len(features)
    if patch_count <= coreset_size:
        return features.copy()

    spatial = _normalised_grid_coordinates(grid_size)
    descriptor = np.concatenate(
        (features, spatial_weight * spatial), axis=1
    ).astype(np.float32, copy=False)
    centre = descriptor.mean(axis=0, dtype=np.float64).astype(np.float32)
    first_index = int(
        np.argmin(np.square(descriptor - centre).sum(axis=1))
    )
    selected = np.empty(coreset_size, dtype=np.int64)
    selected[0] = first_index
    chosen = np.zeros(patch_count, dtype=bool)
    chosen[first_index] = True
    minimum_distances = np.full(patch_count, np.inf, dtype=np.float32)

    for selection_index in range(1, coreset_size):
        latest = descriptor[selected[selection_index - 1]]
        distances = np.square(descriptor - latest).sum(axis=1)
        np.minimum(minimum_distances, distances, out=minimum_distances)
        minimum_distances[chosen] = -np.inf
        next_index = int(np.argmax(minimum_distances))
        selected[selection_index] = next_index
        chosen[next_index] = True

    # Spatial order keeps the selected set deterministic.
    selected.sort()
    return features[selected].copy()


@dataclass(frozen=True)
class TLFMEMemoryRecord:
    """External provenance bound to one TLFME feature grid by the writer."""

    image_id: str
    round_id: int
    augmentation_id: int
    grid_size: tuple[int, int]
    texture_start: int
    texture_stop: int
    texture_source_count: int
    logical_slot: int


class TLFMEMemoryBank:
    """Compressed texture prototypes plus complete logical feature grids."""

    def __init__(
        self,
        *,
        texture_coreset_size: int = DEFAULT_TLFME_TEXTURE_CORESET_SIZE,
        texture_spatial_weight: float = DEFAULT_TLFME_TEXTURE_SPATIAL_WEIGHT,
    ) -> None:
        if (
            isinstance(texture_coreset_size, bool)
            or not isinstance(texture_coreset_size, int)
            or texture_coreset_size <= 0
        ):
            raise ValueError("TLFME texture_coreset_size must be positive")
        if (
            not np.isfinite(texture_spatial_weight)
            or texture_spatial_weight < 0
        ):
            raise ValueError(
                "TLFME texture_spatial_weight must be finite and non-negative"
            )
        self.texture_coreset_size = texture_coreset_size
        self.texture_spatial_weight = float(texture_spatial_weight)
        self._records: list[TLFMEMemoryRecord] = []
        self._texture_chunks: list[np.ndarray] = []
        self._logical_grids: list[np.ndarray] = []
        self._keys: set[tuple[str, int]] = set()
        self._texture_count = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[TLFMEMemoryRecord, ...]:
        return tuple(self._records)

    @property
    def logical_grids(self) -> tuple[np.ndarray, ...]:
        return tuple(self._logical_grids)

    @property
    def texture_features(self) -> np.ndarray:
        if not self._texture_chunks:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(self._texture_chunks, axis=0)

    @property
    def texture_count(self) -> int:
        return self._texture_count

    def add(
        self,
        features: DualBranchFeatures,
        *,
        image_id: str,
        round_id: int,
        augmentation_id: int = 0,
    ) -> TLFMEMemoryRecord:
        """Store a selected image without scoring or filtering its patches."""

        image_id = str(image_id)
        if not image_id:
            raise ValueError("TLFME memory image_id must be non-empty")
        if int(round_id) <= 0:
            raise ValueError("TLFME memory round_id must be positive")
        if int(augmentation_id) < 0:
            raise ValueError("TLFME augmentation_id must be non-negative")
        key = (image_id, int(augmentation_id))
        if key in self._keys:
            raise ValueError(
                "TLFME memory already contains image/augmentation pair: "
                f"{key}"
            )

        texture_chunk = _select_texture_coreset(
            features.texture,
            features.grid_size,
            coreset_size=self.texture_coreset_size,
            spatial_weight=self.texture_spatial_weight,
        )
        texture_start = self._texture_count
        texture_stop = texture_start + len(texture_chunk)
        logical_slot = len(self._logical_grids)
        record = TLFMEMemoryRecord(
            image_id=image_id,
            round_id=int(round_id),
            augmentation_id=int(augmentation_id),
            grid_size=features.grid_size,
            texture_start=texture_start,
            texture_stop=texture_stop,
            texture_source_count=len(features.texture),
            logical_slot=logical_slot,
        )
        logical_grid = features.logical_grid.copy()
        texture_chunk.setflags(write=False)
        logical_grid.setflags(write=False)
        self._texture_chunks.append(texture_chunk)
        self._logical_grids.append(logical_grid)
        self._records.append(record)
        self._keys.add(key)
        self._texture_count = texture_stop
        return record

    def texture_for_record(self, record: TLFMEMemoryRecord) -> np.ndarray:
        """Return the texture prototypes stored for one record."""

        if record not in self._records:
            raise ValueError("TLFME record does not belong to this memory bank")
        return self._texture_chunks[record.logical_slot]

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "image_id": record.image_id,
                "round_id": record.round_id,
                "augmentation_id": record.augmentation_id,
                "grid_size": list(record.grid_size),
                "texture_patch_count": (
                    record.texture_stop - record.texture_start
                ),
                "texture_source_patch_count": record.texture_source_count,
                "texture_storage": "spatial_feature_kcenter",
                "logical_slot": record.logical_slot,
                "logical_patch_count": (
                    record.grid_size[0] * record.grid_size[1]
                ),
                "logical_storage": "full_grid",
            }
            for record in self._records
        ]


# Compatibility aliases for callers and old cached imports.  TLFME is the
# canonical paper-facing name; the storage contract is unchanged.
DPFEMemoryRecord = TLFMEMemoryRecord
DPFEMemoryBank = TLFMEMemoryBank
