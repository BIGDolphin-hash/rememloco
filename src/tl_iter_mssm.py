"""Texture--Logic Iterative MSSM for structured TLFME memory.

The scorer is deliberately separate from TLFME.  It reads the memory that
TLFME has already written and scores a query in two complementary branches.
It never selects or writes reference images itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.dpfe import TLFMEMemoryBank
from src.dual_branch_features import DualBranchFeatures


TL_ITER_MSSM_ALGORITHM = (
    "tl_iter_mssm_anomaly_score_v1_texture_p10_logic_global_bidir_"
    "patchmax_top1p_pool_top3_tau_v17"
)
TL_ITER_ANOMALY_SCORE_VERSION = "v1"
TL_ITER_MAP_MODE = "tl_iter"
TL_ITER_DEFAULT_MEMORY_FRACTION = 0.10
TL_ITER_INITIAL_K = 3
TL_ITER_TOP_K = 3
TL_ITER_SELECTION_POLICY = "lowest_anomaly_score_up_to_top3"
TL_ITER_TEXTURE_SIMILARITY_RANK_FRACTION = 0.10
TL_ITER_IMAGE_PATCH_POOL_FRACTION = 0.01


@dataclass(frozen=True)
class TLIterMSSMResult:
    """Patch maps and image scores for one query image."""

    texture_map: np.ndarray
    logical_map: np.ndarray
    final_map: np.ndarray
    texture_score: float
    logical_score: float
    score: float


def memory_budget(
    sample_count: int,
    fraction: float = TL_ITER_DEFAULT_MEMORY_FRACTION,
) -> int:
    """Return floor(sample_count * fraction) for dynamic TLFME capacity."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("TL-IterMSSM sample_count must be an integer")
    if sample_count <= 0:
        raise ValueError("TL-IterMSSM sample_count must be positive")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0 < float(fraction) <= 1
    ):
        raise ValueError("TL-IterMSSM memory fraction must be in (0, 1]")
    capacity = math.floor(sample_count * float(fraction))
    if capacity < TL_ITER_INITIAL_K:
        raise ValueError(
            "TL-IterMSSM memory fraction leaves fewer than "
            f"{TL_ITER_INITIAL_K} initial reference images"
        )
    return capacity


def select_lowest_score_top_k(
    ranked_entries: list[dict[str, Any]],
    *,
    existing_image_ids: set[str],
    remaining_slots: int,
) -> list[str]:
    """Select up to three unused images with the lowest anomaly scores."""

    if remaining_slots < 0:
        raise ValueError("TL-IterMSSM remaining_slots cannot be negative")
    selection_limit = min(TL_ITER_TOP_K, remaining_slots)
    if selection_limit == 0:
        return []

    candidates: list[tuple[float, int, str]] = []
    for order, entry in enumerate(ranked_entries):
        image_id = entry.get("path")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("TL-IterMSSM ranking entry has an invalid path")
        score = entry.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("TL-IterMSSM ranking entry has an invalid score")
        if image_id in existing_image_ids:
            continue
        candidates.append((float(score), order, image_id))

    selected: list[str] = []
    seen = set(existing_image_ids)
    for _, _, image_id in sorted(candidates):
        if image_id in seen:
            continue
        selected.append(image_id)
        seen.add(image_id)
        if len(selected) == selection_limit:
            break
    return selected


def _normalise_rows(features: np.ndarray, *, name: str) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not len(features) or not features.shape[1]:
        raise ValueError(f"{name} must be a non-empty feature matrix")
    if not np.isfinite(features).all():
        raise ValueError(f"{name} contains NaN or infinity")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero-norm feature")
    return np.ascontiguousarray(features / norms, dtype=np.float32)


def _top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    """Return the mean of the largest ceil(fraction * N) finite values."""

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if not len(flat) or not np.isfinite(flat).all():
        raise ValueError("TL-IterMSSM pooling values must be finite and non-empty")
    count = max(1, math.ceil(fraction * len(flat)))
    first_top_index = len(flat) - count
    top_values = np.partition(flat, first_top_index)[first_top_index:]
    return float(np.mean(top_values, dtype=np.float64))


def _flat_texture_map(
    query: np.ndarray,
    texture_index: Any,
    memory: TLFMEMemoryBank,
    *,
    exclude_image_id: str | None,
) -> np.ndarray:
    """Return distances at the top-10% similarity rank in texture memory."""

    query = _normalise_rows(query, name="TL-IterMSSM texture query")
    total_patches = memory.texture_count
    if total_patches <= 0:
        raise ValueError("TL-IterMSSM texture memory is empty")

    excluded_ranges = tuple(
        (record.texture_start, record.texture_stop)
        for record in memory.records
        if exclude_image_id is not None and record.image_id == exclude_image_id
    )
    excluded_count = sum(stop - start for start, stop in excluded_ranges)
    available_count = total_patches - excluded_count
    if available_count <= 0:
        raise ValueError("TL-IterMSSM has no texture patch available for matching")
    similarity_rank = max(
        1,
        math.ceil(
            TL_ITER_TEXTURE_SIMILARITY_RANK_FRACTION * available_count
        ),
    )
    search_k = min(total_patches, excluded_count + similarity_rank)
    distances, indices = texture_index.search(query, search_k)
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.shape != indices.shape or distances.shape != (
        len(query),
        search_k,
    ):
        raise RuntimeError("TL-IterMSSM texture index returned invalid shapes")

    valid = (indices >= 0) & (indices < total_patches)
    for start, stop in excluded_ranges:
        valid &= (indices < start) | (indices >= stop)
    usable = np.where(valid, distances, np.inf)
    ranked_distance = np.partition(
        usable, similarity_rank - 1, axis=1
    )[:, similarity_rank - 1]
    if not np.isfinite(ranked_distance).all():
        raise RuntimeError(
            "TL-IterMSSM could not find the requested texture similarity rank"
        )
    # FAISS IndexFlatL2 returns squared L2. Unit-normalized vectors therefore
    # map to cosine distance after division by two.
    return np.maximum(ranked_distance / 2.0, 0.0).astype(
        np.float32, copy=False
    )


def _bidirectional_logical_map(
    query_grid: np.ndarray,
    memory_grid: np.ndarray,
) -> np.ndarray:
    """Compute appearance-only global bidirectional logical matching."""

    query_grid = np.asarray(query_grid, dtype=np.float32)
    memory_grid = np.asarray(memory_grid, dtype=np.float32)
    if query_grid.ndim != 3 or memory_grid.shape != query_grid.shape:
        raise ValueError(
            "TL-IterMSSM logical query and memory grids must share [H, W, C]"
        )
    height, width, channels = query_grid.shape
    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError("TL-IterMSSM logical grids must be non-empty")
    query = _normalise_rows(
        query_grid.reshape(-1, channels), name="TL-IterMSSM logical query"
    )
    memory = _normalise_rows(
        memory_grid.reshape(-1, channels), name="TL-IterMSSM logical memory"
    )

    cell_count = height * width
    query_nearest = np.empty(cell_count, dtype=np.float32)
    memory_nearest = np.full(cell_count, np.inf, dtype=np.float32)
    memory_transposed = memory.T
    chunk_size = 256
    for start in range(0, cell_count, chunk_size):
        stop = min(start + chunk_size, cell_count)
        costs = 1.0 - query[start:stop] @ memory_transposed
        np.clip(costs, 0.0, 2.0, out=costs)
        query_nearest[start:stop] = np.min(costs, axis=1)
        memory_nearest = np.minimum(
            memory_nearest,
            np.min(costs, axis=0),
        )

    if (
        not np.isfinite(query_nearest).all()
        or not np.isfinite(memory_nearest).all()
    ):
        raise RuntimeError("TL-IterMSSM logical matching left an uncovered cell")

    combined = np.maximum(query_nearest, memory_nearest)
    return combined.reshape(height, width).astype(np.float32, copy=False)


def _independent_logical_grid_map(
    query_grid: np.ndarray,
    memory_grids: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Match each stored grid separately and return the best single template."""

    if not memory_grids:
        raise ValueError("TL-IterMSSM logical memory is empty")
    maps = [
        _bidirectional_logical_map(query_grid, grid)
        for grid in memory_grids
    ]
    scores = [
        _top_fraction_mean(values, TL_ITER_IMAGE_PATCH_POOL_FRACTION)
        for values in maps
    ]
    return maps[int(np.argmin(scores))]


class TLIterMSSM:
    """Read-only two-branch scorer over a live TLFME memory bank."""

    def __init__(
        self,
        memory: TLFMEMemoryBank,
        texture_index: Any,
    ) -> None:
        self.memory = memory
        self.texture_index = texture_index
        minimum_initial_references = 2
        if len(memory.records) < minimum_initial_references:
            raise ValueError(
                "TL-IterMSSM requires at least "
                f"{minimum_initial_references} memory records for "
                "self-match exclusion"
            )

    def _raw_score(
        self,
        query: DualBranchFeatures,
        *,
        exclude_image_id: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        texture_values = _flat_texture_map(
            query.texture,
            self.texture_index,
            self.memory,
            exclude_image_id=exclude_image_id,
        )
        texture_map = texture_values.reshape(query.grid_size)
        memory_grids = tuple(
            self.memory.logical_grids[record.logical_slot]
            for record in self.memory.records
            if record.image_id != exclude_image_id
        )
        logical_map = _independent_logical_grid_map(
            query.logical_grid,
            memory_grids,
        )
        return texture_map, logical_map

    def score(
        self,
        query: DualBranchFeatures,
        *,
        image_id: str | None = None,
    ) -> TLIterMSSMResult:
        """Score a query without modifying TLFME or its search index."""

        texture_raw, logical_raw = self._raw_score(
            query, exclude_image_id=image_id
        )
        texture_map = texture_raw
        logical_map = logical_raw
        final_map = np.maximum(texture_map, logical_map).astype(
            np.float32, copy=False
        )
        texture_score = _top_fraction_mean(
            texture_map, TL_ITER_IMAGE_PATCH_POOL_FRACTION
        )
        logical_score = _top_fraction_mean(
            logical_map, TL_ITER_IMAGE_PATCH_POOL_FRACTION
        )
        score = _top_fraction_mean(
            final_map, TL_ITER_IMAGE_PATCH_POOL_FRACTION
        )
        return TLIterMSSMResult(
            texture_map=np.asarray(texture_map, dtype=np.float32),
            logical_map=np.asarray(logical_map, dtype=np.float32),
            final_map=final_map,
            texture_score=texture_score,
            logical_score=logical_score,
            score=score,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "algorithm": TL_ITER_MSSM_ALGORITHM,
            "anomaly_score_version": TL_ITER_ANOMALY_SCORE_VERSION,
            "fusion": "raw_elementwise_max",
            "image_score_fusion": "top_1_percent_mean_of_patchwise_max",
            "texture_similarity_rank_fraction": (
                TL_ITER_TEXTURE_SIMILARITY_RANK_FRACTION
            ),
            "branch_image_score_pooling": "top_1_percent_patch_mean",
            "logical_template_selection": (
                "minimum_template_top_1_percent_patch_mean"
            ),
            "logical_matching": "appearance_only_global_bidirectional",
            "uses_position_constraint": False,
            "image_patch_pool_fraction": TL_ITER_IMAGE_PATCH_POOL_FRACTION,
        }
