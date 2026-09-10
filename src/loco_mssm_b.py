"""MSSM-B: bidirectional position-constrained matching for MVTec LOCO.

The module is deliberately label-free.  It builds one robust spatial template
from the complete test pool and then freezes that template while every image
is scored.  The two matching directions expose complementary logical errors:
query-to-template responds to unexpected content, while template-to-query
responds to expected content that is missing from the query.
"""

from __future__ import annotations

import math

import numpy as np


def _normalise_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not len(features):
        raise ValueError(f"Expected a non-empty feature matrix, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("MSSM-B features contain NaN or infinity")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("MSSM-B encountered a zero-norm feature vector")
    return (features / norms).astype(np.float32, copy=False)


def build_consensus_template(pooled_tokens: np.ndarray) -> np.ndarray:
    """Build and freeze a robust feature vector at every grid position."""

    pooled_tokens = np.asarray(pooled_tokens, dtype=np.float32)
    if pooled_tokens.ndim != 3:
        raise ValueError(
            "pooled_tokens must have shape [images, cells, channels]"
        )
    image_count, cell_count, channels = pooled_tokens.shape
    if image_count < 2 or cell_count == 0 or channels == 0:
        raise ValueError("MSSM-B requires at least two non-empty token grids")
    if not np.isfinite(pooled_tokens).all():
        raise ValueError("MSSM-B pooled tokens contain NaN or infinity")

    normalised = _normalise_rows(
        pooled_tokens.reshape(image_count * cell_count, channels)
    ).reshape(image_count, cell_count, channels)
    template = np.median(normalised, axis=0).astype(np.float32)

    # Opposing feature directions can make a median row degenerate.  The mean
    # is only a numerical fallback for that individual spatial position.
    template_norms = np.linalg.norm(template, axis=1)
    weak_rows = template_norms <= 1e-12
    if np.any(weak_rows):
        template[weak_rows] = normalised[:, weak_rows].mean(axis=0)
    still_weak = np.linalg.norm(template, axis=1) <= 1e-12
    if np.any(still_weak):
        template[still_weak] = normalised[0, still_weak]
    return _normalise_rows(template)


def _grid_position_costs(grid_size: int, radius_cells: float) -> np.ndarray:
    rows, columns = np.indices((grid_size, grid_size), dtype=np.float32)
    coordinates = np.stack((rows.reshape(-1), columns.reshape(-1)), axis=1)
    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    squared_distances = np.square(offsets).sum(axis=2)
    radius_squared = float(radius_cells) ** 2
    normalised_costs = squared_distances / radius_squared
    normalised_costs[squared_distances > radius_squared + 1e-6] = np.inf
    return normalised_costs.astype(np.float32, copy=False)


def _top_tail_mean(cell_scores: np.ndarray, tail_count: int) -> np.ndarray:
    cell_count = cell_scores.shape[1]
    return np.partition(cell_scores, cell_count - tail_count, axis=1)[
        :, -tail_count:
    ].mean(axis=1, dtype=np.float64)


def bidirectional_position_knn_scores(
    pooled_tokens: np.ndarray,
    consensus_template: np.ndarray,
    *,
    grid_size: int,
    neighbors: int,
    position_radius_cells: float,
    position_weight: float,
    tail_fraction: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Score images with local query-to-template and template-to-query KNN."""

    pooled_tokens = np.asarray(pooled_tokens, dtype=np.float32)
    consensus_template = np.asarray(consensus_template, dtype=np.float32)
    if pooled_tokens.ndim != 3:
        raise ValueError(
            "pooled_tokens must have shape [images, cells, channels]"
        )
    image_count, cell_count, channels = pooled_tokens.shape
    if grid_size <= 0 or cell_count != grid_size * grid_size:
        raise ValueError("MSSM-B cells do not match grid_size")
    if consensus_template.shape != (cell_count, channels):
        raise ValueError(
            "MSSM-B template shape does not match the pooled token grids"
        )
    if neighbors <= 0:
        raise ValueError("MSSM-B neighbors must be positive")
    if not math.isfinite(position_radius_cells) or position_radius_cells <= 0:
        raise ValueError("MSSM-B position radius must be finite and positive")
    if not math.isfinite(position_weight) or position_weight < 0:
        raise ValueError("MSSM-B position weight must be finite and non-negative")
    if not 0 < tail_fraction <= 1:
        raise ValueError("MSSM-B tail fraction must be in (0, 1]")

    normalised_queries = _normalise_rows(
        pooled_tokens.reshape(image_count * cell_count, channels)
    ).reshape(image_count, cell_count, channels)
    normalised_template = _normalise_rows(consensus_template)
    position_costs = _grid_position_costs(grid_size, position_radius_cells)
    candidates_per_position = np.isfinite(position_costs).sum(axis=1)
    if neighbors > int(candidates_per_position.min()):
        raise ValueError(
            "MSSM-B neighbors exceeds the number of candidates inside the "
            "position radius"
        )

    query_cell_scores = np.empty((image_count, cell_count), dtype=np.float32)
    template_cell_scores = np.empty_like(query_cell_scores)
    for image_index, query_tokens in enumerate(normalised_queries):
        appearance_costs = 1.0 - query_tokens @ normalised_template.T
        np.clip(appearance_costs, 0.0, 2.0, out=appearance_costs)
        costs = appearance_costs + position_weight * position_costs

        query_nearest = np.partition(costs, neighbors - 1, axis=1)[
            :, :neighbors
        ]
        template_nearest = np.partition(costs.T, neighbors - 1, axis=1)[
            :, :neighbors
        ]
        query_cell_scores[image_index] = query_nearest.mean(axis=1)
        template_cell_scores[image_index] = template_nearest.mean(axis=1)

    tail_count = max(1, math.ceil(tail_fraction * cell_count))
    query_to_template = _top_tail_mean(query_cell_scores, tail_count)
    template_to_query = _top_tail_mean(template_cell_scores, tail_count)
    scores = (query_to_template + template_to_query) / 2.0
    diagnostics = {
        "query_to_template_score": query_to_template,
        "template_to_query_score": template_to_query,
        "directional_gap": query_to_template - template_to_query,
    }
    return scores, diagnostics
