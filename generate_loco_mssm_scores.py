"""Generate zero-shot MSSM rankings for native MVTec LOCO AD.

Only good and logical_anomalies test images are consumed. Directory labels
define this subset but are never used by MSSM scoring or reference selection.

The LOCO ranking contains only two label-free, transductive signals:

* MSSM-G: Layer-12 spatial-pyramid descriptors and multi-neighbour density.
* MSSM-C: robust spatial consensus plus unsupervised component/count changes.

No train/good image, anomaly label, or ground-truth mask is used for scoring.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from config import Config
from generate_mssm_scores import (
    _as_patch_feature_matrix,
    _atomic_save_numpy,
    _atomic_write_json,
    _load_resume_state,
    _validate_mssm_layers_for_model,
)
from src.dataset_info import (
    MVTec_LOCO_OBJECT_ANOMALIES,
    MVTec_LOCO_SCOPE,
    MVTec_LOCO_TEST_TYPES,
    sort_scored_paths_label_agnostic,
    sorted_image_paths,
    validate_mvtec_loco_rankings,
)


LOCO_DEFAULT_GLOBAL_LAYER = 12
LOCO_MSSM_GC_ALGORITHM = (
    "mssm_gc_global_density_spatial_count_consensus_v2"
)
LOCO_MSSM_GC_DEFAULT_OUTPUT = (
    "json/MVTecLOCO/dinov2_vits14/batch-0-shot/logical_only/"
    "mssm_gc/global_layer=12/results.json"
)


def _normalise_rows(features: np.ndarray) -> np.ndarray:
    """L2-normalize rows while rejecting invalid or zero feature vectors."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not len(features):
        raise ValueError(f"Expected a non-empty feature matrix, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Encountered a zero-norm feature vector")
    return (features / norms).astype(np.float32, copy=False)


def _average_percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks in [0, 1], preserving exact ties."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        raise ValueError("Cannot rank an empty score vector")
    if not np.isfinite(values).all():
        raise ValueError("Scores contain NaN or infinity")
    if len(values) == 1:
        return np.zeros(1, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks / (len(values) - 1)


def fuse_ranked_anomaly_scores(
    score_components: dict[str, np.ndarray],
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fuse differently-scaled anomaly signals without label calibration."""

    if not score_components:
        raise ValueError("At least one score component is required")
    lengths = {len(np.asarray(values).reshape(-1)) for values in score_components.values()}
    if len(lengths) != 1:
        raise ValueError("Score components must have the same length")
    unknown_weights = set(weights) - set(score_components)
    if unknown_weights:
        raise ValueError(f"Weights supplied for unknown components: {unknown_weights}")

    ranked = {
        name: _average_percentile_ranks(values)
        for name, values in score_components.items()
    }
    total_weight = 0.0
    fused = np.zeros(next(iter(lengths)), dtype=np.float64)
    for name, values in ranked.items():
        weight = float(weights.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid non-negative weight for {name}: {weight}")
        fused += weight * values
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("At least one score weight must be positive")
    return fused / total_weight, ranked


def adaptive_pool_patch_tokens(
    patch_tokens: np.ndarray,
    grid_size: tuple[int, int],
    output_size: int,
) -> np.ndarray:
    """Average a variable-resolution DINO grid into a square fixed grid."""

    if output_size <= 0:
        raise ValueError("output_size must be positive")
    height, width = map(int, grid_size)
    patch_tokens = np.asarray(patch_tokens, dtype=np.float32)
    if patch_tokens.ndim != 2 or len(patch_tokens) != height * width:
        raise ValueError(
            "Patch token count does not match grid size: "
            f"tokens={patch_tokens.shape}, grid={grid_size}"
        )
    feature_grid = patch_tokens.reshape(height, width, -1)
    pooled = np.empty(
        (output_size, output_size, patch_tokens.shape[1]), dtype=np.float32
    )
    for row in range(output_size):
        row_start = math.floor(row * height / output_size)
        row_end = math.ceil((row + 1) * height / output_size)
        for column in range(output_size):
            column_start = math.floor(column * width / output_size)
            column_end = math.ceil((column + 1) * width / output_size)
            pooled[row, column] = feature_grid[
                row_start:row_end, column_start:column_end
            ].mean(axis=(0, 1))
    return pooled.reshape(output_size * output_size, -1)


def build_global_descriptor(
    patch_tokens: np.ndarray,
    grid_size: tuple[int, int],
    pyramid_levels: tuple[int, ...],
) -> np.ndarray:
    """Build a Layer-12 global/layout descriptor without a learned head."""

    if not pyramid_levels or any(level <= 0 for level in pyramid_levels):
        raise ValueError("Global pyramid levels must be positive")
    if len(pyramid_levels) != len(set(pyramid_levels)):
        raise ValueError("Global pyramid levels must be unique")
    parts = []
    for level in pyramid_levels:
        pooled = adaptive_pool_patch_tokens(patch_tokens, grid_size, level)
        parts.append(_normalise_rows(pooled).reshape(-1))
    descriptor = np.concatenate(parts).astype(np.float32, copy=False)
    return _normalise_rows(descriptor[None])[0]


def global_density_scores(
    descriptors: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    """Mean cosine distance to several other images, excluding self."""

    descriptors = _normalise_rows(descriptors)
    if len(descriptors) < 2:
        raise ValueError("MSSM-G requires at least two images")
    if neighbors <= 0:
        raise ValueError("Global neighbour count must be positive")
    neighbor_count = min(neighbors, len(descriptors) - 1)
    distances = 1.0 - descriptors @ descriptors.T
    np.clip(distances, 0.0, 2.0, out=distances)
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, neighbor_count - 1, axis=1)[
        :, :neighbor_count
    ]
    return nearest.mean(axis=1, dtype=np.float64)


def _connected_component_counts(
    label_maps: np.ndarray,
    clusters: int,
    minimum_cells: int,
) -> np.ndarray:
    from scipy import ndimage

    if minimum_cells <= 0:
        raise ValueError("minimum component size must be positive")
    counts = np.zeros((len(label_maps), clusters), dtype=np.float64)
    connectivity = np.ones((3, 3), dtype=np.uint8)
    for image_index, label_map in enumerate(label_maps):
        for cluster_index in range(clusters):
            components, component_count = ndimage.label(
                label_map == cluster_index,
                structure=connectivity,
            )
            if component_count:
                sizes = np.bincount(components.reshape(-1))[1:]
                counts[image_index, cluster_index] = np.count_nonzero(
                    sizes >= minimum_cells
                )
    return counts


def count_consensus_scores(
    pooled_tokens: np.ndarray,
    *,
    grid_size: int,
    clusters: int,
    minimum_component_cells: int,
    spatial_tail_fraction: float,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute label-free spatial, abundance, and component-count deviations."""

    from sklearn.cluster import MiniBatchKMeans

    pooled_tokens = np.asarray(pooled_tokens, dtype=np.float32)
    if pooled_tokens.ndim != 3:
        raise ValueError("pooled_tokens must have shape [images, cells, channels]")
    image_count, cell_count, channels = pooled_tokens.shape
    if image_count < 2:
        raise ValueError("MSSM-C requires at least two images")
    if cell_count != grid_size * grid_size:
        raise ValueError("MSSM-C cells do not match count grid size")
    if clusters < 2 or clusters > image_count * cell_count:
        raise ValueError("Invalid MSSM-C component cluster count")
    if not 0 < spatial_tail_fraction <= 1:
        raise ValueError("MSSM-C spatial tail fraction must be in (0, 1]")

    flat_tokens = _normalise_rows(pooled_tokens.reshape(-1, channels))
    normalized_tokens = flat_tokens.reshape(image_count, cell_count, channels)

    # A coordinate-wise median is robust to different anomalies affecting
    # different images. It does not designate any complete image as normal.
    consensus = np.median(normalized_tokens, axis=0).astype(np.float32)
    consensus_norms = np.linalg.norm(consensus, axis=1, keepdims=True)
    weak_rows = consensus_norms[:, 0] <= 1e-12
    if np.any(weak_rows):
        consensus[weak_rows] = normalized_tokens[:, weak_rows].mean(axis=0)
    consensus = _normalise_rows(consensus)
    spatial_distances = 1.0 - np.einsum(
        "ncd,cd->nc", normalized_tokens, consensus, optimize=True
    )
    np.clip(spatial_distances, 0.0, 2.0, out=spatial_distances)
    tail_count = max(1, math.ceil(spatial_tail_fraction * cell_count))
    spatial_scores = np.partition(
        spatial_distances, cell_count - tail_count, axis=1
    )[:, -tail_count:].mean(axis=1, dtype=np.float64)

    component_model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=random_state,
        n_init=3,
        batch_size=min(2048, len(flat_tokens)),
    )
    component_labels = component_model.fit_predict(flat_tokens).reshape(
        image_count, grid_size, grid_size
    )
    histograms = np.stack(
        [
            np.bincount(labels.reshape(-1), minlength=clusters) / cell_count
            for labels in component_labels
        ]
    )
    consensus_histogram = np.median(histograms, axis=0)
    histogram_total = float(consensus_histogram.sum())
    if histogram_total <= 0:
        raise RuntimeError("MSSM-C produced an empty consensus histogram")
    consensus_histogram /= histogram_total
    histogram_scores = np.sqrt(
        0.5
        * np.square(
            np.sqrt(histograms) - np.sqrt(consensus_histogram[None])
        ).sum(axis=1)
    )

    component_counts = _connected_component_counts(
        component_labels,
        clusters,
        minimum_component_cells,
    )
    consensus_counts = np.median(component_counts, axis=0)
    count_scores = (
        np.abs(component_counts - consensus_counts[None])
        / np.maximum(consensus_counts[None], 1.0)
    ).mean(axis=1)

    count_score, ranked = fuse_ranked_anomaly_scores(
        {
            "spatial": spatial_scores,
            "histogram": histogram_scores,
            "components": count_scores,
        },
        {"spatial": 1.0, "histogram": 1.0, "components": 1.0},
    )
    diagnostics = {
        "spatial_score": spatial_scores,
        "histogram_score": histogram_scores,
        "component_count_score": count_scores,
        "spatial_rank": ranked["spatial"],
        "histogram_rank": ranked["histogram"],
        "component_count_rank": ranked["components"],
        "component_counts": component_counts,
        "consensus_component_counts": consensus_counts,
    }
    return count_score, diagnostics


def get_object_test_samples(
    data_root: Path,
    object_name: str,
) -> list[tuple[str, Path]]:
    """List normal and logical LOCO test samples in deterministic order."""

    test_root = data_root / object_name / "test"
    return [
        (image_path.relative_to(test_root).as_posix(), image_path)
        for test_type in MVTec_LOCO_TEST_TYPES
        for image_path in sorted_image_paths(test_root / test_type)
    ]


def extract_loco_ranking_tokens(
    model: Any,
    image_tensor: Any,
    grid_size: tuple[int, int],
    layer: int,
) -> np.ndarray:
    """Extract one DINO token grid shared by native LOCO rankers."""

    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError("layer must be an integer")
    if layer <= 0:
        raise ValueError("layer uses one-based positive layer numbers")
    _validate_mssm_layers_for_model(model, (layer,))
    layer_outputs = model.extract_features(
        image_tensor,
        feature_list=[layer - 1],
    )
    if not isinstance(layer_outputs, (list, tuple)):
        layer_outputs = [layer_outputs]
    if len(layer_outputs) != 1:
        raise RuntimeError(
            "DINO returned a different number of LOCO ranking layers than "
            f"requested: requested={[layer]}, returned={len(layer_outputs)}"
        )

    global_tokens = _as_patch_feature_matrix(
        layer_outputs[0], layer=layer
    )
    expected_patches = int(grid_size[0]) * int(grid_size[1])
    if len(global_tokens) != expected_patches:
        raise RuntimeError(
            "LOCO ranking features do not match the prepared image grid: "
            f"tokens={len(global_tokens)}, grid={grid_size}"
        )
    return _normalise_rows(global_tokens)


def _loco_token_cache_paths(
    cache_dir: Path,
    relative_path: str,
) -> tuple[Path, Path]:
    relative_stem = Path(relative_path).with_suffix("")
    return (
        cache_dir / "global" / relative_stem.with_suffix(".npy"),
        cache_dir / "grid" / relative_stem.with_suffix(".npy"),
    )


def _load_or_extract_loco_tokens(
    model: Any,
    image_path: Path,
    relative_path: str,
    cache_dir: Path,
    layer: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    import cv2

    global_path, grid_path = _loco_token_cache_paths(cache_dir, relative_path)
    cache_paths = (global_path, grid_path)
    cache_is_fresh = all(
        path.is_file()
        and path.stat().st_mtime_ns >= image_path.stat().st_mtime_ns
        for path in cache_paths
    )
    if cache_is_fresh:
        global_tokens = np.load(global_path, allow_pickle=False)
        grid_array = np.load(grid_path, allow_pickle=False)
    else:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"OpenCV could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor, grid_size = model.prepare_image(image_rgb)
        global_tokens = extract_loco_ranking_tokens(
            model,
            image_tensor,
            grid_size,
            layer,
        )
        grid_array = np.asarray(grid_size, dtype=np.int32)
        _atomic_save_numpy(global_path, global_tokens)
        _atomic_save_numpy(grid_path, grid_array)

    global_tokens = np.asarray(global_tokens, dtype=np.float32)
    grid_array = np.asarray(grid_array)
    if global_tokens.ndim != 2 or not len(global_tokens):
        raise ValueError(f"Invalid LOCO token cache at {global_path}")
    if not np.isfinite(global_tokens).all():
        raise ValueError(
            f"LOCO token cache contains NaN or infinity: {cache_dir}"
        )
    if grid_array.shape != (2,) or np.any(grid_array <= 0):
        raise ValueError(
            f"Invalid LOCO grid cache at {grid_path}: {grid_array}"
        )
    grid_size = (int(grid_array[0]), int(grid_array[1]))
    if len(global_tokens) != math.prod(grid_size):
        raise ValueError(
            f"LOCO cached token/grid mismatch for {relative_path}: "
            f"global={global_tokens.shape}, grid={grid_size}"
        )
    return global_tokens, grid_size


def _prepare_loco_token_cache(
    cache_root: Path,
    *,
    data_root: Path,
    model_name: str,
    resolution: int,
    layer: int,
) -> None:
    """Bind the shared token cache to every feature-producing input."""

    settings = {
        "cache_format": "loco_dino_token_grid_v1",
        "data_root": str(data_root),
        "model_name": model_name,
        "resolution": resolution,
        "layer": layer,
        "layer_numbering": "one_based_transformer_blocks",
    }
    metadata_path = cache_root / "cache.meta.json"
    if metadata_path.is_file():
        import json

        with metadata_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing != settings:
            raise RuntimeError(
                f"LOCO token cache metadata mismatch at {cache_root}. "
                "Choose a different --feature_cache directory."
            )
        return
    if cache_root.is_dir() and any(cache_root.rglob("*.npy")):
        raise RuntimeError(
            f"LOCO token cache contains arrays but no metadata: {cache_root}. "
            "Choose a different --feature_cache directory."
        )
    _atomic_write_json(metadata_path, settings)


def _ranking_tag_from_args(args: argparse.Namespace) -> str:
    return (
        f"gc-g{args.global_layer}-k{args.global_neighbors}-"
        f"c{args.count_clusters}-grid{args.count_grid_size}"
    )


def _resolve_output_path(args: argparse.Namespace) -> Path:
    return Path(args.output_json).expanduser().resolve()


def _settings_from_args(
    args: argparse.Namespace,
    data_root: Path,
) -> dict[str, Any]:
    return {
        "dataset": "MVTecLOCO",
        "experiment_scope": MVTec_LOCO_SCOPE,
        "test_types": list(MVTec_LOCO_TEST_TYPES),
        "ranking_algorithm": LOCO_MSSM_GC_ALGORITHM,
        "ranking_mode": "gc_only",
        "ranking_tag": _ranking_tag_from_args(args),
        "data_root": str(data_root),
        "model_name": args.model_name,
        "resolution": args.resolution,
        "global_layer": args.global_layer,
        "layer_numbering": "one_based_transformer_blocks",
        "global_neighbors": args.global_neighbors,
        "global_pyramid_levels": list(args.global_pyramid_levels),
        "count_grid_size": args.count_grid_size,
        "count_clusters": args.count_clusters,
        "count_min_component_cells": args.count_min_component_cells,
        "count_spatial_tail_fraction": args.count_spatial_tail_fraction,
        "score_fusion": "average_empirical_percentile_ranks_v1",
        "score_weights": {
            "global": args.global_weight,
            "count": args.count_weight,
        },
        "uses_original_patch_mssm": False,
        "uses_train_good": False,
        "uses_ground_truth_labels_for_scoring": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate native MVTec LOCO AD MSSM-G + MSSM-C rankings"
    )
    parser.add_argument(
        "--dataset", default="MVTecLOCO", choices=["MVTecLOCO"]
    )
    parser.add_argument("--data_root", default=Config.ROOTS["MVTecLOCO"])
    parser.add_argument("--model_name", default="dinov2_vits14")
    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument(
        "--global_layer",
        type=int,
        default=LOCO_DEFAULT_GLOBAL_LAYER,
        help="One-based DINO block used by MSSM-G and MSSM-C",
    )
    parser.add_argument("--global_neighbors", type=int, default=10)
    parser.add_argument(
        "--global_pyramid_levels",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    parser.add_argument("--count_grid_size", type=int, default=12)
    parser.add_argument("--count_clusters", type=int, default=8)
    parser.add_argument("--count_min_component_cells", type=int, default=1)
    parser.add_argument("--count_spatial_tail_fraction", type=float, default=0.1)
    parser.add_argument("--global_weight", type=float, default=1.0)
    parser.add_argument("--count_weight", type=float, default=1.0)
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--output_json", default=LOCO_MSSM_GC_DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch
    from tqdm import tqdm

    from src.backbones import get_model
    from validate_mvtec_loco import validate_mvtec_loco_layout

    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    objects = args.objects or list(MVTec_LOCO_OBJECT_ANOMALIES)
    validation_report = validate_mvtec_loco_layout(data_root, objects=objects)
    if not validation_report["valid"]:
        for error in validation_report["errors"]:
            print(f"[ERROR] {error}")
        raise RuntimeError(
            "MVTec LOCO AD validation failed; fix the dataset before MSSM "
            "scoring"
        )

    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.global_layer <= 0:
        raise ValueError("--global_layer uses one-based positive layer numbers")
    if args.global_neighbors <= 0:
        raise ValueError("--global_neighbors must be positive")
    pyramid_levels = tuple(args.global_pyramid_levels)
    if (
        not pyramid_levels
        or any(level <= 0 for level in pyramid_levels)
        or len(pyramid_levels) != len(set(pyramid_levels))
    ):
        raise ValueError(
            "--global_pyramid_levels must contain unique positive integers"
        )
    if args.count_grid_size <= 0:
        raise ValueError("--count_grid_size must be positive")
    if args.count_clusters < 2:
        raise ValueError("--count_clusters must be at least 2")
    if args.count_min_component_cells <= 0:
        raise ValueError("--count_min_component_cells must be positive")
    if not 0 < args.count_spatial_tail_fraction <= 1:
        raise ValueError("--count_spatial_tail_fraction must be in (0, 1]")
    gc_weights = (args.global_weight, args.count_weight)
    if any(not math.isfinite(weight) or weight < 0 for weight in gc_weights):
        raise ValueError("MSSM-GC score weights must be finite and non-negative")
    if sum(gc_weights) <= 0:
        raise ValueError("At least one MSSM-GC score weight must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but CUDA is unavailable: {args.device}"
        )

    output_path = _resolve_output_path(args)
    metadata_path = output_path.with_name(output_path.stem + ".meta.json")
    default_cache_root = (
        Path(Config.FEATURE_ROOT)
        / "MVTecLOCO"
        / args.model_name
        / MVTec_LOCO_SCOPE
        / f"loco_tokens_resolution={args.resolution}"
        / f"layer={args.global_layer}"
    )
    cache_root = Path(args.feature_cache or default_cache_root).expanduser().resolve()
    settings = _settings_from_args(args, data_root)
    results, metadata = _load_resume_state(
        output_path, metadata_path, settings, args.force
    )
    _prepare_loco_token_cache(
        cache_root,
        data_root=data_root,
        model_name=args.model_name,
        resolution=args.resolution,
        layer=args.global_layer,
    )
    _atomic_write_json(output_path, results)
    _atomic_write_json(metadata_path, metadata)

    model = get_model(
        args.model_name,
        args.device,
        smaller_edge_size=args.resolution,
    )
    _validate_mssm_layers_for_model(model, (args.global_layer,))

    for object_name in objects:
        samples = get_object_test_samples(data_root, object_name)
        expected_paths = [relative_path for relative_path, _ in samples]
        existing_entries = results.get(object_name)
        if existing_entries is not None:
            entries_are_objects = isinstance(existing_entries, list) and all(
                isinstance(entry, dict) for entry in existing_entries
            )
            existing_paths = (
                [entry.get("path") for entry in existing_entries]
                if entries_are_objects
                else []
            )
            existing_scores = (
                [entry.get("score") for entry in existing_entries]
                if entries_are_objects
                else []
            )
            gc_fields_are_complete = all(
                {
                    "global_score",
                    "count_score",
                    "component_counts",
                }.issubset(entry)
                for entry in existing_entries
            )
            if (
                entries_are_objects
                and gc_fields_are_complete
                and sorted(existing_paths) == sorted(expected_paths)
                and all(
                    not isinstance(score, bool)
                    and isinstance(score, (int, float))
                    and math.isfinite(score)
                    for score in existing_scores
                )
                and existing_scores == sorted(existing_scores)
            ):
                try:
                    validate_mvtec_loco_rankings(
                        {object_name: existing_entries},
                        [object_name],
                        data_root,
                    )
                except ValueError as error:
                    raise RuntimeError(
                        f"Existing entries for {object_name} do not follow "
                        "the current label-agnostic ranking contract. Use "
                        "--force to rebuild the output."
                    ) from error
                print(
                    f"[Resume] {object_name} already has a complete MSSM-GC "
                    "ranking"
                )
                continue
            raise RuntimeError(
                f"Existing entries for {object_name} are incomplete or "
                "invalid. Use --force to rebuild the output."
            )

        print(
            f"Extracting MSSM-GC features for {object_name} "
            f"({len(samples)} images)"
        )
        token_features = [
            _load_or_extract_loco_tokens(
                model,
                image_path,
                relative_path,
                cache_root / object_name,
                args.global_layer,
            )
            for relative_path, image_path in tqdm(
                samples, desc=f"GC features: {object_name}"
            )
        ]
        global_tokens_per_image = [item[0] for item in token_features]
        grid_sizes = [item[1] for item in token_features]

        global_descriptors = np.stack(
            [
                build_global_descriptor(tokens, grid_size, pyramid_levels)
                for tokens, grid_size in zip(
                    global_tokens_per_image, grid_sizes
                )
            ]
        )
        global_scores = global_density_scores(
            global_descriptors,
            args.global_neighbors,
        )
        count_tokens = np.stack(
            [
                adaptive_pool_patch_tokens(
                    tokens,
                    grid_size,
                    args.count_grid_size,
                )
                for tokens, grid_size in zip(
                    global_tokens_per_image, grid_sizes
                )
            ]
        )
        count_scores, count_diagnostics = count_consensus_scores(
            count_tokens,
            grid_size=args.count_grid_size,
            clusters=args.count_clusters,
            minimum_component_cells=args.count_min_component_cells,
            spatial_tail_fraction=args.count_spatial_tail_fraction,
        )
        fused_scores, fused_ranks = fuse_ranked_anomaly_scores(
            {
                "global": global_scores,
                "count": count_scores,
            },
            {
                "global": args.global_weight,
                "count": args.count_weight,
            },
        )
        object_entries = []
        component_counts = count_diagnostics["component_counts"]
        for index, (relative_path, _) in enumerate(samples):
            object_entries.append(
                {
                    "path": relative_path,
                    "score": float(fused_scores[index]),
                    "global_score": float(global_scores[index]),
                    "count_score": float(count_scores[index]),
                    "spatial_consensus_score": float(
                        count_diagnostics["spatial_score"][index]
                    ),
                    "component_histogram_score": float(
                        count_diagnostics["histogram_score"][index]
                    ),
                    "component_count_score": float(
                        count_diagnostics["component_count_score"][index]
                    ),
                    "global_rank": float(fused_ranks["global"][index]),
                    "count_rank": float(fused_ranks["count"][index]),
                    "component_counts": [
                        int(value) for value in component_counts[index]
                    ],
                }
            )
        metadata.setdefault("gc_diagnostics", {})[object_name] = {
            "consensus_component_counts": [
                float(value)
                for value in count_diagnostics[
                    "consensus_component_counts"
                ]
            ],
            "image_count": len(samples),
        }

        object_entries = sort_scored_paths_label_agnostic(
            object_entries,
            data_root / object_name / "test",
        )
        results[object_name] = object_entries
        completed_objects = set(metadata.get("completed_objects", []))
        completed_objects.add(object_name)
        metadata["completed_objects"] = sorted(completed_objects)
        _atomic_write_json(output_path, results)
        _atomic_write_json(metadata_path, metadata)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    validate_mvtec_loco_rankings(results, objects, data_root)
    print(f"Wrote MSSM-G + MSSM-C rankings to {output_path}")
    print(f"Wrote run metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
