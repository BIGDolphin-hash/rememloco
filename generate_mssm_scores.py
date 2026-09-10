"""Generate the batched zero-shot MSSM ranking required by ReMem.

For each MPDD object, every test image is scored against patch features from
all *other* test images of the same object. Per-patch scores are the mean of
the closest 0.1% cosine distances; an image score is the mean of its highest
1% patch scores. The resulting lists are sorted from most likely normal to
most likely anomalous and written in the format consumed by ReMem.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from config import Config
from src.dataset_info import (
    MPDD_OBJECT_ANOMALIES,
    resolve_mpdd_anomaly_types,
    sorted_image_paths,
    validate_mpdd_rankings,
)
from validate_mpdd import validate_mpdd_layout


DEFAULT_MSSM_LAYERS = (12,)
MSSM_LAYER_FUSION = "per_layer_l2_normalized_concat_equal_weight_v1"


def _atomic_write_json(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)


def _atomic_save_numpy(output_path: Path, array: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    os.replace(temporary_path, output_path)


def _normalise_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encountered a zero-norm patch feature")
    return features / norms


def validate_mssm_layers(layers: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Validate one-based transformer block numbers used by MSSM."""

    layers = tuple(layers)
    if not layers:
        raise ValueError("--mssm_layers must contain at least one layer")
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
        raise ValueError("--mssm_layers must contain integer layer numbers")
    if any(layer <= 0 for layer in layers):
        raise ValueError("--mssm_layers uses one-based positive layer numbers")
    if len(layers) != len(set(layers)):
        raise ValueError("--mssm_layers must contain unique layer numbers")
    if tuple(sorted(layers)) != layers:
        raise ValueError("--mssm_layers must be in ascending order")
    return layers


def mssm_layers_tag(layers: list[int] | tuple[int, ...]) -> str:
    return "-".join(map(str, validate_mssm_layers(layers)))


def _validate_mssm_layers_for_model(model: Any, layers: tuple[int, ...]) -> None:
    """Reject layer numbers beyond the loaded transformer's depth."""

    blocks = getattr(getattr(model, "model", None), "blocks", None)
    if blocks is None:
        return
    depth = len(blocks)
    if max(layers) > depth:
        raise ValueError(
            f"--mssm_layers contains layer {max(layers)}, but the loaded model "
            f"has only {depth} transformer blocks"
        )


def _as_patch_feature_matrix(features: Any, *, layer: int) -> np.ndarray:
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 2 or not len(features):
        raise ValueError(
            f"Invalid MSSM patch features from layer {layer}: {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"MSSM layer {layer} features contain NaN or infinity")
    return features


def extract_mssm_features(
    model: Any,
    image_tensor: Any,
    layers: list[int] | tuple[int, ...] = DEFAULT_MSSM_LAYERS,
) -> np.ndarray:
    """Extract and equally fuse one or more one-based DINO block outputs.

    Every layer is L2-normalized independently before concatenation. The
    existing downstream normalization therefore makes cosine similarity of the
    concatenated vector equal to the arithmetic mean of the per-layer cosine
    similarities.
    """

    layers = validate_mssm_layers(layers)
    _validate_mssm_layers_for_model(model, layers)
    zero_based_layers = [layer - 1 for layer in layers]
    layer_outputs = model.extract_features(
        image_tensor,
        feature_list=zero_based_layers,
    )
    if not isinstance(layer_outputs, (list, tuple)):
        layer_outputs = [layer_outputs]
    if len(layer_outputs) != len(layers):
        raise RuntimeError(
            "DINO returned a different number of feature layers than MSSM "
            f"requested: requested={list(layers)}, returned={len(layer_outputs)}"
        )

    normalized_layers = [
        _normalise_features(_as_patch_feature_matrix(features, layer=layer))
        for layer, features in zip(layers, layer_outputs)
    ]
    patch_counts = {features.shape[0] for features in normalized_layers}
    if len(patch_counts) != 1:
        raise RuntimeError(
            "MSSM feature layers do not have the same number of spatial patches"
        )
    return np.concatenate(normalized_layers, axis=1).astype(
        np.float32,
        copy=False,
    )


def mutual_patch_scores(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    device: str,
    quantile: float = 0.001,
    query_chunk_size: int = 256,
    reference_chunk_size: int = 8192,
    excluded_reference_range: tuple[int, int] | None = None,
    features_are_normalised: bool = False,
) -> np.ndarray:
    """Compute memory-bounded MSSM patch scores.

    ``excluded_reference_range`` identifies the query image's patch rows in a
    global reference matrix. Those rows receive infinite distance and cannot
    contribute to the nearest-neighbour average.
    """

    import torch

    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    if query_chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive")

    query_features = np.asarray(query_features, dtype=np.float32)
    reference_features = np.asarray(reference_features, dtype=np.float32)
    if query_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("Features must be two-dimensional [patches, channels]")
    if query_features.shape[1] != reference_features.shape[1]:
        raise ValueError("Query and reference feature dimensions do not match")
    if not np.isfinite(query_features).all() or not np.isfinite(reference_features).all():
        raise ValueError("Features contain NaN or infinity")

    if excluded_reference_range is None:
        excluded_count = 0
    else:
        excluded_start, excluded_end = excluded_reference_range
        if not 0 <= excluded_start <= excluded_end <= len(reference_features):
            raise ValueError("Invalid excluded reference range")
        excluded_count = excluded_end - excluded_start

    effective_reference_count = len(reference_features) - excluded_count
    if effective_reference_count <= 0:
        raise ValueError("MSSM requires reference patches from at least one other image")
    nearest_count = max(1, math.ceil(quantile * effective_reference_count))

    if not features_are_normalised:
        query_features = _normalise_features(query_features)
        reference_features = _normalise_features(reference_features)

    torch_device = torch.device(device)
    patch_score_chunks = []
    for query_start in range(0, len(query_features), query_chunk_size):
        query_end = min(query_start + query_chunk_size, len(query_features))
        query_tensor = torch.from_numpy(
            np.ascontiguousarray(query_features[query_start:query_end])
        ).to(torch_device)
        best_distances = torch.full(
            (len(query_tensor), nearest_count),
            float("inf"),
            dtype=torch.float32,
            device=torch_device,
        )

        for reference_start in range(
            0, len(reference_features), reference_chunk_size
        ):
            reference_end = min(
                reference_start + reference_chunk_size, len(reference_features)
            )
            reference_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    reference_features[reference_start:reference_end]
                )
            ).to(torch_device)
            distances = 1.0 - torch.matmul(query_tensor, reference_tensor.T)
            distances.clamp_(min=0.0, max=2.0)

            if excluded_reference_range is not None:
                overlap_start = max(reference_start, excluded_start)
                overlap_end = min(reference_end, excluded_end)
                if overlap_start < overlap_end:
                    local_start = overlap_start - reference_start
                    local_end = overlap_end - reference_start
                    distances[:, local_start:local_end] = float("inf")

            candidates = torch.cat((best_distances, distances), dim=1)
            best_distances = torch.topk(
                candidates,
                k=nearest_count,
                dim=1,
                largest=False,
                sorted=False,
            ).values

        if not torch.isfinite(best_distances).all():
            raise RuntimeError("MSSM nearest-neighbour search produced invalid distances")
        patch_score_chunks.append(best_distances.mean(dim=1).cpu().numpy())

    return np.concatenate(patch_score_chunks).astype(np.float32, copy=False)


def aggregate_image_score(patch_scores: np.ndarray, tail_fraction: float = 0.01) -> float:
    """Aggregate patch scores using the highest-scoring image tail."""

    patch_scores = np.asarray(patch_scores, dtype=np.float32).reshape(-1)
    if not len(patch_scores):
        raise ValueError("Cannot aggregate an empty patch-score array")
    if not 0 < tail_fraction <= 1:
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}")
    tail_count = max(1, math.ceil(tail_fraction * len(patch_scores)))
    tail = np.partition(patch_scores, len(patch_scores) - tail_count)[-tail_count:]
    return float(np.mean(tail, dtype=np.float64))


def get_object_test_samples(
    data_root: Path,
    object_name: str,
) -> list[tuple[str, Path]]:
    """List all object test samples without using their labels for scoring."""

    relative_and_absolute_paths = []
    test_types = ["good", *resolve_mpdd_anomaly_types(data_root, object_name)]
    for test_type in test_types:
        for image_path in sorted_image_paths(data_root / object_name / "test" / test_type):
            relative_path = image_path.relative_to(data_root / object_name / "test")
            relative_and_absolute_paths.append((relative_path.as_posix(), image_path))
    return relative_and_absolute_paths


def _load_or_extract_features(
    model: Any,
    image_path: Path,
    relative_path: str,
    cache_dir: Path,
    mssm_layers: list[int] | tuple[int, ...] = DEFAULT_MSSM_LAYERS,
) -> np.ndarray:
    import cv2

    cache_path = cache_dir / Path(relative_path).with_suffix(".npy")
    cache_is_fresh = (
        cache_path.is_file()
        and cache_path.stat().st_mtime_ns >= image_path.stat().st_mtime_ns
    )
    if cache_is_fresh:
        features = np.load(cache_path, allow_pickle=False)
    else:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"OpenCV could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor, _ = model.prepare_image(image_rgb)
        features = extract_mssm_features(model, image_tensor, mssm_layers)
        _atomic_save_numpy(cache_path, features)

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not len(features):
        raise ValueError(f"Invalid cached features at {cache_path}: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Cached features contain NaN or infinity: {cache_path}")
    return features


def _prepare_feature_cache(
    cache_root: Path,
    *,
    data_root: Path,
    model_name: str,
    resolution: int,
    mssm_layers: list[int] | tuple[int, ...] = DEFAULT_MSSM_LAYERS,
) -> None:
    """Bind a cache directory to the inputs that determine its features."""

    mssm_layers = validate_mssm_layers(mssm_layers)
    cache_settings = {
        "data_root": str(data_root),
        "model_name": model_name,
        "resolution": resolution,
        "mssm_layers": list(mssm_layers),
        "layer_numbering": "one_based_transformer_blocks",
        "layer_fusion": MSSM_LAYER_FUSION,
    }
    cache_metadata_path = cache_root / "cache.meta.json"
    if cache_metadata_path.is_file():
        with cache_metadata_path.open("r", encoding="utf-8") as file:
            existing_settings = json.load(file)
        if existing_settings != cache_settings:
            raise RuntimeError(
                f"Feature cache metadata mismatch at {cache_root}. Choose a "
                "different --feature_cache directory."
            )
        return

    if cache_root.is_dir() and any(cache_root.rglob("*.npy")):
        raise RuntimeError(
            f"Feature cache contains arrays but no metadata: {cache_root}. "
            "Choose a different --feature_cache directory."
        )
    _atomic_write_json(cache_metadata_path, cache_settings)


def _settings_from_args(args: argparse.Namespace, data_root: Path) -> dict[str, Any]:
    mssm_layers = validate_mssm_layers(args.mssm_layers)
    return {
        "dataset": "MPDD",
        "data_root": str(data_root),
        "model_name": args.model_name,
        "resolution": args.resolution,
        "mssm_layers": list(mssm_layers),
        "layer_numbering": "one_based_transformer_blocks",
        "layer_fusion": MSSM_LAYER_FUSION,
        "quantile": args.quantile,
        "image_tail_fraction": args.image_tail_fraction,
    }


def _load_resume_state(
    output_path: Path,
    metadata_path: Path,
    settings: dict[str, Any],
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if force or not output_path.exists():
        return {}, {"settings": settings, "completed_objects": []}
    if not metadata_path.is_file():
        raise RuntimeError(
            f"Found {output_path} without {metadata_path}. Use --force to replace it."
        )
    with output_path.open("r", encoding="utf-8") as file:
        results = json.load(file)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("settings") != settings:
        raise RuntimeError(
            "Existing MSSM metadata does not match this run. Use --force or a "
            "different --output_json path."
        )
    if not isinstance(results, dict):
        raise ValueError(f"Invalid MSSM results file: {output_path}")
    return results, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MPDD MSSM initial scores")
    parser.add_argument("--dataset", default="MPDD", choices=["MPDD"])
    parser.add_argument("--data_root", default=Config.ROOTS["MPDD"])
    parser.add_argument("--model_name", default="dinov2_vits14")
    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument(
        "--mssm_layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_MSSM_LAYERS),
        help=(
            "One-based DINO transformer block numbers. Multiple layers are "
            "L2-normalized independently and fused with equal weight"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--quantile", type=float, default=0.001)
    parser.add_argument("--image_tail_fraction", type=float, default=0.01)
    parser.add_argument("--query_chunk_size", type=int, default=256)
    parser.add_argument("--reference_chunk_size", type=int, default=8192)
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--output_json", default=Config.JSON_STARTS["MPDD"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch
    from tqdm import tqdm

    from src.backbones import get_model

    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    validation_report = validate_mpdd_layout(data_root)
    if not validation_report["valid"]:
        for error in validation_report["errors"]:
            print(f"[ERROR] {error}")
        raise RuntimeError("MPDD validation failed; fix the dataset before MSSM scoring")

    objects = args.objects or list(MPDD_OBJECT_ANOMALIES)
    unknown_objects = sorted(set(objects) - set(MPDD_OBJECT_ANOMALIES))
    if unknown_objects:
        raise ValueError(f"Unknown MPDD objects: {unknown_objects}")
    if len(objects) != len(set(objects)):
        raise ValueError("--objects contains duplicate names")
    if not 0 < args.quantile <= 1:
        raise ValueError("--quantile must be in (0, 1]")
    if not 0 < args.image_tail_fraction <= 1:
        raise ValueError("--image_tail_fraction must be in (0, 1]")
    if args.query_chunk_size <= 0 or args.reference_chunk_size <= 0:
        raise ValueError("MSSM chunk sizes must be positive")
    mssm_layers = validate_mssm_layers(args.mssm_layers)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")

    output_path = Path(args.output_json).expanduser().resolve()
    metadata_path = output_path.with_name(output_path.stem + ".meta.json")
    cache_root = Path(
        args.feature_cache
        or Path(Config.FEATURE_ROOT)
        / "MPDD"
        / args.model_name
        / f"mssm_resolution={args.resolution}"
        / f"layers={mssm_layers_tag(mssm_layers)}"
    ).expanduser().resolve()
    settings = _settings_from_args(args, data_root)
    results, metadata = _load_resume_state(
        output_path, metadata_path, settings, args.force
    )
    _prepare_feature_cache(
        cache_root,
        data_root=data_root,
        model_name=args.model_name,
        resolution=args.resolution,
        mssm_layers=mssm_layers,
    )
    # Persist a valid empty/partial checkpoint immediately. In particular,
    # --force must not leave stale rankings active if extraction is interrupted.
    _atomic_write_json(output_path, results)
    _atomic_write_json(metadata_path, metadata)

    model = get_model(
        args.model_name,
        args.device,
        smaller_edge_size=args.resolution,
    )
    _validate_mssm_layers_for_model(model, mssm_layers)

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
            if (
                entries_are_objects
                and sorted(existing_paths) == sorted(expected_paths)
                and all(
                    isinstance(score, (int, float)) and math.isfinite(score)
                    for score in existing_scores
                )
                and existing_scores == sorted(existing_scores)
            ):
                print(f"[Resume] {object_name} already has a complete MSSM ranking")
                continue
            raise RuntimeError(
                f"Existing entries for {object_name} are incomplete or invalid. "
                "Use --force to rebuild the output."
            )

        print(f"Extracting MSSM features for {object_name} ({len(samples)} images)")
        features_per_image = []
        for relative_path, image_path in tqdm(samples, desc=f"Features: {object_name}"):
            features_per_image.append(
                _load_or_extract_features(
                    model,
                    image_path,
                    relative_path,
                    cache_root / object_name,
                    mssm_layers,
                )
            )

        normalised_features = [_normalise_features(item) for item in features_per_image]
        all_features = np.concatenate(normalised_features, axis=0)
        offsets = np.cumsum([0, *[len(item) for item in normalised_features]])

        object_entries = []
        print(f"Mutual scoring {object_name}")
        for sample_index, (relative_path, _) in enumerate(
            tqdm(samples, desc=f"MSSM: {object_name}")
        ):
            start = int(offsets[sample_index])
            end = int(offsets[sample_index + 1])
            patch_scores = mutual_patch_scores(
                all_features[start:end],
                all_features,
                device=args.device,
                quantile=args.quantile,
                query_chunk_size=args.query_chunk_size,
                reference_chunk_size=args.reference_chunk_size,
                excluded_reference_range=(start, end),
                features_are_normalised=True,
            )
            image_score = aggregate_image_score(
                patch_scores,
                tail_fraction=args.image_tail_fraction,
            )
            object_entries.append({"path": relative_path, "score": image_score})

        object_entries.sort(key=lambda entry: (entry["score"], entry["path"]))
        results[object_name] = object_entries
        completed_objects = set(metadata.get("completed_objects", []))
        completed_objects.add(object_name)
        metadata["completed_objects"] = sorted(completed_objects)
        _atomic_write_json(output_path, results)
        _atomic_write_json(metadata_path, metadata)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    validate_mpdd_rankings(results, objects, data_root)
    print(f"Wrote MSSM rankings to {output_path}")
    print(f"Wrote run metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
