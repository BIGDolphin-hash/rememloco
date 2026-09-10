"""Generate an MSSM-B-only zero-shot ranking for native MVTec LOCO AD."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from config import Config
from generate_loco_mssm_scores import (
    _load_or_extract_loco_tokens,
    _prepare_loco_token_cache,
    adaptive_pool_patch_tokens,
    get_object_test_samples,
)
from generate_mssm_scores import (
    _atomic_write_json,
    _load_resume_state,
    _validate_mssm_layers_for_model,
)
from src.dataset_info import (
    MVTec_LOCO_OBJECT_ANOMALIES,
    MVTec_LOCO_SCOPE,
    MVTec_LOCO_TEST_TYPES,
    sort_scored_paths_label_agnostic,
    validate_mvtec_loco_rankings,
)
from src.loco_mssm_b import (
    bidirectional_position_knn_scores,
    build_consensus_template,
)


LOCO_MSSM_B_ALGORITHM = "mssm_b_bidirectional_position_constrained_knn_v1"
LOCO_MSSM_B_DEFAULT_LAYER = 12
LOCO_MSSM_B_DEFAULT_OUTPUT = (
    "json/MVTecLOCO/dinov2_vits14/batch-0-shot/logical_only/"
    "mssm_b/layer=12/results.json"
)


def _ranking_tag_from_args(args: argparse.Namespace) -> str:
    radius = f"{args.position_radius_cells:g}"
    weight = f"{args.position_weight:g}"
    tail = f"{args.tail_fraction:g}"
    return (
        f"b-l{args.layer}-k{args.neighbors}-grid{args.grid_size}-"
        f"r{radius}-p{weight}-tail{tail}"
    )


def _settings_from_args(
    args: argparse.Namespace,
    data_root: Path,
) -> dict[str, Any]:
    return {
        "dataset": "MVTecLOCO",
        "experiment_scope": MVTec_LOCO_SCOPE,
        "test_types": list(MVTec_LOCO_TEST_TYPES),
        "ranking_algorithm": LOCO_MSSM_B_ALGORITHM,
        "ranking_mode": "b_only",
        "ranking_tag": _ranking_tag_from_args(args),
        "data_root": str(data_root),
        "model_name": args.model_name,
        "resolution": args.resolution,
        "layer": args.layer,
        "layer_numbering": "one_based_transformer_blocks",
        "grid_size": args.grid_size,
        "neighbors": args.neighbors,
        "position_radius_cells": args.position_radius_cells,
        "position_weight": args.position_weight,
        "tail_fraction": args.tail_fraction,
        "template_builder": "coordinatewise_median_all_unlabelled_test_images_v1",
        "template_update": "frozen_before_scoring",
        "direction_fusion": "arithmetic_mean_query_to_template_and_reverse_v1",
        "uses_original_patch_mssm": False,
        "uses_mssm_g": False,
        "uses_mssm_c": False,
        "uses_train_good": False,
        "uses_ground_truth_labels_for_scoring": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate native MVTec LOCO AD MSSM-B-only rankings with "
            "bidirectional position-constrained KNN"
        )
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
        "--layer",
        type=int,
        default=LOCO_MSSM_B_DEFAULT_LAYER,
        help="One-based DINO block used only by MSSM-B",
    )
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--position_radius_cells", type=float, default=2.0)
    parser.add_argument("--position_weight", type=float, default=0.25)
    parser.add_argument("--tail_fraction", type=float, default=0.1)
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--output_json", default=LOCO_MSSM_B_DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    import torch

    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.layer <= 0:
        raise ValueError("--layer uses one-based positive layer numbers")
    if args.grid_size <= 0:
        raise ValueError("--grid_size must be positive")
    if args.neighbors <= 0:
        raise ValueError("--neighbors must be positive")
    if (
        not math.isfinite(args.position_radius_cells)
        or args.position_radius_cells <= 0
    ):
        raise ValueError("--position_radius_cells must be finite and positive")
    if not math.isfinite(args.position_weight) or args.position_weight < 0:
        raise ValueError("--position_weight must be finite and non-negative")
    if not 0 < args.tail_fraction <= 1:
        raise ValueError("--tail_fraction must be in (0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but CUDA is unavailable: {args.device}"
        )


def _is_complete_object_ranking(
    entries: Any,
    expected_paths: list[str],
) -> bool:
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        return False
    required_fields = {
        "path",
        "score",
        "query_to_template_score",
        "template_to_query_score",
        "directional_gap",
    }
    if not all(required_fields.issubset(entry) for entry in entries):
        return False
    if sorted(entry["path"] for entry in entries) != sorted(expected_paths):
        return False
    scores = [entry["score"] for entry in entries]
    return (
        all(
            not isinstance(score, bool)
            and isinstance(score, (int, float))
            and math.isfinite(score)
            for score in scores
        )
        and scores == sorted(scores)
    )


def main() -> int:
    import torch
    from tqdm import tqdm

    from src.backbones import get_model
    from validate_mvtec_loco import validate_mvtec_loco_layout

    args = parse_args()
    _validate_args(args)
    data_root = Path(args.data_root).expanduser().resolve()
    objects = args.objects or list(MVTec_LOCO_OBJECT_ANOMALIES)
    unknown_objects = sorted(set(objects) - set(MVTec_LOCO_OBJECT_ANOMALIES))
    if unknown_objects:
        raise ValueError(f"Unknown MVTec LOCO objects: {unknown_objects}")
    if len(objects) != len(set(objects)):
        raise ValueError("--objects contains duplicate names")

    validation_report = validate_mvtec_loco_layout(data_root, objects=objects)
    if not validation_report["valid"]:
        for error in validation_report["errors"]:
            print(f"[ERROR] {error}")
        raise RuntimeError(
            "MVTec LOCO AD validation failed; fix the dataset before MSSM-B "
            "scoring"
        )

    output_path = Path(args.output_json).expanduser().resolve()
    metadata_path = output_path.with_name(output_path.stem + ".meta.json")
    default_cache_root = (
        Path(Config.FEATURE_ROOT)
        / "MVTecLOCO"
        / args.model_name
        / MVTec_LOCO_SCOPE
        / f"loco_tokens_resolution={args.resolution}"
        / f"layer={args.layer}"
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
        layer=args.layer,
    )
    _atomic_write_json(output_path, results)
    _atomic_write_json(metadata_path, metadata)

    model = get_model(
        args.model_name,
        args.device,
        smaller_edge_size=args.resolution,
    )
    _validate_mssm_layers_for_model(model, (args.layer,))

    for object_name in objects:
        samples = get_object_test_samples(data_root, object_name)
        expected_paths = [relative_path for relative_path, _ in samples]
        existing_entries = results.get(object_name)
        if existing_entries is not None:
            if _is_complete_object_ranking(existing_entries, expected_paths):
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
                    f"[Resume] {object_name} already has a complete MSSM-B "
                    "ranking"
                )
                continue
            raise RuntimeError(
                f"Existing entries for {object_name} are incomplete or "
                "invalid. Use --force to rebuild the output."
            )

        print(
            f"Extracting MSSM-B features for {object_name} "
            f"({len(samples)} images)"
        )
        pooled_per_image = []
        for relative_path, image_path in tqdm(
            samples, desc=f"MSSM-B features: {object_name}"
        ):
            tokens, source_grid_size = _load_or_extract_loco_tokens(
                model,
                image_path,
                relative_path,
                cache_root / object_name,
                args.layer,
            )
            pooled_per_image.append(
                adaptive_pool_patch_tokens(
                    tokens,
                    source_grid_size,
                    args.grid_size,
                )
            )
        pooled_tokens = np.stack(pooled_per_image)
        consensus_template = build_consensus_template(pooled_tokens)
        scores, diagnostics = bidirectional_position_knn_scores(
            pooled_tokens,
            consensus_template,
            grid_size=args.grid_size,
            neighbors=args.neighbors,
            position_radius_cells=args.position_radius_cells,
            position_weight=args.position_weight,
            tail_fraction=args.tail_fraction,
        )

        object_entries = []
        for index, (relative_path, _) in enumerate(samples):
            query_score = float(diagnostics["query_to_template_score"][index])
            template_score = float(
                diagnostics["template_to_query_score"][index]
            )
            object_entries.append(
                {
                    "path": relative_path,
                    "score": float(scores[index]),
                    "query_to_template_score": query_score,
                    "template_to_query_score": template_score,
                    "directional_gap": float(query_score - template_score),
                }
            )
        object_entries = sort_scored_paths_label_agnostic(
            object_entries,
            data_root / object_name / "test",
        )
        results[object_name] = object_entries
        completed_objects = set(metadata.get("completed_objects", []))
        completed_objects.add(object_name)
        metadata["completed_objects"] = sorted(completed_objects)
        metadata.setdefault("mssm_b_diagnostics", {})[object_name] = {
            "image_count": len(samples),
            "template_shape": list(consensus_template.shape),
            "mean_query_to_template_score": float(
                diagnostics["query_to_template_score"].mean()
            ),
            "mean_template_to_query_score": float(
                diagnostics["template_to_query_score"].mean()
            ),
        }
        _atomic_write_json(output_path, results)
        _atomic_write_json(metadata_path, metadata)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    validate_mvtec_loco_rankings(results, objects, data_root)
    print(f"Wrote MSSM-B-only rankings to {output_path}")
    print(f"Wrote run metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
