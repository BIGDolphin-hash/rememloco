"""Fuse only MSSM-G and MSSM-B rankings for native MVTec LOCO AD.

The two raw scores are converted to empirical percentile ranks independently
inside each object category, then fused with the fixed 9:1 ratio requested for
the lightweight G+B variant.  MSSM-C scores are neither read nor emitted.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from config import Config
from generate_loco_mssm_scores import _average_percentile_ranks
from generate_mssm_scores import _atomic_write_json, _load_resume_state
from src.dataset_info import (
    MVTec_LOCO_OBJECT_ANOMALIES,
    MVTec_LOCO_SCOPE,
    MVTec_LOCO_TEST_TYPES,
    sha256_file,
    sort_scored_paths_label_agnostic,
    validate_mvtec_loco_ranking_metadata,
    validate_mvtec_loco_rankings,
)


MSSM_G_SOURCE_ALGORITHM = "mssm_gc_global_density_spatial_count_consensus_v2"
MSSM_B_ALGORITHM = "mssm_b_bidirectional_position_constrained_knn_v1"
MSSM_GB_ALGORITHM = "mssm_gb_per_object_rank_fusion_90_10_v1"

GLOBAL_WEIGHT = 0.90
BIDIRECTIONAL_WEIGHT = 0.10

DEFAULT_GLOBAL_JSON = (
    "json/MVTecLOCO/dinov2_vits14/batch-0-shot/logical_only/"
    "mssm_gc/global_layer=12/results.json"
)
DEFAULT_B_JSON = (
    "json/MVTecLOCO/dinov2_vits14/batch-0-shot/logical_only/"
    "mssm_b/layer=12/results.json"
)
DEFAULT_OUTPUT_JSON = (
    "json/MVTecLOCO/dinov2_vits14/batch-0-shot/logical_only/"
    "mssm_gb/weights=9-1/results.json"
)


def _load_json_object(path: Path) -> dict[str, Any]:
    import json

    if not path.is_file():
        raise FileNotFoundError(f"MSSM ranking file was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"MSSM ranking must be a JSON object: {path}")
    return payload


def _index_entries(
    entries: Any,
    *,
    source_name: str,
    required_scores: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    if not isinstance(entries, list):
        raise ValueError(f"{source_name} entries must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"{source_name} contains an invalid entry")
        path = entry["path"]
        if path in indexed:
            raise ValueError(f"{source_name} contains duplicate path: {path}")
        for field in required_scores:
            value = entry.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"{source_name} has invalid {field} for {path}: {value}"
                )
        indexed[path] = entry
    return indexed


def fuse_object_rankings(
    global_entries: list[dict[str, Any]],
    b_entries: list[dict[str, Any]],
    *,
    test_root: Path,
) -> list[dict[str, Any]]:
    """Return one category's fixed 9G:1B percentile-rank fusion."""

    global_by_path = _index_entries(
        global_entries,
        source_name="MSSM-G",
        required_scores=("global_score",),
    )
    b_by_path = _index_entries(
        b_entries,
        source_name="MSSM-B",
        required_scores=(
            "score",
            "query_to_template_score",
            "template_to_query_score",
            "directional_gap",
        ),
    )
    if set(global_by_path) != set(b_by_path):
        missing_from_b = sorted(set(global_by_path) - set(b_by_path))
        missing_from_global = sorted(set(b_by_path) - set(global_by_path))
        raise ValueError(
            "MSSM-G and MSSM-B paths do not match: "
            f"missing_from_b={missing_from_b[:5]}, "
            f"missing_from_global={missing_from_global[:5]}"
        )

    paths = sorted(global_by_path)
    global_scores = np.asarray(
        [global_by_path[path]["global_score"] for path in paths],
        dtype=np.float64,
    )
    bidirectional_scores = np.asarray(
        [b_by_path[path]["score"] for path in paths],
        dtype=np.float64,
    )
    global_ranks = _average_percentile_ranks(global_scores)
    bidirectional_ranks = _average_percentile_ranks(bidirectional_scores)
    fused_scores = (
        GLOBAL_WEIGHT * global_ranks
        + BIDIRECTIONAL_WEIGHT * bidirectional_ranks
    )

    fused_entries = []
    for index, path in enumerate(paths):
        b_entry = b_by_path[path]
        fused_entries.append(
            {
                "path": path,
                "score": float(fused_scores[index]),
                "global_score": float(global_scores[index]),
                "bidirectional_score": float(bidirectional_scores[index]),
                "global_rank": float(global_ranks[index]),
                "bidirectional_rank": float(bidirectional_ranks[index]),
                "query_to_template_score": float(
                    b_entry["query_to_template_score"]
                ),
                "template_to_query_score": float(
                    b_entry["template_to_query_score"]
                ),
                "directional_gap": float(b_entry["directional_gap"]),
            }
        )
    return sort_scored_paths_label_agnostic(fused_entries, test_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse MVTec LOCO MSSM-G and MSSM-B per-object ranks with fixed "
            "weights 9:1; MSSM-C is not used"
        )
    )
    parser.add_argument(
        "--dataset", default="MVTecLOCO", choices=["MVTecLOCO"]
    )
    parser.add_argument("--data_root", default=Config.ROOTS["MVTecLOCO"])
    parser.add_argument("--model_name", default="dinov2_vits14")
    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--global_json", default=DEFAULT_GLOBAL_JSON)
    parser.add_argument("--b_json", default=DEFAULT_B_JSON)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _source_settings(
    path: Path,
    metadata: dict[str, Any],
    *,
    score_field: str,
) -> dict[str, Any]:
    metadata_path = path.with_name(path.stem + ".meta.json")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "ranking_algorithm": metadata["settings"]["ranking_algorithm"],
        "ranking_tag": metadata["settings"].get("ranking_tag"),
        "score_field": score_field,
    }


def main() -> int:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    data_root = Path(args.data_root).expanduser().resolve()
    objects = args.objects or list(MVTec_LOCO_OBJECT_ANOMALIES)
    unknown_objects = sorted(set(objects) - set(MVTec_LOCO_OBJECT_ANOMALIES))
    if unknown_objects:
        raise ValueError(f"Unknown MVTec LOCO objects: {unknown_objects}")
    if len(objects) != len(set(objects)):
        raise ValueError("--objects contains duplicate names")

    global_path = Path(args.global_json).expanduser().resolve()
    b_path = Path(args.b_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path in (global_path, b_path):
        raise ValueError("Fused output must not overwrite an input ranking")

    global_metadata = validate_mvtec_loco_ranking_metadata(
        global_path,
        data_root=data_root,
        model_name=args.model_name,
        resolution=args.resolution,
    )
    b_metadata = validate_mvtec_loco_ranking_metadata(
        b_path,
        data_root=data_root,
        model_name=args.model_name,
        resolution=args.resolution,
    )
    if (
        global_metadata["settings"].get("ranking_algorithm")
        != MSSM_G_SOURCE_ALGORITHM
    ):
        raise ValueError("--global_json does not contain the MSSM-G source")
    if b_metadata["settings"].get("ranking_algorithm") != MSSM_B_ALGORITHM:
        raise ValueError("--b_json is not an MSSM-B ranking")

    global_results = _load_json_object(global_path)
    b_results = _load_json_object(b_path)
    validate_mvtec_loco_rankings(global_results, objects, data_root)
    validate_mvtec_loco_rankings(b_results, objects, data_root)

    settings = {
        "dataset": "MVTecLOCO",
        "experiment_scope": MVTec_LOCO_SCOPE,
        "test_types": list(MVTec_LOCO_TEST_TYPES),
        "ranking_algorithm": MSSM_GB_ALGORITHM,
        "ranking_mode": "gb_fixed_rank_fusion",
        "ranking_tag": "gb-rankfusion-g90-b10",
        "data_root": str(data_root),
        "model_name": args.model_name,
        "resolution": args.resolution,
        "rank_scope": "per_object_empirical_percentile_average_ties_v1",
        "score_weights": {
            "global": GLOBAL_WEIGHT,
            "bidirectional": BIDIRECTIONAL_WEIGHT,
        },
        "source_rankings": {
            "global": _source_settings(
                global_path,
                global_metadata,
                score_field="global_score",
            ),
            "bidirectional": _source_settings(
                b_path,
                b_metadata,
                score_field="score",
            ),
        },
        "uses_original_patch_mssm": False,
        "uses_mssm_c": False,
        "uses_train_good": False,
        "uses_ground_truth_labels_for_score_computation": False,
        "weight_selection_origin": "user_fixed_9_to_1",
    }
    metadata_path = output_path.with_name(output_path.stem + ".meta.json")
    results, metadata = _load_resume_state(
        output_path,
        metadata_path,
        settings,
        args.force,
    )

    for object_name in objects:
        if object_name in results:
            validate_mvtec_loco_rankings(
                {object_name: results[object_name]},
                [object_name],
                data_root,
            )
            print(f"[Resume] {object_name} already has a G+B ranking")
            continue
        results[object_name] = fuse_object_rankings(
            global_results[object_name],
            b_results[object_name],
            test_root=data_root / object_name / "test",
        )
        completed_objects = set(metadata.get("completed_objects", []))
        completed_objects.add(object_name)
        metadata["completed_objects"] = sorted(completed_objects)
        _atomic_write_json(output_path, results)
        _atomic_write_json(metadata_path, metadata)

    validate_mvtec_loco_rankings(results, objects, data_root)
    print(f"Wrote fixed 9G:1B rankings to {output_path}")
    print(f"Wrote fusion metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
