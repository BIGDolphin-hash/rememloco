"""Lightweight dataset metadata shared by runtime and validation tools."""

from __future__ import annotations

import math
import json
import hashlib
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}


MPDD_OBJECT_ANOMALIES = {
    "bracket_black": ["hole", "scratches"],
    "bracket_brown": ["bend_and_parts_mismatch", "parts_mismatch"],
    "bracket_white": ["defective_painting", "scratches"],
    "connector": ["parts_mismatch"],
    "metal_plate": ["major_rust", "scratches", "total_rust"],
    "tubes": ["flattening"],
}


# This research protocol includes normal controls and logical anomalies only.
MVTec_LOCO_SCOPE = "logical_only"
MVTec_LOCO_OBJECT_ANOMALIES = {
    "breakfast_box": ["logical_anomalies"],
    "juice_bottle": ["logical_anomalies"],
    "pushpins": ["logical_anomalies"],
    "screw_bag": ["logical_anomalies"],
    "splicing_connectors": ["logical_anomalies"],
}

MVTec_LOCO_TEST_TYPES = (
    "good",
    "logical_anomalies",
)
MVTec_LOCO_RANKING_ALGORITHMS = (
    "mssm_gc_global_density_spatial_count_consensus_v2",
    "mssm_b_bidirectional_position_constrained_knn_v1",
    "mssm_gb_per_object_rank_fusion_90_10_v1",
)
# Native archives may retain these directories; experiments never consume them.
MVTec_LOCO_IGNORED_TYPES = ("structural_anomalies",)

# MPDD archives in circulation use two directory names for the same tubes
# defect. Keep ``flattening`` as the semantic name, but consume either layout
# without modifying the downloaded dataset.
MPDD_DEFECT_DIRECTORY_ALIASES = {
    "tubes": {
        "flattening": ("flattening", "anomalous"),
    },
}


def is_image_file(path: str | Path) -> bool:
    """Return whether ``path`` has a supported image extension."""

    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def sorted_image_paths(directory: str | Path) -> list[Path]:
    """Return image files in ``directory`` in deterministic name order."""

    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        (item for item in directory.iterdir() if item.is_file() and is_image_file(item)),
        key=lambda item: item.name,
    )


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading a file at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sort_scored_paths_label_agnostic(
    entries: list[dict[str, Any]],
    test_root: str | Path,
) -> list[dict[str, Any]]:
    """Sort by score and resolve exact ties from image content, not labels.

    File hashing is lazy: image bytes are read only for exact score ties. If
    both content and basename are identical, either sample is an equivalent
    probe and stable order has no methodological effect.
    """

    ordered = sorted(entries, key=lambda entry: entry["score"])
    test_root = Path(test_root)
    group_start = 0
    while group_start < len(ordered):
        group_end = group_start + 1
        while (
            group_end < len(ordered)
            and ordered[group_end]["score"] == ordered[group_start]["score"]
        ):
            group_end += 1
        if group_end - group_start > 1:
            tie_group = ordered[group_start:group_end]

            def tie_key(entry: dict[str, Any]) -> tuple[str, str]:
                relative_path = Path(entry["path"])
                image_path = test_root / relative_path
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Tie-break image does not exist: {image_path}"
                    )
                return sha256_file(image_path), relative_path.name

            ordered[group_start:group_end] = sorted(
                tie_group, key=tie_key
            )
        group_start = group_end
    return ordered


def resolve_mpdd_anomaly_types(
    data_root: str | Path,
    object_name: str,
) -> list[str]:
    """Resolve MPDD defect directory aliases against an extracted dataset."""

    test_root = Path(data_root) / object_name / "test"
    resolved = []
    object_aliases = MPDD_DEFECT_DIRECTORY_ALIASES.get(object_name, {})
    for anomaly_type in MPDD_OBJECT_ANOMALIES[object_name]:
        candidates = object_aliases.get(anomaly_type, (anomaly_type,))
        existing = [
            candidate
            for candidate in candidates
            if (test_root / candidate).is_dir()
        ]
        resolved.extend(existing or [anomaly_type])
    return resolved


def get_mpdd_object_anomalies(
    data_root: str | Path,
) -> dict[str, list[str]]:
    """Return the defect directory names present in one MPDD checkout."""

    return {
        object_name: resolve_mpdd_anomaly_types(data_root, object_name)
        for object_name in MPDD_OBJECT_ANOMALIES
    }


def validate_mpdd_rankings(
    rankings: dict[str, Any],
    objects: list[str],
    data_root: str | Path,
) -> None:
    """Validate that MSSM rankings exactly cover selected MPDD test sets."""

    root = Path(data_root)
    for object_name in objects:
        entries = rankings.get(object_name)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Missing or empty MPDD ranking for {object_name}")

        expected_paths = []
        test_root = root / object_name / "test"
        anomaly_types = resolve_mpdd_anomaly_types(root, object_name)
        for test_type in ["good", *anomaly_types]:
            expected_paths.extend(
                path.relative_to(test_root).as_posix()
                for path in sorted_image_paths(test_root / test_type)
            )

        found_paths = []
        scores = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"MPDD ranking entry {object_name}[{index}] is not an object"
                )
            relative_path = entry.get("path")
            score = entry.get("score")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(
                    f"MPDD ranking entry {object_name}[{index}] has an invalid path"
                )
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(
                    f"MPDD ranking entry {object_name}[{index}] has an invalid score"
                )
            found_paths.append(relative_path)
            scores.append(float(score))

        if len(found_paths) != len(set(found_paths)):
            raise ValueError(f"MPDD ranking contains duplicate paths for {object_name}")
        if set(found_paths) != set(expected_paths):
            missing = sorted(set(expected_paths) - set(found_paths))
            extra = sorted(set(found_paths) - set(expected_paths))
            raise ValueError(
                f"MPDD ranking paths do not match {object_name}/test: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if scores != sorted(scores):
            raise ValueError(
                f"MPDD ranking for {object_name} must be sorted by ascending score"
            )


def validate_mpdd_ranking_metadata(
    ranking_path: str | Path,
    *,
    data_root: str | Path,
    model_name: str,
    resolution: int,
) -> dict[str, Any]:
    """Ensure a ranking was generated for the current model inputs."""

    ranking_path = Path(ranking_path).expanduser().resolve()
    metadata_path = ranking_path.with_name(ranking_path.stem + ".meta.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"MPDD ranking metadata was not found: {metadata_path}. Regenerate "
            "the ranking with generate_mssm_scores.py."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"Invalid MPDD ranking metadata: {metadata_path}")

    expected = {
        "dataset": "MPDD",
        "data_root": str(Path(data_root).expanduser().resolve()),
        "model_name": model_name,
        "resolution": resolution,
    }
    mismatches = {
        key: {"expected": value, "found": settings.get(key)}
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"MPDD ranking metadata does not match this run: {mismatches}"
        )
    return metadata


def validate_mvtec_loco_rankings(
    rankings: dict[str, Any],
    objects: list[str],
    data_root: str | Path,
) -> None:
    """Validate that MSSM rankings exactly cover selected LOCO test sets."""

    root = Path(data_root)
    for object_name in objects:
        entries = rankings.get(object_name)
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"Missing or empty MVTecLOCO ranking for {object_name}"
            )

        test_root = root / object_name / "test"
        expected_paths = [
            path.relative_to(test_root).as_posix()
            for test_type in MVTec_LOCO_TEST_TYPES
            for path in sorted_image_paths(test_root / test_type)
        ]

        found_paths = []
        scores = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"MVTecLOCO ranking entry {object_name}[{index}] "
                    "is not an object"
                )
            relative_path = entry.get("path")
            score = entry.get("score")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(
                    f"MVTecLOCO ranking entry {object_name}[{index}] "
                    "has an invalid path"
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(
                    f"MVTecLOCO ranking entry {object_name}[{index}] "
                    "has an invalid score"
                )
            found_paths.append(relative_path)
            scores.append(float(score))

        if len(found_paths) != len(set(found_paths)):
            raise ValueError(
                f"MVTecLOCO ranking contains duplicate paths for {object_name}"
            )
        if set(found_paths) != set(expected_paths):
            missing = sorted(set(expected_paths) - set(found_paths))
            extra = sorted(set(found_paths) - set(expected_paths))
            raise ValueError(
                f"MVTecLOCO ranking paths do not match {object_name}/test: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if scores != sorted(scores):
            raise ValueError(
                f"MVTecLOCO ranking for {object_name} must be sorted by "
                "ascending score"
            )
        label_agnostic_paths = [
            entry["path"]
            for entry in sort_scored_paths_label_agnostic(entries, test_root)
        ]
        if found_paths != label_agnostic_paths:
            raise ValueError(
                f"MVTecLOCO ranking exact-score ties for {object_name} must "
                "be ordered by image content"
            )


def validate_mvtec_loco_ranking_metadata(
    ranking_path: str | Path,
    *,
    data_root: str | Path,
    model_name: str,
    resolution: int,
) -> dict[str, Any]:
    """Ensure a LOCO ranking was generated for the current model inputs."""

    ranking_path = Path(ranking_path).expanduser().resolve()
    metadata_path = ranking_path.with_name(ranking_path.stem + ".meta.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"MVTecLOCO ranking metadata was not found: {metadata_path}. "
            "Regenerate it with the matching LOCO ranking generator."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"Invalid MVTecLOCO ranking metadata: {metadata_path}")

    expected = {
        "dataset": "MVTecLOCO",
        "experiment_scope": MVTec_LOCO_SCOPE,
        "test_types": list(MVTec_LOCO_TEST_TYPES),
        "data_root": str(Path(data_root).expanduser().resolve()),
        "model_name": model_name,
        "resolution": resolution,
    }
    mismatches = {
        key: {"expected": value, "found": settings.get(key)}
        for key, value in expected.items()
        if settings.get(key) != value
    }
    ranking_algorithm = settings.get("ranking_algorithm")
    if ranking_algorithm not in MVTec_LOCO_RANKING_ALGORITHMS:
        mismatches["ranking_algorithm"] = {
            "expected_one_of": list(MVTec_LOCO_RANKING_ALGORITHMS),
            "found": ranking_algorithm,
        }
    if mismatches:
        raise ValueError(
            f"MVTecLOCO ranking metadata does not match this run: {mismatches}. "
            "Regenerate it with the matching LOCO ranking generator; filtering "
            "old rankings cannot remove structural samples' influence on scores."
        )
    return metadata
