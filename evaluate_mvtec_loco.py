"""Logical-only evaluation using the official MVTec LOCO AD v2.0 metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import tifffile

from src.dataset_info import (
    MVTec_LOCO_OBJECT_ANOMALIES,
    MVTec_LOCO_SCOPE,
    MVTec_LOCO_TEST_TYPES,
    sorted_image_paths,
)


OFFICIAL_EVALUATOR_DIR = (
    Path(__file__).resolve().parent
    / "third_party"
    / "mvtec_loco_ad_evaluation"
)
TIFF_EXTENSIONS = {".tif", ".tiff"}
SPRO_FPR_LIMITS = ("0.01", "0.05", "0.1", "0.3", "1.0")
ANOMALY_MODES = (
    "logical_anomalies",
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(temporary_path, path)


def _visible_directories(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        item.name
        for item in path.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    }


def validate_loco_anomaly_maps(
    dataset_base_dir: str | Path,
    anomaly_maps_dir: str | Path,
    objects: list[str],
) -> dict[str, Any]:
    """Validate exact test-image/TIFF coverage before official evaluation.

    The official evaluator checks anomalous ground truth coverage but can
    silently accept missing normal maps. This preflight closes that gap and
    also rejects malformed maps before the relatively expensive sPRO pass.
    """

    dataset_root = Path(dataset_base_dir).expanduser().resolve()
    maps_root = Path(anomaly_maps_dir).expanduser().resolve()
    errors: list[str] = []
    object_reports: dict[str, Any] = {}
    expected_types = set(MVTec_LOCO_TEST_TYPES)

    for object_name in objects:
        dataset_test_root = dataset_root / object_name / "test"
        prediction_test_root = maps_root / object_name / "test"
        if not dataset_test_root.is_dir():
            errors.append(
                f"{object_name}: missing dataset test directory "
                f"{dataset_test_root}"
            )
        present_types = _visible_directories(prediction_test_root)
        if not prediction_test_root.is_dir():
            errors.append(
                f"{object_name}: missing anomaly-map directory "
                f"{prediction_test_root}"
            )
        else:
            missing_types = sorted(expected_types - present_types)
            extra_types = sorted(present_types - expected_types)
            if missing_types:
                errors.append(
                    f"{object_name}: missing anomaly-map test directories "
                    f"{missing_types}"
                )
            if extra_types:
                errors.append(
                    f"{object_name}: unexpected anomaly-map test "
                    f"directories {extra_types}"
                )

        type_counts: dict[str, int] = {}
        for test_type in MVTec_LOCO_TEST_TYPES:
            image_paths = sorted_image_paths(
                dataset_test_root / test_type
            )
            if not image_paths:
                errors.append(
                    f"{object_name}/{test_type}: dataset contains no test "
                    "images"
                )
            expected_by_stem: dict[str, Path] = {}
            for image_path in image_paths:
                if image_path.stem in expected_by_stem:
                    errors.append(
                        f"{object_name}/{test_type}: source images have "
                        f"duplicate stem {image_path.stem!r}"
                    )
                expected_by_stem[image_path.stem] = image_path

            prediction_dir = prediction_test_root / test_type
            prediction_by_stem: dict[str, Path] = {}
            if prediction_dir.is_dir():
                nested_directories = sorted(
                    item.name
                    for item in prediction_dir.iterdir()
                    if item.is_dir() and not item.name.startswith(".")
                )
                if nested_directories:
                    errors.append(
                        f"{object_name}/{test_type}: unexpected nested "
                        f"directories {nested_directories}"
                    )
                for prediction_path in sorted(prediction_dir.iterdir()):
                    if (
                        not prediction_path.is_file()
                        or prediction_path.suffix.lower()
                        not in TIFF_EXTENSIONS
                    ):
                        continue
                    if prediction_path.stem in prediction_by_stem:
                        errors.append(
                            f"{object_name}/{test_type}: multiple TIFFs use "
                            f"stem {prediction_path.stem!r}"
                        )
                    prediction_by_stem[prediction_path.stem] = prediction_path

            expected_stems = set(expected_by_stem)
            prediction_stems = set(prediction_by_stem)
            missing = sorted(expected_stems - prediction_stems)
            extra = sorted(prediction_stems - expected_stems)
            if missing:
                errors.append(
                    f"{object_name}/{test_type}: missing TIFF predictions "
                    f"{missing[:10]}"
                )
            if extra:
                errors.append(
                    f"{object_name}/{test_type}: unexpected TIFF predictions "
                    f"{extra[:10]}"
                )

            for stem in sorted(expected_stems & prediction_stems):
                image_path = expected_by_stem[stem]
                prediction_path = prediction_by_stem[stem]
                try:
                    with Image.open(image_path) as image:
                        expected_shape = (image.height, image.width)
                except Exception as error:
                    errors.append(
                        f"{object_name}/{test_type}/{image_path.name}: "
                        f"cannot read source image ({error})"
                    )
                    continue
                try:
                    anomaly_map = tifffile.imread(prediction_path)
                except Exception as error:
                    errors.append(
                        f"{object_name}/{test_type}/{prediction_path.name}: "
                        f"cannot read TIFF ({error})"
                    )
                    continue
                if anomaly_map.ndim != 2:
                    errors.append(
                        f"{object_name}/{test_type}/{prediction_path.name}: "
                        f"TIFF must be 2-D, got shape {anomaly_map.shape}"
                    )
                    continue
                if not np.issubdtype(anomaly_map.dtype, np.floating):
                    errors.append(
                        f"{object_name}/{test_type}/{prediction_path.name}: "
                        f"TIFF must use floating point scores, got "
                        f"{anomaly_map.dtype}"
                    )
                if anomaly_map.shape != expected_shape:
                    errors.append(
                        f"{object_name}/{test_type}/{prediction_path.name}: "
                        f"TIFF shape {anomaly_map.shape} does not match source "
                        f"shape {expected_shape}"
                    )
                if not np.isfinite(anomaly_map).all():
                    errors.append(
                        f"{object_name}/{test_type}/{prediction_path.name}: "
                        "TIFF contains NaN or infinity"
                    )
            type_counts[test_type] = len(image_paths)

        object_reports[object_name] = {"test_images": type_counts}

    return {
        "valid": not errors,
        "experiment_scope": MVTec_LOCO_SCOPE,
        "test_types": list(MVTec_LOCO_TEST_TYPES),
        "dataset_root": str(dataset_root),
        "anomaly_maps_root": str(maps_root),
        "objects": object_reports,
        "errors": errors,
    }


def _validate_official_metrics(
    metrics: Any,
    object_name: str,
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise RuntimeError(
            f"Official metrics for {object_name} are not a JSON object"
        )
    try:
        classification = metrics["classification"]["auc_roc"]
        localization = metrics["localization"]["auc_spro"]
        values = [classification[mode] for mode in ANOMALY_MODES]
        values.extend(
            localization[mode][limit]
            for mode in ANOMALY_MODES
            for limit in SPRO_FPR_LIMITS
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"Official metrics for {object_name} have an unexpected schema"
        ) from error
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise RuntimeError(
            f"Official metrics for {object_name} contain invalid numbers"
        )
    return metrics


def _aggregate_metrics(
    metrics_by_object: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    classification = {
        mode: float(
            np.mean(
                [
                    metrics["classification"]["auc_roc"][mode]
                    for metrics in metrics_by_object.values()
                ]
            )
        )
        for mode in ANOMALY_MODES
    }
    localization = {
        mode: {
            limit: float(
                np.mean(
                    [
                        metrics["localization"]["auc_spro"][mode][limit]
                        for metrics in metrics_by_object.values()
                    ]
                )
            )
            for limit in SPRO_FPR_LIMITS
        }
        for mode in ANOMALY_MODES
    }
    return {
        "classification": {"auc_roc": classification},
        "localization": {"auc_spro": localization},
    }


def _load_raw_classification_scores(
    path: str | Path,
    objects: list[str],
) -> dict[str, dict[str, list[float]]]:
    score_path = Path(path).expanduser().resolve()
    if not score_path.is_file():
        raise FileNotFoundError(
            f"Missing raw image classification scores: {score_path}"
        )
    with score_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    scores_by_object: dict[str, dict[str, list[float]]] = {}
    for object_name in objects:
        try:
            entries = payload[object_name]["all"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"Raw classification scores are missing {object_name}"
            ) from error
        if not isinstance(entries, list):
            raise RuntimeError(
                f"Raw classification scores for {object_name} must be a list"
            )

        scores = {test_type: [] for test_type in MVTec_LOCO_TEST_TYPES}
        seen_paths: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Raw classification entry for {object_name} is invalid"
                )
            relative_path = entry.get("path")
            score = entry.get("score")
            if not isinstance(relative_path, str) or not relative_path:
                raise RuntimeError(
                    f"Raw classification path for {object_name} is invalid"
                )
            parts = Path(relative_path).parts
            if len(parts) != 2 or parts[0] not in scores:
                raise RuntimeError(
                    f"Raw classification path is out of scope: {relative_path}"
                )
            if relative_path in seen_paths:
                raise RuntimeError(
                    f"Duplicate raw classification path: {relative_path}"
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise RuntimeError(
                    f"Raw classification score is invalid: {relative_path}"
                )
            seen_paths.add(relative_path)
            scores[parts[0]].append(float(score))

        if any(not values for values in scores.values()):
            raise RuntimeError(
                f"Raw classification scores for {object_name} are incomplete"
            )
        scores_by_object[object_name] = scores
    return scores_by_object


def _classification_auc_from_raw_scores(
    good_scores: list[float],
    anomaly_scores: list[float],
) -> float:
    good = np.asarray(good_scores, dtype=np.float64)
    anomaly = np.asarray(anomaly_scores, dtype=np.float64)
    differences = anomaly[:, None] - good[None, :]
    return float(
        np.mean((differences > 0).astype(np.float64) + 0.5 * (differences == 0))
    )


def evaluate_mvtec_loco_image_level(
    *,
    output_dir: str | Path,
    classification_scores_path: str | Path,
    objects: list[str] | None = None,
) -> Path:
    """Evaluate only image-level good-vs-logical AUROC from raw scores."""

    objects = (
        list(MVTec_LOCO_OBJECT_ANOMALIES)
        if objects is None
        else list(objects)
    )
    unknown_objects = sorted(
        set(objects) - set(MVTec_LOCO_OBJECT_ANOMALIES)
    )
    if unknown_objects:
        raise ValueError(f"Unknown MVTecLOCO objects: {unknown_objects}")
    if not objects:
        raise ValueError("At least one MVTecLOCO object must be selected")
    if len(objects) != len(set(objects)):
        raise ValueError("Objects contain duplicate names")

    scores_by_object = _load_raw_classification_scores(
        classification_scores_path, objects
    )
    metrics_root = Path(output_dir).expanduser().resolve()
    metrics_by_object: dict[str, dict[str, Any]] = {}
    for object_name in objects:
        object_scores = scores_by_object[object_name]
        auc_roc = _classification_auc_from_raw_scores(
            object_scores["good"],
            object_scores["logical_anomalies"],
        )
        metrics = {
            "classification": {
                "auc_roc": {"logical_anomalies": auc_roc},
            },
            "sample_counts": {
                test_type: len(object_scores[test_type])
                for test_type in MVTec_LOCO_TEST_TYPES
            },
        }
        _atomic_write_json(metrics_root / object_name / "metrics.json", metrics)
        metrics_by_object[object_name] = metrics

    macro_auc = float(
        np.mean(
            [
                metrics["classification"]["auc_roc"]["logical_anomalies"]
                for metrics in metrics_by_object.values()
            ]
        )
    )
    summary = {
        "protocol": {
            "name": "MVTec LOCO AD logical-only image-level evaluation",
            "version": "image-only-v1",
            "experiment_scope": MVTec_LOCO_SCOPE,
            "test_types": list(MVTec_LOCO_TEST_TYPES),
            "objects": objects,
            "classification_metric": "image-level AUROC",
            "classification_score": (
                "TL-IterMSSM Top-1% mean of the raw patchwise max map (S_raw)"
            ),
            "pixel_level_metrics_computed": False,
            "anomaly_maps_generated": False,
        },
        "objects": metrics_by_object,
        "macro_average": {
            "classification": {
                "auc_roc": {"logical_anomalies": macro_auc},
            }
        },
    }
    summary_path = metrics_root / "metrics_summary.json"
    _atomic_write_json(summary_path, summary)
    return summary_path


def evaluate_mvtec_loco(
    *,
    dataset_base_dir: str | Path,
    anomaly_maps_dir: str | Path,
    output_dir: str | Path,
    objects: list[str] | None = None,
    classification_scores_path: str | Path | None = None,
    official_eval_dir: str | Path = OFFICIAL_EVALUATOR_DIR,
    num_parallel_workers: int | None = None,
    seed: int = 0,
) -> Path:
    """Run upstream metric primitives on good/logical images per object."""

    objects = (
        list(MVTec_LOCO_OBJECT_ANOMALIES)
        if objects is None
        else list(objects)
    )
    unknown_objects = sorted(
        set(objects) - set(MVTec_LOCO_OBJECT_ANOMALIES)
    )
    if unknown_objects:
        raise ValueError(f"Unknown MVTecLOCO objects: {unknown_objects}")
    if not objects:
        raise ValueError("At least one MVTecLOCO object must be selected")
    if len(objects) != len(set(objects)):
        raise ValueError("Objects contain duplicate names")
    if num_parallel_workers is not None and num_parallel_workers <= 0:
        raise ValueError("num_parallel_workers must be positive or None")

    raw_scores = (
        _load_raw_classification_scores(classification_scores_path, objects)
        if classification_scores_path is not None
        else None
    )

    dataset_root = Path(dataset_base_dir).expanduser().resolve()
    maps_root = Path(anomaly_maps_dir).expanduser().resolve()
    metrics_root = Path(output_dir).expanduser().resolve()
    evaluator_root = Path(official_eval_dir).expanduser().resolve()
    evaluator_script = evaluator_root / "evaluate_experiment.py"
    compatibility_script = (
        Path(__file__).resolve().parent / "loco_official_eval_compat.py"
    )
    logical_driver = Path(__file__).resolve().parent / "loco_logical_eval.py"
    license_path = evaluator_root / "LICENSE.txt"
    required_evaluator_files = [
        evaluator_script,
        license_path,
        evaluator_root / "src" / "aggregation.py",
        evaluator_root / "src" / "__init__.py",
        evaluator_root / "src" / "image.py",
        evaluator_root / "src" / "metrics.py",
        evaluator_root / "src" / "util.py",
    ]
    missing_evaluator_files = [
        str(path) for path in required_evaluator_files if not path.is_file()
    ]
    if missing_evaluator_files:
        raise FileNotFoundError(
            "Official MVTec LOCO AD evaluator v2.0 is incomplete. Expected "
            f"all vendored files; missing={missing_evaluator_files}"
        )
    if not compatibility_script.is_file():
        raise FileNotFoundError(
            "Missing LOCO evaluator compatibility module: "
            f"{compatibility_script}"
        )
    if not logical_driver.is_file():
        raise FileNotFoundError(f"Missing LOCO logical-only driver: {logical_driver}")

    preflight = validate_loco_anomaly_maps(dataset_root, maps_root, objects)
    if not preflight["valid"]:
        details = "\n".join(f"  - {error}" for error in preflight["errors"])
        raise RuntimeError(
            "MVTec LOCO anomaly-map validation failed:\n" + details
        )
    metrics_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(metrics_root / "anomaly_maps_validation.json", preflight)

    # Seed the upstream sampler, isolate its generic ``src`` package, and load
    # the compatibility guards only in the child process. The guards remove
    # exact duplicate initial thresholds and restore mathematically exact
    # FPR-sPRO endpoints if upstream dynamic refinement loses either endpoint.
    # Keep the vendored official source files unchanged.
    seeded_launcher = (
        "import importlib,os,runpy,sys,types,numpy as np; "
        "np.random.seed(int(sys.argv[1])); "
        "script=sys.argv[2]; "
        "package=types.ModuleType('src'); "
        "package.__path__=[os.path.join(os.path.dirname(script),'src')]; "
        "sys.modules['src']=package; "
        "aggregation=importlib.import_module('src.aggregation'); "
        "compat=runpy.run_path(sys.argv[3]); "
        "compat['install'](aggregation); "
        "driver=runpy.run_path(sys.argv[4]); "
        "sys.argv=[script,*sys.argv[5:]]; "
        "driver['run'](runpy.run_path(script))"
    )
    metrics_by_object: dict[str, dict[str, Any]] = {}
    for object_name in objects:
        object_output_dir = metrics_root / object_name
        command = [
            sys.executable,
            "-c",
            seeded_launcher,
            str(seed),
            str(evaluator_script),
            str(compatibility_script),
            str(logical_driver),
            "--object_name",
            object_name,
            "--dataset_base_dir",
            str(dataset_root),
            "--anomaly_maps_dir",
            str(maps_root),
            "--output_dir",
            str(object_output_dir),
        ]
        if num_parallel_workers is not None:
            command.extend(
                ["--num_parallel_workers", str(num_parallel_workers)]
            )
        subprocess.run(command, cwd=evaluator_root, check=True)

        metrics_path = object_output_dir / "metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError(
                f"Official evaluator did not write {metrics_path}"
            )
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        if raw_scores is not None:
            object_scores = raw_scores[object_name]
            metrics["classification"]["auc_roc"]["logical_anomalies"] = (
                _classification_auc_from_raw_scores(
                    object_scores["good"],
                    object_scores["logical_anomalies"],
                )
            )
            _atomic_write_json(metrics_path, metrics)
        metrics_by_object[object_name] = _validate_official_metrics(
            metrics, object_name
        )

    summary = {
        "protocol": {
            "name": "MVTec LOCO AD logical-only evaluation (official metric primitives)",
            "version": "2.0",
            "experiment_scope": MVTec_LOCO_SCOPE,
            "test_types": list(MVTec_LOCO_TEST_TYPES),
            "full_benchmark": False,
            "seed": seed,
            "objects": objects,
            "primary_localization_metric": "AUC-sPRO@0.05 FPR",
            "classification_score": (
                "TL-IterMSSM Top-1% mean of the raw patchwise max map (S_raw)"
                if raw_scores is not None
                else "maximum TIFF pixel value"
            ),
            "initial_threshold_policy": (
                "official_v2.0_exact_duplicates_removed"
            ),
            "threshold_endpoint_policy": (
                "official_v2.0_infinite_endpoints_on_demand"
            ),
        },
        "objects": metrics_by_object,
        "macro_average": _aggregate_metrics(metrics_by_object),
    }
    summary_path = metrics_root / "metrics_summary.json"
    _atomic_write_json(summary_path, summary)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate good/logical LOCO maps with official v2.0 metrics"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--anomaly_maps_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument(
        "--official_eval_dir", default=str(OFFICIAL_EVALUATOR_DIR)
    )
    parser.add_argument("--num_parallel_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = evaluate_mvtec_loco(
        dataset_base_dir=args.data_root,
        anomaly_maps_dir=args.anomaly_maps_dir,
        output_dir=args.output_dir,
        objects=args.objects,
        official_eval_dir=args.official_eval_dir,
        num_parallel_workers=args.num_parallel_workers,
        seed=args.seed,
    )
    print(f"Wrote official MVTec LOCO metrics to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
