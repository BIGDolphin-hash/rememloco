"""Strict, read-only validation for an MVTec LOCO AD checkout.

MVTec LOCO AD differs from MVTec AD/MPDD in two important ways: test
anomalies are split into logical and structural groups, and a ground-truth
map is a directory of PNG channels rather than a single ``*_mask.png`` file.
This validator checks the logical-only experiment subset before feature
extraction or GPU inference. Native structural directories may be present but
are not inspected and are not required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import Config
from src.dataset_info import (
    MVTec_LOCO_OBJECT_ANOMALIES,
    MVTec_LOCO_SCOPE,
    MVTec_LOCO_IGNORED_TYPES,
    MVTec_LOCO_TEST_TYPES as DATASET_LOCO_TEST_TYPES,
    is_image_file,
    sorted_image_paths,
)


MVTEC_LOCO_OBJECTS = tuple(MVTec_LOCO_OBJECT_ANOMALIES)
MVTEC_LOCO_ANOMALY_TYPES = tuple(
    MVTec_LOCO_OBJECT_ANOMALIES[MVTEC_LOCO_OBJECTS[0]]
)
MVTEC_LOCO_TEST_TYPES = tuple(DATASET_LOCO_TEST_TYPES)
GENERATED_ROOT_DIRECTORIES = {"results_MVTecLOCO"}

_DEFECT_CONFIG_KEYS = {
    "defect_name",
    "pixel_value",
    "saturation_threshold",
    "relative_saturation",
}


def _atomic_write_json(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)


def _check_readable_image(image_path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            return image.size
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
        return None


def _report_unexpected_subdirectories(
    directory: Path,
    allowed: Iterable[str],
    errors: list[str],
    *,
    context: str,
) -> None:
    if not directory.is_dir():
        return
    found = {item.name for item in directory.iterdir() if item.is_dir()}
    unexpected = sorted(found - set(allowed))
    if unexpected:
        errors.append(f"{context}: unexpected directories: {unexpected}")


def _report_leaf_directory_contents(
    directory: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Report nested directories and ignored files in an image leaf."""

    if not directory.is_dir():
        return
    nested = sorted(item.name for item in directory.iterdir() if item.is_dir())
    if nested:
        errors.append(f"{directory}: unexpected nested directories: {nested}")
    ignored = sorted(
        item.name
        for item in directory.iterdir()
        if item.is_file() and not is_image_file(item)
    )
    if ignored:
        warnings.append(f"{directory}: ignored non-image files: {ignored[:5]}")


def _validate_defects_config(
    config_path: Path,
    errors: list[str],
) -> dict[int, dict[str, Any]]:
    if not config_path.is_file():
        errors.append(f"Missing defects configuration: {config_path}")
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        errors.append(f"Unreadable defects configuration {config_path}: {error}")
        return {}

    if not isinstance(payload, list):
        errors.append(f"{config_path}: top-level value must be a list")
        return {}
    if not payload:
        errors.append(f"{config_path}: defects configuration must not be empty")
        return {}

    by_pixel_value: dict[int, dict[str, Any]] = {}
    for index, entry in enumerate(payload):
        prefix = f"{config_path}: entry {index}"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue

        keys = set(entry)
        missing = sorted(_DEFECT_CONFIG_KEYS - keys)
        unexpected = sorted(keys - _DEFECT_CONFIG_KEYS)
        if missing:
            errors.append(f"{prefix} missing fields: {missing}")
        if unexpected:
            errors.append(f"{prefix} has unexpected fields: {unexpected}")
        if missing:
            continue

        defect_name = entry["defect_name"]
        if not isinstance(defect_name, str) or not defect_name.strip():
            errors.append(f"{prefix}.defect_name must be a non-empty string")

        pixel_value = entry["pixel_value"]
        valid_pixel_value = (
            isinstance(pixel_value, int)
            and not isinstance(pixel_value, bool)
            and 1 <= pixel_value <= 255
        )
        if not valid_pixel_value:
            errors.append(f"{prefix}.pixel_value must be an integer in [1, 255]")
        elif pixel_value in by_pixel_value:
            errors.append(f"{prefix}.pixel_value is duplicated: {pixel_value}")

        relative = entry["relative_saturation"]
        if not isinstance(relative, bool):
            errors.append(f"{prefix}.relative_saturation must be a boolean")

        threshold = entry["saturation_threshold"]
        valid_threshold = False
        if isinstance(relative, bool):
            if relative:
                valid_threshold = (
                    isinstance(threshold, float)
                    and math.isfinite(threshold)
                    and 0.0 < threshold <= 1.0
                )
                if not valid_threshold:
                    errors.append(
                        f"{prefix}.saturation_threshold must be a float in "
                        "(0, 1] when relative_saturation is true"
                    )
            else:
                valid_threshold = isinstance(threshold, int) and not isinstance(
                    threshold, bool
                )
                if not valid_threshold:
                    errors.append(
                        f"{prefix}.saturation_threshold must be an integer "
                        "when relative_saturation is false"
                    )

        if valid_pixel_value and pixel_value not in by_pixel_value:
            by_pixel_value[pixel_value] = entry

    return by_pixel_value


def _validate_image_collection(
    directory: Path,
    errors: list[str],
    warnings: list[str],
    *,
    require_nonempty: bool = True,
) -> tuple[list[Path], dict[str, tuple[int, int]]]:
    if not directory.is_dir():
        errors.append(f"Missing directory: {directory}")
        return [], {}

    _report_leaf_directory_contents(directory, errors, warnings)
    image_paths = sorted_image_paths(directory)
    if require_nonempty and not image_paths:
        errors.append(f"{directory}: contains no supported images")

    stem_to_paths: dict[str, list[Path]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for image_path in image_paths:
        stem_to_paths.setdefault(image_path.stem, []).append(image_path)
        image_size = _check_readable_image(image_path)
        if image_size is None:
            errors.append(f"Unreadable image: {image_path}")
        else:
            sizes[image_path.stem] = image_size

    duplicates = {
        stem: [path.name for path in paths]
        for stem, paths in stem_to_paths.items()
        if len(paths) > 1
    }
    if duplicates:
        errors.append(
            f"{directory}: duplicate image stems with different files: {duplicates}"
        )
    return image_paths, sizes


def _validate_gt_channel(
    channel_path: Path,
    expected_size: tuple[int, int] | None,
    defect_configs: dict[int, dict[str, Any]],
    errors: list[str],
) -> None:
    try:
        with Image.open(channel_path) as image:
            image.load()
            image_size = image.size
            array = np.asarray(image)
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as error:
        errors.append(f"Unreadable ground-truth channel {channel_path}: {error}")
        return

    if array.ndim != 2:
        errors.append(f"{channel_path}: ground-truth channel must be two-dimensional")
        return
    if not np.issubdtype(array.dtype, np.integer):
        errors.append(f"{channel_path}: ground-truth channel must use integer pixels")
        return
    if array.size == 0:
        errors.append(f"{channel_path}: ground-truth channel is empty")
        return
    if expected_size is not None and image_size != expected_size:
        errors.append(
            f"Image/channel size mismatch: expected {expected_size}, "
            f"{channel_path}={image_size}"
        )

    unique_values = np.unique(array)
    if np.any(unique_values < 0):
        errors.append(f"{channel_path}: non-positive pixels must be zero")
        return
    positive_ids = unique_values[unique_values > 0]
    if len(positive_ids) == 0:
        errors.append(f"{channel_path}: ground-truth channel has no foreground")
        return
    if len(positive_ids) != 1:
        errors.append(
            f"{channel_path}: expected exactly one positive pixel id, "
            f"found {[int(value) for value in positive_ids]}"
        )
        return

    pixel_id = int(positive_ids[0])
    if pixel_id not in defect_configs:
        errors.append(
            f"{channel_path}: pixel id {pixel_id} is absent from defects_config.json"
        )


def _validate_ground_truth_type(
    gt_type_dir: Path,
    anomaly_images: list[Path],
    anomaly_sizes: dict[str, tuple[int, int]],
    defect_configs: dict[int, dict[str, Any]],
    errors: list[str],
) -> int:
    if not gt_type_dir.is_dir():
        errors.append(f"Missing directory: {gt_type_dir}")
        return 0

    expected_stems = {path.stem for path in anomaly_images}
    gt_items = list(gt_type_dir.iterdir())
    found_dirs = {item.name for item in gt_items if item.is_dir()}
    unexpected_files = sorted(item.name for item in gt_items if not item.is_dir())
    if unexpected_files:
        errors.append(
            f"{gt_type_dir}: ground-truth type directory may contain only "
            f"image directories, found files: {unexpected_files[:5]}"
        )

    missing = sorted(expected_stems - found_dirs)
    orphan = sorted(found_dirs - expected_stems)
    if missing:
        errors.append(f"{gt_type_dir}: missing ground-truth directories: {missing[:5]}")
    if orphan:
        errors.append(f"{gt_type_dir}: orphan ground-truth directories: {orphan[:5]}")

    channel_count = 0
    for stem in sorted(expected_stems & found_dirs):
        image_gt_dir = gt_type_dir / stem
        contents = list(image_gt_dir.iterdir())
        nested = sorted(item.name for item in contents if item.is_dir())
        if nested:
            errors.append(f"{image_gt_dir}: unexpected nested directories: {nested}")

        png_channels = sorted(
            item for item in contents if item.is_file() and item.suffix == ".png"
        )
        invalid_files = sorted(
            item.name
            for item in contents
            if item.is_file() and item.suffix != ".png"
        )
        if invalid_files:
            errors.append(
                f"{image_gt_dir}: only lower-case .png channel files are allowed, "
                f"found: {invalid_files[:5]}"
            )
        if not png_channels:
            errors.append(f"{image_gt_dir}: contains no PNG ground-truth channels")
            continue

        channel_count += len(png_channels)
        for channel_path in png_channels:
            _validate_gt_channel(
                channel_path,
                anomaly_sizes.get(stem),
                defect_configs,
                errors,
            )
    return channel_count


def validate_mvtec_loco_layout(
    data_root: str | Path,
    objects: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a strict validation report for all or selected LOCO objects."""

    selected_objects = tuple(MVTEC_LOCO_OBJECTS if objects is None else objects)
    if not selected_objects:
        raise ValueError("At least one MVTec LOCO AD object must be selected")
    if len(selected_objects) != len(set(selected_objects)):
        raise ValueError("MVTec LOCO AD object selection contains duplicates")
    unknown_objects = sorted(set(selected_objects) - set(MVTEC_LOCO_OBJECTS))
    if unknown_objects:
        raise ValueError(f"Unknown MVTec LOCO AD objects: {unknown_objects}")

    root = Path(data_root).expanduser().resolve()
    report: dict[str, Any] = {
        "data_root": str(root),
        "selected_objects": list(selected_objects),
        "experiment_scope": MVTec_LOCO_SCOPE,
        "test_types": list(MVTEC_LOCO_TEST_TYPES),
        "ignored_types": list(MVTec_LOCO_IGNORED_TYPES),
        "objects": {},
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]

    if not root.is_dir():
        errors.append(f"Dataset root does not exist or is not a directory: {root}")
        report["totals"] = {
            "train_good": 0,
            "validation_good": 0,
            "test_good": 0,
            "test_logical_anomalies": 0,
            "ground_truth_channels": 0,
            "dataset_images": 0,
        }
        report["valid"] = False
        return report

    expected_objects = set(selected_objects)
    root_directories = {item.name for item in root.iterdir() if item.is_dir()}
    generated_directories = sorted(
        root_directories & GENERATED_ROOT_DIRECTORIES
    )
    if generated_directories:
        warnings.append(
            "Ignoring ReMem-generated root directories: "
            f"{generated_directories}"
        )
    found_objects = root_directories - GENERATED_ROOT_DIRECTORIES
    missing_objects = sorted(expected_objects - found_objects)
    unexpected_objects = sorted(found_objects - set(MVTEC_LOCO_OBJECTS))
    if missing_objects:
        errors.append(f"Missing MVTec LOCO AD object directories: {missing_objects}")
    if unexpected_objects:
        errors.append(f"Unexpected dataset object directories: {unexpected_objects}")

    for object_name in selected_objects:
        object_dir = root / object_name
        object_report: dict[str, Any] = {
            "train_good": 0,
            "validation_good": 0,
            "test": {test_type: 0 for test_type in MVTEC_LOCO_TEST_TYPES},
            "ground_truth_channels": 0,
        }
        report["objects"][object_name] = object_report
        if not object_dir.is_dir():
            continue

        _report_unexpected_subdirectories(
            object_dir,
            {"train", "validation", "test", "ground_truth"},
            errors,
            context=object_name,
        )

        defect_configs = _validate_defects_config(
            object_dir / "defects_config.json",
            errors,
        )

        train_dir = object_dir / "train"
        _report_unexpected_subdirectories(
            train_dir,
            {"good"},
            errors,
            context=f"{object_name}/train",
        )
        train_images, _ = _validate_image_collection(
            train_dir / "good", errors, warnings
        )
        object_report["train_good"] = len(train_images)

        validation_dir = object_dir / "validation"
        if validation_dir.exists():
            if not validation_dir.is_dir():
                errors.append(f"Expected a directory: {validation_dir}")
            else:
                _report_unexpected_subdirectories(
                    validation_dir,
                    {"good"},
                    errors,
                    context=f"{object_name}/validation",
                )
                validation_images, _ = _validate_image_collection(
                    validation_dir / "good", errors, warnings
                )
                object_report["validation_good"] = len(validation_images)

        test_dir = object_dir / "test"
        if not test_dir.is_dir():
            errors.append(f"Missing directory: {test_dir}")
        else:
            found_test_dirs = {
                item.name for item in test_dir.iterdir() if item.is_dir()
            }
            missing_test_dirs = sorted(set(MVTEC_LOCO_TEST_TYPES) - found_test_dirs)
            unexpected_test_dirs = sorted(
                found_test_dirs - set(MVTEC_LOCO_TEST_TYPES) - set(MVTec_LOCO_IGNORED_TYPES)
            )
            if missing_test_dirs:
                errors.append(
                    f"{object_name}/test: missing directories: {missing_test_dirs}"
                )
            if unexpected_test_dirs:
                errors.append(
                    f"{object_name}/test: unexpected directories: "
                    f"{unexpected_test_dirs}"
                )

        test_images: dict[str, list[Path]] = {}
        test_sizes: dict[str, dict[str, tuple[int, int]]] = {}
        for test_type in MVTEC_LOCO_TEST_TYPES:
            images, sizes = _validate_image_collection(
                test_dir / test_type,
                errors,
                warnings,
            )
            test_images[test_type] = images
            test_sizes[test_type] = sizes
            object_report["test"][test_type] = len(images)

        ground_truth_dir = object_dir / "ground_truth"
        if not ground_truth_dir.is_dir():
            errors.append(f"Missing directory: {ground_truth_dir}")
        else:
            found_gt_dirs = {
                item.name for item in ground_truth_dir.iterdir() if item.is_dir()
            }
            missing_gt_dirs = sorted(
                set(MVTEC_LOCO_ANOMALY_TYPES) - found_gt_dirs
            )
            unexpected_gt_dirs = sorted(
                found_gt_dirs - set(MVTEC_LOCO_ANOMALY_TYPES) - set(MVTec_LOCO_IGNORED_TYPES)
            )
            unexpected_gt_files = sorted(
                item.name for item in ground_truth_dir.iterdir() if not item.is_dir()
            )
            if missing_gt_dirs:
                errors.append(
                    f"{object_name}/ground_truth: missing directories: "
                    f"{missing_gt_dirs}"
                )
            if unexpected_gt_dirs:
                errors.append(
                    f"{object_name}/ground_truth: unexpected directories: "
                    f"{unexpected_gt_dirs}"
                )
            if unexpected_gt_files:
                errors.append(
                    f"{object_name}/ground_truth: unexpected files: "
                    f"{unexpected_gt_files[:5]}"
                )

        for anomaly_type in MVTEC_LOCO_ANOMALY_TYPES:
            object_report["ground_truth_channels"] += _validate_ground_truth_type(
                ground_truth_dir / anomaly_type,
                test_images[anomaly_type],
                test_sizes[anomaly_type],
                defect_configs,
                errors,
            )

    totals = {
        "train_good": sum(item["train_good"] for item in report["objects"].values()),
        "validation_good": sum(
            item["validation_good"] for item in report["objects"].values()
        ),
        "test_good": sum(
            item["test"]["good"] for item in report["objects"].values()
        ),
        "test_logical_anomalies": sum(
            item["test"]["logical_anomalies"]
            for item in report["objects"].values()
        ),
        "ground_truth_channels": sum(
            item["ground_truth_channels"] for item in report["objects"].values()
        ),
    }
    totals["dataset_images"] = (
        totals["train_good"]
        + totals["validation_good"]
        + totals["test_good"]
        + totals["test_logical_anomalies"]
    )
    report["totals"] = totals

    report["valid"] = not errors
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"MVTec LOCO AD root: {report['data_root']}")
    for object_name, item in report["objects"].items():
        print(
            f"  {object_name}: train_good={item['train_good']}, "
            f"validation_good={item['validation_good']}, test={item['test']}, "
            f"gt_channels={item['ground_truth_channels']}"
        )
    print(f"Totals: {report['totals']}")
    for warning in report["warnings"]:
        print(f"[WARNING] {warning}")
    for error in report["errors"]:
        print(f"[ERROR] {error}")
    print("Validation: PASS" if report["valid"] else "Validation: FAIL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly validate an MVTec LOCO AD dataset layout"
    )
    parser.add_argument(
        "--data_root",
        default=Config.ROOTS.get(
            "MVTecLOCO",
            "/root/lxq/rememsimple/data/mvtec_loco_anomaly_detection",
        ),
        help="Path to the MVTec LOCO AD dataset root",
    )
    parser.add_argument(
        "--json_report",
        default=None,
        help="Optional path at which to save the validation report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_mvtec_loco_layout(args.data_root)
    _print_report(report)
    if args.json_report:
        _atomic_write_json(Path(args.json_report), report)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
