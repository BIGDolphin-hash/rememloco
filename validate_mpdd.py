"""Validate an MPDD checkout before running ReMem.

The validator is intentionally read-only. It checks the MVTec-style directory
layout, image readability, and the one-to-one correspondence between anomalous
test images and ``*_mask.png`` ground-truth files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from config import Config
from src.dataset_info import (
    MPDD_OBJECT_ANOMALIES,
    is_image_file,
    resolve_mpdd_anomaly_types,
    sorted_image_paths,
)


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


def _mask_has_foreground(mask_path: Path) -> bool | None:
    try:
        with Image.open(mask_path) as image:
            _, maximum = image.convert("L").getextrema()
        return maximum > 0
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
        return None


def validate_mpdd_layout(data_root: str | Path) -> dict[str, Any]:
    """Return a serializable validation report for an MPDD directory."""

    root = Path(data_root).expanduser().resolve()
    report: dict[str, Any] = {
        "data_root": str(root),
        "objects": {},
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]

    if not root.is_dir():
        errors.append(f"Dataset root does not exist or is not a directory: {root}")
        report["valid"] = False
        return report

    found_object_dirs = {item.name for item in root.iterdir() if item.is_dir()}
    expected_objects = set(MPDD_OBJECT_ANOMALIES)
    missing_objects = sorted(expected_objects - found_object_dirs)
    if missing_objects:
        errors.append(f"Missing MPDD object directories: {missing_objects}")

    for object_name in MPDD_OBJECT_ANOMALIES:
        anomaly_types = resolve_mpdd_anomaly_types(root, object_name)
        object_dir = root / object_name
        object_report = {
            "train_good": 0,
            "test_good": 0,
            "test_anomalous": 0,
            "anomaly_types": {},
        }
        report["objects"][object_name] = object_report

        if not object_dir.is_dir():
            continue

        train_good_dir = object_dir / "train" / "good"
        test_dir = object_dir / "test"
        test_good_dir = test_dir / "good"
        ground_truth_dir = object_dir / "ground_truth"

        if not test_dir.is_dir() and (object_dir / "validation").is_dir():
            errors.append(
                f"{object_name}: found 'validation' but no 'test'. Create a "
                "read-only compatibility symlink named 'test' -> 'validation'."
            )

        required_dirs = [train_good_dir, test_good_dir, ground_truth_dir]
        for required_dir in required_dirs:
            if not required_dir.is_dir():
                errors.append(f"{object_name}: missing directory {required_dir}")

        train_images = sorted_image_paths(train_good_dir)
        good_test_images = sorted_image_paths(test_good_dir)
        object_report["train_good"] = len(train_images)
        object_report["test_good"] = len(good_test_images)

        if train_good_dir.is_dir() and not train_images:
            errors.append(f"{object_name}: train/good contains no supported images")
        if test_good_dir.is_dir() and not good_test_images:
            errors.append(f"{object_name}: test/good contains no supported images")

        if test_dir.is_dir():
            found_test_subdirs = {
                item.name for item in test_dir.iterdir() if item.is_dir()
            }
            expected_test_subdirs = {"good", *anomaly_types}
            unexpected = sorted(found_test_subdirs - expected_test_subdirs)
            missing = sorted(expected_test_subdirs - found_test_subdirs)
            if unexpected:
                errors.append(
                    f"{object_name}: unexpected test defect directories would be "
                    f"ignored by ReMem: {unexpected}"
                )
            if missing:
                errors.append(f"{object_name}: missing test directories: {missing}")

        for image_path in [*train_images, *good_test_images]:
            if _check_readable_image(image_path) is None:
                errors.append(f"Unreadable image: {image_path}")

        for anomaly_type in anomaly_types:
            anomaly_dir = test_dir / anomaly_type
            mask_dir = ground_truth_dir / anomaly_type
            anomaly_images = sorted_image_paths(anomaly_dir)
            mask_images = sorted_image_paths(mask_dir)
            object_report["anomaly_types"][anomaly_type] = len(anomaly_images)
            object_report["test_anomalous"] += len(anomaly_images)

            if not anomaly_dir.is_dir():
                errors.append(f"{object_name}: missing directory {anomaly_dir}")
                continue
            if not mask_dir.is_dir():
                errors.append(f"{object_name}: missing directory {mask_dir}")
                continue
            if not anomaly_images:
                errors.append(f"{object_name}/{anomaly_type}: no anomaly images found")

            expected_mask_names = {
                f"{image_path.stem}_mask.png" for image_path in anomaly_images
            }
            found_mask_names = {mask_path.name for mask_path in mask_images}
            missing_masks = sorted(expected_mask_names - found_mask_names)
            orphan_masks = sorted(found_mask_names - expected_mask_names)
            if missing_masks:
                errors.append(
                    f"{object_name}/{anomaly_type}: missing masks: {missing_masks[:5]}"
                )
            if orphan_masks:
                errors.append(
                    f"{object_name}/{anomaly_type}: orphan masks: {orphan_masks[:5]}"
                )

            for image_path in anomaly_images:
                image_size = _check_readable_image(image_path)
                if image_size is None:
                    errors.append(f"Unreadable image: {image_path}")
                    continue
                mask_path = mask_dir / f"{image_path.stem}_mask.png"
                if not mask_path.is_file():
                    continue
                mask_size = _check_readable_image(mask_path)
                if mask_size is None:
                    errors.append(f"Unreadable mask: {mask_path}")
                elif mask_size != image_size:
                    errors.append(
                        f"Image/mask size mismatch: {image_path}={image_size}, "
                        f"{mask_path}={mask_size}"
                    )
                elif not _mask_has_foreground(mask_path):
                    errors.append(f"Ground-truth mask has no foreground: {mask_path}")

        for directory in [train_good_dir, test_good_dir]:
            if directory.is_dir():
                ignored = sorted(
                    item.name
                    for item in directory.iterdir()
                    if item.is_file() and not is_image_file(item)
                )
                if ignored:
                    warnings.append(
                        f"{directory}: ignored non-image files: {ignored[:5]}"
                    )

    report["totals"] = {
        "train_good": sum(
            item["train_good"] for item in report["objects"].values()
        ),
        "test_good": sum(
            item["test_good"] for item in report["objects"].values()
        ),
        "test_anomalous": sum(
            item["test_anomalous"] for item in report["objects"].values()
        ),
    }
    report["valid"] = not errors
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"MPDD root: {report['data_root']}")
    for object_name, item in report["objects"].items():
        print(
            f"  {object_name}: train_good={item['train_good']}, "
            f"test_good={item['test_good']}, "
            f"test_anomalous={item['test_anomalous']}, "
            f"types={item['anomaly_types']}"
        )
    if "totals" in report:
        print(f"Totals: {report['totals']}")
    for warning in report["warnings"]:
        print(f"[WARNING] {warning}")
    for error in report["errors"]:
        print(f"[ERROR] {error}")
    print("Validation: PASS" if report["valid"] else "Validation: FAIL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an MPDD dataset layout")
    parser.add_argument(
        "--data_root",
        default=Config.ROOTS["MPDD"],
        help="Path to the MPDD dataset root",
    )
    parser.add_argument(
        "--json_report",
        default=None,
        help="Optional path at which to save the validation report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_mpdd_layout(args.data_root)
    _print_report(report)
    if args.json_report:
        _atomic_write_json(Path(args.json_report), report)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
