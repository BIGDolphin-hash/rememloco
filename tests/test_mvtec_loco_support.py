from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import tifffile

from config import Config
from evaluate_mvtec_loco import (
    OFFICIAL_EVALUATOR_DIR,
    evaluate_mvtec_loco,
    evaluate_mvtec_loco_image_level,
    validate_loco_anomaly_maps,
)
from generate_loco_mssm_scores import (
    adaptive_pool_patch_tokens,
    build_global_descriptor,
    count_consensus_scores,
    extract_loco_ranking_tokens,
    fuse_ranked_anomaly_scores,
    get_object_test_samples,
    global_density_scores,
    parse_args as parse_loco_mssm_args,
)
from generate_loco_mssm_b_scores import parse_args as parse_loco_mssm_b_args
from fuse_loco_mssm_gcb_scores import (
    BIDIRECTIONAL_WEIGHT,
    COUNT_WEIGHT,
    GLOBAL_WEIGHT,
    fuse_object_rankings,
    parse_args as parse_loco_mssm_gcb_args,
)
from fuse_loco_mssm_gb_scores import (
    BIDIRECTIONAL_WEIGHT as GB_BIDIRECTIONAL_WEIGHT,
    GLOBAL_WEIGHT as GB_GLOBAL_WEIGHT,
    fuse_object_rankings as fuse_gb_object_rankings,
    parse_args as parse_loco_mssm_gb_args,
)
from src.dataset_info import (
    MVTec_LOCO_OBJECT_ANOMALIES,
    MVTec_LOCO_TEST_TYPES,
    sha256_file,
    sort_scored_paths_label_agnostic,
    validate_mvtec_loco_ranking_metadata,
    validate_mvtec_loco_rankings,
)
from src.loco_mssm_b import (
    bidirectional_position_knn_scores,
    build_consensus_template,
)
from src.utils import get_dataset_info


OBJECT_NAME = "breakfast_box"
IMAGE_SIZE = (16, 12)


def _write_image(
    path: Path,
    *,
    size: tuple[int, int] = IMAGE_SIZE,
    value: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(value, value, value)).save(path)


def _write_loco_test_images(root: Path, object_name: str = OBJECT_NAME) -> None:
    for index, test_type in enumerate((*MVTec_LOCO_TEST_TYPES, "structural_anomalies")):
        _write_image(
            root / object_name / "test" / test_type / "000.png",
            value=10 + index,
        )


def _write_complete_anomaly_maps(
    dataset_root: Path,
    maps_root: Path,
    object_name: str = OBJECT_NAME,
) -> None:
    _write_loco_test_images(dataset_root, object_name)
    height, width = IMAGE_SIZE[1], IMAGE_SIZE[0]
    for index, test_type in enumerate(MVTec_LOCO_TEST_TYPES):
        output_path = maps_root / object_name / "test" / test_type / "000.tiff"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            output_path,
            np.full((height, width), index / 10, dtype=np.float32),
        )


def _install_fake_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFlatIndex:
        def __init__(self, dimension: int) -> None:
            self.dimension = dimension

    fake_faiss = types.ModuleType("faiss")
    fake_faiss.IndexFlatL2 = FakeFlatIndex
    fake_faiss.normalize_L2 = lambda array: None
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)


def _fresh_runtime_modules(monkeypatch: pytest.MonkeyPatch):
    _install_fake_faiss(monkeypatch)
    for module_name in (
        "ReRem_run",
        "src.detection_per_object_test",
        "src.ReRem_detection_test",
    ):
        sys.modules.pop(module_name, None)
    main_module = importlib.import_module("ReRem_run")
    loop_module = importlib.import_module("src.detection_per_object_test")
    detector_module = importlib.import_module("src.ReRem_detection_test")
    return main_module, loop_module, detector_module


def _cached_runtime_modules(monkeypatch: pytest.MonkeyPatch):
    """Reuse native-heavy modules in tests that do not need a fresh import."""

    _install_fake_faiss(monkeypatch)
    main_module = importlib.import_module("ReRem_run")
    loop_module = importlib.import_module("src.detection_per_object_test")
    detector_module = importlib.import_module("src.ReRem_detection_test")
    return main_module, loop_module, detector_module


def test_loco_metadata_has_exact_official_objects_and_test_types() -> None:
    assert list(MVTec_LOCO_OBJECT_ANOMALIES) == [
        "breakfast_box",
        "juice_bottle",
        "pushpins",
        "screw_bag",
        "splicing_connectors",
    ]
    assert MVTec_LOCO_TEST_TYPES == (
        "good",
        "logical_anomalies",
    )
    assert all(
        anomaly_types == ["logical_anomalies"]
        for anomaly_types in MVTec_LOCO_OBJECT_ANOMALIES.values()
    )

    objects, anomalies = get_dataset_info("MVTecLOCO")
    assert objects == list(MVTec_LOCO_OBJECT_ANOMALIES)
    assert anomalies == MVTec_LOCO_OBJECT_ANOMALIES


def test_loco_mssm_gc_cli_defaults_to_global_layer_12(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_loco_mssm_scores.py"])
    args = parse_loco_mssm_args()
    assert args.global_layer == 12
    assert args.output_json.endswith(
        "logical_only/mssm_gc/global_layer=12/results.json"
    )
    assert not hasattr(args, "mssm_layers")
    assert not hasattr(args, "mssm_weight")


def test_loco_mssm_gc_extracts_only_the_global_layer() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.model = SimpleNamespace(blocks=[object()] * 12)
            self.requested_layers = None

        def extract_features(self, image_tensor, feature_list):
            assert image_tensor == "image"
            self.requested_layers = feature_list
            return [
                np.array([[[0.0, 5.0], [12.0, 5.0]]], dtype=np.float32),
            ]

    model = FakeModel()
    global_tokens = extract_loco_ranking_tokens(model, "image", (1, 2), 12)

    assert model.requested_layers == [11]
    np.testing.assert_allclose(
        global_tokens,
        np.array([[0.0, 1.0], [12.0 / 13.0, 5.0 / 13.0]], dtype=np.float32),
    )


def test_loco_mssm_b_cli_is_an_independent_layer_12_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_loco_mssm_b_scores.py"])
    args = parse_loco_mssm_b_args()

    assert args.layer == 12
    assert args.grid_size == 12
    assert args.neighbors == 3
    assert args.position_radius_cells == pytest.approx(2.0)
    assert args.position_weight == pytest.approx(0.25)
    assert args.tail_fraction == pytest.approx(0.1)
    assert args.output_json.endswith("logical_only/mssm_b/layer=12/results.json")
    assert not hasattr(args, "global_weight")
    assert not hasattr(args, "count_weight")
    assert not hasattr(args, "mssm_layers")


def test_loco_mssm_gcb_cli_has_fixed_53_18_29_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["fuse_loco_mssm_gcb_scores.py"])
    args = parse_loco_mssm_gcb_args()

    assert GLOBAL_WEIGHT == pytest.approx(0.53)
    assert COUNT_WEIGHT == pytest.approx(0.18)
    assert BIDIRECTIONAL_WEIGHT == pytest.approx(0.29)
    assert not hasattr(args, "global_weight")
    assert not hasattr(args, "count_weight")
    assert not hasattr(args, "bidirectional_weight")
    assert args.output_json.endswith(
        "logical_only/mssm_gcb/weights=53-18-29/results.json"
    )
    assert Config.JSON_STARTS["MVTecLOCO"] != args.output_json


def test_loco_mssm_gcb_fuses_per_object_percentile_ranks(
    tmp_path: Path,
) -> None:
    _write_loco_test_images(tmp_path)
    gc_entries = [
        {"path": "good/000.png", "score": 0.5, "global_score": 1.0, "count_score": 9.0},
        {
            "path": "logical_anomalies/000.png",
            "score": 0.6,
            "global_score": 2.0,
            "count_score": 3.0,
        },
    ]
    b_entries = [
        {
            "path": "logical_anomalies/000.png",
            "score": 4.0,
            "query_to_template_score": 4.5,
            "template_to_query_score": 3.5,
            "directional_gap": 1.0,
        },
        {
            "path": "good/000.png",
            "score": 8.0,
            "query_to_template_score": 7.0,
            "template_to_query_score": 9.0,
            "directional_gap": -2.0,
        },
    ]

    fused = fuse_object_rankings(
        gc_entries,
        b_entries,
        test_root=tmp_path / OBJECT_NAME / "test",
    )

    assert [entry["path"] for entry in fused] == [
        "good/000.png",
        "logical_anomalies/000.png",
    ]
    assert fused[0]["score"] == pytest.approx(0.47)
    assert fused[1]["score"] == pytest.approx(0.53)
    assert fused[0]["global_rank"] == pytest.approx(0.0)
    assert fused[0]["count_rank"] == pytest.approx(1.0)
    assert fused[0]["bidirectional_rank"] == pytest.approx(1.0)
    assert fused[0]["bidirectional_score"] == pytest.approx(8.0)


def test_loco_mssm_gb_cli_has_fixed_9_to_1_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["fuse_loco_mssm_gb_scores.py"])
    args = parse_loco_mssm_gb_args()

    assert GB_GLOBAL_WEIGHT == pytest.approx(0.9)
    assert GB_BIDIRECTIONAL_WEIGHT == pytest.approx(0.1)
    assert not hasattr(args, "count_weight")
    assert not hasattr(args, "global_weight")
    assert not hasattr(args, "bidirectional_weight")
    assert args.output_json.endswith(
        "logical_only/mssm_gb/weights=9-1/results.json"
    )
    assert Config.JSON_STARTS["MVTecLOCO"] == args.output_json


def test_loco_mssm_gb_does_not_read_or_emit_count_scores(
    tmp_path: Path,
) -> None:
    _write_loco_test_images(tmp_path)
    global_entries = [
        {"path": "good/000.png", "score": 0.5, "global_score": 1.0},
        {
            "path": "logical_anomalies/000.png",
            "score": 0.6,
            "global_score": 2.0,
        },
    ]
    b_entries = [
        {
            "path": "logical_anomalies/000.png",
            "score": 4.0,
            "query_to_template_score": 4.5,
            "template_to_query_score": 3.5,
            "directional_gap": 1.0,
        },
        {
            "path": "good/000.png",
            "score": 8.0,
            "query_to_template_score": 7.0,
            "template_to_query_score": 9.0,
            "directional_gap": -2.0,
        },
    ]

    fused = fuse_gb_object_rankings(
        global_entries,
        b_entries,
        test_root=tmp_path / OBJECT_NAME / "test",
    )

    assert [entry["path"] for entry in fused] == [
        "good/000.png",
        "logical_anomalies/000.png",
    ]
    assert fused[0]["score"] == pytest.approx(0.1)
    assert fused[1]["score"] == pytest.approx(0.9)
    assert all("count_score" not in entry for entry in fused)
    assert all("count_rank" not in entry for entry in fused)


def test_mssm_gc_global_density_uses_multiple_images() -> None:
    patch_tokens = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    pooled = adaptive_pool_patch_tokens(patch_tokens, (2, 2), 1)
    np.testing.assert_allclose(pooled, [[0.5, 0.5]])
    descriptor = build_global_descriptor(patch_tokens, (2, 2), (1, 2))
    assert descriptor.shape == (10,)
    assert np.linalg.norm(descriptor) == pytest.approx(1.0)

    descriptors = np.array(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.0, 1.0]],
        dtype=np.float32,
    )
    scores = global_density_scores(descriptors, neighbors=2)
    assert scores[-1] > max(scores[:-1])


def test_mssm_gc_count_consensus_penalizes_missing_components() -> None:
    background = np.array([1.0, 0.0], dtype=np.float32)
    component = np.array([0.0, 1.0], dtype=np.float32)
    normal_map = np.tile(background, (16, 1))
    normal_map[[0, 3, 12, 15]] = component
    empty_map = np.tile(background, (16, 1))
    pooled_tokens = np.stack(
        [normal_map, normal_map.copy(), normal_map.copy(), empty_map]
    )

    scores, diagnostics = count_consensus_scores(
        pooled_tokens,
        grid_size=4,
        clusters=2,
        minimum_component_cells=1,
        spatial_tail_fraction=0.25,
    )

    assert scores[-1] > max(scores[:-1])
    assert diagnostics["spatial_score"][-1] > max(
        diagnostics["spatial_score"][:-1]
    )
    assert diagnostics["component_count_score"][-1] > max(
        diagnostics["component_count_score"][:-1]
    )


def test_mssm_b_reverse_direction_penalizes_a_missing_component() -> None:
    background = np.array([1.0, 0.0], dtype=np.float32)
    component = np.array([0.0, 1.0], dtype=np.float32)
    normal_map = np.tile(background, (16, 1))
    normal_map[[0, 3, 12, 15]] = component
    missing_map = np.tile(background, (16, 1))
    pooled_tokens = np.stack(
        [normal_map, normal_map.copy(), normal_map.copy(), missing_map]
    )

    template = build_consensus_template(pooled_tokens)
    scores, diagnostics = bidirectional_position_knn_scores(
        pooled_tokens,
        template,
        grid_size=4,
        neighbors=1,
        position_radius_cells=1.0,
        position_weight=0.25,
        tail_fraction=0.25,
    )

    assert scores[-1] > max(scores[:-1])
    assert diagnostics["template_to_query_score"][-1] > diagnostics[
        "query_to_template_score"
    ][-1]


def test_mssm_b_position_constraint_penalizes_a_displaced_component() -> None:
    background = np.array([1.0, 0.0], dtype=np.float32)
    component = np.array([0.0, 1.0], dtype=np.float32)
    aligned = np.tile(background, (25, 1))
    aligned[0] = component
    displaced = np.tile(background, (25, 1))
    displaced[-1] = component
    pooled_tokens = np.stack([aligned, aligned.copy(), aligned.copy()])
    template = build_consensus_template(pooled_tokens)

    scores, _ = bidirectional_position_knn_scores(
        np.stack([aligned, displaced]),
        template,
        grid_size=5,
        neighbors=1,
        position_radius_cells=1.0,
        position_weight=0.25,
        tail_fraction=0.1,
    )

    assert scores[1] > scores[0]


def test_mssm_gc_fuses_component_ranks_without_scale_tuning() -> None:
    fused, ranked = fuse_ranked_anomaly_scores(
        {
            "mssm": np.array([0.1, 0.2, 0.3]),
            "global": np.array([100.0, 300.0, 200.0]),
            "count": np.array([5.0, 5.0, 10.0]),
        },
        {"mssm": 1.0, "global": 1.0, "count": 1.0},
    )

    np.testing.assert_allclose(ranked["mssm"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(ranked["global"], [0.0, 1.0, 0.5])
    np.testing.assert_allclose(ranked["count"], [0.25, 0.25, 1.0])
    assert fused[2] > fused[0]


def test_loco_ranking_requires_exact_coverage_order_and_matching_metadata(
    tmp_path: Path,
) -> None:
    _write_loco_test_images(tmp_path)
    entries = [
        {"path": "good/000.png", "score": 0.1},
        {"path": "logical_anomalies/000.png", "score": 0.2},
    ]
    rankings = {OBJECT_NAME: entries}
    validate_mvtec_loco_rankings(rankings, [OBJECT_NAME], tmp_path)

    with pytest.raises(ValueError, match="paths do not match"):
        validate_mvtec_loco_rankings(
            {OBJECT_NAME: [*entries, {"path": "structural_anomalies/000.png", "score": 0.3}]},
            [OBJECT_NAME], tmp_path,
        )

    with pytest.raises(ValueError, match="ascending score"):
        validate_mvtec_loco_rankings(
            {OBJECT_NAME: list(reversed(entries))}, [OBJECT_NAME], tmp_path
        )
    with pytest.raises(ValueError, match="paths do not match"):
        validate_mvtec_loco_rankings(
            {OBJECT_NAME: entries[:-1]}, [OBJECT_NAME], tmp_path
        )
    with pytest.raises(ValueError, match="duplicate paths"):
        validate_mvtec_loco_rankings(
            {OBJECT_NAME: [entries[0], entries[0], entries[1]]},
            [OBJECT_NAME],
            tmp_path,
        )
    invalid_score = [dict(entry) for entry in entries]
    invalid_score[0]["score"] = True
    with pytest.raises(ValueError, match="invalid score"):
        validate_mvtec_loco_rankings(
            {OBJECT_NAME: invalid_score}, [OBJECT_NAME], tmp_path
        )

    ranking_path = tmp_path / "results.json"
    ranking_path.write_text(json.dumps(rankings), encoding="utf-8")
    metadata_path = tmp_path / "results.meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "settings": {
                    "dataset": "MVTecLOCO",
                    "experiment_scope": "logical_only",
                    "test_types": ["good", "logical_anomalies"],
                    "ranking_algorithm": (
                        "mssm_gc_global_density_spatial_count_consensus_v2"
                    ),
                    "data_root": str(tmp_path.resolve()),
                    "model_name": "dinov2_vits14",
                    "resolution": 672,
                }
            }
        ),
        encoding="utf-8",
    )
    validate_mvtec_loco_ranking_metadata(
        ranking_path,
        data_root=tmp_path,
        model_name="dinov2_vits14",
        resolution=672,
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_mvtec_loco_ranking_metadata(
            ranking_path,
            data_root=tmp_path,
            model_name="dinov2_vits14",
            resolution=448,
        )
    # Old scores remain invalid even if somebody deletes structural entries.
    metadata = json.loads(metadata_path.read_text())
    del metadata["settings"]["experiment_scope"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Regenerate"):
        validate_mvtec_loco_ranking_metadata(
            ranking_path, data_root=tmp_path, model_name="dinov2_vits14", resolution=672,
        )


def test_loco_ranking_metadata_accepts_known_gc_algorithm(tmp_path: Path) -> None:
    _write_loco_test_images(tmp_path)
    ranking_path = tmp_path / "gc" / "results.json"
    ranking_path.parent.mkdir(parents=True)
    ranking_path.write_text(
        json.dumps(
            {
                OBJECT_NAME: [
                    {"path": "good/000.png", "score": 0.1},
                    {"path": "logical_anomalies/000.png", "score": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )
    ranking_path.with_name("results.meta.json").write_text(
        json.dumps(
            {
                "settings": {
                    "dataset": "MVTecLOCO",
                    "experiment_scope": "logical_only",
                    "test_types": ["good", "logical_anomalies"],
                    "ranking_algorithm": (
                        "mssm_gc_global_density_spatial_count_consensus_v2"
                    ),
                    "ranking_tag": "gc-g12-k10-c8-grid12",
                    "data_root": str(tmp_path.resolve()),
                    "model_name": "dinov2_vits14",
                    "resolution": 672,
                }
            }
        ),
        encoding="utf-8",
    )

    metadata = validate_mvtec_loco_ranking_metadata(
        ranking_path,
        data_root=tmp_path,
        model_name="dinov2_vits14",
        resolution=672,
    )
    assert metadata["settings"]["ranking_tag"] == "gc-g12-k10-c8-grid12"

    metadata["settings"]["ranking_algorithm"] = (
        "mssm_leave_one_image_out_content_ties_v1"
    )
    ranking_path.with_name("results.meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ranking_algorithm"):
        validate_mvtec_loco_ranking_metadata(
            ranking_path,
            data_root=tmp_path,
            model_name="dinov2_vits14",
            resolution=672,
        )


def test_loco_ranking_metadata_accepts_mssm_b_algorithm(tmp_path: Path) -> None:
    _write_loco_test_images(tmp_path)
    ranking_path = tmp_path / "b" / "results.json"
    ranking_path.parent.mkdir(parents=True)
    ranking_path.write_text(
        json.dumps(
            {
                OBJECT_NAME: [
                    {"path": "good/000.png", "score": 0.1},
                    {"path": "logical_anomalies/000.png", "score": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )
    ranking_path.with_name("results.meta.json").write_text(
        json.dumps(
            {
                "settings": {
                    "dataset": "MVTecLOCO",
                    "experiment_scope": "logical_only",
                    "test_types": ["good", "logical_anomalies"],
                    "ranking_algorithm": (
                        "mssm_b_bidirectional_position_constrained_knn_v1"
                    ),
                    "ranking_tag": "b-l12-k3-grid12-r2-p0.25-tail0.1",
                    "data_root": str(tmp_path.resolve()),
                    "model_name": "dinov2_vits14",
                    "resolution": 672,
                }
            }
        ),
        encoding="utf-8",
    )

    metadata = validate_mvtec_loco_ranking_metadata(
        ranking_path,
        data_root=tmp_path,
        model_name="dinov2_vits14",
        resolution=672,
    )
    assert metadata["settings"]["ranking_tag"] == (
        "b-l12-k3-grid12-r2-p0.25-tail0.1"
    )


def test_loco_mssm_enumerates_every_native_test_sample_deterministically(
    tmp_path: Path,
) -> None:
    _write_loco_test_images(tmp_path)
    _write_image(tmp_path / OBJECT_NAME / "test" / "good" / "001.jpg")
    (tmp_path / OBJECT_NAME / "test" / "logical_anomalies" / "README.txt").write_text(
        "not an image", encoding="utf-8"
    )

    samples = get_object_test_samples(tmp_path, OBJECT_NAME)

    assert [relative_path for relative_path, _ in samples] == [
        "good/000.png",
        "good/001.jpg",
        "logical_anomalies/000.png",
    ]
    assert all(path.is_file() for _, path in samples)


def test_exact_score_ties_are_resolved_from_content_not_type_labels(
    tmp_path: Path,
) -> None:
    test_root = tmp_path / OBJECT_NAME / "test"
    first_path = test_root / "good" / "same.png"
    second_path = test_root / "logical_anomalies" / "same.png"
    _write_image(first_path, value=0)
    _write_image(second_path, value=255)
    entries = [
        {"path": "good/same.png", "score": 0.5},
        {"path": "logical_anomalies/same.png", "score": 0.5},
    ]
    _write_image(
        test_root / "structural_anomalies" / "same.png", value=127
    )

    first_order = sort_scored_paths_label_agnostic(entries, test_root)
    selected_digest = sha256_file(test_root / first_order[0]["path"])
    first_bytes = first_path.read_bytes()
    second_bytes = second_path.read_bytes()
    first_path.write_bytes(second_bytes)
    second_path.write_bytes(first_bytes)
    second_order = sort_scored_paths_label_agnostic(entries, test_root)

    assert sha256_file(test_root / second_order[0]["path"]) == selected_digest
    complete_order = second_order
    validate_mvtec_loco_rankings(
        {OBJECT_NAME: complete_order}, [OBJECT_NAME], tmp_path
    )
    with pytest.raises(ValueError, match="ordered by image content"):
        validate_mvtec_loco_rankings(
            {OBJECT_NAME: list(reversed(second_order))},
            [OBJECT_NAME],
            tmp_path,
        )


def test_loco_cli_defaults_to_gpu_faiss_and_dataset_specific_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module, _, _ = _fresh_runtime_modules(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ReRem_run.py", "--dataset", "MVTecLOCO"],
    )

    args = main_module.parse_args()
    main_module.validate_runtime_args(args)

    assert args.device == "cuda:0"
    assert args.faiss_on_cpu is False
    assert args.rotation is None
    assert main_module.resolve_rotation(args) is False
    assert main_module.resolve_loco_map_mode(args) == "tl_iter"
    assert args.loco_fusion_alpha == pytest.approx(0.5)
    assert args.dpfe_logical_block == 12
    assert "scope=logical_only" in main_module.loco_experiment_name(args, False)
    assert "tlfme=tex7-10-log12_fullgrid" in main_module.loco_experiment_name(
        args, False
    )
    assert main_module.loco_experiment_name(args, False).endswith(
        "map=tl_iter_s=t3_m=tau_x=t10-gbi-pm1-v1_e=img_nopos"
    )
    assert args.tlfme_texture_coreset_size == 256
    assert args.tlfme_texture_spatial_weight == pytest.approx(0.25)
    args.K = 2
    with pytest.raises(ValueError, match="fixes --K to 3"):
        main_module.validate_runtime_args(args)
    args.K = 3
    args.tau = 0.2
    main_module.validate_runtime_args(args)
    args.mssm_ranking_layers = (6, 9)
    assert "mssm_layers=6-9" in main_module.loco_experiment_name(args, False)
    args.mssm_ranking_tag = "gc-g12-k10-c8-grid12"
    assert "mssm_rank=gc-g12-k10-c8-grid12" in main_module.loco_experiment_name(
        args, False
    )
    assert main_module.resolve_mssm_ranking_layers(
        {"settings": {"mssm_layers": [6, 9]}}
    ) == (6, 9)
    assert main_module.resolve_mssm_ranking_tag(
        {"settings": {"ranking_tag": "gc-g12-k10-c8-grid12"}}
    ) == "gc-g12-k10-c8-grid12"
    with pytest.raises(ValueError, match="Invalid ranking_tag"):
        main_module.resolve_mssm_ranking_tag(
            {"settings": {"ranking_tag": "../unsafe"}}
        )
    with pytest.raises(ValueError, match="Invalid mssm_layers"):
        main_module.resolve_mssm_ranking_layers(
            {"settings": {"mssm_layers": [9, 6]}}
        )
    monkeypatch.setattr(
        sys,
        "argv",
        ["ReRem_run.py", "--dataset", "MVTecLOCO", "--rotation"],
    )
    invalid_rotation_args = main_module.parse_args()
    with pytest.raises(ValueError, match="requires rotation to remain disabled"):
        main_module.validate_runtime_args(invalid_rotation_args)
    del args.mssm_ranking_layers
    del args.mssm_ranking_tag
    all_loco_objects = list(MVTec_LOCO_OBJECT_ANOMALIES)
    assert main_module.resolve_objects(args, all_loco_objects) == [
        "breakfast_box"
    ]

    explicit_loco_args = SimpleNamespace(
        dataset="MVTecLOCO",
        objects=["juice_bottle", "pushpins"],
    )
    assert main_module.resolve_objects(
        explicit_loco_args, all_loco_objects
    ) == ["juice_bottle", "pushpins"]

    for legacy_dataset in ("MVTec", "VisA", "MPDD"):
        assert main_module.resolve_rotation(
            SimpleNamespace(dataset=legacy_dataset, rotation=None)
        ) is True
        assert main_module.resolve_objects(
            SimpleNamespace(dataset=legacy_dataset, objects=None),
            ["first", "second"],
        ) == ["first", "second"]

    monkeypatch.setattr(
        sys,
        "argv",
        ["ReRem_run.py", "--dataset", "MPDD", "--loco_map_mode", "deep"],
    )
    invalid_args = main_module.parse_args()
    with pytest.raises(ValueError, match="only supported for MVTecLOCO"):
        main_module.validate_runtime_args(invalid_args)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ReRem_run.py",
            "--dataset",
            "MVTecLOCO",
            "--loco_map_mode",
            "deep",
            "--loco_fusion_alpha",
            "0.25",
        ],
    )
    invalid_alpha_args = main_module.parse_args()
    with pytest.raises(ValueError, match="configurable only"):
        main_module.validate_runtime_args(invalid_alpha_args)

    invalid_alpha_args.loco_map_mode = "fused"
    main_module.validate_runtime_args(invalid_alpha_args)
    experiment_name = main_module.loco_experiment_name(
        invalid_alpha_args, False
    )
    assert "map=fused_alpha=0.25" in experiment_name


def test_loco_evaluation_dispatches_image_level_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    main_module, _, _ = _fresh_runtime_modules(monkeypatch)
    evaluator_module = importlib.import_module("evaluate_mvtec_loco")
    calls: list[dict[str, object]] = []

    def fake_image_level_evaluator(**kwargs):
        calls.append(kwargs)
        return tmp_path / "image_level_metrics" / "metrics_summary.json"

    monkeypatch.setattr(
        evaluator_module,
        "evaluate_mvtec_loco_image_level",
        fake_image_level_evaluator,
    )
    monkeypatch.setattr(
        main_module,
        "eval_finished_run",
        lambda **unused: pytest.fail("legacy post_eval must not handle LOCO"),
    )
    args = SimpleNamespace(
        dataset="MVTecLOCO",
        eval_clf=True,
        eval_segm=True,
        loco_official_eval_dir=None,
        loco_eval_workers=3,
    )

    result = main_module.evaluate_completed_run(
        args,
        str(tmp_path / "dataset"),
        str(tmp_path / "run"),
        [OBJECT_NAME],
    )

    assert result == tmp_path / "image_level_metrics" / "metrics_summary.json"
    assert calls[0]["objects"] == [OBJECT_NAME]
    assert str(calls[0]["classification_scores_path"]).endswith(
        "run/final_object_results.json"
    )
    assert "anomaly_maps_dir" not in calls[0]


@pytest.mark.parametrize(
    ("eval_clf", "eval_segm"),
    [(True, False), (False, False)],
)
def test_loco_loop_skips_pixel_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    eval_clf: bool,
    eval_segm: bool,
) -> None:
    _, loop_module, _ = _fresh_runtime_modules(monkeypatch)
    monkeypatch.setattr(
        loop_module, "memory_budget", lambda unused, unused_fraction: 3
    )
    ranking_path = tmp_path / "ranking.json"
    ranked_samples = [
        {"path": f"good/{index:03d}.png", "score": index / 100}
        for index in range(60)
    ]
    ranking_path.write_text(
        json.dumps({OBJECT_NAME: ranked_samples}),
        encoding="utf-8",
    )
    inference_calls: list[dict[str, object]] = []
    constructor_calls: list[dict[str, object]] = []

    class FakeDetector:
        def __init__(self, **kwargs) -> None:
            constructor_calls.append(kwargs)
            self.results: dict[str, object] = {}

        def init_reference_memory(self, samples) -> None:
            self.results["img_ref_samples"] = list(samples)

        def run_inference(self, **kwargs):
            inference_calls.append(kwargs)
            self.results["all"].append(
                {"path": "good/000.png", "score": 0.01}
            )
            return {"good/000.png": 0.01}, self.results

    monkeypatch.setattr(loop_module, "ReRemAnomalyDetector", FakeDetector)
    args = SimpleNamespace(
        dataset="MVTecLOCO",
        model_name="dinov2_vits14",
        resolution=672,
        tau=0.05,
        gamma=0.1,
        feature_list=[6, 9],
        scales=[1, 5],
        knn_metric="L2_normalized",
        k_neighbors=1,
        K=3,
        data_root=str(tmp_path),
        initial_json=str(ranking_path),
        faiss_on_cpu=False,
        eval_clf=eval_clf,
        eval_segm=eval_segm,
    )
    config = SimpleNamespace(
        RESULTS_ROOT=str(tmp_path / "iter_results"),
        FEATURE_ROOT=str(tmp_path / "features"),
        ROOTS={"MVTecLOCO": str(tmp_path)},
        JSON_STARTS={"MVTecLOCO": str(ranking_path)},
    )
    final_dir = tmp_path / "final"
    final_dir.mkdir()

    results = loop_module.run_per_object_adaptive_loop(
        model=object(),
        args=args,
        Config=config,
        objects=[OBJECT_NAME],
        overall_output_dir=str(final_dir),
        object_anomalies=MVTec_LOCO_OBJECT_ANOMALIES,
        rotation_default={OBJECT_NAME: False},
        stop_condition=lambda *unused: True,
    )

    assert results[OBJECT_NAME]["all"] == [
        {"path": "good/000.png", "score": 0.01}
    ]
    assert len(inference_calls) == 1
    assert inference_calls[0]["save_tiffs"] is False
    assert inference_calls[0]["save_patch_dists"] is False
    assert constructor_calls[0]["faiss_on_cpu"] is False
    assert constructor_calls[0]["rotation"] is False
    assert constructor_calls[0]["label_agnostic_ties"] is True
    assert constructor_calls[0]["feature_fuse"] is True
    assert constructor_calls[0]["loco_map_mode"] == "tl_iter"
    assert constructor_calls[0]["loco_fusion_alpha"] == pytest.approx(0.5)
    assert constructor_calls[0]["dpfe_logical_block"] == 12
    assert constructor_calls[0]["tlfme_texture_coreset_size"] == 256
    assert constructor_calls[0]["tlfme_texture_spatial_weight"] == (
        pytest.approx(0.25)
    )
    assert "masking" not in constructor_calls[0]
    assert "mask_ref_images" not in constructor_calls[0]
    assert "loco_memory_area_z" not in constructor_calls[0]
    assert "reference_split" not in constructor_calls[0]
    assert "resolution=672" in constructor_calls[0]["features_dir"]
    assert "layers=6-9_scales=1-5" in constructor_calls[0]["features_dir"]
    assert "rotation=0_fullgrid=1" in constructor_calls[0]["features_dir"]
    assert "dualbranch=1" in constructor_calls[0]["features_dir"]
    assert "logicalblock=12" in constructor_calls[0]["features_dir"]

    diagnostics_path = (
        final_dir / "diagnostics" / OBJECT_NAME / "cnt=1.json"
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["selection_labels_used"] is False
    assert diagnostics["feature_extraction"] == {
        "full_grid": True,
        "dual_branch": True,
        "texture_blocks": [7, 10],
        "logical_block": 12,
    }
    assert diagnostics["memory"]["reference_samples"] == [
        "good/000.png",
        "good/001.png",
        "good/002.png",
    ]
    assert diagnostics["memory"]["reference_split"] == "test"
    assert diagnostics["memory"]["frozen"] is False
    assert diagnostics["memory"]["update_policy"] == (
        "texture_coreset_logical_full_grid"
    )
    assert diagnostics["memory"]["texture_coreset_size"] == 256
    assert diagnostics["memory"]["texture_spatial_weight"] == (
        pytest.approx(0.25)
    )
    assert diagnostics["memory"]["logical_storage"] == "full_grid"
    assert diagnostics["memory"]["capacity"] == 3
    assert diagnostics["stop_reasons"] == {
        "memory_fraction_reached": True,
    }
    assert diagnostics["memory"]["audit_only_reference_type_counts"] == {
        "good": 3,
        "logical_anomalies": 0,
        "unexpected": 0,
    }


def test_loco_tl_iter_adds_three_lowest_score_candidates_per_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, loop_module, _ = _fresh_runtime_modules(monkeypatch)
    monkeypatch.setattr(
        loop_module, "memory_budget", lambda unused, unused_fraction: 6
    )
    anchors = [
        "logical_anomalies/anchor_a.png",
        "logical_anomalies/anchor_b.png",
        "logical_anomalies/anchor_c.png",
    ]
    candidates = [
        "good/lowest_a.png",
        "logical_anomalies/lowest_b.png",
        "good/lowest_c.png",
        "good/lowest_d.png",
    ]
    filler = [f"good/filler_{index:03d}.png" for index in range(113)]
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(
        json.dumps(
            {
                OBJECT_NAME: [
                    {"path": path, "score": index / 100}
                    for index, path in enumerate(
                        [*anchors, *candidates, *filler]
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    added_batches: list[list[str]] = []

    class FakeDetector:
        def __init__(self, **kwargs) -> None:
            self.results = {}
            self.last_inference_diagnostics = {"map_mode": "tl_iter"}

        def init_reference_memory(self, samples) -> None:
            self.results["img_ref_samples"] = list(samples)

        def add_selected_reference_samples(
            self, samples, *, round_id
        ) -> None:
            assert round_id == 2
            added_batches.append(list(samples))

        def run_inference(self, **kwargs):
            self.results["all"].extend(
                {
                    "path": path,
                    "score": index / 100,
                }
                for index, path in enumerate([*anchors, *candidates])
            )
            return {}, self.results

    monkeypatch.setattr(loop_module, "ReRemAnomalyDetector", FakeDetector)
    args = SimpleNamespace(
        dataset="MVTecLOCO",
        model_name="dinov2_vits14",
        resolution=672,
        tau=0.05,
        gamma=0.1,
        feature_list=[6, 9],
        scales=[1, 5],
        knn_metric="L2_normalized",
        k_neighbors=1,
        K=3,
        data_root=str(tmp_path),
        initial_json=str(ranking_path),
        faiss_on_cpu=False,
        eval_clf=True,
        eval_segm=True,
    )
    config = SimpleNamespace(
        RESULTS_ROOT=str(tmp_path / "iter_results"),
        FEATURE_ROOT=str(tmp_path / "features"),
        ROOTS={"MVTecLOCO": str(tmp_path)},
        JSON_STARTS={"MVTecLOCO": str(ranking_path)},
    )
    final_dir = tmp_path / "final"
    final_dir.mkdir()

    results = loop_module.run_per_object_adaptive_loop(
        model=object(),
        args=args,
        Config=config,
        objects=[OBJECT_NAME],
        overall_output_dir=str(final_dir),
        object_anomalies=MVTec_LOCO_OBJECT_ANOMALIES,
        rotation_default={OBJECT_NAME: False},
        stop_condition=lambda unused_p, unused_n, iteration, unused_k: (
            iteration >= 2
        ),
    )

    accepted = candidates[:3]
    assert added_batches == [accepted]
    assert results[OBJECT_NAME]["img_ref_samples"] == [
        *anchors,
        *accepted,
    ]
    diagnostics = json.loads(
        (final_dir / "diagnostics" / OBJECT_NAME / "cnt=2.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["selection_labels_used"] is False
    assert diagnostics["memory"]["new_reference_count"] == 3
    assert diagnostics["memory"]["new_reference_samples"] == accepted
    assert diagnostics["memory"]["capacity"] == 6
    assert diagnostics["stop_reasons"]["memory_fraction_reached"] is True


def test_loco_tl_iter_continues_with_lowest_score_until_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, loop_module, _ = _fresh_runtime_modules(monkeypatch)
    ranked_samples = [
        {"path": f"good/{index:03d}.png", "score": index / 100}
        for index in range(80)
    ]
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(
        json.dumps({OBJECT_NAME: ranked_samples}), encoding="utf-8"
    )
    added_batches: list[list[str]] = []

    class FakeDetector:
        def __init__(self, **kwargs) -> None:
            self.results = {}
            self.last_inference_diagnostics = {"map_mode": "tl_iter"}

        def init_reference_memory(self, samples) -> None:
            self.results["img_ref_samples"] = list(samples)

        def add_selected_reference_samples(
            self, samples, *, round_id
        ) -> None:
            assert 2 <= round_id <= 4
            added_batches.append(list(samples))

        def run_inference(self, **kwargs):
            self.results["all"].extend(
                {
                    "path": item["path"],
                    "score": item["score"],
                }
                for item in ranked_samples
            )
            return {}, self.results

    monkeypatch.setattr(loop_module, "ReRemAnomalyDetector", FakeDetector)
    args = SimpleNamespace(
        dataset="MVTecLOCO",
        model_name="dinov2_vits14",
        resolution=672,
        tau=0.15,
        gamma=0.1,
        feature_list=[6, 9],
        scales=[1, 5],
        knn_metric="L2_normalized",
        k_neighbors=1,
        K=3,
        data_root=str(tmp_path),
        initial_json=str(ranking_path),
        faiss_on_cpu=False,
        eval_clf=True,
        eval_segm=True,
    )
    config = SimpleNamespace(
        RESULTS_ROOT=str(tmp_path / "iter_results"),
        FEATURE_ROOT=str(tmp_path / "features"),
        ROOTS={"MVTecLOCO": str(tmp_path)},
        JSON_STARTS={"MVTecLOCO": str(ranking_path)},
    )
    final_dir = tmp_path / "final"
    final_dir.mkdir()

    results = loop_module.run_per_object_adaptive_loop(
        model=object(),
        args=args,
        Config=config,
        objects=[OBJECT_NAME],
        overall_output_dir=str(final_dir),
        object_anomalies=MVTec_LOCO_OBJECT_ANOMALIES,
        rotation_default={OBJECT_NAME: False},
        stop_condition=lambda *unused: False,
    )

    assert results[OBJECT_NAME]["img_ref_samples"] == [
        "good/000.png",
        "good/001.png",
        "good/002.png",
        "good/003.png",
        "good/004.png",
        "good/005.png",
        "good/006.png",
        "good/007.png",
        "good/008.png",
        "good/009.png",
        "good/010.png",
        "good/011.png",
    ]
    assert added_batches == [
        ["good/003.png", "good/004.png", "good/005.png"],
        ["good/006.png", "good/007.png", "good/008.png"],
        ["good/009.png", "good/010.png", "good/011.png"],
    ]
    diagnostics = json.loads(
        (final_dir / "diagnostics" / OBJECT_NAME / "cnt=4.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["memory"]["capacity"] == 12
    assert diagnostics["memory"]["capacity_fraction"] == pytest.approx(0.15)
    assert diagnostics["stop_reasons"] == {
        "memory_fraction_reached": True,
    }


def test_loco_feature_cache_metadata_binds_the_resolved_dataset_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, loop_module, _ = _fresh_runtime_modules(monkeypatch)
    features_dir = tmp_path / "features"
    first_root = tmp_path / "dataset_a"
    second_root = tmp_path / "dataset_b"
    first_root.mkdir()
    second_root.mkdir()
    args = SimpleNamespace(
        model_name="dinov2_vits14",
        resolution=672,
        K=3,
        tau=0.05,
        gamma=0.1,
        feature_list=[6, 9],
        scales=[1, 5],
        knn_metric="L2_normalized",
        k_neighbors=1,
    )

    loop_module._write_or_validate_loco_feature_cache_metadata(
        features_dir,
        args=args,
        data_root=first_root,
        rotation=False,
    )
    loop_module._write_or_validate_loco_feature_cache_metadata(
        features_dir,
        args=args,
        data_root=first_root,
        rotation=False,
    )
    metadata = json.loads(
        (features_dir / "cache.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["data_root"] == str(first_root.resolve())
    assert metadata["dual_branch"] is True
    assert metadata["full_grid_features"] is True
    assert metadata["experiment_scope"] == "logical_only"
    assert metadata["test_types"] == ["good", "logical_anomalies"]
    assert metadata["dpfe_texture_blocks"] == [7, 10]
    assert metadata["dpfe_logical_block"] == 12
    assert metadata["dpfe_update_policy"] == "selected_top_k_full_grid"

    with pytest.raises(RuntimeError, match="does not match this run"):
        loop_module._write_or_validate_loco_feature_cache_metadata(
            features_dir,
            args=args,
            data_root=second_root,
            rotation=False,
        )

    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text("{}", encoding="utf-8")
    results_dir = tmp_path / "iteration_results"
    final_eval_dir = results_dir / "final_eval"
    final_eval_dir.mkdir(parents=True)
    (final_eval_dir / "run.meta.json").write_text(
        '{"owned_by": "final_eval"}', encoding="utf-8"
    )
    loop_module._write_or_validate_loco_results_metadata(
        results_dir,
        args=args,
        data_root=first_root,
        json_path=ranking_path,
        rotation=False,
    )
    results_metadata = json.loads(
        (results_dir / "run.meta.json").read_text(encoding="utf-8")
    )
    assert results_metadata["loco_map_mode"] == "tl_iter"
    assert results_metadata["experiment_scope"] == "logical_only"
    assert results_metadata["loco_fusion_alpha"] is None
    assert results_metadata["loco_dual_branch"] is True
    assert results_metadata["dpfe_texture_blocks"] == [7, 10]
    assert results_metadata["dpfe_logical_block"] == 12
    assert results_metadata["dpfe_update_policy"] == (
        "selected_top_k_full_grid"
    )
    assert results_metadata["tlfme_texture_blocks"] == [7, 10]
    assert results_metadata["tlfme_logical_block"] == 12
    assert results_metadata["tlfme_update_policy"] == (
        "texture_coreset_logical_full_grid"
    )
    assert results_metadata["tlfme_texture_coreset_size"] == 256
    assert results_metadata["tlfme_texture_spatial_weight"] == (
        pytest.approx(0.25)
    )
    assert results_metadata["tl_iter_mssm"]["memory_capacity_policy"] == (
        "floor_sample_count_times_tau"
    )
    assert results_metadata["tl_iter_mssm"][
        "memory_capacity_fraction"
    ] == pytest.approx(args.tau)
    assert results_metadata["tl_iter_mssm"]["initial_k"] == 3
    assert results_metadata["tl_iter_mssm"]["top_k"] == 3
    assert results_metadata["tl_iter_mssm"]["selection_policy"] == (
        "lowest_anomaly_score_up_to_top3"
    )
    assert "loco_memory_filter" not in results_metadata
    assert "loco_stage3a" not in results_metadata
    assert json.loads(
        (final_eval_dir / "run.meta.json").read_text(encoding="utf-8")
    ) == {"owned_by": "final_eval"}
    loop_module._write_or_validate_loco_results_metadata(
        results_dir,
        args=args,
        data_root=first_root,
        json_path=ranking_path,
        rotation=False,
    )
    args.loco_map_mode = "fused"
    args.loco_fusion_alpha = 0.25
    with pytest.raises(RuntimeError, match="does not match this run"):
        loop_module._write_or_validate_loco_results_metadata(
            results_dir,
            args=args,
            data_root=first_root,
            json_path=ranking_path,
            rotation=False,
        )

    unbound_results_dir = tmp_path / "unbound_iteration_results"
    unbound_cnt_dir = unbound_results_dir / "cnt=1"
    unbound_cnt_dir.mkdir(parents=True)
    (unbound_cnt_dir / "results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contains artifacts"):
        loop_module._write_or_validate_loco_results_metadata(
            unbound_results_dir,
            args=args,
            data_root=first_root,
            json_path=ranking_path,
            rotation=False,
        )
    args.loco_map_mode = None
    args.loco_fusion_alpha = 0.5
    ranking_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match this run"):
        loop_module._write_or_validate_loco_results_metadata(
            results_dir,
            args=args,
            data_root=first_root,
            json_path=ranking_path,
            rotation=False,
        )
    with pytest.raises(RuntimeError, match="does not match this run"):
        loop_module._write_or_validate_loco_results_metadata(
            results_dir,
            args=args,
            data_root=second_root,
            json_path=ranking_path,
            rotation=False,
        )


def test_tiff_writer_preserves_source_shape_float32_and_finite_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, detector_module = _fresh_runtime_modules(monkeypatch)
    detector_module._save_anomaly_map(
        patch_distances=np.array([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]]),
        deep_patch_distances=None,
        image_shape=(11, 17, 3),
        plots_dir=str(tmp_path),
        object_name=OBJECT_NAME,
        type_anomaly="good",
        img_name="000.png",
        save_patch_dists=False,
        save_tiffs=True,
        feature_fuse=False,
    )

    tiff_path = (
        tmp_path
        / "Anomaly_maps"
        / OBJECT_NAME
        / "test"
        / "good"
        / "000.tiff"
    )
    anomaly_map = tifffile.imread(tiff_path)
    assert anomaly_map.shape == (11, 17)
    assert anomaly_map.dtype == np.float32
    assert np.isfinite(anomaly_map).all()
    assert not tiff_path.with_suffix(".npy").exists()

    with pytest.raises(RuntimeError, match="NaN or infinity"):
        detector_module._save_anomaly_map(
            patch_distances=np.array([[0.0, np.nan], [0.5, 1.0]]),
            deep_patch_distances=None,
            image_shape=(11, 17, 3),
            plots_dir=str(tmp_path),
            object_name=OBJECT_NAME,
            type_anomaly="good",
            img_name="nan.png",
            save_patch_dists=False,
            save_tiffs=True,
            feature_fuse=False,
        )


@pytest.mark.parametrize(
    ("map_mode", "fusion_alpha", "expected_value"),
    [
        ("mlmp", 0.25, 0.2),
        ("deep", 0.25, 0.8),
        ("fused", 0.25, 0.65),
    ],
)
def test_loco_map_mode_controls_the_exact_saved_tiff_and_returned_map(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    map_mode: str,
    fusion_alpha: float,
    expected_value: float,
) -> None:
    _, _, detector_module = _fresh_runtime_modules(monkeypatch)
    output_dir = tmp_path / map_mode
    returned_map = detector_module._save_anomaly_map(
        patch_distances=np.full((2, 3), 0.2, dtype=np.float32),
        deep_patch_distances=np.full((2, 3), 0.8, dtype=np.float32),
        image_shape=(11, 17, 3),
        plots_dir=str(output_dir),
        object_name=OBJECT_NAME,
        type_anomaly="good",
        img_name="000.png",
        save_patch_dists=False,
        save_tiffs=True,
        feature_fuse=True,
        loco_map_mode=map_mode,
        loco_fusion_alpha=fusion_alpha,
    )

    saved_map = tifffile.imread(
        output_dir
        / "Anomaly_maps"
        / OBJECT_NAME
        / "test"
        / "good"
        / "000.tiff"
    )
    assert returned_map.dtype == np.float32
    np.testing.assert_allclose(saved_map, returned_map)
    np.testing.assert_allclose(
        returned_map,
        np.full((11, 17), expected_value, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_loco_deep_and_fused_modes_require_a_matching_deep_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, detector_module = _fresh_runtime_modules(monkeypatch)
    with pytest.raises(RuntimeError, match="requires the deep branch"):
        detector_module._save_anomaly_map(
            patch_distances=np.zeros((2, 3), dtype=np.float32),
            deep_patch_distances=None,
            image_shape=(11, 17, 3),
            plots_dir=str(tmp_path),
            object_name=OBJECT_NAME,
            type_anomaly="good",
            img_name="000.png",
            save_patch_dists=False,
            save_tiffs=True,
            feature_fuse=False,
            loco_map_mode="deep",
        )


def test_loco_memory_score_equals_the_official_saved_tiff_maximum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, detector_module = _fresh_runtime_modules(monkeypatch)
    _write_loco_test_images(tmp_path)
    structural_path = tmp_path / OBJECT_NAME / "test" / "structural_anomalies" / "000.png"
    structural_path.write_bytes(b"must never be decoded during logical inference")

    monkeypatch.setattr(
        detector_module,
        "_load_or_extract_features",
        lambda *args, **kwargs: (
            np.zeros((4, 2), dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
            (2, 2),
        ),
    )
    monkeypatch.setattr(
        detector_module,
        "_compute_anomaly_score",
        lambda *args, **kwargs: (
            np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32),
            np.array([[0.9], [0.8], [0.7], [0.6]], dtype=np.float32),
        ),
    )

    detector = detector_module.ReRemAnomalyDetector(
        model=object(), object_name=OBJECT_NAME, data_root=str(tmp_path),
        features_dir=str(tmp_path / "features"), feature_fuse=True,
        feature_list=[6, 9], scales=[1, 5],
        object_anomalies=MVTec_LOCO_OBJECT_ANOMALIES,
        knn_metric="L2_normalized", knn_neighbors=1, faiss_on_cpu=True,
        rotation=False,
        label_agnostic_ties=True, loco_map_mode="fused", loco_fusion_alpha=0.25,
    )
    detector.results = {"all": []}
    detector.last_inference_diagnostics = None

    scores, results = detector.run_inference(
        overall_output_dir=str(tmp_path / "run"),
        save_patch_dists=False,
        save_tiffs=True,
    )
    assert set(scores) == {"good/000.png", "logical_anomalies/000.png"}
    prediction_root = tmp_path / "run" / "Anomaly_maps" / OBJECT_NAME / "test"
    assert {path.name for path in prediction_root.iterdir()} == {
        "good", "logical_anomalies",
    }

    saved_map = tifffile.imread(
        tmp_path
        / "run"
        / "Anomaly_maps"
        / OBJECT_NAME
        / "test"
        / "good"
        / "000.tiff"
    )
    expected_score = float(np.max(saved_map))
    assert scores["good/000.png"] == pytest.approx(expected_score)
    assert results["all"][0]["score"] == pytest.approx(expected_score)
    assert "candidate_diagnostics" not in results
    assert detector.last_inference_diagnostics["image_score_definition"] == (
        "maximum_saved_tiff_pixel"
    )
    assert detector.last_inference_diagnostics["score_statistics"]["selected"][
        "max"
    ] == pytest.approx(expected_score)


def test_loco_anomaly_map_preflight_accepts_complete_float_maps(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    maps_root = tmp_path / "maps"
    _write_complete_anomaly_maps(dataset_root, maps_root)

    report = validate_loco_anomaly_maps(
        dataset_root, maps_root, [OBJECT_NAME]
    )

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["objects"][OBJECT_NAME]["test_images"] == {
        "good": 1,
        "logical_anomalies": 1,
    }

    with pytest.raises(ValueError, match="At least one"):
        evaluate_mvtec_loco(
            dataset_base_dir=dataset_root,
            anomaly_maps_dir=maps_root,
            output_dir=tmp_path / "metrics",
            objects=[],
        )


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("missing_good", "missing TIFF predictions"),
        ("wrong_shape", "does not match source shape"),
        ("integer_dtype", "must use floating point scores"),
        ("nan", "contains NaN or infinity"),
        ("structural_maps", "unexpected anomaly-map test directories"),
    ],
)
def test_loco_anomaly_map_preflight_rejects_incomplete_or_invalid_maps(
    tmp_path: Path,
    failure: str,
    expected_error: str,
) -> None:
    dataset_root = tmp_path / "dataset"
    maps_root = tmp_path / "maps"
    _write_complete_anomaly_maps(dataset_root, maps_root)
    good_map = maps_root / OBJECT_NAME / "test" / "good" / "000.tiff"
    logical_map = (
        maps_root
        / OBJECT_NAME
        / "test"
        / "logical_anomalies"
        / "000.tiff"
    )

    if failure == "missing_good":
        good_map.unlink()
    elif failure == "wrong_shape":
        tifffile.imwrite(logical_map, np.zeros((5, 7), dtype=np.float32))
    elif failure == "integer_dtype":
        tifffile.imwrite(logical_map, np.zeros((12, 16), dtype=np.uint8))
    elif failure == "nan":
        values = np.zeros((12, 16), dtype=np.float32)
        values[0, 0] = np.nan
        tifffile.imwrite(logical_map, values)
    elif failure == "structural_maps":
        structural_dir = maps_root / OBJECT_NAME / "test" / "structural_anomalies"
        structural_dir.mkdir()
        tifffile.imwrite(structural_dir / "000.tiff", np.zeros((12, 16), dtype=np.float32))
    else:  # pragma: no cover - protects the parametrization itself
        raise AssertionError(f"Unknown failure fixture: {failure}")

    report = validate_loco_anomaly_maps(
        dataset_root, maps_root, [OBJECT_NAME]
    )

    assert report["valid"] is False
    assert any(expected_error in error for error in report["errors"])


def test_vendored_official_evaluator_keeps_license_and_v2_provenance() -> None:
    evaluator_root = Path(OFFICIAL_EVALUATOR_DIR)
    license_path = evaluator_root / "LICENSE.txt"
    provenance_path = evaluator_root / "UPSTREAM.md"

    assert (evaluator_root / "evaluate_experiment.py").is_file()
    assert license_path.is_file()
    license_text = license_path.read_text(encoding="utf-8")
    assert "Copyright 2022 MVTec Software GmbH" in license_text
    assert "Redistribution and use in source and binary forms" in license_text
    assert provenance_path.is_file()
    provenance = provenance_path.read_text(encoding="utf-8")
    assert "v2.0" in provenance
    assert "3a3bd3816cc8d9e3abc10d308b9315ca46c5b7fedd98128920bbc9bc2bad3823" in provenance


@pytest.mark.parametrize("structural_present", [False, True])
def test_official_evaluator_subprocess_writes_per_object_and_macro_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    structural_present: bool,
) -> None:
    dataset_root = tmp_path / "dataset"
    maps_root = tmp_path / "maps"
    metrics_root = tmp_path / "metrics"
    rogue_python_path = tmp_path / "rogue_python_path"
    rogue_src = rogue_python_path / "src"
    rogue_src.mkdir(parents=True)
    (rogue_src / "__init__.py").write_text(
        "raise RuntimeError('wrong src package imported')\n",
        encoding="utf-8",
    )
    existing_python_path = os.environ.get("PYTHONPATH")
    python_path_parts = [str(rogue_python_path)]
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(python_path_parts))
    object_root = dataset_root / OBJECT_NAME
    object_root.mkdir(parents=True)
    (object_root / "defects_config.json").write_text(
        json.dumps(
            [
                {
                    "defect_name": "synthetic_defect",
                    "pixel_value": 1,
                    "saturation_threshold": 1,
                    "relative_saturation": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    height, width = 16, 16
    gradient = (
        np.arange(height * width, dtype=np.float32).reshape(height, width)
        / (height * width * 10)
    )
    for index, test_type in enumerate(MVTec_LOCO_TEST_TYPES):
        _write_image(
            object_root / "test" / test_type / "000.png",
            size=(width, height),
        )
        anomaly_map = gradient.copy()
        if test_type != "good":
            channel = np.zeros((height, width), dtype=np.uint8)
            channel[3:7, 4:9] = 1
            channel_path = (
                object_root
                / "ground_truth"
                / test_type
                / "000"
                / "000.png"
            )
            channel_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(channel).save(channel_path)
            anomaly_map[channel.astype(bool)] += 0.8 + index * 0.05
        map_path = maps_root / OBJECT_NAME / "test" / test_type / "000.tiff"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(map_path, anomaly_map.astype(np.float32))

    if structural_present:
        # Full native archives remain usable; excluded corrupt data is never read.
        for split in ("test", "ground_truth"):
            structural_dir = object_root / split / "structural_anomalies"
            structural_dir.mkdir(parents=True)
            (structural_dir / "invalid.png").write_bytes(b"not an image")

    summary_path = evaluate_mvtec_loco(
        dataset_base_dir=dataset_root,
        anomaly_maps_dir=maps_root,
        output_dir=metrics_root,
        objects=[OBJECT_NAME],
        official_eval_dir=OFFICIAL_EVALUATOR_DIR,
        num_parallel_workers=None,
        seed=7,
    )

    assert summary_path == metrics_root.resolve() / "metrics_summary.json"
    assert (metrics_root / OBJECT_NAME / "metrics.json").is_file()
    assert (metrics_root / "anomaly_maps_validation.json").is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["protocol"]["version"] == "2.0"
    assert summary["protocol"]["objects"] == [OBJECT_NAME]
    assert summary["protocol"]["experiment_scope"] == "logical_only"
    assert summary["protocol"]["full_benchmark"] is False
    object_metrics = summary["objects"][OBJECT_NAME]
    assert set(object_metrics) == {"classification", "localization"}
    assert set(object_metrics["classification"]["auc_roc"]) == {
        "logical_anomalies",
    }
    assert set(object_metrics["localization"]["auc_spro"]) == {"logical_anomalies"}
    assert set(object_metrics["localization"]["auc_spro"]["logical_anomalies"]) == {
        "0.01",
        "0.05",
        "0.1",
        "0.3",
        "1.0",
    }
    assert summary["macro_average"] == object_metrics
    assert object_metrics["classification"]["auc_roc"]["logical_anomalies"] == pytest.approx(1.0)
    assert object_metrics["localization"]["auc_spro"]["logical_anomalies"]["0.05"] == pytest.approx(1.0)


def test_official_adapter_handles_repeated_initial_thresholds(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    maps_root = tmp_path / "maps"
    metrics_root = tmp_path / "metrics"
    object_root = dataset_root / OBJECT_NAME
    object_root.mkdir(parents=True)
    (object_root / "defects_config.json").write_text(
        json.dumps(
            [
                {
                    "defect_name": "synthetic_defect",
                    "pixel_value": 1,
                    "saturation_threshold": 1,
                    "relative_saturation": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    height, width = 8, 6
    for index, test_type in enumerate(MVTec_LOCO_TEST_TYPES):
        _write_image(
            object_root / "test" / test_type / "000.png",
            size=(width, height),
        )
        anomaly_map = np.zeros((height, width), dtype=np.float32)
        if test_type != "good":
            channel = np.zeros((height, width), dtype=np.uint8)
            channel[2:5, 1:4] = 1
            channel_path = (
                object_root
                / "ground_truth"
                / test_type
                / "000"
                / "000.png"
            )
            channel_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(channel).save(channel_path)
            anomaly_map[channel.astype(bool)] = 0.8 + index * 0.05
        map_path = maps_root / OBJECT_NAME / "test" / test_type / "000.tiff"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(map_path, anomaly_map)

    summary_path = evaluate_mvtec_loco(
        dataset_base_dir=dataset_root,
        anomaly_maps_dir=maps_root,
        output_dir=metrics_root,
        objects=[OBJECT_NAME],
        official_eval_dir=OFFICIAL_EVALUATOR_DIR,
        num_parallel_workers=4,
        seed=0,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["protocol"]["initial_threshold_policy"] == (
        "official_v2.0_exact_duplicates_removed"
    )
    assert (metrics_root / OBJECT_NAME / "metrics.json").is_file()
    for section in ("classification", "localization"):
        assert section in summary["objects"][OBJECT_NAME]


def test_official_compatibility_restores_missing_curve_endpoints() -> None:
    from loco_official_eval_compat import install

    class FakeMetrics:
        def __init__(self, fp_rates):
            self._fp_rates = np.asarray(fp_rates, dtype=np.float64)

        def get_fp_rates(self):
            return self._fp_rates

    class FakeMetricsAggregator:
        def __init__(self):
            self.threshold_metrics = FakeMetrics([0.2, 0.8])
            self.endpoint_queries = []

        def _get_initial_thresholds(self):
            return [3.0, 2.0, 2.0, 1.0]

        def _refinement_callback(self, thresholds):
            self.endpoint_queries.append(thresholds)
            self.threshold_metrics = FakeMetrics([0.0, 0.2, 0.8, 1.0])

        def run(self, curve_max_distance):
            return self.threshold_metrics

    fake_aggregation = SimpleNamespace(
        MetricsAggregator=FakeMetricsAggregator
    )
    install(fake_aggregation)

    aggregator = fake_aggregation.MetricsAggregator()
    assert aggregator._get_initial_thresholds() == [3.0, 2.0, 1.0]

    metrics = aggregator.run(curve_max_distance=0.001)

    assert aggregator.endpoint_queries == [[np.inf, -np.inf]]
    assert metrics.get_fp_rates().tolist() == [0.0, 0.2, 0.8, 1.0]
