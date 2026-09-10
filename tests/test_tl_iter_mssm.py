from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.dpfe import TLFMEMemoryBank
from src.dual_branch_features import DualBranchFeatures
from src.tl_iter_mssm import (
    TL_ITER_DEFAULT_MEMORY_FRACTION,
    TLIterMSSM,
    _bidirectional_logical_map,
    memory_budget,
    select_lowest_score_top_k,
)


class _NumpyFlatL2:
    def __init__(self) -> None:
        self.features = np.empty((0, 0), dtype=np.float32)

    def add(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float32)
        if not len(self.features):
            self.features = features.copy()
        else:
            self.features = np.concatenate((self.features, features), axis=0)

    def search(self, query: np.ndarray, neighbors: int):
        query = np.asarray(query, dtype=np.float32)
        squared = np.square(
            query[:, None, :] - self.features[None, :, :]
        ).sum(axis=2)
        order = np.argsort(squared, axis=1, kind="mergesort")[:, :neighbors]
        return np.take_along_axis(squared, order, axis=1), order


def _features(values: np.ndarray) -> DualBranchFeatures:
    values = np.asarray(values, dtype=np.float32)
    return DualBranchFeatures(
        texture=values,
        logical=values,
        grid_size=(2, 2),
    )


def _three_reference_memory() -> tuple[TLFMEMemoryBank, _NumpyFlatL2]:
    base = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    memory = TLFMEMemoryBank()
    index = _NumpyFlatL2()
    for reference_index in range(3):
        features = _features(base)
        memory.add(
            features,
            image_id=f"good/{reference_index:03d}.png",
            round_id=1,
        )
        index.add(features.texture)
    return memory, index


def test_tl_iter_memory_budget_uses_ten_percent_floor() -> None:
    assert TL_ITER_DEFAULT_MEMORY_FRACTION == pytest.approx(0.10)
    assert memory_budget(100) == 10
    assert memory_budget(229) == 22
    assert memory_budget(300) == 30
    with pytest.raises(ValueError, match="fewer than 3 initial"):
        memory_budget(29)


def test_select_lowest_score_top_k_returns_three_and_respects_slots() -> None:
    entries = [
        {"path": "good/c.png", "score": 0.4},
        {"path": "good/existing.png", "score": 0.0},
        {"path": "logical_anomalies/a.png", "score": 0.1},
        {"path": "good/a.png", "score": 0.2},
        {"path": "good/b.png", "score": 0.3},
        {"path": "good/d.png", "score": 0.5},
    ]
    assert select_lowest_score_top_k(
        entries,
        existing_image_ids={"good/existing.png"},
        remaining_slots=10,
    ) == ["logical_anomalies/a.png", "good/a.png", "good/b.png"]
    assert select_lowest_score_top_k(
        entries,
        existing_image_ids={"good/existing.png"},
        remaining_slots=2,
    ) == ["logical_anomalies/a.png", "good/a.png"]
    assert select_lowest_score_top_k(
        entries,
        existing_image_ids={"good/existing.png"},
        remaining_slots=0,
    ) == []


def test_logical_reverse_direction_marks_a_missing_expected_component() -> None:
    memory = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    query = memory.copy()
    query[0, 1] = [1.0, 0.0, 0.0]
    score_map = _bidirectional_logical_map(query, memory)

    assert score_map[0, 1] > score_map[1, 0]
    assert score_map[0, 1] > 0.2


def test_global_logical_matching_is_position_invariant() -> None:
    memory = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    displaced = memory.copy()
    displaced[0, 0], displaced[0, 1] = (
        memory[0, 1].copy(),
        memory[0, 0].copy(),
    )
    identical_score = _bidirectional_logical_map(memory, memory).max()
    displaced_score = _bidirectional_logical_map(displaced, memory).max()

    assert displaced_score == pytest.approx(identical_score)


def test_tl_iter_scores_an_unknown_query_without_branch_calibration() -> None:
    memory, index = _three_reference_memory()
    scorer = TLIterMSSM(memory, index)
    normal = memory.texture_features[:4]
    normal_result = scorer.score(_features(normal))
    anomaly = np.tile(
        np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        (4, 1),
    )
    anomaly_result = scorer.score(_features(anomaly))

    assert anomaly_result.score > normal_result.score
    np.testing.assert_allclose(
        anomaly_result.final_map,
        np.maximum(anomaly_result.texture_map, anomaly_result.logical_map),
    )


def test_tl_iter_requires_two_references_for_self_match_exclusion() -> None:
    memory, index = _three_reference_memory()
    truncated_memory = TLFMEMemoryBank()
    for record in memory.records[:1]:
        truncated_memory.add(
            _features(
                memory.texture_features[
                    record.texture_start:record.texture_stop
                ]
            ),
            image_id=record.image_id,
            round_id=1,
        )
    truncated_index = _NumpyFlatL2()
    truncated_index.add(truncated_memory.texture_features)

    with pytest.raises(ValueError, match="at least 2 memory records"):
        TLIterMSSM(truncated_memory, truncated_index)


def test_detector_tl_iter_path_reads_structured_memory_not_flat_logical_knn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detector_module = importlib.import_module("src.ReRem_detection_test")
    for test_type, names in {
        "good": ["000.png", "001.png", "002.png"],
        "logical_anomalies": ["003.png"],
    }.items():
        for name in names:
            path = tmp_path / "breakfast_box" / "test" / test_type / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(10, 10, 10)).save(path)

    base = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        detector_module,
        "_load_or_extract_features",
        lambda *args, **kwargs: (base.copy(), base.copy(), (2, 2)),
    )

    def fail_if_legacy_knn_runs(*args, **kwargs):
        raise AssertionError("TL-IterMSSM must not run flat logical KNN")

    monkeypatch.setattr(
        detector_module, "_compute_anomaly_score", fail_if_legacy_knn_runs
    )
    texture_index = _NumpyFlatL2()
    monkeypatch.setattr(
        detector_module,
        "_build_faiss_index",
        lambda *args, **kwargs: (texture_index, None),
    )
    detector = detector_module.ReRemAnomalyDetector(
        model=object(),
        object_name="breakfast_box",
        data_root=str(tmp_path),
        features_dir=str(tmp_path / "features"),
        feature_fuse=True,
        feature_list=[6, 9],
        scales=[1, 5],
        object_anomalies={"breakfast_box": ["logical_anomalies"]},
        knn_metric="L2_normalized",
        knn_neighbors=1,
        faiss_on_cpu=True,
        rotation=False,
        label_agnostic_ties=True,
        loco_map_mode="tl_iter",
        dpfe_logical_block=12,
        tlfme_texture_coreset_size=2,
    )
    detector.init_reference_memory(
        ["good/000.png", "good/001.png", "good/002.png"]
    )
    detector.results = {"all": [], "img_ref_samples": [
        "good/000.png",
        "good/001.png",
        "good/002.png",
    ]}

    _, results = detector.run_inference(
        overall_output_dir=str(tmp_path / "run"),
        save_patch_dists=False,
        save_tiffs=True,
    )

    assert detector.class_index is None
    assert len(detector.knn_index.features) == 6
    assert all(
        item["texture_patch_count"] == 2
        for item in detector.tlfme_memory.manifest()
    )
    assert len(detector.dpfe_memory.logical_grids) == 3
    assert all(
        {"path", "score", "texture_score", "logical_score"}
        <= set(entry)
        for entry in results["all"]
    )
    assert detector.last_inference_diagnostics["map_mode"] == "tl_iter"
    assert detector.last_inference_diagnostics["algorithm"].startswith(
        "tl_iter_mssm_"
    )
