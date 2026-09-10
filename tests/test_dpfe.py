from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.dpfe import DPFEMemoryBank
from src.dual_branch_features import (
    DualBranchFeatures,
    extract_dual_branch_features,
)


def _nonzero_layers() -> dict[int, torch.Tensor]:
    return {
        6: torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]]
        ),
        9: torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0], [1.0, -1.0], [0.0, 2.0]]]
        ),
        11: torch.tensor(
            [[[3.0, 4.0], [5.0, 12.0], [8.0, 15.0], [7.0, 24.0]]]
        ),
    }


class _FakeDPFEModel:
    def __init__(self) -> None:
        self.model = type("Backbone", (), {"blocks": [object()] * 12})()
        self.requested_layers = None
        self.received_scales = None
        self.layers = _nonzero_layers()

    def prepare_image(self, image):
        return image, (2, 2)

    def extract_features(self, image_tensor, feature_list):
        self.requested_layers = list(feature_list)
        return [self.layers[index] for index in feature_list]

    def MLMP(self, features, grid_size, scales):
        assert grid_size == (2, 2)
        self.received_scales = list(scales)
        return sum(feature.squeeze(0).numpy() for feature in features)


def test_dpfe_extracts_texture_blocks_7_10_and_logical_block_12() -> None:
    model = _FakeDPFEModel()

    features = extract_dual_branch_features(
        model,
        image_tensor="image",
        grid_size=(2, 2),
        texture_layer_indices=(6, 9),
        logical_block=12,
        scales=(1, 5),
    )

    assert model.requested_layers == [6, 9, 11]
    assert model.received_scales == [1, 5]
    assert features.texture.shape == (4, 2)
    assert features.logical.shape == (4, 2)
    assert features.logical_grid.shape == (2, 2, 2)
    np.testing.assert_allclose(
        np.linalg.norm(features.texture, axis=1), np.ones(4), atol=1e-6
    )
    np.testing.assert_allclose(
        np.linalg.norm(features.logical, axis=1), np.ones(4), atol=1e-6
    )
    np.testing.assert_allclose(features.logical[0], [0.6, 0.8])


def test_dpfe_memory_preserves_image_groups_positions_and_full_grids() -> None:
    features = DualBranchFeatures(
        texture=np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            dtype=np.float32,
        ),
        logical=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        grid_size=(2, 2),
    )
    memory = DPFEMemoryBank()

    memory.add(features, image_id="good/000.png", round_id=1)
    memory.add(features, image_id="good/001.png", round_id=2)

    assert len(memory) == 2
    assert memory.texture_features.shape == (8, 2)
    assert [grid.shape for grid in memory.logical_grids] == [
        (2, 2, 3),
        (2, 2, 3),
    ]
    assert memory.manifest()[1] == {
        "image_id": "good/001.png",
        "round_id": 2,
        "augmentation_id": 0,
        "grid_size": [2, 2],
        "texture_patch_count": 4,
        "texture_source_patch_count": 4,
        "texture_storage": "spatial_feature_kcenter",
        "logical_slot": 1,
        "logical_patch_count": 4,
        "logical_storage": "full_grid",
    }
    with pytest.raises(ValueError, match="already contains"):
        memory.add(features, image_id="good/001.png", round_id=3)


def test_tlfme_compresses_texture_but_preserves_the_complete_logical_grid(
) -> None:
    patch_count = 16
    texture = np.eye(patch_count, dtype=np.float32)
    logical = np.roll(texture, shift=1, axis=1)
    features = DualBranchFeatures(
        texture=texture,
        logical=logical,
        grid_size=(4, 4),
    )
    memory = DPFEMemoryBank(
        texture_coreset_size=4,
        texture_spatial_weight=0.25,
    )

    record = memory.add(features, image_id="good/000.png", round_id=1)

    assert memory.texture_features.shape == (4, patch_count)
    assert memory.logical_grids[0].shape == (4, 4, patch_count)
    assert len(np.unique(memory.texture_features, axis=0)) == 4
    assert record.texture_source_count == patch_count
    assert memory.manifest()[0]["texture_patch_count"] == 4
    assert memory.manifest()[0]["logical_patch_count"] == patch_count


@pytest.mark.parametrize(
    "texture, logical, message",
    [
        (
            np.ones((3, 2), dtype=np.float32),
            np.ones((4, 2), dtype=np.float32),
            "texture patch count",
        ),
        (
            np.ones((4, 2), dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
            "zero-norm",
        ),
        (
            np.full((4, 2), np.nan, dtype=np.float32),
            np.ones((4, 2), dtype=np.float32),
            "NaN or infinity",
        ),
    ],
)
def test_dpfe_rejects_invalid_full_grid_features(
    texture: np.ndarray,
    logical: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DualBranchFeatures(texture=texture, logical=logical, grid_size=(2, 2))


def test_dpfe_cache_is_bound_to_layers_scales_and_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detector_module = importlib.import_module("src.ReRem_detection_test")
    model = _FakeDPFEModel()
    cache_dir = tmp_path / "cache"

    first = detector_module._load_or_extract_features(
        model,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        feature_path=str(cache_dir),
        feature_list=[6, 9],
        feature_fuse=True,
        scales=[1, 5],
        dpfe_logical_block=12,
    )
    assert model.requested_layers == [6, 9, 11]
    assert (cache_dir / "dual_branch_features.pt").is_file()
    assert not (cache_dir / "feature_fuse.pt").exists()
    assert len(first) == 3

    def fail_if_extracted(*args, **kwargs):
        raise AssertionError("valid DPFE cache should have been reused")

    monkeypatch.setattr(model, "extract_features", fail_if_extracted)
    second = detector_module._load_or_extract_features(
        model,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        feature_path=str(cache_dir),
        feature_list=[6, 9],
        feature_fuse=True,
        scales=[1, 5],
        dpfe_logical_block=12,
    )
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert first[2] == second[2] == (2, 2)

    with pytest.raises(RuntimeError, match="metadata does not match"):
        detector_module._load_or_extract_features(
            model,
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            feature_path=str(cache_dir),
            feature_list=[6, 9],
            feature_fuse=True,
            scales=[1, 7],
            dpfe_logical_block=12,
        )


def test_detector_dpfe_update_adds_every_patch_from_selected_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detector_module = importlib.import_module("src.ReRem_detection_test")
    for index in range(3):
        path = (
            tmp_path
            / "breakfast_box"
            / "test"
            / "good"
            / f"{index:03d}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(index, index, index)).save(path)

    texture = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    logical = np.array(
        [[3.0, 4.0], [5.0, 12.0], [8.0, 15.0], [7.0, 24.0]],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        detector_module,
        "_load_or_extract_features",
        lambda *args, **kwargs: (texture.copy(), logical.copy(), (2, 2)),
    )

    def fail_if_scored(*args, **kwargs):
        raise AssertionError("DPFE memory updates must not perform scoring")

    monkeypatch.setattr(
        detector_module, "_compute_anomaly_score", fail_if_scored
    )

    class RecordingIndex:
        def __init__(self) -> None:
            self.added = []

        def add(self, features) -> None:
            self.added.append(np.asarray(features).copy())

    texture_index = RecordingIndex()
    logical_index = RecordingIndex()
    monkeypatch.setattr(
        detector_module,
        "_build_faiss_index",
        lambda *args, **kwargs: (texture_index, logical_index),
    )
    monkeypatch.setattr(
        detector_module.faiss, "normalize_L2", lambda features: None
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
        dpfe_logical_block=12,
    )
    detector.init_reference_memory(["good/000.png"])
    detector.add_selected_reference_samples(
        ["good/001.png", "good/002.png"], round_id=2
    )

    assert not hasattr(detector, "add_reference_samples_greedy")
    assert detector.features_ref_len == 12
    assert [len(batch) for batch in texture_index.added] == [4, 8]
    assert [len(batch) for batch in logical_index.added] == [4, 8]
    assert [record.round_id for record in detector.dpfe_memory.records] == [
        1,
        2,
        2,
    ]
    assert len(detector.dpfe_memory.logical_grids) == 3
