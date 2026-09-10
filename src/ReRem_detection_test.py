import os
import json
from pathlib import Path
import torch
import faiss
import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm
from src.utils import augment_image, dists2map
from src.post_eval import mean_top1p
from src.dataset_info import (
    sort_scored_paths_label_agnostic,
    sorted_image_paths,
)
from src.dpfe import (
    DEFAULT_TLFME_TEXTURE_CORESET_SIZE,
    DEFAULT_TLFME_TEXTURE_SPATIAL_WEIGHT,
    TLFMEMemoryBank,
)
from src.dual_branch_features import (
    DUAL_BRANCH_CACHE_FORMAT_VERSION,
    DualBranchFeatures,
    extract_dual_branch_features,
)
from src.tl_iter_mssm import (
    TL_ITER_MAP_MODE,
    TLIterMSSM,
)


LOCO_MAP_MODES = (TL_ITER_MAP_MODE, "mlmp", "deep", "fused")

def _load_or_extract_features(
    model,
    image=None,
    image_path=None,
    feature_path=None,
    feature_list=None,
    feature_fuse=False,
    scales=None,
    dpfe_logical_block=None,
):
    os.makedirs(feature_path, exist_ok=True)
    grid_json = os.path.join(feature_path, "grid_size.json")
    feature_pt = os.path.join(feature_path, "feature.pt")
    feature_fuse_pt = os.path.join(feature_path, "feature_fuse.pt")
    dual_branch_pt = os.path.join(feature_path, "dual_branch_features.pt")
    if image is None and image_path is not None:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"OpenCV could not read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if dpfe_logical_block is not None and not feature_fuse:
        raise ValueError("TLFME logical features require the two-branch path")

    cache_pt = (
        dual_branch_pt
        if dpfe_logical_block is not None
        else feature_fuse_pt if feature_fuse else feature_pt
    )

    if os.path.exists(cache_pt):
        try:
            torch.load(cache_pt, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"[WARNING] 损坏的特征缓存: {cache_pt}")
            print(f"[WARNING] 原因: {e}")
            print("[WARNING] 删除该缓存并重新提取...")
            os.remove(cache_pt)

    need_extract = not os.path.exists(grid_json) or not os.path.exists(cache_pt)

    if need_extract:
        image_tensor, grid_size = model.prepare_image(image)
        with open(grid_json, "w") as f:
            json.dump({"grid_height": grid_size[0], "grid_width": grid_size[1]}, f)

        if dpfe_logical_block is not None:
            dual_branch_features = extract_dual_branch_features(
                model,
                image_tensor,
                grid_size,
                texture_layer_indices=feature_list,
                logical_block=dpfe_logical_block,
                scales=scales or [1, 5],
            )
            torch.save(
                {
                    "format_version": DUAL_BRANCH_CACHE_FORMAT_VERSION,
                    "texture_layer_indices": list(feature_list),
                    "logical_block": int(dpfe_logical_block),
                    "scales": list(scales or [1, 5]),
                    "grid_size": list(grid_size),
                    "texture": torch.from_numpy(
                        dual_branch_features.texture
                    ),
                    "logical": torch.from_numpy(
                        dual_branch_features.logical
                    ),
                },
                dual_branch_pt,
            )
            features = dual_branch_features.texture
            feature_origin = dual_branch_features.logical
        elif feature_fuse:
            features = model.extract_features(image_tensor, feature_list)
            features = [feature.detach().cpu() for feature in features]
            torch.save(features, feature_fuse_pt)
            feature_origin = features[-1].squeeze().cpu().numpy()
            features = model.MLMP(
                features,
                grid_size=grid_size,
                scales=scales or [1, 5],
            )
        else:
            features = model.extract_features(image_tensor)
            torch.save(features, feature_pt)
            feature_origin = features
    else:
        with open(grid_json, "r") as f:
            g = json.load(f)
        grid_size = (g["grid_height"], g["grid_width"])
        if dpfe_logical_block is not None:
            payload = torch.load(
                dual_branch_pt,
                weights_only=False,
                map_location="cpu",
            )
            expected = {
                "format_version": DUAL_BRANCH_CACHE_FORMAT_VERSION,
                "texture_layer_indices": list(feature_list),
                "logical_block": int(dpfe_logical_block),
                "scales": list(scales or [1, 5]),
                "grid_size": list(grid_size),
            }
            if not isinstance(payload, dict) or any(
                payload.get(key) != value for key, value in expected.items()
            ):
                raise RuntimeError(
                    "Dual-branch feature cache metadata does not match the current "
                    f"configuration: {dual_branch_pt}"
                )
            dual_branch_features = DualBranchFeatures(
                texture=payload.get("texture"),
                logical=payload.get("logical"),
                grid_size=grid_size,
            )
            features = dual_branch_features.texture
            feature_origin = dual_branch_features.logical
        elif feature_fuse:
            features = torch.load(
                feature_fuse_pt,
                weights_only=False,
                map_location="cpu",
            )
            feature_origin = features[-1].squeeze().cpu().numpy()
            features = model.MLMP(
                features,
                grid_size=grid_size,
                scales=scales or [1, 5],
            )
        else:
            features = torch.load(
                feature_pt,
                weights_only=False,
                map_location="cpu",
            )
            feature_origin = features

    return features, feature_origin, grid_size


def _build_faiss_index(features_ref, features_origin_ref=None, feature_fuse=False, faiss_on_cpu=False):
    features_dim = features_ref.shape[1]

    if faiss_on_cpu:
        # IDs are never used by ReMem, so a plain flat index is sufficient and
        # supports the ``add`` call used below.
        knn_index = faiss.IndexFlatL2(features_dim)
        class_index = None
        if feature_fuse and features_origin_ref is not None:
            class_index = faiss.IndexFlatL2(features_origin_ref.shape[1])
    else:
        res1 = faiss.StandardGpuResources()
        knn_index = faiss.GpuIndexFlatL2(res1, features_dim)


        class_index = None
        if feature_fuse and features_origin_ref is not None:
            res2 = faiss.StandardGpuResources()
            class_index = faiss.GpuIndexFlatL2(
                res2, features_origin_ref.shape[1]
            )


    return knn_index, class_index



def _compute_anomaly_score(features2, feature_origin, knn_index, class_index, feature_fuse, knn_neighbors=1):
    faiss.normalize_L2(features2)
    distances, _ = knn_index.search(features2, k=knn_neighbors)
    if knn_neighbors > 1:
        distances = distances.mean(axis=1)
    distances = distances / 2

    if feature_fuse and feature_origin is not None:
        faiss.normalize_L2(feature_origin)
        distances2, _ = class_index.search(feature_origin, k=knn_neighbors)
        if knn_neighbors > 1:
            distances2 = distances2.mean(axis=1)
        distances2 = distances2 / 2
        return distances, distances2

    return distances, None


def _select_loco_patch_distances(
    distances_mlmp,
    distances_deep,
    map_mode,
    fusion_alpha,
):
    """Select the one patch-score tensor used by all LOCO consumers."""

    if map_mode not in LOCO_MAP_MODES:
        raise ValueError(
            f"Unknown MVTecLOCO map mode {map_mode!r}; "
            f"expected one of {LOCO_MAP_MODES}"
        )
    if not np.isfinite(fusion_alpha) or not 0 <= fusion_alpha <= 1:
        raise ValueError("LOCO fusion alpha must be finite and in [0, 1]")
    if map_mode == TL_ITER_MAP_MODE:
        raise RuntimeError(
            "TL-IterMSSM supplies its final patch map directly; it cannot be "
            "selected from legacy flat-KNN branch distances"
        )

    mlmp = np.asarray(distances_mlmp, dtype=np.float32)
    if not np.isfinite(mlmp).all():
        raise RuntimeError("MLMP patch distances contain NaN or infinity")
    if map_mode == "mlmp":
        return mlmp

    if distances_deep is None:
        raise RuntimeError(
            f"MVTecLOCO map mode {map_mode!r} requires the deep branch"
        )
    deep = np.asarray(distances_deep, dtype=np.float32)
    if deep.shape != mlmp.shape:
        raise RuntimeError(
            "MVTecLOCO branch distance shapes do not match: "
            f"MLMP={mlmp.shape}, deep={deep.shape}"
        )
    if not np.isfinite(deep).all():
        raise RuntimeError("Deep patch distances contain NaN or infinity")
    if map_mode == "deep":
        return deep

    return (
        fusion_alpha * mlmp + (1.0 - fusion_alpha) * deep
    ).astype(np.float32, copy=False)


def _full_anomaly_map(patch_map, image_shape):
    full_map = np.asarray(
        dists2map(np.asarray(patch_map), image_shape), dtype=np.float32
    )
    expected_shape = tuple(image_shape[:2])
    if full_map.ndim != 2 or full_map.shape != expected_shape:
        raise RuntimeError(
            f"Invalid anomaly-map shape: expected {expected_shape}, "
            f"got {full_map.shape}"
        )
    if not np.isfinite(full_map).all():
        raise RuntimeError("Anomaly map contains NaN or infinity")
    return full_map


def _summarize_scores(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Cannot summarize empty or non-vector scores")
    if not np.isfinite(values).all():
        raise RuntimeError("Diagnostic scores contain NaN or infinity")
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _save_anomaly_map(patch_distances, deep_patch_distances, image_shape, plots_dir,
                      object_name, type_anomaly, img_name,
                      save_patch_dists=True, save_tiffs=False,
                      feature_fuse=False, loco_map_mode=None,
                      loco_fusion_alpha=0.5):

    base = os.path.join(plots_dir, "Anomaly_maps", object_name, "test", type_anomaly)
    os.makedirs(base, exist_ok=True)
    img_base = os.path.splitext(img_name)[0]

    if save_patch_dists:
        np.save(os.path.join(base, f"{img_base}.npy"), patch_distances)
        if feature_fuse and deep_patch_distances is not None:
            np.save(os.path.join(base, f"{img_base}_class.npy"), deep_patch_distances)

    selected_patch_map = patch_distances
    if loco_map_mode is not None:
        selected_patch_map = _select_loco_patch_distances(
            patch_distances,
            deep_patch_distances,
            loco_map_mode,
            loco_fusion_alpha,
        )
    try:
        full_map = _full_anomaly_map(selected_patch_map, image_shape)
    except RuntimeError as error:
        raise RuntimeError(
            f"{error} for {object_name}/{type_anomaly}/{img_name}"
        ) from error

    if save_tiffs:
        tiff.imwrite(os.path.join(base, f"{img_base}.tiff"), full_map)

    return full_map

class ReRemAnomalyDetector:
    def __init__(
        self,
        model,
        object_name,
        data_root,
        features_dir,
        feature_fuse,
        feature_list,
        scales,
        object_anomalies,
        knn_metric,
        knn_neighbors,
        faiss_on_cpu,
        rotation,
        label_agnostic_ties=False,
        loco_map_mode=None,
        loco_fusion_alpha=0.5,
        dpfe_logical_block=None,
        tlfme_texture_coreset_size=DEFAULT_TLFME_TEXTURE_CORESET_SIZE,
        tlfme_texture_spatial_weight=DEFAULT_TLFME_TEXTURE_SPATIAL_WEIGHT,
    ):
        self.model = model
        self.object_name = object_name
        self.object_anomalies = object_anomalies[object_name] + ['good']
        self.object_anomalies = list(dict.fromkeys(self.object_anomalies))

        self.features_dir = features_dir
        self.data_root = data_root
        self.feature_list = feature_list
        self.scales = scales
        self.feature_fuse = feature_fuse
        self.rotation = rotation
        self.faiss_on_cpu = faiss_on_cpu
        self.knn_metric = knn_metric
        self.knn_neighbors = knn_neighbors
        self.label_agnostic_ties = label_agnostic_ties
        self.loco_map_mode = loco_map_mode
        self.loco_fusion_alpha = float(loco_fusion_alpha)
        self.dpfe_logical_block = dpfe_logical_block
        self.use_tl_iter_mssm = self.loco_map_mode == TL_ITER_MAP_MODE
        if self.dpfe_logical_block is not None:
            self.dpfe_logical_block = int(self.dpfe_logical_block)
            if self.dpfe_logical_block <= 0:
                raise ValueError("TLFME logical block must be positive")
            if not self.feature_fuse:
                raise ValueError(
                    "TLFME logical features require the two-branch path"
                )
            if self.rotation:
                raise ValueError(
                    "TLFME logical memory does not support rotation augmentation"
                )
        if self.use_tl_iter_mssm and self.dpfe_logical_block is None:
            raise ValueError(
                "TL-IterMSSM requires the structured TLFME logical branch"
            )
        if self.loco_map_mode is not None:
            if self.loco_map_mode not in LOCO_MAP_MODES:
                raise ValueError(
                    f"Unknown MVTecLOCO map mode: {self.loco_map_mode}"
                )
            if not self.feature_fuse:
                raise ValueError(
                    "MVTecLOCO map modes require both MLMP and deep branches"
                )
            if not np.isfinite(self.loco_fusion_alpha) or not (
                0 <= self.loco_fusion_alpha <= 1
            ):
                raise ValueError(
                    "MVTecLOCO fusion alpha must be finite and in [0, 1]"
                )

        self.img_cnt = 1
        self.knn_index = None
        self.class_index = None
        self.features_ref_len = 0
        self.tlfme_memory = TLFMEMemoryBank(
            texture_coreset_size=tlfme_texture_coreset_size,
            texture_spatial_weight=tlfme_texture_spatial_weight,
        )
        # Temporary compatibility alias for existing diagnostics and callers.
        self.dpfe_memory = self.tlfme_memory
        self.tl_iter_mssm = None
        self.results = {}
        self.last_inference_diagnostics = None

    def _reference_paths(self, relative_path):
        split_root = (
            Path(self.data_root)
            .expanduser()
            .resolve()
            / self.object_name
            / "test"
        ).resolve()
        image_path = (split_root / relative_path).resolve()
        if split_root not in image_path.parents:
            raise ValueError(
                f"Reference path escapes test: {relative_path}"
            )
        safe_relative_path = image_path.relative_to(split_root)
        feature_path = (
            Path(self.features_dir)
            / self.object_name
            / "test"
            / safe_relative_path.with_suffix("")
        )
        return str(image_path), str(feature_path)

    def _extract_selected_reference_features(
        self,
        img_ref_samples,
        *,
        round_id,
    ):
        """Encode externally selected images without scoring or filtering."""

        features_ref = []
        features_origin_ref = []
        for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank"):
            ref_img_path, ref_feat_path = self._reference_paths(img_ref_n)
            image_bgr = cv2.imread(ref_img_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"OpenCV could not read image: {ref_img_path}")
            image_ref = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_augmented = augment_image(image_ref) if self.rotation else [image_ref]

            for i, img in enumerate(img_augmented):
                feat_path = os.path.join(ref_feat_path, str(i))
                features, feature_origin, grid_size = _load_or_extract_features(
                    self.model, img, None, feat_path, self.feature_list,
                    self.feature_fuse, self.scales,
                    dpfe_logical_block=self.dpfe_logical_block,
                )
                if self.dpfe_logical_block is not None:
                    dual_features = DualBranchFeatures(
                        texture=features,
                        logical=feature_origin,
                        grid_size=grid_size,
                    )
                    record = self.tlfme_memory.add(
                        dual_features,
                        image_id=img_ref_n,
                        round_id=round_id,
                        augmentation_id=i,
                    )
                    if self.use_tl_iter_mssm:
                        features_ref.append(
                            self.tlfme_memory.texture_for_record(record)
                        )
                    else:
                        features_ref.append(features)
                else:
                    features_ref.append(features)
                if self.feature_fuse and not self.use_tl_iter_mssm:
                    features_origin_ref.append(feature_origin)
            self.img_cnt += 1

        if not features_ref:
            raise ValueError("TLFME received no selected reference images")
        features_ref = np.concatenate(features_ref, axis=0).astype("float32")
        if features_origin_ref:
            features_origin_ref = np.concatenate(
                features_origin_ref, axis=0
            ).astype("float32")
        else:
            features_origin_ref = None
        return features_ref, features_origin_ref

    def _normalise_reference_features(
        self,
        features_ref,
        features_origin_ref,
    ):
        if self.knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)
            if self.feature_fuse and features_origin_ref is not None:
                faiss.normalize_L2(features_origin_ref)

    def init_reference_memory(self, img_ref_samples, *, round_id=1):
        """Build the first memory exclusively from TriCue-selected images."""

        if self.knn_index is not None or len(self.tlfme_memory):
            raise RuntimeError("Reference memory has already been initialized")
        self.results["img_ref_samples"] = list(img_ref_samples)
        features_ref, features_origin_ref = (
            self._extract_selected_reference_features(
                img_ref_samples,
                round_id=round_id,
            )
        )
        self.features_ref_len = features_ref.shape[0]

        legacy_logical_index = self.feature_fuse and not self.use_tl_iter_mssm
        self.knn_index, self.class_index = _build_faiss_index(
            features_ref,
            features_origin_ref if legacy_logical_index else None,
            feature_fuse=legacy_logical_index,
            faiss_on_cpu=self.faiss_on_cpu,
        )
        self._normalise_reference_features(
            features_ref, features_origin_ref
        )
        self.knn_index.add(features_ref)
        if legacy_logical_index:
            self.class_index.add(features_origin_ref)
        if self.use_tl_iter_mssm:
            self.tl_iter_mssm = TLIterMSSM(
                self.tlfme_memory,
                self.knn_index,
            )

    def add_selected_reference_samples(
        self,
        img_ref_samples,
        *,
        round_id,
    ):
        """Append texture prototypes and a full logical grid for each image."""

        if not img_ref_samples:
            return
        if self.knn_index is None:
            raise RuntimeError("Reference memory must be initialized first")
        features_ref, features_origin_ref = (
            self._extract_selected_reference_features(
                img_ref_samples,
                round_id=round_id,
            )
        )
        self._normalise_reference_features(
            features_ref, features_origin_ref
        )
        self.knn_index.add(features_ref)
        if (
            self.feature_fuse
            and not self.use_tl_iter_mssm
            and features_origin_ref is not None
        ):
            self.class_index.add(features_origin_ref)
        self.features_ref_len += features_ref.shape[0]


    def run_inference(self, overall_output_dir, out_samples=None, save_patch_dists=True, save_tiffs=False):
        anomaly_scores = {}
        if self.use_tl_iter_mssm:
            diagnostic_scores = {
                "texture": [],
                "logical": [],
                "selected": [],
            }
        else:
            diagnostic_scores = {
                "mlmp": [],
                "deep": [],
                "fused": [],
                "selected": [],
            }
        for type_anomaly in tqdm(self.object_anomalies, desc=f"Processing test samples ({self.object_name})"):
            test_dir = os.path.join(self.data_root, self.object_name, "test", type_anomaly)
            feat_dir = os.path.join(self.features_dir, self.object_name, "test", type_anomaly)

            for img_path_object in sorted_image_paths(test_dir):
                img_name = img_path_object.name
                if out_samples is not None:
                    temp_samples = f"{type_anomaly}/{img_name}"
                    if temp_samples in out_samples:
                        continue
                img_path = str(img_path_object)
                feat_path = os.path.join(
                    feat_dir,
                    os.path.splitext(img_name)[0],
                    "0",
                )
                image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"OpenCV could not read image: {img_path}")
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                features, feature_origin, grid_size = _load_or_extract_features(
                    self.model, image, None, feat_path, self.feature_list,
                    self.feature_fuse, self.scales,
                    dpfe_logical_block=self.dpfe_logical_block,
                )
                sample_key = f"{type_anomaly}/{img_name}"
                result_entry = {"path": sample_key}
                if self.use_tl_iter_mssm:
                    if self.tl_iter_mssm is None:
                        raise RuntimeError(
                            "TL-IterMSSM reference memory is not initialized"
                        )
                    tl_result = self.tl_iter_mssm.score(
                        DualBranchFeatures(
                            texture=features,
                            logical=feature_origin,
                            grid_size=grid_size,
                        ),
                        image_id=sample_key,
                    )
                    if save_patch_dists or save_tiffs:
                        _save_anomaly_map(
                            tl_result.final_map,
                            None,
                            image.shape,
                            overall_output_dir,
                            self.object_name,
                            type_anomaly,
                            img_name,
                            save_patch_dists,
                            save_tiffs,
                            False,
                            None,
                            self.loco_fusion_alpha,
                        )
                    score = float(tl_result.score)
                    diagnostic_scores["texture"].append(
                        float(tl_result.texture_score)
                    )
                    diagnostic_scores["logical"].append(
                        float(tl_result.logical_score)
                    )
                    diagnostic_scores["selected"].append(score)
                    result_entry.update(
                        {
                            "texture_score": float(
                                tl_result.texture_score
                            ),
                            "logical_score": float(
                                tl_result.logical_score
                            ),
                        }
                    )
                else:
                    d, d2 = _compute_anomaly_score(
                        features,
                        feature_origin,
                        self.knn_index,
                        self.class_index,
                        self.feature_fuse,
                        self.knn_neighbors,
                    )
                    patch_map = d.squeeze().reshape(grid_size)
                    deep_patch_map = None
                    if self.feature_fuse and d2 is not None:
                        deep_patch_map = d2.squeeze().reshape(grid_size)
                    full_map = _save_anomaly_map(
                        patch_map,
                        deep_patch_map,
                        image.shape,
                        overall_output_dir,
                        self.object_name,
                        type_anomaly,
                        img_name,
                        save_patch_dists,
                        save_tiffs,
                        self.feature_fuse,
                        self.loco_map_mode,
                        self.loco_fusion_alpha,
                    )
                    if self.loco_map_mode is None:
                        score = mean_top1p(
                            d2 if self.feature_fuse else d
                        )
                    else:
                        # Rank with the exact map consumed by the evaluator.
                        score = float(np.max(full_map))
                        branch_patch_maps = {
                            "mlmp": patch_map,
                            "deep": deep_patch_map,
                            "fused": _select_loco_patch_distances(
                                patch_map,
                                deep_patch_map,
                                "fused",
                                self.loco_fusion_alpha,
                            ),
                        }
                        for branch_name, branch_patch_map in (
                            branch_patch_maps.items()
                        ):
                            branch_map = _full_anomaly_map(
                                branch_patch_map, image.shape
                            )
                            diagnostic_scores[branch_name].append(
                                float(np.max(branch_map))
                            )
                        diagnostic_scores["selected"].append(score)

                anomaly_scores[sample_key] = score
                result_entry["score"] = float(score)
                self.results["all"].append(result_entry)

        if self.label_agnostic_ties:
            self.results["all"] = sort_scored_paths_label_agnostic(
                self.results["all"],
                os.path.join(self.data_root, self.object_name, "test"),
            )
        else:
            self.results["all"] = sorted(
                self.results["all"], key=lambda x: x["score"]
            )
        if self.loco_map_mode is not None:
            self.last_inference_diagnostics = {
                "map_mode": self.loco_map_mode,
                "image_score_definition": (
                    "top_1_percent_mean_of_patchwise_texture_logical_max"
                    if self.use_tl_iter_mssm
                    else "maximum_saved_tiff_pixel"
                ),
                "pixel_maps_generated": bool(save_tiffs),
                "patch_distance_arrays_saved": bool(save_patch_dists),
                "score_statistics": {
                    name: _summarize_scores(values)
                    for name, values in diagnostic_scores.items()
                },
            }
            if self.use_tl_iter_mssm:
                self.last_inference_diagnostics.update(
                    self.tl_iter_mssm.manifest()
                )
            else:
                self.last_inference_diagnostics[
                    "fusion_alpha"
                ] = self.loco_fusion_alpha
        return anomaly_scores, self.results
