import os
import json
from pathlib import Path
from src.ReRem_detection_test import ReRemAnomalyDetector
from src.dataset_info import MVTec_LOCO_SCOPE, MVTec_LOCO_TEST_TYPES, sha256_file
from src.dpfe import (
    DEFAULT_TLFME_TEXTURE_CORESET_SIZE,
    DEFAULT_TLFME_TEXTURE_SPATIAL_WEIGHT,
)
from src.tl_iter_mssm import (
    TL_ITER_ANOMALY_SCORE_VERSION,
    TL_ITER_INITIAL_K,
    TL_ITER_IMAGE_PATCH_POOL_FRACTION,
    TL_ITER_MAP_MODE,
    TL_ITER_MSSM_ALGORITHM,
    TL_ITER_SELECTION_POLICY,
    TL_ITER_TEXTURE_SIMILARITY_RANK_FRACTION,
    TL_ITER_TOP_K,
    memory_budget,
    select_lowest_score_top_k,
)


STRICT_DATASETS = {"MPDD", "MVTecLOCO"}


def _loco_scoring_settings(args):
    map_mode = getattr(args, "loco_map_mode", None) or TL_ITER_MAP_MODE
    fusion_alpha = float(getattr(args, "loco_fusion_alpha", 0.5))
    return map_mode, fusion_alpha


def _loco_dpfe_logical_block(args):
    return int(getattr(args, "dpfe_logical_block", 12))


def _loco_tlfme_texture_coreset_size(args):
    return int(
        getattr(
            args,
            "tlfme_texture_coreset_size",
            DEFAULT_TLFME_TEXTURE_CORESET_SIZE,
        )
    )


def _loco_tlfme_texture_spatial_weight(args):
    return float(
        getattr(
            args,
            "tlfme_texture_spatial_weight",
            DEFAULT_TLFME_TEXTURE_SPATIAL_WEIGHT,
        )
    )


def _loco_tl_iter_metadata(args):
    return {
        "algorithm": TL_ITER_MSSM_ALGORITHM,
        "anomaly_score_version": TL_ITER_ANOMALY_SCORE_VERSION,
        "memory_capacity_policy": "floor_sample_count_times_tau",
        "memory_capacity_fraction": args.tau,
        "initial_k": TL_ITER_INITIAL_K,
        "top_k": TL_ITER_TOP_K,
        "selection_policy": TL_ITER_SELECTION_POLICY,
        "stop_conditions": ["memory_fraction_reached"],
        "texture_similarity_rank_fraction": (
            TL_ITER_TEXTURE_SIMILARITY_RANK_FRACTION
        ),
        "branch_image_score_pooling": "top_1_percent_patch_mean",
        "logical_template_selection": (
            "minimum_template_top_1_percent_patch_mean"
        ),
        "logical_matching": "appearance_only_global_bidirectional",
        "uses_position_constraint": False,
        "image_patch_pool_fraction": TL_ITER_IMAGE_PATCH_POOL_FRACTION,
    }


def _atomic_write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(temporary_path, output_path)


def _audit_reference_types(reference_samples):
    counts = {
        "good": 0,
        "logical_anomalies": 0,
        "unexpected": 0,
    }
    for relative_path in reference_samples:
        test_type = Path(relative_path).parts[0] if relative_path else ""
        if test_type in counts and test_type != "unexpected":
            counts[test_type] += 1
        else:
            counts["unexpected"] += 1
    return counts


def _write_or_validate_loco_feature_cache_metadata(
    features_dir,
    *,
    args,
    data_root,
    rotation,
):
    """Prevent LOCO feature reuse across incompatible run inputs."""

    features_root = Path(features_dir)
    metadata_path = features_root / "cache.meta.json"
    settings = {
        "dataset": "MVTecLOCO",
        "experiment_scope": MVTec_LOCO_SCOPE,
        "test_types": list(MVTec_LOCO_TEST_TYPES),
        "data_root": str(Path(data_root).expanduser().resolve()),
        "model_name": args.model_name,
        "resolution": args.resolution,
        "feature_list": list(args.feature_list),
        "scales": list(args.scales),
        "dpfe_texture_blocks": [index + 1 for index in args.feature_list],
        "dpfe_logical_block": _loco_dpfe_logical_block(args),
        "dpfe_update_policy": "selected_top_k_full_grid",
        "rotation": bool(rotation),
        "full_grid_features": True,
        "dual_branch": True,
    }
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing != settings:
            raise RuntimeError(
                "MVTecLOCO feature cache metadata does not match this run: "
                f"{metadata_path}"
            )
        return

    if features_root.is_dir() and any(
        item.is_file() for item in features_root.rglob("*")
    ):
        raise RuntimeError(
            "MVTecLOCO feature cache contains artifacts but no metadata: "
            f"{features_root}. Move it aside before running this "
            "configuration."
        )
    features_root.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, metadata_path)


def _write_or_validate_loco_results_metadata(
    results_dir,
    *,
    args,
    data_root,
    json_path,
    rotation,
):
    """Bind resumable LOCO iteration results to all scoring inputs."""

    results_root = Path(results_dir)
    metadata_path = results_root / "run.meta.json"
    map_mode, fusion_alpha = _loco_scoring_settings(args)
    settings = {
        "dataset": "MVTecLOCO",
        "experiment_scope": MVTec_LOCO_SCOPE,
        "test_types": list(MVTec_LOCO_TEST_TYPES),
        "data_root": str(Path(data_root).expanduser().resolve()),
        "initial_json": str(Path(json_path).expanduser().resolve()),
        "initial_json_sha256": sha256_file(json_path),
        "model_name": args.model_name,
        "resolution": args.resolution,
        "K": args.K,
        "tau": args.tau,
        "feature_list": list(args.feature_list),
        "scales": list(args.scales),
        "knn_metric": args.knn_metric,
        "k_neighbors": args.k_neighbors,
        "full_grid_features": True,
        "rotation": bool(rotation),
        "loco_map_mode": map_mode,
        "loco_fusion_alpha": (
            fusion_alpha if map_mode == "fused" else None
        ),
        "loco_dual_branch": True,
        "dpfe_texture_blocks": [index + 1 for index in args.feature_list],
        "dpfe_logical_block": _loco_dpfe_logical_block(args),
        "dpfe_update_policy": "selected_top_k_full_grid",
        "tlfme_texture_blocks": [index + 1 for index in args.feature_list],
        "tlfme_logical_block": _loco_dpfe_logical_block(args),
        "tlfme_update_policy": (
            "texture_coreset_logical_full_grid"
            if map_mode == TL_ITER_MAP_MODE
            else "selected_top_k_full_grid"
        ),
        "tlfme_texture_coreset_size": (
            _loco_tlfme_texture_coreset_size(args)
            if map_mode == TL_ITER_MAP_MODE
            else None
        ),
        "tlfme_texture_spatial_weight": (
            _loco_tlfme_texture_spatial_weight(args)
            if map_mode == TL_ITER_MAP_MODE
            else None
        ),
    }
    if map_mode == TL_ITER_MAP_MODE:
        settings["tl_iter_mssm"] = _loco_tl_iter_metadata(args)
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing != settings:
            raise RuntimeError(
                "MVTecLOCO iteration metadata does not match this run: "
                f"{metadata_path}"
            )
        return

    # The final-evaluation directory has its own run.meta.json and is created
    # before the adaptive loop starts. It is safe to create the iteration
    # metadata alongside it, but cnt=* artifacts without iteration metadata
    # must still be rejected to prevent incompatible resume state.
    if results_root.is_dir() and any(
        item.is_file()
        and item.relative_to(results_root).parts[0] != "final_eval"
        for item in results_root.rglob("*")
    ):
        raise RuntimeError(
            "MVTecLOCO iteration directory contains artifacts but no "
            f"metadata: {results_root}. Move it aside before running this "
            "configuration."
        )
    results_root.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, metadata_path)

def run_per_object_adaptive_loop(
    model,
    args,
    Config,
    objects,
    overall_output_dir,
    object_anomalies,
    rotation_default,
    stop_condition,
):
    loco_map_mode = None
    loco_fusion_alpha = 0.5
    if args.dataset == "MVTecLOCO":
        loco_map_mode, loco_fusion_alpha = _loco_scoring_settings(args)

    run_namespace = args.model_name
    if args.dataset in STRICT_DATASETS:
        run_namespace = os.path.join(
            args.model_name, f"resolution={args.resolution}"
        )

    experiment_namespace = None
    if args.dataset in STRICT_DATASETS:
        layers = "-".join(map(str, args.feature_list))
        scales = "-".join(map(str, args.scales))
        experiment_namespace = (
            f"tau={args.tau:g}_layers={layers}_"
            f"scales={scales}_knn={args.knn_metric}-{args.k_neighbors}_fullgrid=1"
        )
        mssm_ranking_layers = getattr(args, "mssm_ranking_layers", None)
        if mssm_ranking_layers is not None:
            experiment_namespace += (
                f"_mssm_layers={'-'.join(map(str, mssm_ranking_layers))}"
            )
        mssm_ranking_tag = getattr(args, "mssm_ranking_tag", None)
        if mssm_ranking_tag is not None:
            experiment_namespace += f"_mssm_rank={mssm_ranking_tag}"
        if args.dataset == "MVTecLOCO":
            experiment_namespace += (
                f"_scope={MVTec_LOCO_SCOPE}_rotation={int(rotation_default[objects[0]])}_"
                "tlfme=tex"
                f"{'-'.join(str(index + 1) for index in args.feature_list)}-"
                f"log{_loco_dpfe_logical_block(args)}_fullgrid_"
                f"tcore={_loco_tlfme_texture_coreset_size(args)}_"
                f"tsp={_loco_tlfme_texture_spatial_weight(args):g}_"
                f"map={loco_map_mode}"
            )
            if loco_map_mode == TL_ITER_MAP_MODE:
                experiment_namespace += (
                    "_s=t3_m=tau_x=t10-gbi-pm1-v1_e=img_nopos"
                )
            elif loco_map_mode == "fused":
                experiment_namespace += f"_alpha={loco_fusion_alpha:g}"

    results_dataset_dir = (
        f"results_{args.dataset}"
        if args.dataset == "MVTecLOCO"
        else args.dataset
    )
    results_dir_base = os.path.join(
        str(Path(Config.RESULTS_ROOT).expanduser().resolve()),
        results_dataset_dir,
        run_namespace,
        "n_jicheng",
        f"K={args.K}",
    )
    if experiment_namespace is not None:
        results_dir_base = os.path.join(
            results_dir_base, experiment_namespace
        )

    features_dir = os.path.join(
        Config.FEATURE_ROOT,
        args.dataset,
        run_namespace,
    )
    if args.dataset in STRICT_DATASETS:
        features_dir = os.path.join(
            features_dir,
            f"layers={'-'.join(map(str, args.feature_list))}_"
            f"scales={'-'.join(map(str, args.scales))}",
        )
    if args.dataset == "MVTecLOCO":
        features_dir = os.path.join(
            features_dir,
            MVTec_LOCO_SCOPE,
            f"rotation={int(rotation_default[objects[0]])}_"
            "fullgrid=1_dualbranch=1_"
            f"logicalblock={_loco_dpfe_logical_block(args)}",
        )

    data_root = getattr(args, "data_root", None) or Config.ROOTS[args.dataset]
    json_path = (
        getattr(args, "initial_json", None)
        or Config.JSON_STARTS[args.dataset]
    )
    if args.dataset == "MVTecLOCO":
        rotation_values = {
            rotation_default[object_name] for object_name in objects
        }
        if len(rotation_values) != 1:
            raise ValueError(
                "MVTecLOCO requires one consistent rotation policy per run"
            )
        _write_or_validate_loco_feature_cache_metadata(
            features_dir,
            args=args,
            data_root=data_root,
            rotation=rotation_values.pop(),
        )

    os.makedirs(results_dir_base, exist_ok=True)
    if args.dataset == "MVTecLOCO":
        _write_or_validate_loco_results_metadata(
            results_dir_base,
            args=args,
            data_root=data_root,
            json_path=json_path,
            rotation=rotation_default[objects[0]],
        )

    with open(json_path, "r", encoding="utf-8") as file:
        sort_path_all = json.load(file)

    checkpoint_path = os.path.join(
        overall_output_dir, "final_object_results.json"
    )
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as file:
                final_results = json.load(file)
            print(f"[Resume] 已完成类别: {list(final_results.keys())}")
        except Exception as error:
            print(f"[Resume] 断点读取失败: {error}")
            final_results = {}
    else:
        final_results = {}

    for object_name in objects:
        if object_name in final_results:
            print(f"[Resume] {object_name} 已完成，跳过。")
            continue

        current_cnt = 1
        sample_nums = len(sort_path_all[object_name])
        tl_iter_active = (
            args.dataset == "MVTecLOCO"
            and loco_map_mode == TL_ITER_MAP_MODE
        )
        object_memory_budget = None
        if tl_iter_active:
            object_memory_budget = memory_budget(sample_nums, args.tau)
            minimum_initial_images = TL_ITER_INITIAL_K
            if object_memory_budget < minimum_initial_images:
                raise ValueError(
                    "The TL-IterMSSM memory fraction leaves fewer "
                    f"than {minimum_initial_images} initial images for "
                    f"{object_name}: samples={sample_nums}, "
                    f"budget={object_memory_budget}"
                )
        # Full-grid extraction retains both feature branches. TL-IterMSSM uses
        # a texture index plus structured per-image logical grids; legacy
        # ablations may still build two flat indexes.
        effective_feature_fuse = True
        detector = ReRemAnomalyDetector(
            model=model,
            object_name=object_name,
            data_root=data_root,
            features_dir=features_dir,
            feature_fuse=effective_feature_fuse,
            feature_list=args.feature_list,
            scales=args.scales,
            object_anomalies=object_anomalies,
            knn_metric=args.knn_metric,
            knn_neighbors=args.k_neighbors,
            faiss_on_cpu=args.faiss_on_cpu,
            rotation=rotation_default[object_name],
            label_agnostic_ties=args.dataset == "MVTecLOCO",
            loco_map_mode=loco_map_mode,
            loco_fusion_alpha=loco_fusion_alpha,
            dpfe_logical_block=(
                _loco_dpfe_logical_block(args)
                if args.dataset == "MVTecLOCO"
                else None
            ),
            tlfme_texture_coreset_size=(
                _loco_tlfme_texture_coreset_size(args)
                if args.dataset == "MVTecLOCO"
                else DEFAULT_TLFME_TEXTURE_CORESET_SIZE
            ),
            tlfme_texture_spatial_weight=(
                _loco_tlfme_texture_spatial_weight(args)
                if args.dataset == "MVTecLOCO"
                else DEFAULT_TLFME_TEXTURE_SPATIAL_WEIGHT
            ),
        )

        while True:
            print(f"\n Iteration: {current_cnt}")
            results_dir = f"{results_dir_base}/cnt={current_cnt}"
            os.makedirs(results_dir, exist_ok=True)
            detector.results = {"all": []}
            first_flag = current_cnt == 1

            if first_flag:
                initial_count = (
                    min(TL_ITER_INITIAL_K, object_memory_budget)
                    if tl_iter_active
                    else args.K
                )
                img_ref_samples = [
                    item["path"]
                    for item in sort_path_all[object_name][
                        :initial_count
                    ]
                ]
                new_reference_samples = list(img_ref_samples)
                detector.init_reference_memory(img_ref_samples)
            else:
                previous_json = (
                    f"{results_dir_base}/cnt={current_cnt - 1}/results.json"
                )
                with open(previous_json, "r", encoding="utf-8") as file:
                    previous_results = json.load(file)[object_name]

                img_ref_samples = list(
                    previous_results["img_ref_samples"]
                )
                existing_reference_samples = set(img_ref_samples)
                if tl_iter_active:
                    remaining_slots = max(
                        0,
                        object_memory_budget - len(img_ref_samples),
                    )
                    new_reference_samples = select_lowest_score_top_k(
                        previous_results["all"],
                        existing_image_ids=existing_reference_samples,
                        remaining_slots=remaining_slots,
                    )
                    if not new_reference_samples:
                        raise RuntimeError(
                            "TL-IterMSSM resumed another iteration without "
                            "an unused scored candidate or free memory slot"
                        )
                else:
                    new_reference_samples = [
                        item["path"]
                        for item in previous_results["all"]
                        if item["path"] not in existing_reference_samples
                    ][: args.K]
                img_ref_samples.extend(new_reference_samples)
                detector.results["img_ref_samples"] = list(img_ref_samples)
                detector.add_selected_reference_samples(
                    new_reference_samples,
                    round_id=current_cnt,
                )

            _, results = detector.run_inference(
                overall_output_dir=overall_output_dir,
                save_patch_dists=(
                    args.eval_clf and args.dataset != "MVTecLOCO"
                ),
                save_tiffs=(
                    args.eval_segm and args.dataset != "MVTecLOCO"
                ),
            )

            stop_memory_budget = False
            if tl_iter_active:
                stop_memory_budget = (
                    len(img_ref_samples) >= object_memory_budget
                )
                stop_flag = stop_memory_budget
            else:
                stop_flag = bool(
                    stop_condition(
                        args.tau,
                        sample_nums,
                        current_cnt,
                        args.K,
                    )
                )
            if args.dataset == "MVTecLOCO":
                tlfme_memory = getattr(
                    detector,
                    "tlfme_memory",
                    getattr(detector, "dpfe_memory", None),
                )
                tlfme_manifest = (
                    tlfme_memory.manifest()
                    if tlfme_memory is not None
                    else []
                )
                diagnostic_payload = {
                    "dataset": "MVTecLOCO",
                    "experiment_scope": MVTec_LOCO_SCOPE,
                    "test_types": list(MVTec_LOCO_TEST_TYPES),
                    "object": object_name,
                    "iteration": current_cnt,
                    "selection_labels_used": False,
                    "feature_extraction": {
                        "full_grid": True,
                        "dual_branch": bool(effective_feature_fuse),
                        "texture_blocks": [
                            index + 1 for index in args.feature_list
                        ],
                        "logical_block": _loco_dpfe_logical_block(args),
                    },
                    "memory": {
                        "reference_split": "test",
                        "frozen": False,
                        "reference_count": len(img_ref_samples),
                        "reference_samples": list(img_ref_samples),
                        "new_reference_samples": list(
                            new_reference_samples
                        ),
                        "new_reference_count": len(
                            new_reference_samples
                        ),
                        "capacity": object_memory_budget,
                        "capacity_fraction": args.tau,
                        "update_policy": (
                            "texture_coreset_logical_full_grid"
                            if tl_iter_active
                            else "selected_top_k_full_grid"
                        ),
                        "texture_coreset_size": (
                            _loco_tlfme_texture_coreset_size(args)
                            if tl_iter_active
                            else None
                        ),
                        "texture_spatial_weight": (
                            _loco_tlfme_texture_spatial_weight(args)
                            if tl_iter_active
                            else None
                        ),
                        "logical_storage": (
                            "full_grid" if tl_iter_active else None
                        ),
                        "structured_record_count": len(tlfme_manifest),
                        "structured_records": tlfme_manifest,
                        "audit_only_reference_type_counts": (
                            _audit_reference_types(img_ref_samples)
                        ),
                        "audit_only_new_reference_type_counts": (
                            _audit_reference_types(
                                new_reference_samples
                            )
                        ),
                    },
                    "inference": getattr(
                        detector, "last_inference_diagnostics", None
                    ),
                    "stop_after_iteration": stop_flag,
                    "stop_reasons": {
                        "memory_fraction_reached": (
                            stop_memory_budget
                        ),
                    },
                }
                diagnostics_path = (
                    Path(overall_output_dir)
                    / "diagnostics"
                    / object_name
                    / f"cnt={current_cnt}.json"
                )
                _atomic_write_json(
                    diagnostics_path, diagnostic_payload
                )

            json_output_path = f"{results_dir}/results.json"
            if os.path.exists(json_output_path):
                with open(
                    json_output_path, "r", encoding="utf-8"
                ) as file:
                    existing_data = json.load(file)
            else:
                existing_data = {}
            existing_data[object_name] = results
            with open(
                json_output_path, "w", encoding="utf-8"
            ) as file:
                json.dump(existing_data, file, indent=4)

            if stop_flag:
                final_results[object_name] = results
                temporary_checkpoint = checkpoint_path + ".tmp"
                with open(
                    temporary_checkpoint, "w", encoding="utf-8"
                ) as file:
                    json.dump(final_results, file, indent=4)
                os.replace(temporary_checkpoint, checkpoint_path)
                print(f"[Checkpoint] {object_name} 已完成并保存。")
                break

            current_cnt += 1

    return final_results
