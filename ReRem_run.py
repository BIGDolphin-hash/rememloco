import argparse
import math
import os
from pathlib import Path
from argparse import ArgumentParser
import json
from src.utils import get_dataset_info
from src.detection_per_object_test import run_per_object_adaptive_loop
from src.ReRem_detection_test import LOCO_MAP_MODES
from src.dpfe import (
    DEFAULT_TLFME_TEXTURE_CORESET_SIZE,
    DEFAULT_TLFME_TEXTURE_SPATIAL_WEIGHT,
)
from src.tl_iter_mssm import (
    TL_ITER_ANOMALY_SCORE_VERSION,
    TL_ITER_DEFAULT_MEMORY_FRACTION,
    TL_ITER_INITIAL_K,
    TL_ITER_IMAGE_PATCH_POOL_FRACTION,
    TL_ITER_MAP_MODE,
    TL_ITER_MSSM_ALGORITHM,
    TL_ITER_SELECTION_POLICY,
    TL_ITER_TEXTURE_SIMILARITY_RANK_FRACTION,
    TL_ITER_TOP_K,
)
from src.backbones import get_model
from src.post_eval import eval_finished_run
from config import Config
from src.dataset_info import (
    MVTec_LOCO_SCOPE,
    MVTec_LOCO_TEST_TYPES,
    get_mpdd_object_anomalies,
    sha256_file,
    validate_mvtec_loco_ranking_metadata,
    validate_mvtec_loco_rankings,
    validate_mpdd_ranking_metadata,
    validate_mpdd_rankings,
)
import warnings
warnings.filterwarnings("ignore")


RANKED_DATASETS = {"MPDD", "MVTecLOCO"}
LOCO_EXPERIMENT_OBJECTS = ("breakfast_box",)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='MVTec',
        choices=['VisA', 'MVTec', 'MPDD', 'MVTecLOCO'],
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override Config.ROOTS for the selected dataset",
    )
    parser.add_argument(
        "--initial_json",
        default=None,
        help="Override Config.JSON_STARTS for the selected dataset",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help=(
            "Optional subset of object names to process; MVTecLOCO defaults "
            "to breakfast_box only"
        ),
    )
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument(
        "--K",
        type=int,
        default=TL_ITER_INITIAL_K,
        help=(
            "Initial reference count; TL-IterMSSM starts with 3 and then "
            "admits up to three images per round"
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TL_ITER_DEFAULT_MEMORY_FRACTION,
        help="TL-IterMSSM memory-capacity fraction; default: 0.10",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.10,
        help=(
            "Deprecated compatibility option. Full-grid TLFME updates ignore "
            "patch-selection gamma."
        ),
    )
    parser.add_argument("--feature_list", type=int, nargs="+", default=[6, 9])
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 5])
    parser.add_argument(
        "--tlfme_logical_block",
        "--dpfe_logical_block",
        dest="dpfe_logical_block",
        type=int,
        default=12,
        help=(
            "One-based DINO block for the structured TLFME logical branch "
            "on MVTecLOCO; --dpfe_logical_block is a compatibility alias"
        ),
    )
    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument(
        "--knn_metric",
        choices=["L2_normalized"],
        default="L2_normalized",
        help="ReMem currently implements cosine distance via normalized L2",
    )
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--faiss_on_cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rotation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reference-image rotation augmentation. It must remain disabled "
            "for the position-aware MVTecLOCO logical memory and remains "
            "enabled for MVTec, VisA, and MPDD"
        ),
    )
    parser.add_argument(
        "--loco_map_mode",
        choices=LOCO_MAP_MODES,
        default=None,
        help=(
            "MVTecLOCO anomaly map used consistently for memory ranking and "
            "official TIFF evaluation. Dataset default: tl_iter"
        ),
    )
    parser.add_argument(
        "--loco_fusion_alpha",
        type=float,
        default=0.5,
        help=(
            "MLMP weight for --loco_map_mode fused; the deep branch receives "
            "weight 1-alpha. Configurable only in fused mode"
        ),
    )
    parser.add_argument(
        "--tlfme_texture_coreset_size",
        type=int,
        default=DEFAULT_TLFME_TEXTURE_CORESET_SIZE,
        help="Maximum spatial-feature k-center texture prototypes per image",
    )
    parser.add_argument(
        "--tlfme_texture_spatial_weight",
        type=float,
        default=DEFAULT_TLFME_TEXTURE_SPATIAL_WEIGHT,
        help="Spatial-coordinate weight used only for texture coreset selection",
    )
    parser.add_argument("--eval_clf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_segm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval_every_time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to evaluate the anomaly score after every iteration",
    )
    parser.add_argument(
        "--eval_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only evaluate anomaly maps from a completed run",
    )
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument(
        "--loco_eval_workers",
        type=int,
        default=None,
        help="CPU workers for the official MVTec LOCO evaluator",
    )
    parser.add_argument(
        "--loco_official_eval_dir",
        default=None,
        help="Override the vendored official MVTec LOCO evaluator directory",
    )
    return parser.parse_args()


def load_initial_rankings(json_path, dataset=None):
    json_path = Path(json_path).expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Initial MSSM ranking was not found: {json_path}. "
            "For MPDD, run generate_mssm_scores.py first; for MVTecLOCO, "
            "run the matching LOCO ranking generator first."
        )
    with json_path.open("r", encoding="utf-8") as file:
        rankings = json.load(file)
    if not isinstance(rankings, dict):
        raise ValueError(f"Initial ranking must be a JSON object: {json_path}")
    return rankings, str(json_path)


def resolve_objects(args, all_objects):
    """Resolve the requested subset, with a LOCO single-class experiment default."""

    if args.objects is None:
        if args.dataset == "MVTecLOCO":
            return list(LOCO_EXPERIMENT_OBJECTS)
        return list(all_objects)

    unknown_objects = sorted(set(args.objects) - set(all_objects))
    if unknown_objects:
        raise ValueError(
            f"Unknown {args.dataset} object names: {unknown_objects}. "
            f"Available objects: {all_objects}"
        )
    if len(args.objects) != len(set(args.objects)):
        raise ValueError("--objects contains duplicate names")
    return list(args.objects)


def validate_runtime_args(args):
    if args.K <= 0:
        raise ValueError("--K must be positive")
    if not 0 < args.tau <= 1:
        raise ValueError("--tau must be in (0, 1]")
    if not 0 < args.gamma <= 1:
        raise ValueError("--gamma must be in (0, 1]")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.k_neighbors <= 0:
        raise ValueError("--k_neighbors must be positive")
    if not args.feature_list or len(args.feature_list) != len(set(args.feature_list)):
        raise ValueError("--feature_list must contain unique layer numbers")
    if any(layer < 0 for layer in args.feature_list):
        raise ValueError("--feature_list uses non-negative zero-based indices")
    if not args.scales or any(scale <= 0 for scale in args.scales):
        raise ValueError("--scales must contain positive integers")
    if args.dpfe_logical_block <= 0:
        raise ValueError("--dpfe_logical_block must be positive and one-based")
    if (
        args.dataset == "MVTecLOCO"
        and args.dpfe_logical_block - 1 in args.feature_list
    ):
        raise ValueError(
            "--dpfe_logical_block must be distinct from texture --feature_list"
        )
    if args.loco_eval_workers is not None and args.loco_eval_workers <= 0:
        raise ValueError("--loco_eval_workers must be positive or omitted")
    if not math.isfinite(args.loco_fusion_alpha) or not (
        0 <= args.loco_fusion_alpha <= 1
    ):
        raise ValueError("--loco_fusion_alpha must be finite and in [0, 1]")
    if args.dataset != "MVTecLOCO" and args.loco_map_mode is not None:
        raise ValueError("--loco_map_mode is only supported for MVTecLOCO")
    if (
        args.dataset == "MVTecLOCO"
        and (args.loco_map_mode or TL_ITER_MAP_MODE) != "fused"
        and args.loco_fusion_alpha != 0.5
    ):
        raise ValueError(
            "--loco_fusion_alpha is configurable only when "
            "--loco_map_mode fused"
        )
    if args.dataset == "MVTecLOCO" and (
        args.loco_map_mode or TL_ITER_MAP_MODE
    ) == TL_ITER_MAP_MODE:
        if args.K != TL_ITER_INITIAL_K:
            raise ValueError(
                "MVTecLOCO TL-IterMSSM fixes --K to 3 initial references "
                "and admits up to three lowest-score candidates per later round"
            )
    if args.tlfme_texture_coreset_size <= 0:
        raise ValueError("--tlfme_texture_coreset_size must be positive")
    if (
        not math.isfinite(args.tlfme_texture_spatial_weight)
        or args.tlfme_texture_spatial_weight < 0
    ):
        raise ValueError(
            "--tlfme_texture_spatial_weight must be finite and non-negative"
        )
    if args.dataset != "MVTecLOCO" and args.rotation is not None:
        raise ValueError(
            "--rotation/--no-rotation overrides are only supported for "
            "MVTecLOCO; existing dataset behavior is fixed for compatibility"
        )
    if args.dataset == "MVTecLOCO" and args.rotation is True:
        raise ValueError(
            "MVTecLOCO TLFME logical memory requires rotation to remain disabled "
            "so grid coordinates keep a consistent meaning"
        )


def mpdd_experiment_name(args):
    layers = "-".join(map(str, args.feature_list))
    scales = "-".join(map(str, args.scales))
    name = (
        f"tau={args.tau:g}_layers={layers}_scales={scales}_"
        f"knn={args.knn_metric}-{args.k_neighbors}_fullgrid=1"
    )
    mssm_ranking_layers = getattr(args, "mssm_ranking_layers", None)
    if mssm_ranking_layers is not None:
        name += f"_mssm_layers={'-'.join(map(str, mssm_ranking_layers))}"
    mssm_ranking_tag = getattr(args, "mssm_ranking_tag", None)
    if mssm_ranking_tag is not None:
        name += f"_mssm_rank={mssm_ranking_tag}"
    return name


def resolve_mssm_ranking_layers(ranking_metadata):
    """Read and validate the optional MSSM layer ablation identity."""

    if ranking_metadata is None:
        return None
    settings = ranking_metadata.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Invalid MSSM ranking metadata settings")
    layers = settings.get("mssm_layers")
    if layers is None:
        return None
    if (
        not isinstance(layers, list)
        or not layers
        or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers)
        or any(layer <= 0 for layer in layers)
        or len(layers) != len(set(layers))
        or layers != sorted(layers)
    ):
        raise ValueError(f"Invalid mssm_layers in ranking metadata: {layers}")
    return tuple(layers)


def resolve_mssm_ranking_tag(ranking_metadata):
    """Read a filesystem-safe identity for non-baseline MSSM rankings."""

    if ranking_metadata is None:
        return None
    settings = ranking_metadata.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Invalid MSSM ranking metadata settings")
    ranking_tag = settings.get("ranking_tag")
    if ranking_tag is None:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._")
    if (
        not isinstance(ranking_tag, str)
        or not ranking_tag
        or len(ranking_tag) > 80
        or any(character not in allowed for character in ranking_tag)
    ):
        raise ValueError(f"Invalid ranking_tag in MSSM metadata: {ranking_tag}")
    return ranking_tag


def resolve_loco_map_mode(args):
    """Return the explicit LOCO map mode used by scoring and evaluation."""

    if args.dataset != "MVTecLOCO":
        return None
    return getattr(args, "loco_map_mode", None) or TL_ITER_MAP_MODE


def loco_experiment_name(args, rotation):
    map_mode = resolve_loco_map_mode(args)
    map_suffix = f"map={map_mode}"
    if map_mode == TL_ITER_MAP_MODE:
        map_suffix += (
            "_s=t3_m=tau_x=t10-gbi-pm1-v1_e=img_nopos"
        )
    elif map_mode == "fused":
        map_suffix += f"_alpha={args.loco_fusion_alpha:g}"
    return (
        f"{mpdd_experiment_name(args)}_scope={MVTec_LOCO_SCOPE}_rotation={int(rotation)}_"
        f"tlfme=tex{'-'.join(str(index + 1) for index in args.feature_list)}-"
        f"log{args.dpfe_logical_block}_fullgrid_"
        f"tcore={args.tlfme_texture_coreset_size}_"
        f"tsp={args.tlfme_texture_spatial_weight:g}_{map_suffix}"
    )


def resolve_rotation(args):
    """Resolve the dataset default without changing legacy behavior."""

    if args.rotation is not None:
        return bool(args.rotation)
    return args.dataset != "MVTecLOCO"


def write_or_validate_run_metadata(output_dir, settings):
    output_dir = Path(output_dir)
    metadata_path = output_dir / "run.meta.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing != settings:
            raise RuntimeError(
                f"Existing run metadata does not match this run: "
                f"{metadata_path}"
            )
        return
    if any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory contains artifacts but no run metadata: "
            f"{output_dir}. Move it aside before starting this configuration."
        )
    temporary_path = metadata_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, metadata_path)


def atomic_write_json(output_path, payload):
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
    os.replace(temporary_path, output_path)


def stop_condition(p, sample_num, n, cnt):
    if n * cnt > sample_num * p:
        return True
    else:
        return False

def evaluate_completed_run(args, data_root, overall_output_dir, objects):
    """Dispatch evaluation through the dataset-specific implementation."""

    if args.dataset == "MVTecLOCO":
        if not args.eval_clf:
            print("MVTec LOCO image-level evaluation disabled (--no-eval_clf).")
            return None
        from evaluate_mvtec_loco import evaluate_mvtec_loco_image_level

        return evaluate_mvtec_loco_image_level(
            output_dir=os.path.join(overall_output_dir, "image_level_metrics"),
            objects=objects,
            classification_scores_path=os.path.join(
                overall_output_dir, "final_object_results.json"
            ),
        )

    if not args.eval_clf and not args.eval_segm:
        print("Evaluation disabled (--no-eval_clf --no-eval_segm).")
        return None

    anomaly_maps_dir = os.path.join(overall_output_dir, "Anomaly_maps")
    return eval_finished_run(
        dataset=args.dataset,
        dataset_base_dir=data_root,
        anomaly_maps_dir=anomaly_maps_dir,
        output_dir=overall_output_dir,
        seed=0,
        pro_integration_limit=0.3,
        eval_clf=args.eval_clf,
        eval_segm=args.eval_segm,
        delete_tiff_files=args.dataset != "MPDD",
        object_name=objects,
    )

if __name__ == "__main__":
    args = parse_args()
    validate_runtime_args(args)

    all_objects, object_anomalies = get_dataset_info(args.dataset)
    objects = resolve_objects(args, all_objects)

    data_root = str(
        Path(args.data_root or Config.ROOTS[args.dataset]).expanduser().resolve()
    )
    if not Path(data_root).is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    args.data_root = data_root

    if args.dataset == "MPDD":
        from validate_mpdd import validate_mpdd_layout

        validation_report = validate_mpdd_layout(data_root)
        if not validation_report["valid"]:
            details = "\n".join(
                f"  - {error}" for error in validation_report["errors"]
            )
            raise RuntimeError(f"MPDD dataset validation failed:\n{details}")
        object_anomalies = get_mpdd_object_anomalies(data_root)
    elif args.dataset == "MVTecLOCO":
        from validate_mvtec_loco import validate_mvtec_loco_layout

        validation_report = validate_mvtec_loco_layout(data_root, objects=objects)
        if not validation_report["valid"]:
            details = "\n".join(
                f"  - {error}" for error in validation_report["errors"]
            )
            raise RuntimeError(
                f"MVTec LOCO AD dataset validation failed:\n{details}"
            )

    initial_rankings, args.initial_json = load_initial_rankings(
        args.initial_json or Config.JSON_STARTS[args.dataset],
        dataset=args.dataset,
    )
    ranking_metadata = None
    if args.dataset == "MPDD":
        ranking_metadata = validate_mpdd_ranking_metadata(
            args.initial_json,
            data_root=data_root,
            model_name=args.model_name,
            resolution=args.resolution,
        )
        validate_mpdd_rankings(
            initial_rankings,
            objects,
            data_root,
        )
    elif args.dataset == "MVTecLOCO":
        ranking_metadata = validate_mvtec_loco_ranking_metadata(
            args.initial_json,
            data_root=data_root,
            model_name=args.model_name,
            resolution=args.resolution,
        )
        validate_mvtec_loco_rankings(
            initial_rankings,
            objects,
            data_root,
        )

    args.mssm_ranking_layers = resolve_mssm_ranking_layers(ranking_metadata)
    args.mssm_ranking_tag = resolve_mssm_ranking_tag(ranking_metadata)

    rotation_enabled = resolve_rotation(args)
    loco_map_mode = resolve_loco_map_mode(args)
    args.loco_map_mode = loco_map_mode
    rotation_default = {o: rotation_enabled for o in objects}

    if args.dataset == "MVTecLOCO":
        output_parts = [
            str(Path(Config.RESULTS_ROOT).expanduser().resolve()),
            f"results_{args.dataset}",
            args.model_name,
        ]
    else:
        output_parts = [
            data_root,
            f"results_{args.dataset}",
            args.model_name,
        ]
    if args.dataset in RANKED_DATASETS:
        output_parts.append(f"resolution={args.resolution}")
    output_parts.extend(["n_jicheng", f"K={args.K}"])
    if args.dataset == "MPDD":
        output_parts.append(mpdd_experiment_name(args))
    elif args.dataset == "MVTecLOCO":
        output_parts.append(loco_experiment_name(args, rotation_enabled))
    output_parts.append("final_eval")
    overall_output_dir = os.path.join(*output_parts)
    os.makedirs(overall_output_dir, exist_ok=True)
    if args.dataset in RANKED_DATASETS:
        run_settings = {
            "dataset": args.dataset,
            "data_root": data_root,
            "initial_json": args.initial_json,
            "ranking_settings": ranking_metadata["settings"],
            "model_name": args.model_name,
            "resolution": args.resolution,
            "K": args.K,
            "tau": args.tau,
            "feature_list": args.feature_list,
            "scales": args.scales,
            "knn_metric": args.knn_metric,
            "k_neighbors": args.k_neighbors,
            "full_grid_features": True,
        }
        if args.dataset == "MVTecLOCO":
            run_settings["experiment_scope"] = MVTec_LOCO_SCOPE
            run_settings["test_types"] = list(MVTec_LOCO_TEST_TYPES)
            run_settings["rotation"] = rotation_enabled
            run_settings["loco_map_mode"] = loco_map_mode
            run_settings["loco_fusion_alpha"] = (
                args.loco_fusion_alpha if loco_map_mode == "fused" else None
            )
            run_settings["loco_dual_branch"] = True
            run_settings["evaluation_scope"] = "image_level_logical_only"
            run_settings["pixel_level_metrics_computed"] = False
            run_settings["anomaly_maps_generated"] = False
            run_settings["dpfe_texture_blocks"] = [
                index + 1 for index in args.feature_list
            ]
            run_settings["dpfe_logical_block"] = args.dpfe_logical_block
            run_settings["dpfe_update_policy"] = (
                "selected_top_k_full_grid"
            )
            run_settings["tlfme_texture_blocks"] = [
                index + 1 for index in args.feature_list
            ]
            run_settings["tlfme_logical_block"] = (
                args.dpfe_logical_block
            )
            run_settings["tlfme_update_policy"] = (
                "texture_coreset_logical_full_grid"
                if loco_map_mode == TL_ITER_MAP_MODE
                else "selected_top_k_full_grid"
            )
            run_settings["tlfme_texture_coreset_size"] = (
                args.tlfme_texture_coreset_size
                if loco_map_mode == TL_ITER_MAP_MODE
                else None
            )
            run_settings["tlfme_texture_spatial_weight"] = (
                args.tlfme_texture_spatial_weight
                if loco_map_mode == TL_ITER_MAP_MODE
                else None
            )
            if loco_map_mode == TL_ITER_MAP_MODE:
                run_settings["tl_iter_mssm"] = {
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
                    "branch_image_score_pooling": (
                        "top_1_percent_patch_mean"
                    ),
                    "logical_template_selection": (
                        "minimum_template_top_1_percent_patch_mean"
                    ),
                    "logical_matching": (
                        "appearance_only_global_bidirectional"
                    ),
                    "uses_position_constraint": False,
                    "image_patch_pool_fraction": (
                        TL_ITER_IMAGE_PATCH_POOL_FRACTION
                    ),
                }
            run_settings["initial_json_sha256"] = sha256_file(
                args.initial_json
            )
        write_or_validate_run_metadata(overall_output_dir, run_settings)

    if not args.eval_mode:
        if args.device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
            model_device = "cuda"
        else:
            model_device = args.device
        model = get_model(
            args.model_name,
            model_device,
            smaller_edge_size=args.resolution,
        )

        final_results = run_per_object_adaptive_loop(
            model,
            args,
            Config,
            objects,
            overall_output_dir,
            object_anomalies,
            rotation_default,
            stop_condition,
        )


        atomic_write_json(
            os.path.join(overall_output_dir, "final_object_results.json"),
            final_results,
        )
        evaluate_completed_run(
            args,
            data_root,
            overall_output_dir,
            objects,
        )
    else:
        evaluate_completed_run(
            args,
            data_root,
            overall_output_dir,
            objects,
        )
