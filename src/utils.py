import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from src.dataset_info import MPDD_OBJECT_ANOMALIES, MVTec_LOCO_OBJECT_ANOMALIES


def augment_image(img_ref, augmentation = "rotate", flip=False, angles = [0, 45, 90, 135, 180, 225, 270, 315]):
    """
    Simply augmentation of images, currently just rotation.
    """
    imgs = []
    if augmentation == "rotate":
        for angle in angles:
            imgs.append(rotate_image(img_ref, angle))
        # for angle in angles:
        #     imgs.append(rotate_image(img_ref, angle+22.5))
    if flip:
        imgs.append(cv2.flip(img_ref, 0))
        imgs.append(cv2.flip(img_ref, 1))
    return imgs


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)
    return result

def dists2map(dists, img_shape):
    # resize and smooth the distance map
    # caution: cv2.resize expects the shape in (width, height) order (not (height, width) as in numpy, so indices here are swapped!
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_LINEAR)
    dists = gaussian_filter(dists, sigma=4)
    return dists


def get_dataset_info(dataset):

    objects = []
    object_anomalies = {}
    if dataset == "MVTec":
        objects = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
        object_anomalies = {"bottle": ["broken_large", "broken_small", "contamination"],
                            "cable": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation", "cut_outer_insulation", "missing_wire", "missing_cable", "poke_insulation"],
                            "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
                            "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
                            "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
                            "hazelnut": ["crack", "cut", "hole", "print"],
                            "leather": ["color", "cut", "fold", "glue", "poke"],
                            "metal_nut": ["bent", "color", "flip", "scratch"],
                            "pill": ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
                            "screw": ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
                            "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
                            "toothbrush": ["defective"], 
                            "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
                            "wood": ["color", "combined", "hole", "liquid", "scratch"],
                            "zipper": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough", "split_teeth", "squeezed_teeth"]
                            }


    elif dataset == "VisA":
        objects = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
        object_anomalies = {"candle": ["bad"],
                            "capsules": ["bad"],
                            "cashew": ["bad"],
                            "chewinggum": ["bad"],
                            "fryum": ["bad"],
                            "macaroni1": ["bad"],
                            "macaroni2": ["bad"],
                            "pcb1": ["bad"],
                            "pcb2": ["bad"],
                            "pcb3": ["bad"],
                            "pcb4": ["bad"],
                            "pipe_fryum": ["bad"],
                            }

    elif dataset == "MPDD":
        objects = list(MPDD_OBJECT_ANOMALIES)
        object_anomalies = {
            object_name: list(anomaly_types)
            for object_name, anomaly_types in MPDD_OBJECT_ANOMALIES.items()
        }

    elif dataset == "MVTecLOCO":
        objects = list(MVTec_LOCO_OBJECT_ANOMALIES)
        object_anomalies = {
            object_name: list(anomaly_types)
            for object_name, anomaly_types in MVTec_LOCO_OBJECT_ANOMALIES.items()
        }

    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Expected one of: MVTec, VisA, "
            "MPDD, MVTecLOCO."
        )

    return objects, object_anomalies
