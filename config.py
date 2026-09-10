
class Config:

    DATA = "VisA"  # 'MVTec', 'VisA', 'MPDD', or 'MVTecLOCO'

    ROOTS = {
        "VisA": '/home/lxq/Project/rememsimple/data/visa_mvtec',
        "MVTec": '/home/lxq/Project/rememsimple/data/mvtec_anomaly_detection',
        "MPDD": '/home/lxq/Project/rememsimple/data/MPDD',
        "MVTecLOCO": '/home/lxq/Project/rememsimple/data/mvtec_loco_anomaly_detection',
    }

    FEATURE_ROOT = "./features"

    RESULTS_ROOT = "./results"

    JSON_STARTS = {
        "VisA": 'json/VisA/dinov2_vits14/batch-0-shot/results.json',
        "MVTec": 'json/MVTec/dinov2_vits14/batch-0-shot/results.json',
        "MPDD": 'json/MPDD/dinov2_vits14/batch-0-shot/results.json',
        "MVTecLOCO": 'json/MVTecLOCO/dinov2_vits14/batch-0-shot/logical_only/mssm_gb/weights=9-1/results.json',
    }
