from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent

# "laptop" albo "pc"
DEVICE_PROFILE = "laptop"

if DEVICE_PROFILE == "pc":
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 48
    WORKERS = 8
    CACHE = "ram"

elif DEVICE_PROFILE == "laptop":
    DEVICE = "cpu"
    BATCH_SIZE = 4
    WORKERS = 2
    CACHE = False

else:
    raise ValueError("Nieznany profil urządzenia")

SOURCE_DATASET_YAML = "kitti.yaml"

CUSTOM_DATASET_DIR = PROJECT_DIR / "datasets" / "kitti_custom"
CUSTOM_DATASET_YAML = CUSTOM_DATASET_DIR / "kitti_custom.yaml"

MODEL_NAME = "yolo11n.pt"
EPOCHS = 1
IMAGE_SIZE = 640

RUNS_DIR = PROJECT_DIR / "runs" / "detect"
RUN_NAME = "kitti_yolo11n"
BEST_MODEL = RUNS_DIR / RUN_NAME / "weights" / "best.pt"

CLASS_NAMES = {
    0: "samochod",
    1: "van",
    2: "ciezarowka",
    3: "czlowiek",
    4: "rowerzysta",
    5: "tramwaj",
}

SOURCE_TO_TARGET = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    6: 5,
}