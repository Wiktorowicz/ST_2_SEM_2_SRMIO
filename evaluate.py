from pathlib import Path
from ultralytics import YOLO

from config import BEST_MODEL, CUSTOM_DATASET_YAML, IMAGE_SIZE, WORKERS


def evaluate_model():
    if not Path(BEST_MODEL).exists():
        print("Nie znaleziono best.pt")
        return

    if not Path(CUSTOM_DATASET_YAML).exists():
        print("Nie znaleziono własnego YAML")
        return

    model = YOLO(str(BEST_MODEL))

    results = model.val(
        data=str(CUSTOM_DATASET_YAML),
        imgsz=IMAGE_SIZE,
        batch=16,
        device=0,
        workers=WORKERS,
        plots=True,
    )

    print("\nWYNIKI WALIDACJI")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")


if __name__ == "__main__":
    evaluate_model()
