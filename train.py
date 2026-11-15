
import torch
from ultralytics import YOLO

from config import (
    MODEL_NAME,
    EPOCHS,
    IMAGE_SIZE,
    BATCH_SIZE,
    WORKERS,
    DEVICE,
    CACHE,
    RUNS_DIR,
    RUN_NAME,
)

from dataset import prepare_custom_dataset


def train_model():
    if DEVICE == 0:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print("[GPU]", gpu_name)
        print("[GPU] VRAM:", round(vram, 2), "GB")
        print("[GPU] CUDA:", torch.version.cuda)

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()
    else:
        print("[CPU] Trening na procesorze")

    data_yaml = prepare_custom_dataset()
    model = YOLO(MODEL_NAME)

    print("\nUSTAWIENIA TRENINGU")
    print("Model:", MODEL_NAME)
    print("Epoki:", EPOCHS)
    print("Rozdzielczość:", IMAGE_SIZE)
    print("Batch:", BATCH_SIZE)
    print("Workers:", WORKERS)
    print("Urządzenie:", DEVICE)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        amp=(DEVICE == 0),
        workers=WORKERS,
        cache=CACHE,
        optimizer="auto",
        cos_lr=True,
        deterministic=False,
        patience=12,
        pretrained=True,
        seed=42,
        plots=True,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )

    print("TRENING ZAKOŃCZONY")
    print("Model best.pt:")
    print(RUNS_DIR / RUN_NAME / "weights" / "best.pt")

    return results


if __name__ == "__main__":
    try:
        train_model()
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print("\nBrak pamięci")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise
