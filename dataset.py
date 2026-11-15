from pathlib import Path
import os
import shutil
import yaml
from ultralytics.data.utils import check_det_dataset
from config import SOURCE_DATASET_YAML, CUSTOM_DATASET_DIR, CUSTOM_DATASET_YAML, CLASS_NAMES, SOURCE_TO_TARGET

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _resolve_image_dir(value, dataset_root):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Oczekiwano jednej ścieżki")
        value = value[0]

    path = Path(value)
    if not path.is_absolute():
        path = Path(dataset_root) / path

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError("Nie znaleziono katalogu obrazów")
    return path


def _label_dir_from_image_dir(image_dir):
    parts = list(image_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts)
    raise ValueError("Nie mogę znaleźć katalogu images")


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _convert_label_file(source_label, destination_label):
    destination_label.parent.mkdir(parents=True, exist_ok=True)

    if not source_label.exists():
        destination_label.write_text("", encoding="utf-8")
        return 0

    converted_lines = []
    kept_objects = 0

    for line_number, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue

        try:
            source_class = int(float(parts[0]))
        except ValueError:
            raise ValueError("Błędna etykieta")

        target_class = SOURCE_TO_TARGET.get(source_class)
        if target_class is None:
            continue

        if len(parts) < 5:
            raise ValueError("Za mało pól w etykiecie")

        converted_lines.append(" ".join([str(target_class)] + parts[1:]))
        kept_objects += 1

    destination_label.write_text(
        "\n".join(converted_lines) + ("\n" if converted_lines else ""),
        encoding="utf-8",
    )
    return kept_objects


def _prepare_split(source_images, split_name):
    source_labels = _label_dir_from_image_dir(source_images)
    target_images = CUSTOM_DATASET_DIR / "images" / split_name
    target_labels = CUSTOM_DATASET_DIR / "labels" / split_name

    if not source_labels.exists():
        raise FileNotFoundError("Nie znaleziono etykiet ")

    image_files = [
        path for path in source_images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        raise RuntimeError("Brak obrazów w katalogu")

    total_objects = 0
    for image_path in image_files:
        relative_path = image_path.relative_to(source_images)
        source_label = source_labels / relative_path.with_suffix(".txt")
        target_image = target_images / relative_path
        target_label = target_labels / relative_path.with_suffix(".txt")

        _link_or_copy(image_path, target_image)
        total_objects += _convert_label_file(source_label, target_label)

    print(f"{split_name}: przygotowano {len(image_files)} obrazów, zachowano {total_objects} obiektów.")


def prepare_custom_dataset():
    source_data = check_det_dataset(SOURCE_DATASET_YAML)
    train_images = _resolve_image_dir(source_data["train"], source_data["path"])
    val_images = _resolve_image_dir(source_data["val"], source_data["path"])

    _prepare_split(train_images, "train")
    _prepare_split(val_images, "val")

    CUSTOM_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    custom_yaml = {
        "path": str(CUSTOM_DATASET_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }

    CUSTOM_DATASET_YAML.write_text(
        yaml.safe_dump(custom_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"[DATASET] Gotowy własny zbiór: {CUSTOM_DATASET_YAML}")
    print("[DATASET] Klasy: samochod, van, ciezarowka, rowerzysta, czlowiek, tramwaj")
    return str(CUSTOM_DATASET_YAML)