
from pathlib import Path
from tkinter import Tk, filedialog
from ultralytics import YOLO
import cv2

from config import BEST_MODEL, IMAGE_SIZE, CLASS_NAMES, DEVICE


def choose_image():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    image_path = filedialog.askopenfilename(
        title="Wybierz zdjęcie do rozpoznania",
        filetypes=[
            ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Wszystkie pliki", "*.*")
        ]
    )

    root.destroy()
    return image_path


def detect_objects():
    if not Path(BEST_MODEL).exists():
        print("Nie znaleziono wytrenowanego modelu best.pt.")
        return

    image_path = choose_image()

    if not image_path:
        return

    model = YOLO(str(BEST_MODEL))

    results = model.predict(
        source=image_path,
        conf=0.25,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        save=True,
        show=False
    )

    for result in results:
        counts = {name: 0 for name in CLASS_NAMES.values()}

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                counts[class_name] = counts.get(class_name, 0) + 1

        print("\nWYKRYTE OBIEKTY")
        print("Łączna liczba obiektów:", sum(counts.values()))

        for name, count in counts.items():
            print(f"{name}: {count}")

        print("=====================================")

        annotated_image = result.plot()
        cv2.imshow("Wynik rozpoznawania obiektow", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_objects()
