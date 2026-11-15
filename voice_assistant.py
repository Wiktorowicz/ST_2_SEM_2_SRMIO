
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import speech_recognition as sr
import unicodedata
import threading
import sounddevice as sd

from ultralytics import YOLO
from config import BEST_MODEL, IMAGE_SIZE, CLASS_NAMES, DEVICE


def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFD", text)

    return "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )


def choose_image():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Wybierz zdjęcie",
        filetypes=[
            ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.webp")
        ]
    )

    root.destroy()
    return path


def listen():
    recognizer = sr.Recognizer()

    sample_rate = 16000
    block_duration = 0.1
    silence_duration = 1.2

    block_size = int(sample_rate * block_duration)
    silence_limit = int(silence_duration / block_duration)

    audio_blocks = []
    silence_blocks = 0
    speech_detected = False

    threshold = 500

    print("\nNasłuchuję...")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=block_size
        ) as stream:

            while True:
                audio_block, overflowed = stream.read(block_size)

                audio_block = audio_block.flatten()

                volume = (
                    sum(int(sample) ** 2 for sample in audio_block)
                    / len(audio_block)
                ) ** 0.5

                if volume > threshold:
                    speech_detected = True
                    silence_blocks = 0

                elif speech_detected:
                    silence_blocks += 1

                if speech_detected:
                    audio_blocks.append(audio_block.copy())

                if speech_detected and silence_blocks >= silence_limit:
                    break

        if not audio_blocks:
            return ""

        audio_data = b"".join(
            block.tobytes()
            for block in audio_blocks
        )

        audio = sr.AudioData(
            audio_data,
            sample_rate,
            2
        )

        print("Rozpoznawanie wypowiedzi...")

        text = recognizer.recognize_google(
            audio,
            language="pl-PL"
        )

        print("Rozpoznano:", text)

        return normalize(text)

    except sr.UnknownValueError:
        print("Nie udało się rozpoznać wypowiedzi.")
        return ""

    except sr.RequestError:
        print("Brak połączenia z usługą rozpoznawania mowy.")
        return ""

    except Exception as error:
        print("Błąd mikrofonu:", error)
        return None


def answer_question(question, counts):

    if any(word in question for word in [
        "koniec", "stop", "wyjdz", "zakoncz"
    ]):
        print("Zakończono asystenta.")
        return False

    if any(word in question for word in [
        "wszystkie", "wszystkich", "lacznie", "ogolem"
    ]):
        print("Łączna liczba obiektów:", sum(counts.values()))
        return True

    categories = {
        "samochod": ["samochod", "auto", "auta"],
        "van": ["van", "dostawczy"],
        "ciezarowka": ["ciezarow", "tir"],
        "czlowiek": ["czlowiek", "ludzi", "osob", "pieszy"],
        "rowerzysta": ["rowerzyst", "rower"],
        "tramwaj": ["tramwaj"]
    }

    for category, keywords in categories.items():
        if any(word in question for word in keywords):
            print(
                f"Liczba obiektów ({category}): "
                f"{counts.get(category, 0)}"
            )
            return True

    print("Nie rozumiem pytania.")
    return True


def start_voice_assistant():

    if not Path(BEST_MODEL).exists():
        print("Nie znaleziono modelu best.pt.")
        return

    image_path = choose_image()

    if not image_path:
        return

    model = YOLO(str(BEST_MODEL))

    results = model.predict(
        source=image_path,
        imgsz=IMAGE_SIZE,
        conf=0.25,
        device=DEVICE
    )

    counts = {
        name: 0
        for name in CLASS_NAMES.values()
    }

    for result in results:

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                name = result.names[class_id]

                counts[name] = counts.get(name, 0) + 1

        cv2.imshow(
            "Wynik rozpoznawania",
            result.plot()
        )

    print("\nWYKRYTE OBIEKTY:")

    for name, count in counts.items():
        print(f"{name}: {count}")

    stop_event = threading.Event()

    def voice_loop():

        while not stop_event.is_set():

            question = listen()

            if question is None:
                stop_event.set()
                break

            if question and not answer_question(question, counts):
                stop_event.set()
                break

    voice_thread = threading.Thread(
        target=voice_loop,
        daemon=True
    )

    voice_thread.start()

    print("\nAsystent gotowy. Powiedz 'koniec', aby zakończyć.")

    while not stop_event.is_set():

        cv2.waitKey(20)

        if cv2.getWindowProperty(
            "Wynik rozpoznawania",
            cv2.WND_PROP_VISIBLE
        ) < 1:
            stop_event.set()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_voice_assistant()
