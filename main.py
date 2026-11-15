from dataset import prepare_custom_dataset
from train import train_model
from evaluate import evaluate_model
from detection import detect_objects
from voice_assistant import start_voice_assistant

def main():
    print(" SYSTEM ROZPOZNAWANIA OBIEKTÓW")

    while True:
        print("\nMENU")
        print("1. Przygotuj zbiór KITTI")
        print("2. Trenuj model")
        print("3. Metryki")
        print("4. Wgraj zdjęcie")
        print("5. Asystent głosowy")
        print("0. Wyjście")

        choice = input("\nWybierz opcję: ").strip()

        try:
            if choice == "1":
                prepare_custom_dataset()
            elif choice == "2":
                train_model()
            elif choice == "3":
                evaluate_model()
            elif choice == "4":
                detect_objects()
            elif choice == "5":
                start_voice_assistant()
            elif choice == "0":
                print("Zamykanie programu")
                break
            else:
                print("Nieprawidłowa opcja")

        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                print("\nout of memory")
            else:
                raise


if __name__ == "__main__":
    main()
