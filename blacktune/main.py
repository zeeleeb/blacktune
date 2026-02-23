"""BlackTune - FPV PID Autotuner."""
import sys
from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BlackTune")
    # Window will be created in later tasks
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
