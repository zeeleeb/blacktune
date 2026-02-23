"""BlackTune - FPV PID Autotuner."""
import sys
from PyQt6.QtWidgets import QApplication
from blacktune.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BlackTune")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
