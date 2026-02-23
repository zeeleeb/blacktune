"""BlackTune main application window.

Provides the top-level window shell with a dark theme, menu bar,
four-tab layout (Log Viewer, Analysis, Tune, History), drag-and-drop
file loading, and status bar.
"""
from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from blacktune.ui.theme import dark_stylesheet
from blacktune.ui.viewer_tab import ViewerTab
from blacktune.ui.analysis_tab import AnalysisTab
from blacktune.ui.tune_tab import TuneTab

# Supported file extensions for drag-and-drop filtering
_SUPPORTED_EXTENSIONS = {".bbl", ".bfl", ".csv"}


class MainWindow(QMainWindow):
    """BlackTune main application window."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("BlackTune - FPV PID Autotuner")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(dark_stylesheet())

        # Flight log and analysis references (populated by _load_file)
        self._flight_log = None  # Optional[FlightLog]
        self._analysis_result = None  # Optional[AnalysisResult]
        self._recommendation = None  # Optional[TuneRecommendation]

        # Tab widget exposed as public attribute for later task wiring
        self.tabs: Optional[QTabWidget] = None

        self._setup_menu()
        self._setup_tabs()
        self._setup_statusbar()

        # Accept drag-and-drop
        self.setAcceptDrops(True)

    # ── Menu bar ────────────────────────────────────────────────

    def _setup_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open BBL/CSV...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # ── Tab layout ──────────────────────────────────────────────

    def _setup_tabs(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Log Viewer -- real implementation
        self.viewer_tab = ViewerTab()
        self.tabs.addTab(self.viewer_tab, "Log Viewer")

        # Analysis Dashboard -- real implementation
        self.analysis_tab = AnalysisTab()
        self.tabs.addTab(self.analysis_tab, "Analysis")

        # Tune tab -- real implementation
        self.tune_tab = TuneTab()
        self.tune_tab.set_analyze_callback(self._on_tune_requested)
        self.tabs.addTab(self.tune_tab, "Tune")

        # History tab -- placeholder until Task 13
        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        history_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        history_placeholder = QLabel("Load a BBL file to begin")
        history_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        history_placeholder.setStyleSheet("color: #8888aa; font-size: 16px;")
        history_layout.addWidget(history_placeholder)
        self.tabs.addTab(history_page, "History")

        self.setCentralWidget(self.tabs)

    # ── Status bar ──────────────────────────────────────────────

    def _setup_statusbar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Ready - Open a BBL or CSV file to begin")

    # ── File opening ────────────────────────────────────────────

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Flight Log",
            "",
            "Blackbox Logs (*.bbl *.bfl *.csv);;All Files (*)",
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        """Load a flight log file and update the UI."""
        from blacktune.parser import load_log

        try:
            flight_log = load_log(path)
        except Exception as exc:
            self.statusBar().showMessage(f"Error loading file: {exc}")
            return

        self._flight_log = flight_log
        filename = os.path.basename(path)

        # Build status info
        parts = [
            filename,
            f"{flight_log.sample_rate} Hz",
            f"{flight_log.duration_s:.1f}s",
            flight_log.firmware,
        ]
        self.statusBar().showMessage(" | ".join(parts))

        # Populate viewer tab with flight data
        self.viewer_tab.load_data(flight_log)

        # Run analysis and populate analysis tab
        try:
            from blacktune.analyzer import run_analysis

            analysis_result = run_analysis(flight_log)
            self._analysis_result = analysis_result
            self.analysis_tab.load_results(flight_log, analysis_result)
        except Exception as exc:
            self.statusBar().showMessage(
                f"{' | '.join(parts)} | Analysis error: {exc}"
            )

    # ── Tune recommendation ────────────────────────────────────

    def _on_tune_requested(self, profile) -> None:
        """Generate a tune recommendation when the user clicks Analyze & Tune."""
        if self._flight_log is None or self._analysis_result is None:
            self.statusBar().showMessage("Load and analyze a log first")
            return
        from blacktune.analyzer import generate_recommendation

        rec = generate_recommendation(self._flight_log, profile, self._analysis_result)
        self._recommendation = rec
        self.tune_tab.load_recommendation(
            self._flight_log.current_pids,
            rec,
            current_filters=self._flight_log.current_filters,
        )
        self.tabs.setCurrentWidget(self.tune_tab)
        self.statusBar().showMessage("Tune recommendation generated")

    # ── Drag & Drop ─────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in _SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in _SUPPORTED_EXTENSIONS:
                self._load_file(file_path)
                return
