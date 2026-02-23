"""Tune tab -- quad profile setup, PID comparison tables, CLI export.

Three-column layout:
1. Left: Quad profile dropdowns + Analyze button + confidence indicator
2. Center: PID values table + filter settings table (current vs suggested)
3. Right: Betaflight CLI commands (terminal-style) + copy button
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from blacktune.models import (
    FilterSettings,
    PIDValues,
    QuadProfile,
    TuneRecommendation,
)


# ── Dropdown option mappings ──────────────────────────────────────────────────

_CELL_OPTIONS = ["3S", "4S", "5S", "6S"]
_CELL_VALUES = [3, 4, 5, 6]

_PROP_OPTIONS = ['2"', '2.5"', '3"', '4"', '5"', '6"', '7"']
_PROP_VALUES = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]

_FRAME_OPTIONS = ["Auto", "Micro", '3"', '5"', '7"', '10"+']

_STYLE_OPTIONS = ["Freestyle", "Race", "Cinematic", "Long Range"]
_STYLE_VALUES = ["freestyle", "race", "cinematic", "long_range"]


class TuneTab(QWidget):
    """Tune recommendation tab with quad profile, PID comparison, and CLI export."""

    def __init__(self) -> None:
        super().__init__()

        self._analyze_callback: Optional[Callable[[QuadProfile], None]] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        # ── Left column: Quad Profile Setup ───────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        profile_group = QGroupBox("Quad Profile")
        profile_layout = QVBoxLayout(profile_group)
        profile_layout.setSpacing(8)

        # Cell Count
        profile_layout.addWidget(QLabel("Cell Count"))
        self._cell_combo = QComboBox()
        self._cell_combo.addItems(_CELL_OPTIONS)
        self._cell_combo.setCurrentIndex(1)  # 4S default
        profile_layout.addWidget(self._cell_combo)

        # Prop Size
        profile_layout.addWidget(QLabel("Prop Size"))
        self._prop_combo = QComboBox()
        self._prop_combo.addItems(_PROP_OPTIONS)
        self._prop_combo.setCurrentIndex(4)  # 5" default
        profile_layout.addWidget(self._prop_combo)

        # Frame Size
        profile_layout.addWidget(QLabel("Frame Size"))
        self._frame_combo = QComboBox()
        self._frame_combo.addItems(_FRAME_OPTIONS)
        self._frame_combo.setCurrentIndex(0)  # Auto default
        profile_layout.addWidget(self._frame_combo)

        # Flying Style
        profile_layout.addWidget(QLabel("Flying Style"))
        self._style_combo = QComboBox()
        self._style_combo.addItems(_STYLE_OPTIONS)
        self._style_combo.setCurrentIndex(0)  # Freestyle default
        profile_layout.addWidget(self._style_combo)

        left.addWidget(profile_group)

        # Analyze & Tune button
        self._analyze_btn = QPushButton("Analyze && Tune")
        self._analyze_btn.setObjectName("primary")
        self._analyze_btn.clicked.connect(self._on_analyze_clicked)
        left.addWidget(self._analyze_btn)

        # Confidence indicator
        self._confidence_label = QLabel("")
        self._confidence_label.setStyleSheet(
            "color: #8888aa; font-size: 14px; background: transparent;"
        )
        self._confidence_label.setWordWrap(True)
        left.addWidget(self._confidence_label)

        left.addStretch()
        root.addLayout(left, stretch=1)

        # ── Center column: PID + Filter tables ────────────────────
        center = QVBoxLayout()
        center.setSpacing(8)

        # PID Values header
        pid_header = QLabel("PID Values")
        pid_header.setObjectName("header")
        center.addWidget(pid_header)

        # PID table: 3 rows (Roll, Pitch, Yaw) x 7 columns
        self._pid_table = QTableWidget(3, 7)
        self._pid_table.setHorizontalHeaderLabels([
            "Axis",
            "Current P", "Suggested P",
            "Current I", "Suggested I",
            "Current D", "Suggested D",
        ])
        self._pid_table.verticalHeader().setVisible(False)
        self._pid_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._pid_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._pid_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._pid_table.setMinimumHeight(130)

        # Populate axis labels
        for row, axis in enumerate(("Roll", "Pitch", "Yaw")):
            item = QTableWidgetItem(axis)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._pid_table.setItem(row, 0, item)

        center.addWidget(self._pid_table, stretch=2)

        # Filter Settings header
        filter_header = QLabel("Filter Settings")
        filter_header.setObjectName("header")
        center.addWidget(filter_header)

        # Filter table: 7 rows x 3 columns (Setting | Current | Suggested)
        self._filter_table = QTableWidget(7, 3)
        self._filter_table.setHorizontalHeaderLabels([
            "Setting", "Current", "Suggested",
        ])
        self._filter_table.verticalHeader().setVisible(False)
        self._filter_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._filter_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._filter_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._filter_table.setMinimumHeight(220)

        # Pre-populate setting names
        filter_names = [
            "Gyro LPF1",
            "Gyro LPF2",
            "D-term LPF1",
            "D-term LPF2",
            "Dyn Notch Count",
            "Dyn Notch Q",
            "RPM Harmonics",
        ]
        for row, name in enumerate(filter_names):
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._filter_table.setItem(row, 0, item)

        center.addWidget(self._filter_table, stretch=3)
        root.addLayout(center, stretch=3)

        # ── Right column: CLI Commands ────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        cli_header = QLabel("Betaflight CLI Commands")
        cli_header.setObjectName("header")
        right.addWidget(cli_header)

        self._cli_text = QTextEdit()
        self._cli_text.setReadOnly(True)
        self._cli_text.setFont(QFont("Consolas", 11))
        self._cli_text.setStyleSheet(
            "QTextEdit {"
            "  background-color: #0a0a1a;"
            "  color: #00ff00;"
            "  border: 1px solid #2d2d4e;"
            "  border-radius: 4px;"
            "  padding: 8px;"
            "}"
        )
        right.addWidget(self._cli_text, stretch=1)

        self._copy_btn = QPushButton("Copy CLI Commands")
        self._copy_btn.clicked.connect(self._copy_cli)
        right.addWidget(self._copy_btn)

        root.addLayout(right, stretch=2)

    # ── Public API ────────────────────────────────────────────────

    def get_profile(self) -> QuadProfile:
        """Read current dropdown values and return a QuadProfile."""
        cell_idx = self._cell_combo.currentIndex()
        prop_idx = self._prop_combo.currentIndex()
        frame_text = self._frame_combo.currentText()
        style_idx = self._style_combo.currentIndex()

        frame_size: Optional[str] = None if frame_text == "Auto" else frame_text

        return QuadProfile(
            cell_count=_CELL_VALUES[cell_idx],
            prop_size=_PROP_VALUES[prop_idx],
            frame_size=frame_size,
            flying_style=_STYLE_VALUES[style_idx],
        )

    def set_analyze_callback(self, callback: Callable[[QuadProfile], None]) -> None:
        """Set the callback invoked when 'Analyze & Tune' is clicked.

        The callback receives the QuadProfile built from current dropdown values.
        """
        self._analyze_callback = callback

    def load_recommendation(
        self,
        current_pids: PIDValues,
        rec: TuneRecommendation,
        current_filters: Optional[FilterSettings] = None,
    ) -> None:
        """Populate PID table, filter table, CLI text, and confidence label.

        Parameters
        ----------
        current_pids : PIDValues
            The PID values currently on the flight controller.
        rec : TuneRecommendation
            The generated recommendation with suggested PIDs, filters, CLI, confidence.
        current_filters : FilterSettings, optional
            Current filter settings from the log. Defaults to FilterSettings() if None.
        """
        self._populate_pid_table(current_pids, rec.suggested_pids)
        self._populate_filter_table(
            current_filters or FilterSettings(),
            rec.suggested_filters,
        )
        self._cli_text.setPlainText(rec.cli_commands)
        self._update_confidence(rec.confidence)

    # ── Private helpers ───────────────────────────────────────────

    def _on_analyze_clicked(self) -> None:
        """Handle the 'Analyze & Tune' button click."""
        if self._analyze_callback is not None:
            profile = self.get_profile()
            self._analyze_callback(profile)

    def _populate_pid_table(
        self,
        current: PIDValues,
        suggested: PIDValues,
    ) -> None:
        """Fill the PID comparison table with current vs suggested values."""
        axes = [
            ("Roll", "roll_p", "roll_i", "roll_d"),
            ("Pitch", "pitch_p", "pitch_i", "pitch_d"),
            ("Yaw", "yaw_p", "yaw_i", "yaw_d"),
        ]

        for row, (label, p_attr, i_attr, d_attr) in enumerate(axes):
            # Axis label (column 0) is already set in __init__
            pairs = [
                (getattr(current, p_attr), getattr(suggested, p_attr)),  # P
                (getattr(current, i_attr), getattr(suggested, i_attr)),  # I
                (getattr(current, d_attr), getattr(suggested, d_attr)),  # D
            ]
            col = 1
            for cur_val, sug_val in pairs:
                # Current value (white)
                cur_item = QTableWidgetItem(str(round(cur_val)))
                cur_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._pid_table.setItem(row, col, cur_item)
                col += 1

                # Suggested value (color-coded)
                sug_item = QTableWidgetItem(str(round(sug_val)))
                sug_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                if round(sug_val) > round(cur_val):
                    sug_item.setForeground(QColor("#44ff44"))
                elif round(sug_val) < round(cur_val):
                    sug_item.setForeground(QColor("#ff6b6b"))
                # else white (default)

                self._pid_table.setItem(row, col, sug_item)
                col += 1

    def _populate_filter_table(
        self,
        current: FilterSettings,
        suggested_filters: FilterSettings,
    ) -> None:
        """Fill the filter settings comparison table.

        Parameters
        ----------
        current : FilterSettings
            Current filter settings from the flight log (or defaults).
        suggested_filters : FilterSettings
            Suggested filter settings from the recommendation.
        """
        filter_rows = [
            ("Gyro LPF1", f"{current.gyro_lpf1_type} {round(current.gyro_lpf1_hz)} Hz",
             f"{suggested_filters.gyro_lpf1_type} {round(suggested_filters.gyro_lpf1_hz)} Hz",
             current.gyro_lpf1_hz, suggested_filters.gyro_lpf1_hz),
            ("Gyro LPF2", f"{current.gyro_lpf2_type} {round(current.gyro_lpf2_hz)} Hz",
             f"{suggested_filters.gyro_lpf2_type} {round(suggested_filters.gyro_lpf2_hz)} Hz",
             current.gyro_lpf2_hz, suggested_filters.gyro_lpf2_hz),
            ("D-term LPF1", f"{current.dterm_lpf1_type} {round(current.dterm_lpf1_hz)} Hz",
             f"{suggested_filters.dterm_lpf1_type} {round(suggested_filters.dterm_lpf1_hz)} Hz",
             current.dterm_lpf1_hz, suggested_filters.dterm_lpf1_hz),
            ("D-term LPF2", f"{current.dterm_lpf2_type} {round(current.dterm_lpf2_hz)} Hz",
             f"{suggested_filters.dterm_lpf2_type} {round(suggested_filters.dterm_lpf2_hz)} Hz",
             current.dterm_lpf2_hz, suggested_filters.dterm_lpf2_hz),
            ("Dyn Notch Count", str(current.dyn_notch_count),
             str(suggested_filters.dyn_notch_count),
             current.dyn_notch_count, suggested_filters.dyn_notch_count),
            ("Dyn Notch Q", str(current.dyn_notch_q),
             str(suggested_filters.dyn_notch_q),
             current.dyn_notch_q, suggested_filters.dyn_notch_q),
            ("RPM Harmonics", str(current.rpm_harmonics),
             str(suggested_filters.rpm_harmonics),
             current.rpm_harmonics, suggested_filters.rpm_harmonics),
        ]

        for row, (name, cur_str, sug_str, cur_num, sug_num) in enumerate(filter_rows):
            # Setting name (column 0) already set in __init__

            # Current value
            cur_item = QTableWidgetItem(cur_str)
            cur_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._filter_table.setItem(row, 1, cur_item)

            # Suggested value (color-coded)
            sug_item = QTableWidgetItem(sug_str)
            sug_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            if sug_num > cur_num:
                sug_item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor("#44ff44")
                )
            elif sug_num < cur_num:
                sug_item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor("#ff6b6b")
                )

            self._filter_table.setItem(row, 2, sug_item)

    def _update_confidence(self, confidence: float) -> None:
        """Update the confidence indicator label."""
        pct = int(confidence * 100)
        if confidence >= 0.75:
            level = "High"
            color = "#44ff44"
        elif confidence >= 0.50:
            level = "Medium"
            color = "#ffaa00"
        else:
            level = "Low"
            color = "#ff6b6b"

        self._confidence_label.setText(f"Confidence: {level} ({pct}%)")
        self._confidence_label.setStyleSheet(
            f"color: {color}; font-size: 14px; font-weight: bold; background: transparent;"
        )

    def _copy_cli(self) -> None:
        """Copy CLI commands to the system clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._cli_text.toPlainText())
