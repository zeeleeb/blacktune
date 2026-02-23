"""History tab -- table of past tuning sessions.

Shows a scrollable table of prior tuning runs so pilots can track how
their PID values have evolved across sessions.
"""
from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from blacktune.history import load_history, clear_history


class HistoryTab(QWidget):
    """Tuning history table with refresh and clear controls."""

    def __init__(self) -> None:
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Header
        header = QLabel("Tuning History")
        header.setObjectName("header")
        root.addWidget(header)

        # Table
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels([
            "Date",
            "File",
            "Profile",
            "Roll P",
            "Pitch P",
            "Yaw P",
            "Overshoot %",
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        root.addWidget(self._table, stretch=1)

        # Empty-state label (shown when no history)
        self._empty_label = QLabel("No tuning history yet. Analyze a log to get started.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #8888aa; font-size: 15px;")
        root.addWidget(self._empty_label)

        # Bottom button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._clear_btn = QPushButton("Clear History")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self._clear_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(self._refresh_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # Initial population
        self.refresh()

    # ── Public API ─────────────────────────────────────────────

    def refresh(self) -> None:
        """Reload history from disk and repopulate the table."""
        history = load_history()
        self._table.setRowCount(0)

        if not history:
            self._empty_label.show()
            self._table.hide()
            return

        self._empty_label.hide()
        self._table.show()
        self._table.setRowCount(len(history))

        for row, entry in enumerate(reversed(history)):  # newest first
            # Date -- format as "YYYY-MM-DD HH:MM"
            ts_raw = entry.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts_raw)
                ts_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                ts_str = ts_raw[:16] if ts_raw else "?"

            self._set_cell(row, 0, ts_str)

            # File
            self._set_cell(row, 1, entry.get("filename", "?"))

            # Profile -- e.g. "5S 5\" Freestyle"
            profile = entry.get("profile", {})
            profile_str = self._format_profile(profile)
            self._set_cell(row, 2, profile_str)

            # PID change columns: "curr->sugg"
            current_pids = entry.get("current_pids", {})
            suggested_pids = entry.get("suggested_pids", {})

            for col, key in ((3, "roll_p"), (4, "pitch_p"), (5, "yaw_p")):
                cur = current_pids.get(key)
                sug = suggested_pids.get(key)
                self._set_pid_cell(row, col, cur, sug)

            # Overshoot %
            metrics = entry.get("metrics", {})
            overshoot_parts = []
            for axis in ("roll", "pitch", "yaw"):
                val = metrics.get(f"{axis}_overshoot")
                if val is not None:
                    overshoot_parts.append(f"{axis[0].upper()}:{val:.1f}")
            overshoot_str = " ".join(overshoot_parts) if overshoot_parts else "-"
            self._set_cell(row, 6, overshoot_str)

    # ── Private helpers ────────────────────────────────────────

    def _on_clear(self) -> None:
        """Clear history and refresh the view."""
        clear_history()
        self.refresh()

    def _set_cell(self, row: int, col: int, text: str) -> None:
        """Set a plain text table cell, center-aligned."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, col, item)

    def _set_pid_cell(
        self, row: int, col: int, current, suggested
    ) -> None:
        """Set a PID change cell like '45->48' with color coding."""
        if current is None or suggested is None:
            self._set_cell(row, col, "-")
            return

        cur_r = round(current)
        sug_r = round(suggested)
        text = f"{cur_r}->{sug_r}"
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        if sug_r > cur_r:
            item.setForeground(QColor("#44ff44"))  # green = increased
        elif sug_r < cur_r:
            item.setForeground(QColor("#ff6b6b"))  # red = decreased

        self._table.setItem(row, col, item)

    @staticmethod
    def _format_profile(profile: dict) -> str:
        """Format a profile dict as e.g. '5S 5\" Freestyle'."""
        parts = []
        cell = profile.get("cell_count")
        if cell is not None:
            parts.append(f"{cell}S")
        prop = profile.get("prop_size")
        if prop is not None:
            parts.append(f'{prop:.0f}"')
        style = profile.get("flying_style")
        if style:
            parts.append(style.replace("_", " ").title())
        return " ".join(parts) if parts else "-"
