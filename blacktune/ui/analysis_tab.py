"""Analysis Dashboard tab -- FFT spectra, step response, issues, motor heat.

Provides four visual sections giving the pilot an at-a-glance
view of their tune quality:

1. FFT spectrum plots (roll / pitch / yaw)
2. Overlaid step response plot
3. Color-coded issue cards
4. Motor heat indicator bars
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from blacktune.models import AnalysisResult, FlightLog
from blacktune.analyzers.noise import compute_fft_spectrum, find_noise_peaks
from blacktune.analyzers.step_response import compute_step_response

pg.setConfigOptions(antialias=True)

# Axis colors matching the Log Viewer palette
AXIS_COLORS = {
    "roll": "#ff6b6b",
    "pitch": "#4ecdc4",
    "yaw": "#ffe66d",
}


class IssueCard(QFrame):
    """Color-coded issue card with severity border, header, and detail text."""

    SEVERITY_COLORS = {
        "red": "#ff4444",
        "yellow": "#ffaa00",
        "green": "#44ff44",
    }

    def __init__(
        self,
        axis: str,
        category: str,
        severity: str,
        message: str,
        detail: str,
    ) -> None:
        super().__init__()

        color = self.SEVERITY_COLORS.get(severity, "#888888")

        self.setStyleSheet(
            f"IssueCard {{"
            f"  background-color: #1e2a4a;"
            f"  border-left: 4px solid {color};"
            f"  border-radius: 4px;"
            f"  padding: 8px;"
            f"  margin: 2px 0px;"
            f"}}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        header = QLabel(f"[{axis.upper()}] {message}")
        header.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: 13px; background: transparent;"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        body = QLabel(detail)
        body.setStyleSheet(
            "color: #aaaacc; font-size: 11px; background: transparent;"
        )
        body.setWordWrap(True)
        layout.addWidget(body)


class MotorHeatBar(QWidget):
    """Visual motor heat indicator with colored bar and status label."""

    def __init__(self, motor_idx: int, heat_value: float) -> None:
        super().__init__()

        heat_value = max(0.0, min(1.0, heat_value))
        pct = int(heat_value * 100)

        # Determine color and label based on heat threshold
        if heat_value < 0.3:
            color = "#44ff44"
            status = "Cool"
        elif heat_value < 0.6:
            color = "#ffaa00"
            status = "Warm"
        elif heat_value < 0.8:
            color = "#ff8800"
            status = "Hot"
        else:
            color = "#ff4444"
            status = "DANGER"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        # Motor label
        motor_label = QLabel(f"Motor {motor_idx}")
        motor_label.setFixedWidth(60)
        motor_label.setStyleSheet("color: #e0e0e0; background: transparent;")
        layout.addWidget(motor_label)

        # Bar container (fixed height frame with colored fill inside)
        bar_container = QFrame()
        bar_container.setFixedHeight(18)
        bar_container.setStyleSheet(
            "background-color: #16213e; border: 1px solid #2d2d4e; border-radius: 3px;"
        )

        # Inner colored bar
        bar_fill = QFrame(bar_container)
        bar_fill.setStyleSheet(
            f"background-color: {color}; border-radius: 2px; border: none;"
        )
        # We set geometry after the container is sized; use a layout trick
        bar_inner_layout = QHBoxLayout(bar_container)
        bar_inner_layout.setContentsMargins(1, 1, 1, 1)
        bar_inner_layout.setSpacing(0)

        bar_fill_widget = QWidget()
        bar_fill_widget.setStyleSheet(
            f"background-color: {color}; border-radius: 2px;"
        )
        bar_spacer = QWidget()
        bar_spacer.setStyleSheet("background: transparent;")

        # Use stretch factors to set relative widths
        fill_stretch = max(1, int(heat_value * 100))
        empty_stretch = max(1, 100 - fill_stretch)
        bar_inner_layout.addWidget(bar_fill_widget, stretch=fill_stretch)
        bar_inner_layout.addWidget(bar_spacer, stretch=empty_stretch)

        layout.addWidget(bar_container, stretch=1)

        # Status label
        status_label = QLabel(f"{status} ({pct}%)")
        status_label.setFixedWidth(100)
        status_label.setStyleSheet(f"color: {color}; background: transparent;")
        layout.addWidget(status_label)


class AnalysisTab(QWidget):
    """Analysis Dashboard tab with FFT, step response, issues, and motor heat."""

    def __init__(self) -> None:
        super().__init__()

        self._log: Optional[FlightLog] = None
        self._result: Optional[AnalysisResult] = None

        # -- Build the dashboard layout --
        self._dashboard = QWidget()
        dashboard_layout = QVBoxLayout(self._dashboard)
        dashboard_layout.setContentsMargins(4, 4, 4, 4)
        dashboard_layout.setSpacing(4)

        # ---- Section 1: FFT Spectrum Plots (3 columns) ----
        self._fft_widget = pg.GraphicsLayoutWidget()
        self._fft_widget.setBackground("#16213e")
        self._fft_widget.setMinimumHeight(200)

        self.fft_plots: dict[str, pg.PlotItem] = {}
        for col, axis_name in enumerate(("roll", "pitch", "yaw")):
            plot = self._fft_widget.addPlot(
                row=0, col=col, title=f"{axis_name.capitalize()} FFT Spectrum"
            )
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.getAxis("bottom").setLabel("Frequency", units="Hz")
            plot.getAxis("left").setLabel("Power", units="dB")
            self.fft_plots[axis_name] = plot

        dashboard_layout.addWidget(self._fft_widget, stretch=3)

        # ---- Section 2: Step Response Plot ----
        self._step_widget = pg.GraphicsLayoutWidget()
        self._step_widget.setBackground("#16213e")
        self._step_widget.setMinimumHeight(180)

        self.step_plot: pg.PlotItem = self._step_widget.addPlot(
            row=0, col=0, title="Step Response"
        )
        self.step_plot.showGrid(x=True, y=True, alpha=0.2)
        self.step_plot.getAxis("bottom").setLabel("Time", units="ms")
        self.step_plot.getAxis("left").setLabel("Response")
        self.step_plot.addLegend(offset=(10, 10))

        dashboard_layout.addWidget(self._step_widget, stretch=2)

        # ---- Section 3: Bottom row -- issues (left) + motor heat (right) ----
        bottom_row = QWidget()
        bottom_layout = QHBoxLayout(bottom_row)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        # Issues scroll area
        self._issues_scroll = QScrollArea()
        self._issues_scroll.setWidgetResizable(True)
        self._issues_scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #2d2d4e; border-radius: 4px; background: #1a1a2e; }"
        )

        self._issues_container = QWidget()
        self._issues_layout = QVBoxLayout(self._issues_container)
        self._issues_layout.setContentsMargins(4, 4, 4, 4)
        self._issues_layout.setSpacing(4)
        self._issues_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        issues_header = QLabel("Detected Issues")
        issues_header.setStyleSheet(
            "color: #00d4ff; font-weight: bold; font-size: 14px; background: transparent;"
        )
        self._issues_layout.addWidget(issues_header)

        self._issues_scroll.setWidget(self._issues_container)
        bottom_layout.addWidget(self._issues_scroll, stretch=3)

        # Motor heat frame
        self._heat_frame = QFrame()
        self._heat_frame.setStyleSheet(
            "QFrame { border: 1px solid #2d2d4e; border-radius: 4px; background: #1a1a2e; }"
        )
        self._heat_layout = QVBoxLayout(self._heat_frame)
        self._heat_layout.setContentsMargins(8, 8, 8, 8)
        self._heat_layout.setSpacing(6)
        self._heat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        heat_header = QLabel("Motor Heat Index")
        heat_header.setStyleSheet(
            "color: #00d4ff; font-weight: bold; font-size: 14px; background: transparent;"
        )
        self._heat_layout.addWidget(heat_header)

        bottom_layout.addWidget(self._heat_frame, stretch=2)

        dashboard_layout.addWidget(bottom_row, stretch=2)

        # -- Empty-state label --
        self._empty_label = QLabel("Run analysis on a flight log to view results")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #8888aa; font-size: 18px;")

        # -- Stacked layout: empty label OR dashboard --
        self._stack = QStackedLayout()
        self._stack.addWidget(self._empty_label)   # index 0
        self._stack.addWidget(self._dashboard)      # index 1
        self._stack.setCurrentIndex(0)

        self.setLayout(self._stack)

    # ── Public API ────────────────────────────────────────────────

    def load_results(self, log: FlightLog, result: AnalysisResult) -> None:
        """Populate all dashboard sections from analysis results.

        Parameters
        ----------
        log : FlightLog
            The parsed flight log (needed for raw signal data).
        result : AnalysisResult
            Analysis results containing metrics, peaks, issues, motor heat.
        """
        self._log = log
        self._result = result
        self._stack.setCurrentIndex(1)

        self._populate_fft(log, result)
        self._populate_step_response(log)
        self._populate_issues(result)
        self._populate_motor_heat(result)

    # ── FFT Spectrum Plots ────────────────────────────────────────

    def _populate_fft(self, log: FlightLog, result: AnalysisResult) -> None:
        """Compute and plot FFT spectrum for each axis, marking noise peaks."""
        for axis_name in ("roll", "pitch", "yaw"):
            plot = self.fft_plots[axis_name]
            plot.clear()

            axis_data = getattr(log, axis_name)
            color = AXIS_COLORS[axis_name]

            # Compute spectrum
            freqs, psd_db = compute_fft_spectrum(
                signal=axis_data.gyro,
                sample_rate=log.sample_rate,
            )

            # Plot the spectrum
            plot.plot(
                freqs,
                psd_db,
                pen=pg.mkPen(color, width=1.5),
            )

            # Mark detected noise peaks with vertical dashed red lines
            peaks = result.noise_peaks.get(axis_name, [])
            for freq_hz, _amp_db in peaks:
                peak_line = pg.InfiniteLine(
                    pos=freq_hz,
                    angle=90,
                    pen=pg.mkPen("#ff0000", width=1, style=Qt.PenStyle.DashLine),
                )
                plot.addItem(peak_line)

    # ── Step Response Plot ────────────────────────────────────────

    def _populate_step_response(self, log: FlightLog) -> None:
        """Compute and plot overlaid step responses for all axes."""
        self.step_plot.clear()

        # Re-add legend (clear() removes it)
        if self.step_plot.legend is not None:
            self.step_plot.legend.scene().removeItem(self.step_plot.legend)
        self.step_plot.addLegend(offset=(10, 10))

        # Horizontal dashed white line at y=1.0 (target)
        target_line = pg.InfiniteLine(
            pos=1.0,
            angle=0,
            pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DashLine),
        )
        self.step_plot.addItem(target_line)

        for axis_name in ("roll", "pitch", "yaw"):
            axis_data = getattr(log, axis_name)
            color = AXIS_COLORS[axis_name]

            response, resp_time = compute_step_response(
                setpoint=axis_data.setpoint,
                gyro=axis_data.gyro,
                sample_rate=log.sample_rate,
            )

            # Convert time to milliseconds
            time_ms = resp_time * 1000.0

            self.step_plot.plot(
                time_ms,
                response,
                pen=pg.mkPen(color, width=2),
                name=axis_name.capitalize(),
            )

    # ── Issue Cards ───────────────────────────────────────────────

    def _populate_issues(self, result: AnalysisResult) -> None:
        """Create issue cards for each detected issue."""
        # Remove existing cards (keep the header at index 0)
        while self._issues_layout.count() > 1:
            item = self._issues_layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not result.issues:
            no_issues = QLabel("No issues detected -- tune looks good!")
            no_issues.setStyleSheet(
                "color: #44ff44; font-size: 13px; background: transparent;"
            )
            self._issues_layout.addWidget(no_issues)
            return

        for issue in result.issues:
            card = IssueCard(
                axis=issue.axis,
                category=issue.category,
                severity=issue.severity,
                message=issue.message,
                detail=issue.detail,
            )
            self._issues_layout.addWidget(card)

    # ── Motor Heat Bars ───────────────────────────────────────────

    def _populate_motor_heat(self, result: AnalysisResult) -> None:
        """Create motor heat bars for each motor."""
        # Remove existing bars (keep the header at index 0)
        while self._heat_layout.count() > 1:
            item = self._heat_layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for motor_idx in sorted(result.motor_heat_index.keys()):
            heat_value = result.motor_heat_index[motor_idx]
            bar = MotorHeatBar(motor_idx=motor_idx, heat_value=heat_value)
            self._heat_layout.addWidget(bar)
