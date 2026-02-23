"""Log Viewer tab with time-series pyqtgraph plots.

Provides four synchronized plot rows showing gyro vs setpoint,
PID terms, motor outputs, and throttle -- the core data an FPV
pilot needs to visually assess flight controller tuning.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from blacktune.models import FlightLog

pg.setConfigOptions(antialias=True)


class ViewerTab(QWidget):
    """Log Viewer tab with time-series pyqtgraph plots."""

    # FPV-themed colors
    COLORS = {
        "roll": "#ff6b6b",      # Red
        "pitch": "#4ecdc4",     # Teal
        "yaw": "#ffe66d",       # Yellow
        "setpoint": "#a8e6cf",  # Light green (dashed)
        "throttle": "#ff8b94",  # Pink
        "motor0": "#ff6b6b",
        "motor1": "#4ecdc4",
        "motor2": "#ffe66d",
        "motor3": "#a8e6cf",
    }

    def __init__(self) -> None:
        super().__init__()

        self._log: Optional[FlightLog] = None

        # -- Checkbox row for axis toggles --
        self._cb_roll = QCheckBox("Roll")
        self._cb_roll.setChecked(True)
        self._cb_pitch = QCheckBox("Pitch")
        self._cb_pitch.setChecked(True)
        self._cb_yaw = QCheckBox("Yaw")
        self._cb_yaw.setChecked(True)

        for cb in (self._cb_roll, self._cb_pitch, self._cb_yaw):
            cb.stateChanged.connect(self._update_plots)

        cb_layout = QHBoxLayout()
        cb_layout.setContentsMargins(8, 4, 8, 0)
        cb_layout.addWidget(QLabel("Show axes:"))
        cb_layout.addWidget(self._cb_roll)
        cb_layout.addWidget(self._cb_pitch)
        cb_layout.addWidget(self._cb_yaw)
        cb_layout.addStretch()

        cb_row = QWidget()
        cb_row.setLayout(cb_layout)

        # -- pyqtgraph plot area --
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("#16213e")

        self.gyro_plot: pg.PlotItem = self.plot_widget.addPlot(
            row=0, col=0, title="Gyro vs Setpoint (deg/s)"
        )
        self.pid_plot: pg.PlotItem = self.plot_widget.addPlot(
            row=1, col=0, title="PID Terms"
        )
        self.motor_plot: pg.PlotItem = self.plot_widget.addPlot(
            row=2, col=0, title="Motor Output"
        )
        self.throttle_plot: pg.PlotItem = self.plot_widget.addPlot(
            row=3, col=0, title="Throttle"
        )

        # Link X axes for synchronized scroll/zoom
        self.pid_plot.setXLink(self.gyro_plot)
        self.motor_plot.setXLink(self.gyro_plot)
        self.throttle_plot.setXLink(self.gyro_plot)

        # Grid + styling on every plot
        for plot in (self.gyro_plot, self.pid_plot, self.motor_plot, self.throttle_plot):
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.addLegend(offset=(10, 10))
            plot.getAxis("bottom").setLabel("Time", units="s")

        # -- Empty-state label --
        self._empty_label = QLabel("Open a BBL file to view flight data")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #8888aa; font-size: 18px;")

        # -- Stacked layout: show empty label OR (checkboxes + plots) --
        self._plot_container = QWidget()
        plot_layout = QVBoxLayout(self._plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(cb_row)
        plot_layout.addWidget(self.plot_widget, stretch=1)

        self._stack = QStackedLayout()
        self._stack.addWidget(self._empty_label)    # index 0
        self._stack.addWidget(self._plot_container)  # index 1
        self._stack.setCurrentIndex(0)

        self.setLayout(self._stack)

    # ── Public API ────────────────────────────────────────────────

    def load_data(self, log: FlightLog) -> None:
        """Load a flight log and display its data."""
        self._log = log
        self._stack.setCurrentIndex(1)
        self._update_plots()

    # ── Internal ──────────────────────────────────────────────────

    def _checked_axes(self) -> list[str]:
        """Return list of axis names whose checkboxes are checked."""
        axes = []
        if self._cb_roll.isChecked():
            axes.append("roll")
        if self._cb_pitch.isChecked():
            axes.append("pitch")
        if self._cb_yaw.isChecked():
            axes.append("yaw")
        return axes

    def _update_plots(self) -> None:
        """Clear and re-draw all plots based on current checkbox state."""
        # Clear all plots
        self.gyro_plot.clear()
        self.pid_plot.clear()
        self.motor_plot.clear()
        self.throttle_plot.clear()

        # Re-add legends (clear() removes them)
        for plot in (self.gyro_plot, self.pid_plot, self.motor_plot, self.throttle_plot):
            if plot.legend is not None:
                plot.legend.scene().removeItem(plot.legend)
            plot.addLegend(offset=(10, 10))

        if self._log is None:
            return

        checked = self._checked_axes()

        # ---- Gyro vs Setpoint ----
        for axis_name in checked:
            axis_data = getattr(self._log, axis_name)
            color = self.COLORS[axis_name]

            self.gyro_plot.plot(
                axis_data.time,
                axis_data.gyro,
                pen=pg.mkPen(color, width=1.5),
                name=f"{axis_name.capitalize()} Gyro",
            )
            self.gyro_plot.plot(
                axis_data.time,
                axis_data.setpoint,
                pen=pg.mkPen(
                    self.COLORS["setpoint"],
                    width=1,
                    style=Qt.PenStyle.DashLine,
                ),
                name=f"{axis_name.capitalize()} Setpoint",
            )

        # ---- PID Terms ----
        for axis_name in checked:
            axis_data = getattr(self._log, axis_name)
            color = self.COLORS[axis_name]

            self.pid_plot.plot(
                axis_data.time,
                axis_data.p_term,
                pen=pg.mkPen(color, width=1.5),
                name=f"{axis_name.capitalize()} P",
            )
            self.pid_plot.plot(
                axis_data.time,
                axis_data.d_term,
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                name=f"{axis_name.capitalize()} D",
            )

        # ---- Motor Output (always show all 4) ----
        num_motors = self._log.motors.shape[0]
        # time axis from roll (any axis would work -- same sample count)
        time = self._log.roll.time
        for i in range(min(num_motors, 4)):
            motor_color = self.COLORS.get(f"motor{i}", "#ffffff")
            self.motor_plot.plot(
                time,
                self._log.motors[i],
                pen=pg.mkPen(motor_color, width=1.5),
                name=f"Motor {i}",
            )

        # ---- Throttle ----
        self.throttle_plot.plot(
            time,
            self._log.throttle,
            pen=pg.mkPen(self.COLORS["throttle"], width=1.5),
            name="Throttle",
        )
