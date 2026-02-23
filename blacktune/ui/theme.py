"""Dark theme stylesheet for BlackTune.

FPV pilots live in dark mode. This module provides a deep navy / cyan
accent stylesheet targeting all standard PyQt6 widgets used in the app.
"""

# Colour palette
BG_PRIMARY = "#1a1a2e"
BG_SECONDARY = "#16213e"
ACCENT = "#00d4ff"
TEXT = "#e0e0e0"
TEXT_MUTED = "#8888aa"
BORDER = "#2d2d4e"
BUTTON_BG = "#0f3460"
BUTTON_HOVER = "#1a5276"


def dark_stylesheet() -> str:
    """Return the full QSS stylesheet string."""
    return f"""
/* ── Main Window & generic widgets ──────────────────────────── */

QMainWindow {{
    background-color: {BG_PRIMARY};
    color: {TEXT};
}}

QWidget {{
    background-color: {BG_PRIMARY};
    color: {TEXT};
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}

/* ── Tab Widget ─────────────────────────────────────────────── */

QTabWidget::pane {{
    border: 1px solid {BORDER};
    background-color: {BG_PRIMARY};
    border-radius: 4px;
}}

QTabBar::tab {{
    background-color: {BG_SECONDARY};
    color: {TEXT_MUTED};
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid {BORDER};
    border-bottom: none;
}}

QTabBar::tab:selected {{
    background-color: {BG_PRIMARY};
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background-color: {BUTTON_BG};
    color: {TEXT};
}}

/* ── Buttons ────────────────────────────────────────────────── */

QPushButton {{
    background-color: {BUTTON_BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 16px;
    min-height: 24px;
}}

QPushButton:hover {{
    background-color: {BUTTON_HOVER};
    border-color: {ACCENT};
}}

QPushButton:pressed {{
    background-color: {BG_SECONDARY};
}}

QPushButton#primary {{
    background-color: {ACCENT};
    color: {BG_PRIMARY};
    font-weight: bold;
    border: none;
}}

QPushButton#primary:hover {{
    background-color: #33ddff;
}}

/* ── Combo Box ──────────────────────────────────────────────── */

QComboBox {{
    background-color: {BG_SECONDARY};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
}}

QComboBox:hover {{
    border-color: {ACCENT};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_SECONDARY};
    color: {TEXT};
    selection-background-color: {BUTTON_BG};
    selection-color: {ACCENT};
    border: 1px solid {BORDER};
}}

/* ── Labels ─────────────────────────────────────────────────── */

QLabel {{
    color: {TEXT};
    background: transparent;
}}

QLabel#header {{
    font-size: 18px;
    font-weight: bold;
    color: {ACCENT};
}}

/* ── Group Box ──────────────────────────────────────────────── */

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: {TEXT};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {ACCENT};
}}

/* ── Scroll Bars ────────────────────────────────────────────── */

QScrollBar:vertical {{
    background-color: {BG_SECONDARY};
    width: 10px;
    margin: 0;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background-color: {BORDER};
    min-height: 30px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {ACCENT};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {BG_SECONDARY};
    height: 10px;
    margin: 0;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background-color: {BORDER};
    min-width: 30px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {ACCENT};
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Table Widget ───────────────────────────────────────────── */

QTableWidget {{
    background-color: {BG_SECONDARY};
    color: {TEXT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 4px;
    selection-background-color: {BUTTON_BG};
    selection-color: {ACCENT};
}}

QTableWidget::item {{
    padding: 4px 8px;
}}

QTableWidget::item:selected {{
    background-color: {BUTTON_BG};
    color: {ACCENT};
}}

/* ── Header View (table/tree headers) ───────────────────────── */

QHeaderView::section {{
    background-color: {BG_PRIMARY};
    color: {ACCENT};
    padding: 6px 8px;
    border: none;
    border-bottom: 2px solid {BORDER};
    font-weight: bold;
}}

QHeaderView::section:hover {{
    background-color: {BUTTON_BG};
}}

/* ── Text Edit ──────────────────────────────────────────────── */

QTextEdit {{
    background-color: {BG_SECONDARY};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px;
    selection-background-color: {BUTTON_BG};
    selection-color: {ACCENT};
}}

/* ── Status Bar ─────────────────────────────────────────────── */

QStatusBar {{
    background-color: {BG_SECONDARY};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}

QStatusBar::item {{
    border: none;
}}

/* ── Menu Bar ───────────────────────────────────────────────── */

QMenuBar {{
    background-color: {BG_SECONDARY};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
}}

QMenuBar::item:selected {{
    background-color: {BUTTON_BG};
    color: {ACCENT};
}}

QMenu {{
    background-color: {BG_SECONDARY};
    color: {TEXT};
    border: 1px solid {BORDER};
}}

QMenu::item:selected {{
    background-color: {BUTTON_BG};
    color: {ACCENT};
}}
"""
