"""Session history persistence for BlackTune.

Stores tuning session results as JSON at ``~/.blacktune/history.json``
so pilots can track how their tune evolves over time.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_HISTORY_DIR = Path.home() / ".blacktune"
_HISTORY_FILE = _HISTORY_DIR / "history.json"


def save_session(
    filename: str,
    current_pids: dict,
    suggested_pids: dict,
    profile: dict,
    metrics: dict,
) -> None:
    """Save a tuning session to history.

    Args:
        filename: BBL file name.
        current_pids: from ``PIDValues.as_dict()``.
        suggested_pids: from ``PIDValues.as_dict()``.
        profile: quad profile dict (cell_count, prop_size, ...).
        metrics: per-axis metrics (e.g. ``roll_overshoot``).
    """
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "profile": profile,
        "current_pids": current_pids,
        "suggested_pids": suggested_pids,
        "metrics": metrics,
    }
    history.append(entry)
    _HISTORY_FILE.write_text(json.dumps(history, indent=2))


def load_history() -> list[dict]:
    """Load history list. Returns empty list if no history."""
    if not _HISTORY_FILE.exists():
        return []
    try:
        return json.loads(_HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def clear_history() -> None:
    """Delete all history."""
    if _HISTORY_FILE.exists():
        _HISTORY_FILE.unlink()
