# tests/test_history.py
"""Tests for session history persistence."""
import json

import blacktune.history as history_mod


def test_save_and_load(tmp_path, monkeypatch):
    """save_session persists an entry that load_history can read back."""
    monkeypatch.setattr(history_mod, "_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(history_mod, "_HISTORY_FILE", tmp_path / "history.json")

    history_mod.save_session(
        filename="test_log.bbl",
        current_pids={"roll_p": 45, "roll_i": 80, "roll_d": 30},
        suggested_pids={"roll_p": 48, "roll_i": 82, "roll_d": 28},
        profile={"cell_count": 6, "prop_size": 5.0, "flying_style": "freestyle"},
        metrics={"roll_overshoot": 12.5, "pitch_overshoot": 8.0},
    )

    loaded = history_mod.load_history()
    assert len(loaded) == 1
    entry = loaded[0]
    assert entry["filename"] == "test_log.bbl"
    assert entry["current_pids"]["roll_p"] == 45
    assert entry["suggested_pids"]["roll_p"] == 48
    assert entry["profile"]["cell_count"] == 6
    assert entry["metrics"]["roll_overshoot"] == 12.5
    assert "timestamp" in entry


def test_save_multiple(tmp_path, monkeypatch):
    """Multiple saves accumulate entries."""
    monkeypatch.setattr(history_mod, "_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(history_mod, "_HISTORY_FILE", tmp_path / "history.json")

    for i in range(3):
        history_mod.save_session(
            filename=f"log_{i}.bbl",
            current_pids={"roll_p": 45 + i},
            suggested_pids={"roll_p": 48 + i},
            profile={"cell_count": 4},
            metrics={"roll_overshoot": 10.0 + i},
        )

    loaded = history_mod.load_history()
    assert len(loaded) == 3
    assert loaded[0]["filename"] == "log_0.bbl"
    assert loaded[2]["filename"] == "log_2.bbl"


def test_load_empty(tmp_path, monkeypatch):
    """No file exists -> empty list."""
    monkeypatch.setattr(history_mod, "_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(history_mod, "_HISTORY_FILE", tmp_path / "history.json")

    assert history_mod.load_history() == []


def test_load_corrupt_json(tmp_path, monkeypatch):
    """Corrupt JSON file returns empty list rather than crashing."""
    monkeypatch.setattr(history_mod, "_HISTORY_DIR", tmp_path)
    hist_file = tmp_path / "history.json"
    monkeypatch.setattr(history_mod, "_HISTORY_FILE", hist_file)

    hist_file.write_text("not valid json {{{")
    assert history_mod.load_history() == []


def test_clear_history(tmp_path, monkeypatch):
    """clear_history removes the file; subsequent load returns empty."""
    monkeypatch.setattr(history_mod, "_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(history_mod, "_HISTORY_FILE", tmp_path / "history.json")

    history_mod.save_session(
        filename="test.bbl",
        current_pids={"roll_p": 45},
        suggested_pids={"roll_p": 48},
        profile={"cell_count": 4},
        metrics={"roll_overshoot": 10.0},
    )
    assert len(history_mod.load_history()) == 1

    history_mod.clear_history()
    assert history_mod.load_history() == []


def test_clear_history_no_file(tmp_path, monkeypatch):
    """clear_history is safe when no file exists."""
    monkeypatch.setattr(history_mod, "_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(history_mod, "_HISTORY_FILE", tmp_path / "history.json")

    # Should not raise
    history_mod.clear_history()


def test_timestamp_is_iso_format(tmp_path, monkeypatch):
    """Saved timestamp parses as ISO 8601."""
    from datetime import datetime

    monkeypatch.setattr(history_mod, "_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(history_mod, "_HISTORY_FILE", tmp_path / "history.json")

    history_mod.save_session(
        filename="test.bbl",
        current_pids={},
        suggested_pids={},
        profile={},
        metrics={},
    )

    loaded = history_mod.load_history()
    ts = loaded[0]["timestamp"]
    # Should not raise
    parsed = datetime.fromisoformat(ts)
    assert parsed.year >= 2024


def test_history_file_is_valid_json(tmp_path, monkeypatch):
    """The on-disk file is valid, pretty-printed JSON."""
    monkeypatch.setattr(history_mod, "_HISTORY_DIR", tmp_path)
    hist_file = tmp_path / "history.json"
    monkeypatch.setattr(history_mod, "_HISTORY_FILE", hist_file)

    history_mod.save_session(
        filename="test.bbl",
        current_pids={"roll_p": 45},
        suggested_pids={"roll_p": 48},
        profile={},
        metrics={},
    )

    raw = hist_file.read_text()
    data = json.loads(raw)
    assert isinstance(data, list)
    assert len(data) == 1
    # Pretty-printed => contains newlines
    assert "\n" in raw
