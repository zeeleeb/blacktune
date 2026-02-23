"""Tests for blacktune.parser -- BBL/CSV parsing and header extraction."""
import csv
import os
import tempfile

import numpy as np
import pytest

from blacktune.models import AxisData, FilterSettings, FlightLog, PIDValues
from blacktune.parser import (
    filters_from_headers,
    load_log,
    parse_csv_log,
    parse_headers,
    pids_from_headers,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_csv(path: str, n: int = 200, dt_us: int = 125) -> None:
    """Write a synthetic Betaflight blackbox CSV with *n* rows.

    dt_us=125 -> 8 kHz sample rate (1_000_000 / 125 = 8000).
    """
    fieldnames = [
        "loopIteration", "time (us)",
        "axisP[0]", "axisP[1]", "axisP[2]",
        "axisI[0]", "axisI[1]", "axisI[2]",
        "axisD[0]", "axisD[1]", "axisD[2]",
        "axisF[0]", "axisF[1]", "axisF[2]",
        "rcCommand[0]", "rcCommand[1]", "rcCommand[2]", "rcCommand[3]",
        "setpoint[0]", "setpoint[1]", "setpoint[2]", "setpoint[3]",
        "gyroADC[0]", "gyroADC[1]", "gyroADC[2]",
        "motor[0]", "motor[1]", "motor[2]", "motor[3]",
    ]

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            t = i * dt_us
            row = {
                "loopIteration": str(i),
                "time (us)": str(t),
                "axisP[0]": str(10 + i * 0.1),
                "axisP[1]": str(11 + i * 0.1),
                "axisP[2]": str(5 + i * 0.05),
                "axisI[0]": str(1.0),
                "axisI[1]": str(1.1),
                "axisI[2]": str(0.5),
                "axisD[0]": str(-2.0),
                "axisD[1]": str(-2.1),
                "axisD[2]": str(0.0),
                "axisF[0]": str(3.0),
                "axisF[1]": str(3.1),
                "axisF[2]": str(0.0),
                "rcCommand[0]": str(i % 500),
                "rcCommand[1]": str(i % 500),
                "rcCommand[2]": str(i % 500),
                "rcCommand[3]": str(1500 + i % 500),
                "setpoint[0]": str(100 + i * 0.5),
                "setpoint[1]": str(101 + i * 0.5),
                "setpoint[2]": str(50 + i * 0.2),
                "setpoint[3]": str(0.5),
                "gyroADC[0]": str(99 + i * 0.5),
                "gyroADC[1]": str(100 + i * 0.5),
                "gyroADC[2]": str(49 + i * 0.2),
                "motor[0]": str(1100 + i),
                "motor[1]": str(1100 + i),
                "motor[2]": str(1100 + i),
                "motor[3]": str(1100 + i),
            }
            writer.writerow(row)


def _write_csv_with_extras(path: str, n: int = 100, dt_us: int = 125) -> None:
    """Write CSV that also has gyroUnfilt and eRPM columns."""
    fieldnames = [
        "loopIteration", "time (us)",
        "axisP[0]", "axisP[1]", "axisP[2]",
        "axisI[0]", "axisI[1]", "axisI[2]",
        "axisD[0]", "axisD[1]", "axisD[2]",
        "axisF[0]", "axisF[1]", "axisF[2]",
        "rcCommand[0]", "rcCommand[1]", "rcCommand[2]", "rcCommand[3]",
        "setpoint[0]", "setpoint[1]", "setpoint[2]", "setpoint[3]",
        "gyroADC[0]", "gyroADC[1]", "gyroADC[2]",
        "motor[0]", "motor[1]", "motor[2]", "motor[3]",
        "gyroUnfilt[0]", "gyroUnfilt[1]", "gyroUnfilt[2]",
        "eRPM[0]", "eRPM[1]", "eRPM[2]", "eRPM[3]",
    ]

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            t = i * dt_us
            row = {
                "loopIteration": str(i),
                "time (us)": str(t),
                "axisP[0]": "10", "axisP[1]": "11", "axisP[2]": "5",
                "axisI[0]": "1", "axisI[1]": "1", "axisI[2]": "0.5",
                "axisD[0]": "-2", "axisD[1]": "-2", "axisD[2]": "0",
                "axisF[0]": "3", "axisF[1]": "3", "axisF[2]": "0",
                "rcCommand[0]": "0", "rcCommand[1]": "0",
                "rcCommand[2]": "0", "rcCommand[3]": "1500",
                "setpoint[0]": "100", "setpoint[1]": "101",
                "setpoint[2]": "50", "setpoint[3]": "0.5",
                "gyroADC[0]": "99", "gyroADC[1]": "100", "gyroADC[2]": "49",
                "motor[0]": "1100", "motor[1]": "1100",
                "motor[2]": "1100", "motor[3]": "1100",
                "gyroUnfilt[0]": "102", "gyroUnfilt[1]": "103",
                "gyroUnfilt[2]": "51",
                "eRPM[0]": "25000", "eRPM[1]": "25000",
                "eRPM[2]": "25000", "eRPM[3]": "25000",
            }
            writer.writerow(row)


def _write_bbl_headers(path: str) -> None:
    """Write a fake BBL file whose first bytes are header lines.

    The H-line format matches real Betaflight BBL files.
    After headers, we write a few bytes of dummy binary so it resembles a
    real file (binary frames follow headers in real BBL).
    """
    lines = [
        "H Product:Blackbox flight data recorder by Nicholas Sherlock",
        "H Data version:2",
        "H Firmware revision:Betaflight 4.4.0",
        "H rollPID:45,80,30",
        "H pitchPID:47,84,34",
        "H yawPID:45,80,0",
        "H dMax:40,46,0",
        "H feedforward:120,125,120",
        "H gyro_lpf1_type:PT1",
        "H gyro_lpf1_static_hz:250",
        "H gyro_lpf1_dyn_min_hz:200",
        "H gyro_lpf1_dyn_max_hz:500",
        "H gyro_lpf2_type:BIQUAD",
        "H gyro_lpf2_static_hz:500",
        "H dterm_lpf1_type:PT1",
        "H dterm_lpf1_static_hz:75",
        "H dterm_lpf1_dyn_min_hz:60",
        "H dterm_lpf1_dyn_max_hz:150",
        "H dterm_lpf2_type:PT1",
        "H dterm_lpf2_static_hz:150",
        "H dyn_notch_count:3",
        "H dyn_notch_q:300",
        "H dyn_notch_min_hz:100",
        "H dyn_notch_max_hz:600",
        "H dshot_bidir:1",
        "H rpm_filter_harmonics:3",
        "H rpm_filter_min_hz:100",
        "H rpm_filter_q:500",
    ]
    with open(path, "w") as fh:
        for line in lines:
            fh.write(line + "\n")
        # dummy binary to simulate frame data
        fh.write("\x00\x01\x02\x03")


# ---------------------------------------------------------------------------
# parse_csv_log
# ---------------------------------------------------------------------------

class TestParseCsvLog:
    def test_basic_shape(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        _write_csv(csv_file, n=200, dt_us=125)

        log = parse_csv_log(csv_file)

        assert isinstance(log, FlightLog)
        assert log.roll.name == "roll"
        assert log.pitch.name == "pitch"
        assert log.yaw.name == "yaw"

    def test_array_lengths(self, tmp_path):
        n = 200
        csv_file = str(tmp_path / "test.csv")
        _write_csv(csv_file, n=n, dt_us=125)

        log = parse_csv_log(csv_file)

        assert len(log.roll.gyro) == n
        assert len(log.roll.setpoint) == n
        assert len(log.roll.p_term) == n
        assert len(log.roll.i_term) == n
        assert len(log.roll.d_term) == n
        assert len(log.roll.time) == n
        assert len(log.throttle) == n

    def test_sample_rate_estimation(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        _write_csv(csv_file, n=200, dt_us=125)  # 8kHz

        log = parse_csv_log(csv_file)

        assert log.sample_rate == 8000

    def test_sample_rate_4k(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        _write_csv(csv_file, n=200, dt_us=250)  # 4kHz

        log = parse_csv_log(csv_file)

        assert log.sample_rate == 4000

    def test_duration(self, tmp_path):
        n = 200
        dt_us = 125
        csv_file = str(tmp_path / "test.csv")
        _write_csv(csv_file, n=n, dt_us=dt_us)

        log = parse_csv_log(csv_file)

        expected_duration = (n - 1) * dt_us / 1_000_000
        assert abs(log.duration_s - expected_duration) < 0.001

    def test_motors_shape(self, tmp_path):
        n = 200
        csv_file = str(tmp_path / "test.csv")
        _write_csv(csv_file, n=n)

        log = parse_csv_log(csv_file)

        assert log.motors.shape == (4, n)

    def test_feedforward_parsed(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        _write_csv(csv_file, n=100)

        log = parse_csv_log(csv_file)

        assert log.roll.f_term is not None
        assert len(log.roll.f_term) == 100

    def test_gyro_unfiltered_when_present(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        _write_csv_with_extras(csv_file, n=100)

        log = parse_csv_log(csv_file)

        assert log.roll.gyro_unfiltered is not None
        assert len(log.roll.gyro_unfiltered) == 100

    def test_erpm_when_present(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        _write_csv_with_extras(csv_file, n=100)

        log = parse_csv_log(csv_file)

        assert log.erpm is not None
        assert log.erpm.shape == (4, 100)

    def test_default_pids_and_firmware(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        _write_csv(csv_file, n=100)

        log = parse_csv_log(csv_file)

        # CSV has no header info, so defaults
        assert isinstance(log.current_pids, PIDValues)
        assert log.firmware == "unknown"

    def test_data_values_correct(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        _write_csv(csv_file, n=10, dt_us=125)

        log = parse_csv_log(csv_file)

        # First row: axisP[0] = 10 + 0*0.1 = 10.0
        np.testing.assert_almost_equal(log.roll.p_term[0], 10.0)
        # Second row: axisP[0] = 10 + 1*0.1 = 10.1
        np.testing.assert_almost_equal(log.roll.p_term[1], 10.1)
        # time[0] should be 0.0 seconds
        np.testing.assert_almost_equal(log.roll.time[0], 0.0)


# ---------------------------------------------------------------------------
# parse_headers
# ---------------------------------------------------------------------------

class TestParseHeaders:
    def test_basic_parsing(self, tmp_path):
        bbl_file = str(tmp_path / "test.bbl")
        _write_bbl_headers(bbl_file)

        headers = parse_headers(bbl_file)

        assert headers["Firmware revision"] == "Betaflight 4.4.0"
        assert headers["rollPID"] == "45,80,30"
        assert headers["pitchPID"] == "47,84,34"
        assert headers["yawPID"] == "45,80,0"

    def test_filter_headers(self, tmp_path):
        bbl_file = str(tmp_path / "test.bbl")
        _write_bbl_headers(bbl_file)

        headers = parse_headers(bbl_file)

        assert headers["gyro_lpf1_static_hz"] == "250"
        assert headers["dterm_lpf1_static_hz"] == "75"
        assert headers["dyn_notch_count"] == "3"
        assert headers["rpm_filter_harmonics"] == "3"

    def test_stops_at_non_header(self, tmp_path):
        bbl_file = str(tmp_path / "test.bbl")
        _write_bbl_headers(bbl_file)

        headers = parse_headers(bbl_file)

        # Should not include binary garbage as a header key
        assert all(isinstance(v, str) for v in headers.values())
        assert len(headers) > 10  # We wrote ~28 header lines


# ---------------------------------------------------------------------------
# pids_from_headers
# ---------------------------------------------------------------------------

class TestPidsFromHeaders:
    def test_complete_pids(self):
        headers = {
            "rollPID": "45,80,30",
            "pitchPID": "47,84,34",
            "yawPID": "45,80,0",
            "dMax": "40,46,0",
            "feedforward": "120,125,120",
        }

        pids = pids_from_headers(headers)

        assert isinstance(pids, PIDValues)
        assert pids.roll_p == 45
        assert pids.roll_i == 80
        assert pids.roll_d == 30
        assert pids.roll_d_max == 40
        assert pids.roll_f == 120
        assert pids.pitch_p == 47
        assert pids.pitch_i == 84
        assert pids.pitch_d == 34
        assert pids.pitch_d_max == 46
        assert pids.pitch_f == 125
        assert pids.yaw_p == 45
        assert pids.yaw_i == 80
        assert pids.yaw_d == 0
        assert pids.yaw_f == 120

    def test_d_max_lowercase(self):
        """d_max is an alternative key name to dMax."""
        headers = {
            "rollPID": "45,80,30",
            "pitchPID": "47,84,34",
            "yawPID": "45,80,0",
            "d_max": "40,46,0",
        }

        pids = pids_from_headers(headers)

        assert pids.roll_d_max == 40
        assert pids.pitch_d_max == 46

    def test_feedforward_alt_key(self):
        """Some firmware versions use 'f' instead of 'feedforward'."""
        headers = {
            "rollPID": "45,80,30",
            "pitchPID": "47,84,34",
            "yawPID": "45,80,0",
            "f": "120,125,120",
        }

        pids = pids_from_headers(headers)

        assert pids.roll_f == 120
        assert pids.pitch_f == 125
        assert pids.yaw_f == 120

    def test_missing_optional_fields(self):
        """Only PID triplets required; d_max and feedforward optional."""
        headers = {
            "rollPID": "45,80,30",
            "pitchPID": "47,84,34",
            "yawPID": "45,80,0",
        }

        pids = pids_from_headers(headers)

        assert pids.roll_d_max == 0
        assert pids.roll_f == 0


# ---------------------------------------------------------------------------
# filters_from_headers
# ---------------------------------------------------------------------------

class TestFiltersFromHeaders:
    def test_complete_filters(self):
        headers = {
            "gyro_lpf1_type": "PT1",
            "gyro_lpf1_static_hz": "250",
            "gyro_lpf1_dyn_min_hz": "200",
            "gyro_lpf1_dyn_max_hz": "500",
            "gyro_lpf2_type": "BIQUAD",
            "gyro_lpf2_static_hz": "500",
            "dterm_lpf1_type": "PT1",
            "dterm_lpf1_static_hz": "75",
            "dterm_lpf1_dyn_min_hz": "60",
            "dterm_lpf1_dyn_max_hz": "150",
            "dterm_lpf2_type": "PT1",
            "dterm_lpf2_static_hz": "150",
            "dyn_notch_count": "3",
            "dyn_notch_q": "300",
            "dyn_notch_min_hz": "100",
            "dyn_notch_max_hz": "600",
            "rpm_filter_harmonics": "3",
            "rpm_filter_min_hz": "100",
            "rpm_filter_q": "500",
        }

        fs = filters_from_headers(headers)

        assert isinstance(fs, FilterSettings)
        assert fs.gyro_lpf1_type == "PT1"
        assert fs.gyro_lpf1_hz == 250
        assert fs.gyro_lpf1_dyn_min_hz == 200
        assert fs.gyro_lpf1_dyn_max_hz == 500
        assert fs.gyro_lpf2_type == "BIQUAD"
        assert fs.gyro_lpf2_hz == 500
        assert fs.dterm_lpf1_type == "PT1"
        assert fs.dterm_lpf1_hz == 75
        assert fs.dterm_lpf1_dyn_min_hz == 60
        assert fs.dterm_lpf1_dyn_max_hz == 150
        assert fs.dterm_lpf2_type == "PT1"
        assert fs.dterm_lpf2_hz == 150
        assert fs.dyn_notch_count == 3
        assert fs.dyn_notch_q == 300
        assert fs.dyn_notch_min_hz == 100
        assert fs.dyn_notch_max_hz == 600
        assert fs.rpm_harmonics == 3
        assert fs.rpm_min_hz == 100
        assert fs.rpm_q == 500

    def test_defaults_when_missing(self):
        """Should return FilterSettings with defaults when headers are empty."""
        fs = filters_from_headers({})

        assert isinstance(fs, FilterSettings)
        # defaults from the dataclass
        assert fs.gyro_lpf1_hz == 250
        assert fs.dterm_lpf1_hz == 75
        assert fs.dyn_notch_count == 3

    def test_partial_headers(self):
        """Only some filter headers present."""
        headers = {
            "gyro_lpf1_static_hz": "300",
            "dyn_notch_count": "5",
        }

        fs = filters_from_headers(headers)

        assert fs.gyro_lpf1_hz == 300
        assert fs.dyn_notch_count == 5
        # Everything else stays at defaults
        assert fs.dterm_lpf1_hz == 75


# ---------------------------------------------------------------------------
# load_log auto-detection
# ---------------------------------------------------------------------------

class TestLoadLog:
    def test_csv_detected(self, tmp_path):
        csv_file = str(tmp_path / "flight.csv")
        _write_csv(csv_file, n=100)

        log = load_log(csv_file)

        assert isinstance(log, FlightLog)
        assert log.roll.name == "roll"

    def test_unknown_extension_raises(self, tmp_path):
        txt_file = str(tmp_path / "flight.txt")
        with open(txt_file, "w") as fh:
            fh.write("not a log file")

        with pytest.raises(ValueError, match="Unsupported"):
            load_log(txt_file)


# ---------------------------------------------------------------------------
# orangebox import sanity
# ---------------------------------------------------------------------------

class TestOrangeboxImport:
    def test_orangebox_available(self):
        """Verify orangebox can be imported -- needed for load_bbl_orangebox."""
        from orangebox import Parser  # noqa: F401
