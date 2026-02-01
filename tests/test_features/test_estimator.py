"""Tests for AnalysisTimeEstimator."""

import struct
import tempfile
from pathlib import Path

import pytest

from src.features.estimator import (
    AnalysisTimeEstimator,
    FileMetaQuick,
    PreScanResult,
    TimeEstimate,
    BASE_ANALYZER_RATES,
    FEATURE_RATES,
)


def _create_wav(path: Path, duration_seconds: float = 1.0, sample_rate: int = 44100, channels: int = 2, bits: int = 16):
    """Create a minimal valid WAV file for testing."""
    byte_rate = sample_rate * channels * (bits // 8)
    data_size = int(duration_seconds * byte_rate)
    block_align = channels * (bits // 8)

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


class TestPreScan:
    def test_scan_single_wav(self, tmp_path):
        wav = tmp_path / "test.wav"
        _create_wav(wav, duration_seconds=2.0)

        estimator = AnalysisTimeEstimator()
        result = estimator.pre_scan([wav])

        assert isinstance(result, PreScanResult)
        assert result.total_count == 1
        assert abs(result.total_duration - 2.0) < 0.1
        assert result.format_distribution == {"wav": 1}

    def test_scan_multiple_files(self, tmp_path):
        paths = []
        for i in range(3):
            wav = tmp_path / f"test_{i}.wav"
            _create_wav(wav, duration_seconds=1.0)
            paths.append(wav)

        estimator = AnalysisTimeEstimator()
        result = estimator.pre_scan(paths)

        assert result.total_count == 3
        assert abs(result.total_duration - 3.0) < 0.3

    def test_scan_long_file_flagged(self, tmp_path):
        wav = tmp_path / "long.wav"
        _create_wav(wav, duration_seconds=45.0)

        estimator = AnalysisTimeEstimator()
        result = estimator.pre_scan([wav])

        assert len(result.long_files) == 1

    def test_scan_nonexistent_file(self, tmp_path):
        fake = tmp_path / "nonexistent.wav"
        estimator = AnalysisTimeEstimator()
        result = estimator.pre_scan([fake])

        assert result.total_count == 0

    def test_scan_mp3_heuristic(self, tmp_path):
        mp3 = tmp_path / "test.mp3"
        # Create a fake MP3 file (just raw bytes, no real MP3 header)
        mp3.write_bytes(b"\x00" * (128 * 1000 // 8))  # ~1 second at 128kbps

        estimator = AnalysisTimeEstimator()
        result = estimator.pre_scan([mp3])

        assert result.total_count == 1
        assert result.format_distribution == {"mp3": 1}
        assert abs(result.total_duration - 1.0) < 0.2


class TestEstimate:
    def test_basic_estimate(self, tmp_path):
        wav = tmp_path / "test.wav"
        _create_wav(wav, duration_seconds=5.0)

        estimator = AnalysisTimeEstimator()
        scan = estimator.pre_scan([wav])
        estimate = estimator.estimate(scan, enabled_analyzers=["source", "musical"])

        assert isinstance(estimate, TimeEstimate)
        assert estimate.file_count == 1
        assert estimate.estimated_total_seconds > 0

    def test_estimate_with_features(self, tmp_path):
        wav = tmp_path / "test.wav"
        _create_wav(wav, duration_seconds=5.0)

        estimator = AnalysisTimeEstimator()
        scan = estimator.pre_scan([wav])
        estimate = estimator.estimate(
            scan,
            enabled_analyzers=["source"],
            enabled_features=["production_notes", "sample_pack_curator"],
        )

        assert "production_notes" in estimate.breakdown
        assert "sample_pack_curator" in estimate.breakdown
        assert estimate.estimated_tokens > 0
        assert len(estimate.enabled_features) == 2

    def test_time_threshold_warning(self, tmp_path):
        wav = tmp_path / "test.wav"
        _create_wav(wav, duration_seconds=5.0)

        estimator = AnalysisTimeEstimator(time_threshold=0.001)
        scan = estimator.pre_scan([wav])
        estimate = estimator.estimate(scan)

        assert estimate.exceeds_time_threshold is True
        assert any("time" in w.lower() for w in estimate.warnings)

    def test_cost_threshold_warning(self, tmp_path):
        wav = tmp_path / "test.wav"
        _create_wav(wav, duration_seconds=5.0)

        estimator = AnalysisTimeEstimator(cost_threshold=0.0001)
        scan = estimator.pre_scan([wav])
        estimate = estimator.estimate(
            scan, enabled_features=["sample_pack_curator"]
        )

        assert estimate.exceeds_cost_threshold is True
        assert any("cost" in w.lower() for w in estimate.warnings)

    def test_parallelism_reduces_time(self, tmp_path):
        paths = []
        for i in range(4):
            wav = tmp_path / f"test_{i}.wav"
            _create_wav(wav, duration_seconds=10.0)
            paths.append(wav)

        estimator = AnalysisTimeEstimator()
        scan = estimator.pre_scan(paths)

        est_1 = estimator.estimate(scan, max_workers=1)
        est_4 = estimator.estimate(scan, max_workers=4)

        assert est_4.estimated_total_seconds < est_1.estimated_total_seconds

    def test_empty_scan(self):
        estimator = AnalysisTimeEstimator()
        scan = estimator.pre_scan([])
        estimate = estimator.estimate(scan)

        assert estimate.file_count == 0
        assert estimate.estimated_total_seconds == 0.0
