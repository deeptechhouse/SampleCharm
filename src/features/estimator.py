"""
Analysis time and cost estimator.

Pre-scans files without audio decoding to estimate processing time,
API token usage, and cost before committing to expensive operations.
"""

import logging
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FileMetaQuick:
    """Metadata extracted from file headers without full audio decode."""

    path: Path
    file_size_bytes: int
    duration_estimate: float  # seconds, from header or file size heuristic
    format: str  # wav / mp3 / flac / aiff
    channels: int
    sample_rate: int


@dataclass(frozen=True)
class PreScanResult:
    """Aggregated pre-scan data for a batch of files."""

    files: List[FileMetaQuick]
    total_count: int
    total_duration: float
    total_size_bytes: int
    format_distribution: Dict[str, int]
    long_files: List[Path]  # Files > 30s
    scan_time: float


@dataclass(frozen=True)
class TimeEstimate:
    """Complete time/cost estimate for an analysis run."""

    estimated_total_seconds: float
    estimated_per_file_seconds: float
    estimated_api_cost_usd: float
    estimated_tokens: int
    file_count: int
    enabled_features: List[str]
    warnings: List[str]
    exceeds_time_threshold: bool
    exceeds_cost_threshold: bool
    breakdown: Dict[str, float]  # Per-component time estimates


# Empirical benchmarks: seconds per second of audio
BASE_ANALYZER_RATES: Dict[str, float] = {
    "source": 0.15,      # YAMNet inference
    "musical": 0.08,     # librosa pitch/key
    "percussive": 0.05,  # random forest
    "rhythmic": 0.06,    # librosa tempo/beat
    "speech": 0.80,      # Whisper base model
    "llm": 2.50,         # API call latency (fixed per file)
}

# Feature-level time estimates (seconds per invocation)
FEATURE_RATES: Dict[str, float] = {
    "sample_pack_curator": 5.0,
    "natural_language_search": 3.0,
    "daw_suggestions": 4.0,
    "batch_rename": 4.0,
    "production_notes": 2.5,
    "speech_deep_analyzer": 2.5,
    "similar_sample_finder": 3.5,
    "sample_chain": 4.0,
    "marketplace_description": 5.0,
    "anomaly_reporter": 4.0,
}

# Token estimates per feature invocation
FEATURE_TOKEN_ESTIMATES: Dict[str, int] = {
    "sample_pack_curator": 5000,
    "natural_language_search": 3000,
    "daw_suggestions": 4000,
    "batch_rename": 3500,
    "production_notes": 1500,
    "speech_deep_analyzer": 1200,
    "similar_sample_finder": 3000,
    "sample_chain": 3500,
    "marketplace_description": 4000,
    "anomaly_reporter": 4000,
}

# Features that run per-file (not batch)
PER_FILE_FEATURES = {"production_notes", "speech_deep_analyzer"}


class AnalysisTimeEstimator:
    """
    Pre-scan estimator for time, tokens, and cost.

    Quickly scans file headers to estimate duration without decoding audio.
    Combines base analyzer rates with feature-specific estimates to produce
    a total cost/time prediction.
    """

    def __init__(
        self,
        time_threshold: float = 300.0,   # 5 minutes
        cost_threshold: float = 1.00,    # $1.00 USD
        max_batch_warning: int = 50,
        cost_per_1k_tokens: float = 0.0025,
    ):
        self.time_threshold = time_threshold
        self.cost_threshold = cost_threshold
        self.max_batch_warning = max_batch_warning
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.logger = logging.getLogger("features.estimator")

    def pre_scan(self, paths: List[Path]) -> PreScanResult:
        """
        Fast header-only scan. No audio decoding.

        Args:
            paths: List of audio file paths.

        Returns:
            PreScanResult with aggregated metadata.
        """
        start = time.time()
        files: List[FileMetaQuick] = []
        format_dist: Dict[str, int] = {}
        long_files: List[Path] = []
        total_duration = 0.0
        total_size = 0

        for path in paths:
            meta = self._scan_file(path)
            if meta:
                files.append(meta)
                total_duration += meta.duration_estimate
                total_size += meta.file_size_bytes
                format_dist[meta.format] = format_dist.get(meta.format, 0) + 1
                if meta.duration_estimate > 30.0:
                    long_files.append(path)

        scan_time = time.time() - start
        self.logger.info(
            f"Pre-scan: {len(files)} files, {total_duration:.1f}s total duration, "
            f"{scan_time:.3f}s scan time"
        )

        return PreScanResult(
            files=files,
            total_count=len(files),
            total_duration=total_duration,
            total_size_bytes=total_size,
            format_distribution=format_dist,
            long_files=long_files,
            scan_time=scan_time,
        )

    def estimate(
        self,
        scan: PreScanResult,
        enabled_analyzers: Optional[List[str]] = None,
        enabled_features: Optional[List[str]] = None,
        max_workers: int = 4,
    ) -> TimeEstimate:
        """
        Compute time/cost estimate from pre-scan data.

        Args:
            scan: Pre-scan result.
            enabled_analyzers: Which core analyzers are active.
            enabled_features: Which LLM features are active.
            max_workers: Thread pool size for parallelism factor.

        Returns:
            TimeEstimate with full breakdown.
        """
        analyzers = enabled_analyzers or list(BASE_ANALYZER_RATES.keys())
        features = enabled_features or []
        breakdown: Dict[str, float] = {}
        warnings: List[str] = []

        # Base analysis time (parallelized across files)
        if analyzers:
            slowest_rate = max(
                BASE_ANALYZER_RATES.get(a, 0) for a in analyzers
            )
            base_time = scan.total_duration * slowest_rate
            parallel_factor = min(max_workers, max(scan.total_count, 1))
            base_time /= parallel_factor
            breakdown["core_analysis"] = base_time
        else:
            base_time = 0.0

        # Feature time
        feature_time = 0.0
        for f in features:
            rate = FEATURE_RATES.get(f, 2.5)
            if f in PER_FILE_FEATURES:
                ft = rate * scan.total_count
            else:
                ft = rate  # Batch = single call
            breakdown[f] = ft
            feature_time += ft

        total_time = base_time + feature_time

        # Token / cost estimation
        total_tokens = 0
        for f in features:
            tokens = FEATURE_TOKEN_ESTIMATES.get(f, 2000)
            if f in PER_FILE_FEATURES:
                total_tokens += tokens * scan.total_count
            else:
                total_tokens += tokens

        total_cost = (total_tokens / 1000) * self.cost_per_1k_tokens

        # Generate warnings
        if total_time > self.time_threshold:
            warnings.append(
                f"Estimated time ({total_time:.0f}s) exceeds "
                f"threshold ({self.time_threshold:.0f}s)"
            )
        if total_cost > self.cost_threshold:
            warnings.append(
                f"Estimated cost (${total_cost:.3f}) exceeds "
                f"threshold (${self.cost_threshold:.2f})"
            )
        if scan.long_files:
            warnings.append(
                f"{len(scan.long_files)} file(s) exceed 30 seconds duration"
            )
        if scan.total_count > self.max_batch_warning and features:
            warnings.append(
                f"Batch of {scan.total_count} files with LLM features enabled"
            )

        per_file = total_time / max(scan.total_count, 1)

        return TimeEstimate(
            estimated_total_seconds=total_time,
            estimated_per_file_seconds=per_file,
            estimated_api_cost_usd=total_cost,
            estimated_tokens=total_tokens,
            file_count=scan.total_count,
            enabled_features=features,
            warnings=warnings,
            exceeds_time_threshold=total_time > self.time_threshold,
            exceeds_cost_threshold=total_cost > self.cost_threshold,
            breakdown=breakdown,
        )

    def _scan_file(self, path: Path) -> Optional[FileMetaQuick]:
        """Extract metadata from a single file without decoding audio."""
        try:
            stat = path.stat()
            file_size = stat.st_size
            fmt = path.suffix.lower().lstrip(".")
            if fmt == "aif":
                fmt = "aiff"

            # Try to get duration from header
            duration = self._estimate_duration(path, fmt, file_size)
            channels, sample_rate = self._read_header_info(path, fmt)

            return FileMetaQuick(
                path=path,
                file_size_bytes=file_size,
                duration_estimate=duration,
                format=fmt,
                channels=channels,
                sample_rate=sample_rate,
            )
        except Exception as e:
            self.logger.warning(f"Pre-scan failed for {path}: {e}")
            return None

    def _estimate_duration(
        self, path: Path, fmt: str, file_size: int
    ) -> float:
        """Estimate duration from file header or size heuristic."""
        if fmt == "wav":
            return self._wav_duration(path, file_size)
        elif fmt == "mp3":
            # Rough: 128kbps average bitrate
            return file_size / (128 * 1000 / 8)
        elif fmt == "flac":
            # Rough: ~50% of equivalent WAV
            return file_size / (44100 * 2 * 2 * 0.5)
        elif fmt == "aiff":
            return self._aiff_duration(path, file_size)
        # Fallback: assume 44.1kHz 16-bit stereo WAV
        return file_size / (44100 * 2 * 2)

    def _wav_duration(self, path: Path, file_size: int) -> float:
        """Read WAV header for duration."""
        try:
            with open(path, "rb") as f:
                f.read(4)  # RIFF
                f.read(4)  # file size
                f.read(4)  # WAVE
                f.read(4)  # fmt chunk id
                fmt_size = struct.unpack("<I", f.read(4))[0]
                f.read(2)  # audio format
                channels = struct.unpack("<H", f.read(2))[0]
                sample_rate = struct.unpack("<I", f.read(4))[0]
                byte_rate = struct.unpack("<I", f.read(4))[0]
                if byte_rate > 0:
                    # Subtract header size estimate
                    data_size = file_size - 44
                    return max(0.0, data_size / byte_rate)
        except Exception:
            pass
        return file_size / (44100 * 2 * 2)

    def _aiff_duration(self, path: Path, file_size: int) -> float:
        """Estimate AIFF duration from file size."""
        # AIFF header parsing is complex; use size heuristic
        return file_size / (44100 * 2 * 2)

    def _read_header_info(self, path: Path, fmt: str) -> tuple:
        """Read channels and sample rate from header. Returns (channels, sr)."""
        if fmt == "wav":
            try:
                with open(path, "rb") as f:
                    f.read(22)
                    channels = struct.unpack("<H", f.read(2))[0]
                    sample_rate = struct.unpack("<I", f.read(4))[0]
                    return channels, sample_rate
            except Exception:
                pass
        return 2, 44100  # Default assumption
