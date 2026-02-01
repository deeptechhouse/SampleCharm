"""
Result writer for outputting analysis results to files.

Follows SOLID principles:
- Single Responsibility: Only handles result output formatting and writing
- Open/Closed: New output formats can be added without modifying existing code
- Interface Segregation: Simple interface for writing results
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from src.core.models import AnalysisResult


class ResultWriter(ABC):
    """Abstract base class for result writers (Strategy Pattern)."""

    @abstractmethod
    def write(self, results: Dict[Path, AnalysisResult], output_path: Path) -> None:
        """Write results to the specified path."""
        pass


class TextResultWriter(ResultWriter):
    """Writes analysis results to a human-readable text file."""

    def __init__(self, include_timestamp: bool = True):
        """
        Initialize text writer.

        Args:
            include_timestamp: Whether to include timestamp in output
        """
        self.include_timestamp = include_timestamp
        self.logger = logging.getLogger("result_writer.text")

    def write(self, results: Dict[Path, AnalysisResult], output_path: Path) -> None:
        """
        Write results to a text file.

        Args:
            results: Dictionary mapping file paths to their analysis results
            output_path: Path to output text file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 70 + "\n")
            f.write("SAMPLECHARM BATCH ANALYSIS RESULTS\n")
            f.write("=" * 70 + "\n")

            if self.include_timestamp:
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            f.write(f"Total Files Analyzed: {len(results)}\n")
            f.write("=" * 70 + "\n\n")

            # Write each result
            for file_path, result in results.items():
                self._write_single_result(f, file_path, result)

            # Write footer
            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

        self.logger.info(f"Results written to: {output_path}")

    def _write_single_result(self, f, file_path: Path, result: AnalysisResult) -> None:
        """Write a single analysis result to file."""
        f.write("-" * 70 + "\n")
        f.write(f"FILE: {file_path.name}\n")
        f.write(f"PATH: {file_path}\n")
        f.write("-" * 70 + "\n")

        # Processing info
        f.write(f"Processing Time: {result.processing_time:.3f}s\n")
        f.write(f"Quality: {result.quality_metadata.get('quality_tier', 'Unknown')}\n")

        # Summary
        f.write(f"\nSummary: {result.get_summary()}\n")

        # Source Classification
        if result.source_classification:
            sc = result.source_classification
            f.write(f"\nSource Classification:\n")
            f.write(f"  Type: {sc.source_type}\n")
            f.write(f"  Confidence: {sc.confidence:.2%}\n")
            if sc.characteristics:
                f.write(f"  Characteristics: {', '.join(sc.characteristics)}\n")

        # Musical Analysis
        if result.musical_analysis:
            ma = result.musical_analysis
            f.write(f"\nMusical Analysis:\n")
            f.write(f"  Has Pitch: {ma.has_pitch}\n")
            if ma.has_pitch:
                f.write(f"  Note: {ma.note_name}\n")
                f.write(f"  Frequency: {ma.fundamental_frequency:.1f} Hz\n")
            if ma.estimated_key:
                f.write(f"  Key: {ma.estimated_key}\n")

        # Percussive Analysis
        if result.percussive_analysis:
            pa = result.percussive_analysis
            f.write(f"\nPercussive Analysis:\n")
            f.write(f"  Drum Type: {pa.drum_type}\n")
            f.write(f"  Confidence: {pa.confidence:.2%}\n")
            f.write(f"  Synthesized: {pa.is_synthesized}\n")

        # Rhythmic Analysis
        if result.rhythmic_analysis:
            ra = result.rhythmic_analysis
            f.write(f"\nRhythmic Analysis:\n")
            f.write(f"  One-Shot: {ra.is_one_shot}\n")
            if ra.has_rhythm:
                f.write(f"  Tempo: {ra.tempo_bpm:.1f} BPM\n")
                f.write(f"  Beats: {ra.num_beats:.1f}\n")

        # LLM Analysis
        if result.llm_analysis:
            la = result.llm_analysis
            f.write(f"\nLLM Analysis:\n")
            f.write(f"  Suggested Name: {la.suggested_name}\n")
            f.write(f"  Description: {la.description}\n")
            f.write(f"  Contains Speech: {'Yes' if la.contains_speech else 'No'}\n")
            if la.contains_speech:
                if la.transcription:
                    f.write(f"  Transcription: {la.transcription}\n")
                if la.detected_words:
                    f.write(f"  Detected Words: {', '.join(la.detected_words)}\n")
                if la.speech_language:
                    f.write(f"  Language: {la.speech_language}\n")
                if la.speech_confidence is not None:
                    f.write(f"  Speech Confidence: {la.speech_confidence:.2%}\n")
            if la.tags:
                f.write(f"  Tags: {', '.join(la.tags)}\n")
            f.write(f"  Model: {la.model_used}\n")

        f.write("\n")


class JSONResultWriter(ResultWriter):
    """Writes analysis results to a JSON file."""

    def __init__(self, indent: int = 2):
        """
        Initialize JSON writer.

        Args:
            indent: JSON indentation level
        """
        self.indent = indent
        self.logger = logging.getLogger("result_writer.json")

    def write(self, results: Dict[Path, AnalysisResult], output_path: Path) -> None:
        """
        Write results to a JSON file.

        Args:
            results: Dictionary mapping file paths to their analysis results
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "generated": datetime.now().isoformat(),
            "total_files": len(results),
            "results": {
                str(path): result.to_dict()
                for path, result in results.items()
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=self.indent, default=str)

        self.logger.info(f"Results written to: {output_path}")


def create_result_writer(format: str = "text", **kwargs) -> ResultWriter:
    """
    Factory function to create appropriate result writer.

    Args:
        format: Output format ("text" or "json")
        **kwargs: Additional arguments for the writer

    Returns:
        Appropriate ResultWriter instance
    """
    writers = {
        "text": TextResultWriter,
        "txt": TextResultWriter,
        "json": JSONResultWriter,
    }

    writer_class = writers.get(format.lower())
    if writer_class is None:
        raise ValueError(f"Unknown format: {format}. Supported: {list(writers.keys())}")

    return writer_class(**kwargs)
