"""Shared prompt construction utilities for LLM features.

Extracts structured summaries from AnalysisResult objects into text
blocks suitable for LLM prompts. All 10 features reuse these helpers
instead of duplicating context-building logic.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from src.core.models import AnalysisResult

logger = logging.getLogger("features.prompt_helpers")


def summarize_result(result: AnalysisResult) -> Dict[str, Any]:
    """Build a flat dict summarizing an AnalysisResult for prompt insertion.

    Returns a JSON-serializable dictionary with the most useful fields
    from each analysis component. Missing components are omitted.
    """
    summary: Dict[str, Any] = {
        "hash": result.audio_sample_hash,
        "processing_time": round(result.processing_time, 3),
    }

    # Quality metadata
    qm = result.quality_metadata
    if qm:
        summary["sample_rate"] = qm.get("sample_rate")
        summary["channels"] = qm.get("channels")

    # Source classification
    sc = result.source_classification
    if sc:
        summary["source_type"] = sc.source_type
        summary["source_confidence"] = round(sc.confidence, 2)
        summary["characteristics"] = sc.characteristics

    # Musical analysis
    ma = result.musical_analysis
    if ma:
        summary["has_pitch"] = ma.has_pitch
        summary["estimated_key"] = ma.estimated_key
        summary["key_confidence"] = round(ma.key_confidence, 2)
        summary["note_name"] = ma.note_name
        summary["is_atonal"] = ma.is_atonal

    # Percussive analysis
    pa = result.percussive_analysis
    if pa:
        summary["drum_type"] = pa.drum_type
        summary["drum_confidence"] = round(pa.confidence, 2)
        summary["attack_time_ms"] = round(pa.attack_time, 1)
        summary["brightness_hz"] = round(pa.brightness, 0)
        summary["is_synthesized"] = pa.is_synthesized

    # Rhythmic analysis
    ra = result.rhythmic_analysis
    if ra:
        summary["is_one_shot"] = ra.is_one_shot
        summary["has_rhythm"] = ra.has_rhythm
        summary["tempo_bpm"] = ra.tempo_bpm
        summary["num_beats"] = ra.num_beats

    # LLM analysis
    la = result.llm_analysis
    if la:
        summary["suggested_name"] = la.suggested_name
        summary["description"] = la.description
        summary["contains_speech"] = la.contains_speech
        summary["tags"] = la.tags
        if la.transcription:
            summary["transcription"] = la.transcription

    return summary


def summarize_results(results: List[AnalysisResult]) -> List[Dict[str, Any]]:
    """Summarize a batch of AnalysisResult objects."""
    return [summarize_result(r) for r in results]


def results_to_prompt_block(
    results: Union[AnalysisResult, List[AnalysisResult]],
    max_per_sample: int = 500,
) -> str:
    """Convert results into a formatted text block for prompt insertion.

    Args:
        results: Single or list of AnalysisResult.
        max_per_sample: Max characters per sample summary (truncation guard).

    Returns:
        A multi-line string suitable for embedding in an LLM prompt.
    """
    if isinstance(results, AnalysisResult):
        results = [results]

    lines = []
    for i, r in enumerate(results):
        summary = summarize_result(r)
        text = json.dumps(summary, indent=None, default=str)
        if len(text) > max_per_sample:
            text = text[:max_per_sample] + "..."
        lines.append(f"Sample {i + 1}: {text}")

    return "\n".join(lines)


def parse_json_response(raw: str) -> Dict[str, Any]:
    """Parse an LLM response that should be JSON.

    Handles common LLM quirks: markdown code blocks, trailing text.

    Returns:
        Parsed dict, or empty dict on failure.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON response, preserving raw text")
        return {"_parse_error": True, "_raw_response": raw}
