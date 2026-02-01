"""Tests for src/features/prompt_helpers.py (Phase B8).

Validates JSON parsing, result summarization, and prompt block formatting
used by all 10 LLM features.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.core.models import AnalysisResult
from src.features.prompt_helpers import (
    parse_json_response,
    results_to_prompt_block,
    summarize_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_result(hash_val="abc123"):
    """Create a fully-populated mock AnalysisResult for testing."""
    result = MagicMock(spec=AnalysisResult)
    result.audio_sample_hash = hash_val
    result.processing_time = 0.512
    result.quality_metadata = {"quality_tier": "high", "sample_rate": 44100, "channels": 2}

    result.source_classification = MagicMock()
    result.source_classification.source_type = "drum"
    result.source_classification.confidence = 0.9
    result.source_classification.characteristics = ["punchy"]

    result.musical_analysis = MagicMock()
    result.musical_analysis.has_pitch = True
    result.musical_analysis.note_name = "C4"
    result.musical_analysis.estimated_key = "C major"
    result.musical_analysis.key_confidence = 0.88
    result.musical_analysis.is_atonal = False
    result.musical_analysis.fundamental_frequency = 261.6

    result.percussive_analysis = MagicMock()
    result.percussive_analysis.drum_type = "kick"
    result.percussive_analysis.confidence = 0.85
    result.percussive_analysis.attack_time = 5.2
    result.percussive_analysis.brightness = 3200.0
    result.percussive_analysis.is_synthesized = False

    result.rhythmic_analysis = MagicMock()
    result.rhythmic_analysis.is_one_shot = True
    result.rhythmic_analysis.has_rhythm = False
    result.rhythmic_analysis.tempo_bpm = 0.0
    result.rhythmic_analysis.num_beats = 0

    result.llm_analysis = MagicMock()
    result.llm_analysis.suggested_name = "Kick_C4"
    result.llm_analysis.description = "A punchy kick drum tuned to C4."
    result.llm_analysis.tags = ["kick", "drum"]
    result.llm_analysis.contains_speech = False
    result.llm_analysis.transcription = None

    return result


def make_minimal_result(hash_val="minimal_hash"):
    """Create a mock AnalysisResult with all analysis fields set to None."""
    result = MagicMock(spec=AnalysisResult)
    result.audio_sample_hash = hash_val
    result.processing_time = 0.1
    result.quality_metadata = None
    result.source_classification = None
    result.musical_analysis = None
    result.percussive_analysis = None
    result.rhythmic_analysis = None
    result.llm_analysis = None
    return result


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponsePlainJson:
    """parse_json_response with plain JSON strings."""

    def test_simple_object(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_nested_object(self):
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_json_response(raw)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_with_whitespace(self):
        raw = '  \n {"a": 1}  \n '
        assert parse_json_response(raw) == {"a": 1}


class TestParseJsonResponseMarkdownBlock:
    """parse_json_response handles ```json ... ``` markdown blocks."""

    def test_json_code_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert parse_json_response(raw) == {"a": 1}

    def test_plain_code_fence(self):
        raw = '```\n{"key": "value"}\n```'
        assert parse_json_response(raw) == {"key": "value"}

    def test_code_fence_with_extra_whitespace(self):
        raw = '  ```json\n  {"a": 1}\n  ```  '
        result = parse_json_response(raw)
        assert result == {"a": 1}


class TestParseJsonResponseInvalid:
    """parse_json_response returns _parse_error dict for invalid input."""

    def test_plain_text(self):
        result = parse_json_response("not json at all")
        assert result.get("_parse_error") is True
        assert "not json at all" in result.get("_raw_response", "")

    def test_partial_json(self):
        result = parse_json_response('{"a": 1, "b":')
        assert result.get("_parse_error") is True

    def test_xml_input(self):
        result = parse_json_response("<root><a>1</a></root>")
        assert result.get("_parse_error") is True


class TestParseJsonResponseEmpty:
    """parse_json_response returns _parse_error dict for empty string."""

    def test_empty_string(self):
        result = parse_json_response("")
        assert result.get("_parse_error") is True

    def test_whitespace_only(self):
        result = parse_json_response("   \n\t  ")
        assert result.get("_parse_error") is True


# ---------------------------------------------------------------------------
# summarize_result
# ---------------------------------------------------------------------------


class TestSummarizeResultBasic:
    """summarize_result returns dict with expected keys for a fully populated result."""

    def test_contains_hash(self):
        summary = summarize_result(make_mock_result("test_hash"))
        assert summary["hash"] == "test_hash"

    def test_contains_quality_fields(self):
        summary = summarize_result(make_mock_result())
        assert "sample_rate" in summary
        assert summary["sample_rate"] == 44100

    def test_contains_source_type(self):
        summary = summarize_result(make_mock_result())
        assert summary["source_type"] == "drum"
        assert summary["source_confidence"] == 0.9

    def test_contains_musical_fields(self):
        summary = summarize_result(make_mock_result())
        assert summary["has_pitch"] is True
        assert summary["note_name"] == "C4"
        assert summary["estimated_key"] == "C major"

    def test_contains_percussive_fields(self):
        summary = summarize_result(make_mock_result())
        assert summary["drum_type"] == "kick"
        assert summary["drum_confidence"] == 0.85

    def test_contains_rhythmic_fields(self):
        summary = summarize_result(make_mock_result())
        assert summary["is_one_shot"] is True
        assert summary["has_rhythm"] is False

    def test_contains_llm_fields(self):
        summary = summarize_result(make_mock_result())
        assert summary["suggested_name"] == "Kick_C4"
        assert summary["tags"] == ["kick", "drum"]
        assert summary["contains_speech"] is False

    def test_processing_time_rounded(self):
        summary = summarize_result(make_mock_result())
        assert summary["processing_time"] == 0.512

    def test_no_transcription_when_none(self):
        """When llm_analysis.transcription is None, key should not appear."""
        summary = summarize_result(make_mock_result())
        assert "transcription" not in summary


class TestSummarizeResultMissingAnalyses:
    """summarize_result handles None analysis fields gracefully."""

    def test_all_none_produces_minimal_dict(self):
        summary = summarize_result(make_minimal_result())
        assert "hash" in summary
        assert "processing_time" in summary
        # Fields from None analysis components should be absent
        assert "source_type" not in summary
        assert "has_pitch" not in summary
        assert "drum_type" not in summary
        assert "is_one_shot" not in summary
        assert "suggested_name" not in summary

    def test_none_quality_metadata(self):
        summary = summarize_result(make_minimal_result())
        assert "sample_rate" not in summary
        assert "channels" not in summary

    def test_partial_analyses(self):
        """Only source_classification populated; others None."""
        result = make_minimal_result()
        result.source_classification = MagicMock()
        result.source_classification.source_type = "synth"
        result.source_classification.confidence = 0.75
        result.source_classification.characteristics = ["warm", "analog"]

        summary = summarize_result(result)
        assert summary["source_type"] == "synth"
        assert "has_pitch" not in summary
        assert "drum_type" not in summary


# ---------------------------------------------------------------------------
# results_to_prompt_block
# ---------------------------------------------------------------------------


class TestResultsToPromptBlockSingle:
    """results_to_prompt_block formats a single result into a string block."""

    def test_single_result_produces_one_line(self):
        result = make_mock_result()
        block = results_to_prompt_block(result)
        lines = block.strip().split("\n")
        assert len(lines) == 1

    def test_line_starts_with_sample_1(self):
        result = make_mock_result()
        block = results_to_prompt_block(result)
        assert block.startswith("Sample 1:")

    def test_contains_hash(self):
        result = make_mock_result("my_hash")
        block = results_to_prompt_block(result)
        assert "my_hash" in block

    def test_output_is_parseable_json_after_prefix(self):
        result = make_mock_result()
        block = results_to_prompt_block(result)
        # Strip "Sample 1: " prefix to get the JSON portion
        json_part = block.split(": ", 1)[1]
        # May be truncated, only test if short enough
        if not json_part.endswith("..."):
            parsed = json.loads(json_part)
            assert parsed["hash"] == "abc123"


class TestResultsToPromptBlockList:
    """results_to_prompt_block formats a list of results."""

    def test_multiple_results_produce_multiple_lines(self):
        results = [make_mock_result(f"hash_{i}") for i in range(3)]
        block = results_to_prompt_block(results)
        lines = block.strip().split("\n")
        assert len(lines) == 3

    def test_lines_numbered_sequentially(self):
        results = [make_mock_result(f"hash_{i}") for i in range(3)]
        block = results_to_prompt_block(results)
        lines = block.strip().split("\n")
        assert lines[0].startswith("Sample 1:")
        assert lines[1].startswith("Sample 2:")
        assert lines[2].startswith("Sample 3:")

    def test_each_line_contains_its_hash(self):
        results = [make_mock_result(f"hash_{i}") for i in range(3)]
        block = results_to_prompt_block(results)
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            assert f"hash_{i}" in line

    def test_empty_list(self):
        block = results_to_prompt_block([])
        assert block == ""


class TestResultsToPromptBlockTruncation:
    """results_to_prompt_block respects max_per_sample parameter."""

    def test_truncation_with_small_limit(self):
        result = make_mock_result()
        block = results_to_prompt_block(result, max_per_sample=50)
        # The JSON portion should be truncated with "..."
        json_part = block.split(": ", 1)[1]
        assert json_part.endswith("...")
        # The truncated portion (before "...") should be <= max_per_sample
        assert len(json_part) <= 50 + 3  # 50 chars + "..."

    def test_no_truncation_with_large_limit(self):
        result = make_minimal_result()
        block = results_to_prompt_block(result, max_per_sample=10000)
        json_part = block.split(": ", 1)[1]
        assert not json_part.endswith("...")

    def test_truncation_applies_per_sample(self):
        results = [make_mock_result(f"h{i}") for i in range(3)]
        block = results_to_prompt_block(results, max_per_sample=40)
        lines = block.strip().split("\n")
        for line in lines:
            json_part = line.split(": ", 1)[1]
            # Each sample's JSON is independently truncated
            assert len(json_part) <= 40 + 3  # 40 chars + "..."
