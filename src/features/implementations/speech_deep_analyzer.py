"""Spoken Content Deep Analyzer — LLM feature.

Analyzes speech content within audio samples for sentiment, tone,
language register, and content warnings.
"""

from typing import Any, List, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, SpokenContentResult
from src.features.protocols import BaseLLMFeature


class SpokenContentAnalyzer(BaseLLMFeature):
    """Performs deep analysis of spoken content found in audio samples."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="speech_deep_analyzer",
            display_name="Spoken Content Deep Analyzer",
            feature_type="single",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self, results: Union[AnalysisResult, List[AnalysisResult]], **kwargs: Any
    ) -> SpokenContentResult:
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        result: AnalysisResult = results  # type: ignore[assignment]

        # If no speech detected, return neutral defaults immediately
        if not result.llm_analysis or not result.llm_analysis.contains_speech:
            return SpokenContentResult(
                sample_hash=result.audio_sample_hash,
                sentiment="neutral",
                sentiment_score=0.0,
                tone="neutral",
                language_register="unknown",
                genre_fit=[],
                content_warnings=[],
                licensing_notes="No speech detected",
            )

        # Build prompt from analysis data
        sample_block = results_to_prompt_block(result)

        transcription = result.llm_analysis.transcription or "(transcription unavailable)"

        system_prompt = (
            "You are an expert audio content analyst specializing in spoken word, "
            "vocal samples, and speech analysis for music production. You provide "
            "detailed assessments of spoken content in audio samples."
        )

        user_prompt = (
            f"Analyze the spoken content in this audio sample.\n\n"
            f"Sample data:\n{sample_block}\n\n"
            f"Transcription: {transcription}\n\n"
            f"Provide your analysis as a JSON object with exactly these fields:\n"
            f"- \"sentiment\": one of \"positive\", \"negative\", or \"neutral\"\n"
            f"- \"sentiment_score\": float from -1.0 (most negative) to 1.0 (most positive)\n"
            f"- \"tone\": a short descriptor (e.g. \"aggressive\", \"calm\", \"excited\", \"melancholic\")\n"
            f"- \"language_register\": one of \"formal\", \"informal\", \"slang\", \"poetic\", \"technical\"\n"
            f"- \"genre_fit\": list of genres/contexts this vocal fits (e.g. [\"hip-hop ad-lib\", \"EDM vocal chop\"])\n"
            f"- \"content_warnings\": list of any content warnings (profanity, violence, etc.), empty list if none\n"
            f"- \"licensing_notes\": string with any licensing concerns (recognizable quotes, copyrighted phrases, etc.)\n\n"
            f"Respond with ONLY the JSON object, no extra text."
        )

        # Call LLM
        raw = self._client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1500,
        )

        # Parse response
        data = parse_json_response(raw)

        if data.get("_parse_error"):
            raw_text = data.get("_raw_response", "")[:500]
            self.logger.warning("parse_json_response returned _parse_error for speech_deep_analyzer")
            return SpokenContentResult(
                sample_hash=result.audio_sample_hash,
                sentiment="neutral",
                sentiment_score=0.0,
                tone="neutral",
                language_register="unknown",
                genre_fit=[],
                content_warnings=[],
                licensing_notes=f"[Parse error] Raw LLM response: {raw_text}",
            )

        # Build result, falling back to defaults if parsing failed
        try:
            return SpokenContentResult(
                sample_hash=result.audio_sample_hash,
                sentiment=data.get("sentiment", "neutral"),
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                tone=data.get("tone", "neutral"),
                language_register=data.get("language_register", "unknown"),
                genre_fit=data.get("genre_fit", []),
                content_warnings=data.get("content_warnings", []),
                licensing_notes=data.get("licensing_notes", "Unable to determine licensing concerns"),
            )
        except (TypeError, ValueError) as exc:
            self.logger.warning("Failed to build SpokenContentResult from LLM data: %s", exc)
            return SpokenContentResult(
                sample_hash=result.audio_sample_hash,
                sentiment="neutral",
                sentiment_score=0.0,
                tone="neutral",
                language_register="unknown",
                genre_fit=[],
                content_warnings=[],
                licensing_notes="Analysis failed — could not parse LLM response",
            )

    def estimate_cost(self, results: Union[AnalysisResult, List[AnalysisResult]]) -> CostEstimate:
        count = len(results) if isinstance(results, list) else 1
        tokens = 1200 * count
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=count,
        )
