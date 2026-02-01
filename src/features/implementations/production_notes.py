"""Production Notes & Usage Tips - LLM feature for generating per-sample production guidance."""

from typing import Any, List, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, ProductionNotesResult
from src.features.protocols import BaseLLMFeature


class ProductionNotesGenerator(BaseLLMFeature):
    """Generates production notes, usage tips, and mixing advice for individual samples."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="production_notes",
            display_name="Production Notes & Usage Tips",
            feature_type="single",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self, results: Union[AnalysisResult, List[AnalysisResult]], **kwargs: Any
    ) -> ProductionNotesResult:
        """Generate detailed production notes for a single sample via LLM."""
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        result: AnalysisResult = results  # type: ignore[assignment]
        sample_block = results_to_prompt_block(result)

        system_prompt = (
            "You are a professional music producer and mixing engineer with decades "
            "of experience. You provide actionable production advice including EQ "
            "suggestions, layering techniques, mixing tips, arrangement placement, "
            "and processing chain recommendations for audio samples."
        )

        user_prompt = (
            f"Analyze this audio sample and provide detailed production notes:\n\n"
            f"{sample_block}\n\n"
            "Provide specific, actionable advice in each category. Be precise with "
            "frequency ranges, dB values, and effect settings where applicable.\n\n"
            "Respond with JSON in this exact format:\n"
            "{\n"
            '  "eq_suggestions": [\n'
            '    "Cut 200-400Hz by 2-3dB to reduce muddiness",\n'
            '    "Boost 4-6kHz shelf for presence"\n'
            "  ],\n"
            '  "layering_advice": [\n'
            '    "Layer with a sub bass for low-end weight",\n'
            '    "Combine with a noise layer for texture"\n'
            "  ],\n"
            '  "mixing_tips": [\n'
            '    "Pan slightly left for stereo width",\n'
            '    "Use sidechain compression from kick"\n'
            "  ],\n"
            '  "arrangement_placement": [\n'
            '    "Works well in drops and choruses",\n'
            '    "Use as a transition element between sections"\n'
            "  ],\n"
            '  "processing_chain": [\n'
            '    "1. High-pass filter at 80Hz",\n'
            '    "2. Gentle compression (3:1 ratio, slow attack)",\n'
            '    "3. Plate reverb with 1.5s decay"\n'
            "  ],\n"
            '  "compatible_genres": ["house", "techno", "drum and bass"]\n'
            "}"
        )

        try:
            raw = self._client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1500,
            )
            data = parse_json_response(raw)
        except Exception:
            self.logger.warning("LLM call or JSON parse failed for production_notes, returning empty result")
            return ProductionNotesResult(
                sample_hash=result.audio_sample_hash,
                eq_suggestions=[],
                layering_advice=[],
                mixing_tips=[],
                arrangement_placement=[],
                processing_chain=[],
                compatible_genres=[],
            )

        if not data:
            return ProductionNotesResult(
                sample_hash=result.audio_sample_hash,
                eq_suggestions=[],
                layering_advice=[],
                mixing_tips=[],
                arrangement_placement=[],
                processing_chain=[],
                compatible_genres=[],
            )

        def safe_list(key: str) -> List[str]:
            """Extract a list of strings from data, defaulting to empty list."""
            val = data.get(key, [])
            if not isinstance(val, list):
                return []
            return [str(item) for item in val]

        return ProductionNotesResult(
            sample_hash=result.audio_sample_hash,
            eq_suggestions=safe_list("eq_suggestions"),
            layering_advice=safe_list("layering_advice"),
            mixing_tips=safe_list("mixing_tips"),
            arrangement_placement=safe_list("arrangement_placement"),
            processing_chain=safe_list("processing_chain"),
            compatible_genres=safe_list("compatible_genres"),
        )

    def estimate_cost(self, results: Union[AnalysisResult, List[AnalysisResult]]) -> CostEstimate:
        """Estimate cost for production notes generation per file."""
        count = len(results) if isinstance(results, list) else 1
        tokens = 800 * count
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=count,
        )
