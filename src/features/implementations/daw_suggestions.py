"""DAW-Contextualized Suggestions - LLM feature for context-aware sample recommendations."""

from typing import Any, Dict, List, Optional, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, SuggestionResult, SampleSuggestion, DAWContext
from src.features.protocols import BaseLLMFeature


class DAWContextSuggester(BaseLLMFeature):
    """Provides sample suggestions informed by current DAW session context."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="daw_suggestions",
            display_name="DAW-Contextualized Suggestions",
            feature_type="batch",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self, results: List[AnalysisResult], **kwargs: Any
    ) -> SuggestionResult:
        """Ask LLM which samples fit the DAW context and why."""
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        context: DAWContext = kwargs.get(
            "context",
            DAWContext(bpm=120.0, key="C major", genre="electronic", mood="neutral"),
        )

        sample_block = results_to_prompt_block(results)

        system_prompt = (
            "You are an expert music production assistant specializing in sample "
            "selection and arrangement. You help producers find the right samples "
            "for their current project context, considering tempo, key, genre, and mood."
        )

        context_description = (
            f"BPM: {context.bpm}, Key: {context.key}, "
            f"Genre: {context.genre}, Mood: {context.mood}"
        )
        if context.notes:
            context_description += f", Notes: {context.notes}"

        user_prompt = (
            f"Current DAW project context:\n{context_description}\n\n"
            f"Available samples:\n{sample_block}\n\n"
            "For each sample, evaluate how well it fits the current project context. "
            "Consider key compatibility, tempo matching, genre fit, and mood alignment. "
            "Also suggest which samples would layer well together and flag any conflicts.\n\n"
            "Respond with JSON in this exact format:\n"
            "{\n"
            '  "suggestions": [\n'
            "    {\n"
            '      "sample_hash": "the_hash",\n'
            '      "fit_score": 0.85,\n'
            '      "reason": "Why this sample fits the context",\n'
            '      "conflicts": ["any conflicts with the context"],\n'
            '      "layer_with": ["hashes of samples that layer well with this one"]\n'
            "    }\n"
            "  ],\n"
            '  "layer_groups": [["hash1", "hash2"], ["hash3", "hash4"]]\n'
            "}"
        )

        try:
            raw = self._client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2000,
            )
            data = parse_json_response(raw)
        except Exception as exc:
            self.logger.warning("LLM call failed for daw_suggestions: %s", exc)
            error_suggestion = SampleSuggestion(
                sample_hash="",
                fit_score=0.0,
                reason=f"[LLM error] {exc}",
                conflicts=[],
                layer_with=[],
            )
            return SuggestionResult(context=context, suggestions=[error_suggestion], layer_groups=[])

        if data.get("_parse_error"):
            raw_text = data.get("_raw_response", "")[:500]
            error_suggestion = SampleSuggestion(
                sample_hash="",
                fit_score=0.0,
                reason=f"[LLM parse error] {raw_text}",
                conflicts=[],
                layer_with=[],
            )
            return SuggestionResult(context=context, suggestions=[error_suggestion], layer_groups=[])

        if not data:
            return SuggestionResult(context=context, suggestions=[], layer_groups=[])

        suggestions = []
        for s in data.get("suggestions", []):
            try:
                suggestions.append(SampleSuggestion(
                    sample_hash=s.get("sample_hash", ""),
                    fit_score=float(s.get("fit_score", 0.0)),
                    reason=s.get("reason", ""),
                    conflicts=s.get("conflicts", []),
                    layer_with=s.get("layer_with", []),
                ))
            except (TypeError, ValueError):
                continue

        layer_groups = data.get("layer_groups", [])
        if not isinstance(layer_groups, list):
            layer_groups = []
        # Ensure each group is a list of strings
        validated_groups = []
        for group in layer_groups:
            if isinstance(group, list) and all(isinstance(h, str) for h in group):
                validated_groups.append(group)

        return SuggestionResult(
            context=context,
            suggestions=suggestions,
            layer_groups=validated_groups,
        )

    def estimate_cost(self, results: List[AnalysisResult]) -> CostEstimate:
        """Estimate cost for DAW-contextualized suggestion generation."""
        count = len(results) if isinstance(results, list) else 1
        tokens = 600 + 150 * count
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=count,
        )
