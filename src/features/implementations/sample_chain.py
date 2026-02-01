"""Sample Chain / Transition Suggester — LLM feature.

Suggests an optimal ordering of samples and transition points
to create cohesive chains or sets with a desired energy arc.
"""

from typing import Any, Dict, List, Optional, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, ChainResult, TransitionPoint
from src.features.protocols import BaseLLMFeature


class SampleChainSuggester(BaseLLMFeature):
    """Suggests sample ordering and transitions for cohesive chains."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="sample_chain",
            display_name="Sample Chain / Transition Suggester",
            feature_type="batch",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        **kwargs: Any,
    ) -> ChainResult:
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        energy_preference: str = kwargs.get("energy_preference", "ascending")

        # results is guaranteed to be a list for batch features
        result_list: List[AnalysisResult] = results  # type: ignore[assignment]

        # Build prompt
        sample_block = results_to_prompt_block(result_list)

        system_prompt = (
            "You are an expert DJ and music producer specializing in sample "
            "arrangement, set building, and energy flow. You understand how to "
            "order audio samples for smooth transitions and compelling energy arcs."
        )

        user_prompt = (
            f"Given the following audio samples, suggest an optimal ordering that "
            f"creates smooth transitions with a '{energy_preference}' energy arc.\n\n"
            f"Energy preference: {energy_preference}\n"
            f"(ascending = build energy over time, descending = wind down, "
            f"arc = build then release, flat = maintain consistent energy)\n\n"
            f"Sample data:\n{sample_block}\n\n"
            f"Respond with a JSON object containing:\n"
            f"- \"ordered_hashes\": list of sample hashes in the suggested order\n"
            f"- \"transitions\": list of objects, each with:\n"
            f"  - \"from_hash\": hash of the outgoing sample\n"
            f"  - \"to_hash\": hash of the incoming sample\n"
            f"  - \"transition_note\": brief advice on how to transition between them\n"
            f"  - \"compatibility_score\": float from 0.0 to 1.0\n"
            f"- \"energy_arc\": string describing the achieved energy arc\n"
            f"- \"overall_score\": float from 0.0 to 1.0 rating the overall chain quality\n\n"
            f"Respond with ONLY the JSON object, no extra text."
        )

        # Call LLM
        raw = self._client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2000,
        )

        # Parse response
        data = parse_json_response(raw)

        # Build result with graceful fallback
        try:
            transitions = []
            for t in data.get("transitions", []):
                transitions.append(
                    TransitionPoint(
                        from_hash=t.get("from_hash", ""),
                        to_hash=t.get("to_hash", ""),
                        transition_note=t.get("transition_note", ""),
                        compatibility_score=float(t.get("compatibility_score", 0.0)),
                    )
                )

            return ChainResult(
                ordered_hashes=data.get("ordered_hashes", []),
                transitions=transitions,
                energy_arc=data.get("energy_arc", energy_preference),
                overall_score=float(data.get("overall_score", 0.0)),
            )
        except (TypeError, ValueError, KeyError) as exc:
            self.logger.warning("Failed to build ChainResult from LLM data: %s", exc)
            return ChainResult(
                ordered_hashes=[r.audio_sample_hash for r in result_list],
                transitions=[],
                energy_arc="flat",
                overall_score=0.0,
            )

    def estimate_cost(self, results: Union[AnalysisResult, List[AnalysisResult]]) -> CostEstimate:
        num_files = len(results) if isinstance(results, list) else 1
        tokens = 3500 + (200 * num_files)
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=num_files,
        )
