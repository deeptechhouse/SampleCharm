"""Similar Sample Finder — LLM feature.

Finds samples that are sonically or contextually similar to a given
reference sample, supporting both single and batch modes.
"""

from typing import Any, List, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, SimilarityResult, SimilarMatch
from src.features.protocols import BaseLLMFeature


class SimilarSampleFinder(BaseLLMFeature):
    """Identifies samples similar to a reference sample."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="similar_sample_finder",
            display_name="Similar Sample Finder",
            feature_type="single+batch",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        **kwargs: Any,
    ) -> SimilarityResult:
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        reference_hash: str = kwargs.get("reference_hash", "")

        # Normalize to list
        if isinstance(results, AnalysisResult):
            result_list = [results]
        else:
            result_list = results

        # If no reference hash provided, use the first sample
        if not reference_hash and result_list:
            reference_hash = result_list[0].audio_sample_hash

        # Build prompt
        sample_block = results_to_prompt_block(result_list)

        system_prompt = (
            "You are an expert audio analyst specializing in identifying sonic "
            "similarities between audio samples. You understand timbral qualities, "
            "musical characteristics, rhythmic patterns, and production techniques."
        )

        user_prompt = (
            f"Given the following audio samples, find which samples are most similar "
            f"to the reference sample (hash: {reference_hash}).\n\n"
            f"Sample data:\n{sample_block}\n\n"
            f"Respond with a JSON object containing:\n"
            f"- \"matches\": a list of objects, each with:\n"
            f"  - \"sample_hash\": the hash of the similar sample (exclude the reference itself)\n"
            f"  - \"similarity_score\": float from 0.0 to 1.0\n"
            f"  - \"explanation\": brief explanation of why they are similar\n"
            f"  - \"shared_attributes\": dict of attribute name to shared value "
            f"(e.g. {{\"key\": \"C minor\", \"tempo_range\": \"120-130 BPM\"}})\n"
            f"- \"weighting_strategy\": string describing how you weighted the comparison "
            f"(e.g. \"timbral-focused\", \"rhythm-focused\", \"balanced\")\n\n"
            f"Order matches by similarity_score descending. "
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

        if data.get("_parse_error"):
            raw_text = data.get("_raw_response", "")[:500]
            self.logger.warning("parse_json_response returned _parse_error for similar_sample_finder")
            return SimilarityResult(
                reference_hash=reference_hash,
                matches=[
                    SimilarMatch(
                        sample_hash="",
                        similarity_score=0.0,
                        explanation=f"[Parse error] Raw LLM response: {raw_text}",
                        shared_attributes={},
                    )
                ],
                weighting_strategy="parse_error",
            )

        # Build result with graceful fallback
        try:
            matches = []
            for m in data.get("matches", []):
                matches.append(
                    SimilarMatch(
                        sample_hash=m.get("sample_hash", ""),
                        similarity_score=float(m.get("similarity_score", 0.0)),
                        explanation=m.get("explanation", ""),
                        shared_attributes=m.get("shared_attributes", {}),
                    )
                )

            return SimilarityResult(
                reference_hash=reference_hash,
                matches=matches,
                weighting_strategy=data.get("weighting_strategy", "balanced"),
            )
        except (TypeError, ValueError, KeyError) as exc:
            self.logger.warning("Failed to build SimilarityResult from LLM data: %s", exc)
            return SimilarityResult(
                reference_hash=reference_hash,
                matches=[],
                weighting_strategy="default",
            )

    def estimate_cost(self, results: Union[AnalysisResult, List[AnalysisResult]]) -> CostEstimate:
        count = len(results) if isinstance(results, list) else 1
        tokens = 3000 * count
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=count,
        )
