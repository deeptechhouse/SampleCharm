"""Marketplace Description Generator — LLM feature.

Generates marketing-ready descriptions, tags, and metadata for
sample packs targeting various marketplace platforms.
"""

from typing import Any, Dict, List, Optional, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, MarketplaceResult
from src.features.protocols import BaseLLMFeature


class MarketplaceDescriptionGen(BaseLLMFeature):
    """Generates marketplace-ready descriptions for sample packs."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="marketplace_description",
            display_name="Marketplace Description Generator",
            feature_type="batch",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        **kwargs: Any,
    ) -> MarketplaceResult:
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        pack_name: str = kwargs.get("pack_name", "Untitled Pack")
        platforms: List[str] = kwargs.get("platforms") or ["splice"]

        # results is guaranteed to be a list for batch features
        result_list: List[AnalysisResult] = results  # type: ignore[assignment]

        # Build prompt
        sample_block = results_to_prompt_block(result_list)
        platforms_str = ", ".join(platforms)

        system_prompt = (
            "You are a professional music marketing copywriter specializing in "
            "sample pack descriptions for online marketplaces like Splice, Loopmasters, "
            "LANDR, and similar platforms. You write compelling, SEO-optimized copy "
            "that accurately represents the content."
        )

        user_prompt = (
            f"Generate a marketplace listing for a sample pack called \"{pack_name}\".\n\n"
            f"Target platforms: {platforms_str}\n\n"
            f"Here are the samples in the pack:\n{sample_block}\n\n"
            f"Respond with a JSON object containing:\n"
            f"- \"headline\": a catchy one-line headline (max 80 chars)\n"
            f"- \"description\": a marketing description (2-4 paragraphs, platform-ready)\n"
            f"- \"tags\": list of 10-20 searchable tags for discoverability\n"
            f"- \"genres\": list of relevant genre categories\n"
            f"- \"stats\": object with summary statistics about the pack, e.g.:\n"
            f"  {{\"total_samples\": N, \"one_shots\": N, \"loops\": N, "
            f"\"tempo_range\": \"X-Y BPM\", \"key_signatures\": [...]}}\n\n"
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
            self.logger.warning("parse_json_response returned _parse_error for marketplace_description")
            return MarketplaceResult(
                pack_name=pack_name,
                headline="[Parse error]",
                description=f"[Parse error] Raw LLM response: {raw_text}",
                tags=[],
                genres=[],
                stats={},
                target_platforms=platforms,
            )

        # Build result with graceful fallback
        try:
            return MarketplaceResult(
                pack_name=pack_name,
                headline=data.get("headline", ""),
                description=data.get("description", ""),
                tags=data.get("tags", []),
                genres=data.get("genres", []),
                stats=data.get("stats", {}),
                target_platforms=platforms,
            )
        except (TypeError, ValueError, KeyError) as exc:
            self.logger.warning("Failed to build MarketplaceResult from LLM data: %s", exc)
            return MarketplaceResult(
                pack_name=pack_name,
                headline="",
                description="",
                tags=[],
                genres=[],
                stats={},
                target_platforms=platforms,
            )

    def estimate_cost(self, results: Union[AnalysisResult, List[AnalysisResult]]) -> CostEstimate:
        num_files = len(results) if isinstance(results, list) else 1
        tokens = 4000 + (150 * num_files)
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=num_files,
        )
