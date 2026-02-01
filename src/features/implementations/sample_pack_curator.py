"""Smart Sample Pack Curator - LLM feature for organizing samples into curated packs."""

from typing import Any, Dict, List, Optional, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, PackCurationResult, SamplePack
from src.features.protocols import BaseLLMFeature


class SamplePackCurator(BaseLLMFeature):
    """Organizes analyzed samples into thematic packs using LLM classification."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="sample_pack_curator",
            display_name="Smart Sample Pack Curator",
            feature_type="batch",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self, results: List[AnalysisResult], **kwargs: Any
    ) -> PackCurationResult:
        """Analyze samples and group them into themed packs via LLM."""
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        sample_block = results_to_prompt_block(results)

        system_prompt = (
            "You are an expert sample library curator with deep knowledge of music "
            "production, sound design, and audio sample organization. You group audio "
            "samples into coherent, themed packs that producers would find useful."
        )

        user_prompt = (
            "Analyze the following audio samples and organize them into themed sample packs.\n\n"
            f"{sample_block}\n\n"
            "Group these samples into logical packs based on their sonic characteristics, "
            "musical properties, and production use cases. Each sample should appear in "
            "exactly one pack. If a sample doesn't fit any pack well, list it as uncategorized.\n\n"
            "Respond with JSON in this exact format:\n"
            "{\n"
            '  "packs": [\n'
            "    {\n"
            '      "name": "Pack Name",\n'
            '      "description": "Brief description of the pack theme",\n'
            '      "tags": ["tag1", "tag2"],\n'
            '      "sample_hashes": ["hash1", "hash2"],\n'
            '      "confidence": 0.85\n'
            "    }\n"
            "  ],\n"
            '  "uncategorized": ["hash_of_uncategorized_sample"]\n'
            "}"
        )

        try:
            raw = self._client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2000,
            )
            data = parse_json_response(raw)
        except Exception:
            self.logger.warning("LLM call or JSON parse failed for sample_pack_curator, returning empty result")
            return PackCurationResult(packs=[], uncategorized=[])

        if not data:
            return PackCurationResult(packs=[], uncategorized=[])

        packs = []
        for p in data.get("packs", []):
            try:
                packs.append(SamplePack(
                    name=p.get("name", "Unnamed Pack"),
                    description=p.get("description", ""),
                    tags=p.get("tags", []),
                    sample_hashes=p.get("sample_hashes", []),
                    confidence=float(p.get("confidence", 0.5)),
                ))
            except (TypeError, ValueError):
                continue

        uncategorized = data.get("uncategorized", [])
        if not isinstance(uncategorized, list):
            uncategorized = []

        return PackCurationResult(packs=packs, uncategorized=uncategorized)

    def estimate_cost(self, results: List[AnalysisResult]) -> CostEstimate:
        """Estimate cost for batch curation of samples."""
        count = len(results) if isinstance(results, list) else 1
        tokens = 500 + 150 * count
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=count,
        )
