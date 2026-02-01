"""Natural Language Sample Search - LLM feature for querying samples with plain text."""

from typing import Any, Dict, List, Optional, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, SearchResult, SearchMatch
from src.features.protocols import BaseLLMFeature


class NaturalLanguageSearch(BaseLLMFeature):
    """Enables natural language queries against analyzed sample libraries."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="natural_language_search",
            display_name="Natural Language Sample Search",
            feature_type="single+batch",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self, results: List[AnalysisResult], **kwargs: Any
    ) -> SearchResult:
        """Score each sample against a natural language query via LLM."""
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        query: str = kwargs.get("query", "")
        if not query:
            return SearchResult(query=query, matches=[], total_searched=len(results))

        # Normalize to list
        if isinstance(results, AnalysisResult):
            results = [results]

        sample_block = results_to_prompt_block(results)

        system_prompt = (
            "You are an expert audio sample search engine. Given a natural language "
            "query and a set of audio sample analyses, you score how well each sample "
            "matches the query. You consider sonic characteristics, musical properties, "
            "tags, descriptions, and implied production intent."
        )

        user_prompt = (
            f'Search query: "{query}"\n\n'
            f"Available samples:\n{sample_block}\n\n"
            "Score each sample's relevance to the query from 0.0 (no match) to 1.0 "
            "(perfect match). Only include samples with relevance > 0.1. Explain why "
            "each sample matches and list the attributes that contributed.\n\n"
            "Respond with JSON in this exact format:\n"
            "{\n"
            '  "matches": [\n'
            "    {\n"
            '      "sample_hash": "the_hash",\n'
            '      "relevance_score": 0.85,\n'
            '      "explanation": "Why this sample matches the query",\n'
            '      "matched_attributes": ["attribute1", "attribute2"]\n'
            "    }\n"
            "  ]\n"
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
            self.logger.warning("LLM call or JSON parse failed for natural_language_search, returning empty result")
            return SearchResult(query=query, matches=[], total_searched=len(results))

        if not data:
            return SearchResult(query=query, matches=[], total_searched=len(results))

        matches = []
        for m in data.get("matches", []):
            try:
                matches.append(SearchMatch(
                    sample_hash=m.get("sample_hash", ""),
                    relevance_score=float(m.get("relevance_score", 0.0)),
                    explanation=m.get("explanation", ""),
                    matched_attributes=m.get("matched_attributes", []),
                ))
            except (TypeError, ValueError):
                continue

        # Sort by relevance descending
        matches.sort(key=lambda x: x.relevance_score, reverse=True)

        return SearchResult(
            query=query,
            matches=matches,
            total_searched=len(results),
        )

    def estimate_cost(self, results: List[AnalysisResult]) -> CostEstimate:
        """Estimate cost per query against the sample set."""
        count = len(results) if isinstance(results, list) else 1
        tokens = 400 + 120 * count
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=count,
        )
