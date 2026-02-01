"""Anomaly / Quality Flag Reporter — LLM feature.

Scans a batch of analysis results for quality issues, anomalies,
and potential duplicates, producing a consolidated report.
"""

from typing import Any, Dict, List, Optional, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, AnomalyReport, AnomalyFlag, DuplicateGroup
from src.features.protocols import BaseLLMFeature


class AnomalyReporter(BaseLLMFeature):
    """Reports anomalies, quality flags, and duplicates across a sample batch."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="anomaly_reporter",
            display_name="Anomaly / Quality Flag Reporter",
            feature_type="batch",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        **kwargs: Any,
    ) -> AnomalyReport:
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        # results is guaranteed to be a list for batch features
        result_list: List[AnalysisResult] = results  # type: ignore[assignment]

        # Build prompt
        sample_block = results_to_prompt_block(result_list)

        system_prompt = (
            "You are an expert audio quality engineer and sample library curator. "
            "You specialize in detecting quality issues, anomalies, mislabeled files, "
            "and near-duplicate samples in audio collections. You are thorough and "
            "precise in your assessments."
        )

        user_prompt = (
            f"Scan the following batch of audio samples for quality issues and anomalies.\n\n"
            f"Sample data:\n{sample_block}\n\n"
            f"Check for the following issues:\n"
            f"- Clipping or distortion (unusually high peak levels)\n"
            f"- DC offset problems\n"
            f"- Mislabeled samples (e.g. a drum loop labeled as a one-shot, "
            f"wrong key/tempo metadata)\n"
            f"- Near-duplicate samples that may be redundant\n"
            f"- Outliers that don't fit with the rest of the batch\n"
            f"- Any other quality concerns\n\n"
            f"Respond with a JSON object containing:\n"
            f"- \"flags\": list of objects, each with:\n"
            f"  - \"sample_hash\": the hash of the flagged sample\n"
            f"  - \"flag_type\": one of \"clipping\", \"dc_offset\", \"mislabel\", "
            f"\"duplicate\", \"outlier\"\n"
            f"  - \"severity\": one of \"low\", \"medium\", \"high\"\n"
            f"  - \"description\": brief description of the issue\n"
            f"  - \"recommendation\": suggested action to fix or address the issue\n"
            f"- \"duplicate_groups\": list of objects, each with:\n"
            f"  - \"hashes\": list of sample hashes that are near-duplicates of each other\n"
            f"  - \"similarity_reason\": why they appear to be duplicates\n"
            f"- \"overall_quality_score\": float from 0.0 to 1.0 "
            f"(1.0 = no issues found, lower = more problems)\n"
            f"- \"summary\": a brief overall summary of the batch quality\n\n"
            f"If no issues are found, return empty lists and a score of 1.0. "
            f"Respond with ONLY the JSON object, no extra text."
        )

        # Call LLM
        raw = self._client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2500,
        )

        # Parse response
        data = parse_json_response(raw)

        if data.get("_parse_error"):
            raw_text = data.get("_raw_response", "")[:500]
            self.logger.warning("parse_json_response returned _parse_error for anomaly_reporter")
            return AnomalyReport(
                flags=[],
                duplicate_groups=[],
                overall_quality_score=0.0,
                summary=f"[Parse error] Raw LLM response: {raw_text}",
            )

        # Build result with graceful fallback
        try:
            flags = []
            for f in data.get("flags", []):
                flags.append(
                    AnomalyFlag(
                        sample_hash=f.get("sample_hash", ""),
                        flag_type=f.get("flag_type", "outlier"),
                        severity=f.get("severity", "low"),
                        description=f.get("description", ""),
                        recommendation=f.get("recommendation", ""),
                    )
                )

            duplicate_groups = []
            for g in data.get("duplicate_groups", []):
                duplicate_groups.append(
                    DuplicateGroup(
                        hashes=g.get("hashes", []),
                        similarity_reason=g.get("similarity_reason", ""),
                    )
                )

            return AnomalyReport(
                flags=flags,
                duplicate_groups=duplicate_groups,
                overall_quality_score=float(data.get("overall_quality_score", 1.0)),
                summary=data.get("summary", "No anomalies detected"),
            )
        except (TypeError, ValueError, KeyError) as exc:
            self.logger.warning("Failed to build AnomalyReport from LLM data: %s", exc)
            return AnomalyReport(
                flags=[],
                duplicate_groups=[],
                overall_quality_score=1.0,
                summary="Analysis failed — could not parse LLM response",
            )

    def estimate_cost(self, results: Union[AnalysisResult, List[AnalysisResult]]) -> CostEstimate:
        num_files = len(results) if isinstance(results, list) else 1
        tokens = 4000 + (250 * num_files)
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=num_files,
        )
