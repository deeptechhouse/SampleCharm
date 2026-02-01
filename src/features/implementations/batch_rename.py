"""Automatic Batch Rename - LLM feature for intelligent sample file renaming."""

import os
from typing import Any, Dict, List, Optional, Union

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, RenameResult, RenameEntry
from src.features.protocols import BaseLLMFeature


class BatchRenamer(BaseLLMFeature):
    """Generates descriptive filenames for samples based on their audio analysis."""

    def __init__(self, client: Any, gate: Any) -> None:
        super().__init__(
            feature_id="batch_rename",
            display_name="Automatic Batch Rename",
            feature_type="batch",
            version="1.0.0",
            client=client,
            gate=gate,
        )

    def _execute_impl(
        self, results: List[AnalysisResult], **kwargs: Any
    ) -> RenameResult:
        """Ask LLM to generate descriptive filenames for each sample."""
        from src.features.prompt_helpers import results_to_prompt_block, parse_json_response

        template: str = kwargs.get("template", "standard")
        dry_run: bool = kwargs.get("dry_run", True)

        sample_block = results_to_prompt_block(results)

        system_prompt = (
            "You are an expert audio sample librarian who specializes in creating "
            "clear, descriptive, and consistent filenames for audio samples. You follow "
            "industry naming conventions used by professional sample libraries."
        )

        user_prompt = (
            f"Naming template/convention: {template}\n\n"
            f"Samples to rename:\n{sample_block}\n\n"
            "Generate a descriptive filename for each sample based on its audio analysis. "
            "Filenames should be concise, use underscores for spaces, include relevant "
            "musical information (key, BPM, type), and follow the naming template. "
            "Do not include file extensions in the new name.\n\n"
            "Also check for any naming conflicts (duplicate names) and report them.\n\n"
            "Respond with JSON in this exact format:\n"
            "{\n"
            '  "entries": [\n'
            "    {\n"
            '      "sample_hash": "the_hash",\n'
            '      "new_name": "Kick_Punchy_C2_120bpm",\n'
            '      "confidence": 0.9\n'
            "    }\n"
            "  ],\n"
            '  "conflicts": ["description of any naming conflicts"]\n'
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
            self.logger.warning("LLM call or JSON parse failed for batch_rename, returning empty result")
            return RenameResult(
                entries=[], naming_template=template, conflicts=[], is_dry_run=dry_run,
            )

        if not data:
            return RenameResult(
                entries=[], naming_template=template, conflicts=[], is_dry_run=dry_run,
            )

        # Build a hash -> original_path lookup from the analysis results
        hash_to_path: Dict[str, str] = {}
        for r in results:
            # AnalysisResult doesn't directly store file_path, use hash as placeholder
            # The quality_metadata may contain path info; fall back to hash
            path = r.quality_metadata.get("file_path", r.audio_sample_hash)
            hash_to_path[r.audio_sample_hash] = str(path)

        entries = []
        for e in data.get("entries", []):
            try:
                sample_hash = e.get("sample_hash", "")
                new_name = e.get("new_name", "")
                original_path = hash_to_path.get(sample_hash, sample_hash)
                # Build new_path by replacing the filename in the original path
                if os.sep in original_path or "/" in original_path:
                    directory = os.path.dirname(original_path)
                    # Preserve original extension if present
                    _, ext = os.path.splitext(original_path)
                    new_path = os.path.join(directory, new_name + ext)
                else:
                    new_path = new_name

                entries.append(RenameEntry(
                    original_path=original_path,
                    new_name=new_name,
                    new_path=new_path,
                    confidence=float(e.get("confidence", 0.5)),
                ))
            except (TypeError, ValueError):
                continue

        conflicts = data.get("conflicts", [])
        if not isinstance(conflicts, list):
            conflicts = []

        return RenameResult(
            entries=entries,
            naming_template=template,
            conflicts=conflicts,
            is_dry_run=dry_run,
        )

    def estimate_cost(self, results: List[AnalysisResult]) -> CostEstimate:
        """Estimate cost for batch rename generation."""
        count = len(results) if isinstance(results, list) else 1
        tokens = 400 + 100 * count
        return CostEstimate(
            feature_id=self.feature_id,
            estimated_tokens=tokens,
            estimated_cost_usd=(tokens / 1000) * self._client.cost_per_1k_tokens,
            estimated_time_seconds=tokens * 0.002,
            sample_count=count,
        )
