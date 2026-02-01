"""
Data models for LLM feature results.

All result dataclasses are frozen (immutable) to match the existing
pattern in src/core/models.py and ensure thread safety.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Generic wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostEstimate:
    """Pre-execution cost/time estimate for a feature."""

    feature_id: str
    estimated_tokens: int
    estimated_cost_usd: float
    estimated_time_seconds: float
    sample_count: int
    warning: Optional[str] = None


@dataclass(frozen=True)
class FeatureResult:
    """Wrapper for any feature's output, with metadata."""

    feature_id: str
    data: Any  # Feature-specific frozen dataclass
    processing_time: float
    model_used: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        result = {
            "feature_id": self.feature_id,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
        }
        if hasattr(self.data, "__dataclass_fields__"):
            from dataclasses import asdict

            result["data"] = asdict(self.data)
        else:
            result["data"] = str(self.data)
        return result


# ---------------------------------------------------------------------------
# Feature 1: Smart Sample Pack Curator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplePack:
    """A themed group of samples."""

    name: str
    description: str
    tags: List[str]
    sample_hashes: List[str]
    confidence: float


@dataclass(frozen=True)
class PackCurationResult:
    """Result of pack curation across a batch."""

    packs: List[SamplePack]
    uncategorized: List[str]  # Hashes that didn't fit any pack


# ---------------------------------------------------------------------------
# Feature 2: Natural Language Sample Search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchMatch:
    """A single search result with relevance scoring."""

    sample_hash: str
    relevance_score: float
    explanation: str
    matched_attributes: List[str]


@dataclass(frozen=True)
class SearchResult:
    """Result of a natural language search query."""

    query: str
    matches: List[SearchMatch]
    total_searched: int


# ---------------------------------------------------------------------------
# Feature 3: DAW-Contextualized Suggestions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DAWContext:
    """User-provided project context."""

    bpm: float
    key: str
    genre: str
    mood: str
    notes: Optional[str] = None


@dataclass(frozen=True)
class SampleSuggestion:
    """A sample recommendation for a DAW context."""

    sample_hash: str
    fit_score: float
    reason: str
    conflicts: List[str]
    layer_with: List[str]


@dataclass(frozen=True)
class SuggestionResult:
    """Result of DAW-contextualized suggestions."""

    context: DAWContext
    suggestions: List[SampleSuggestion]
    layer_groups: List[List[str]]


# ---------------------------------------------------------------------------
# Feature 4: Automatic Batch Rename
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenameEntry:
    """A single rename mapping."""

    original_path: str
    new_name: str
    new_path: str
    confidence: float


@dataclass(frozen=True)
class RenameResult:
    """Result of batch rename operation."""

    entries: List[RenameEntry]
    naming_template: str
    conflicts: List[str]
    is_dry_run: bool = True


# ---------------------------------------------------------------------------
# Feature 5: Production Notes & Usage Tips
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionNotesResult:
    """Production advice for a single sample."""

    sample_hash: str
    eq_suggestions: List[str]
    layering_advice: List[str]
    mixing_tips: List[str]
    arrangement_placement: List[str]
    processing_chain: List[str]
    compatible_genres: List[str]


# ---------------------------------------------------------------------------
# Feature 6: Spoken Content Deep Analyzer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpokenContentResult:
    """Deep analysis of spoken content in a sample."""

    sample_hash: str
    sentiment: str  # positive / negative / neutral
    sentiment_score: float  # -1.0 to 1.0
    tone: str  # e.g. "aggressive", "calm", "excited"
    language_register: str  # formal / informal / slang
    genre_fit: List[str]  # e.g. ["hip-hop ad-lib", "podcast"]
    content_warnings: List[str]
    licensing_notes: str


# ---------------------------------------------------------------------------
# Feature 7: Similar Sample Finder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarMatch:
    """A similar sample with explanation."""

    sample_hash: str
    similarity_score: float
    explanation: str
    shared_attributes: Dict[str, str]


@dataclass(frozen=True)
class SimilarityResult:
    """Result of similarity search."""

    reference_hash: str
    matches: List[SimilarMatch]
    weighting_strategy: str


# ---------------------------------------------------------------------------
# Feature 8: Sample Chain / Transition Suggester
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionPoint:
    """Transition between two samples in a chain."""

    from_hash: str
    to_hash: str
    transition_note: str
    compatibility_score: float


@dataclass(frozen=True)
class ChainResult:
    """Optimal sample ordering with transition notes."""

    ordered_hashes: List[str]
    transitions: List[TransitionPoint]
    energy_arc: str
    overall_score: float


# ---------------------------------------------------------------------------
# Feature 9: Marketplace Description Generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketplaceResult:
    """Marketing copy for a sample pack."""

    pack_name: str
    headline: str
    description: str
    tags: List[str]
    genres: List[str]
    stats: Dict[str, Any]
    target_platforms: List[str]


# ---------------------------------------------------------------------------
# Feature 10: Anomaly / Quality Flag Reporter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyFlag:
    """A single quality/anomaly flag."""

    sample_hash: str
    flag_type: str  # dc_offset / clipping / mislabel / duplicate / outlier
    severity: str  # low / medium / high
    description: str
    recommendation: str


@dataclass(frozen=True)
class DuplicateGroup:
    """Group of near-duplicate samples."""

    hashes: List[str]
    similarity_reason: str


@dataclass(frozen=True)
class AnomalyReport:
    """Complete anomaly/quality report for a batch."""

    flags: List[AnomalyFlag]
    duplicate_groups: List[DuplicateGroup]
    overall_quality_score: float
    summary: str
