"""Comprehensive tests for all 10 LLM feature FULL implementations.

Tests the real implementations (not stubs) by mocking the LLM client
to return realistic JSON responses. Each test exercises the full
Template Method path: gate check -> validate -> _execute_impl -> wrap.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.core.models import AnalysisResult
from src.features.gate import FeatureGate, AlwaysEntitled
from src.features.models import (
    PackCurationResult,
    SearchResult,
    SuggestionResult,
    RenameResult,
    ProductionNotesResult,
    SpokenContentResult,
    SimilarityResult,
    ChainResult,
    MarketplaceResult,
    AnomalyReport,
    DAWContext,
    FeatureResult,
)


# ---------------------------------------------------------------------------
# Shared config and helpers
# ---------------------------------------------------------------------------

ALL_ENABLED = {
    "user_id": "test",
    "sample_pack_curator": {"enabled": True},
    "natural_language_search": {"enabled": True},
    "daw_suggestions": {"enabled": True},
    "batch_rename": {"enabled": True},
    "production_notes": {"enabled": True},
    "speech_deep_analyzer": {"enabled": True},
    "similar_sample_finder": {"enabled": True},
    "sample_chain": {"enabled": True},
    "marketplace_description": {"enabled": True},
    "anomaly_reporter": {"enabled": True},
}


@pytest.fixture
def gate():
    return FeatureGate(config=ALL_ENABLED, entitlement=AlwaysEntitled())


def _make_client(response: str) -> MagicMock:
    """Create a MagicMock LLM client that returns the given JSON string."""
    client = MagicMock()
    client.chat.return_value = response
    client.model_id = "mock/test-model"
    client.cost_per_1k_tokens = 0.0
    return client


def make_mock_result(hash_val="abc123"):
    """Build a richly-populated mock AnalysisResult."""
    result = MagicMock(spec=AnalysisResult)
    result.audio_sample_hash = hash_val
    result.processing_time = 0.5
    result.quality_metadata = {"quality_tier": "high", "sample_rate": 44100}

    result.source_classification = MagicMock()
    result.source_classification.source_type = "drum"
    result.source_classification.confidence = 0.9
    result.source_classification.characteristics = ["punchy"]

    result.musical_analysis = MagicMock()
    result.musical_analysis.has_pitch = True
    result.musical_analysis.note_name = "C4"
    result.musical_analysis.estimated_key = "C major"
    result.musical_analysis.key_confidence = 0.85
    result.musical_analysis.fundamental_frequency = 261.6
    result.musical_analysis.is_atonal = False

    result.percussive_analysis = MagicMock()
    result.percussive_analysis.drum_type = "kick"
    result.percussive_analysis.confidence = 0.85
    result.percussive_analysis.attack_time = 5.0
    result.percussive_analysis.brightness = 2000.0
    result.percussive_analysis.is_synthesized = False

    result.rhythmic_analysis = MagicMock()
    result.rhythmic_analysis.is_one_shot = True
    result.rhythmic_analysis.has_rhythm = False
    result.rhythmic_analysis.tempo_bpm = 0.0
    result.rhythmic_analysis.num_beats = 0

    result.llm_analysis = MagicMock()
    result.llm_analysis.suggested_name = "Kick_C4"
    result.llm_analysis.description = "Punchy kick drum in C4"
    result.llm_analysis.tags = ["kick", "drum"]
    result.llm_analysis.contains_speech = False
    result.llm_analysis.transcription = ""

    return result


# ===========================================================================
# Feature 1: SamplePackCurator
# ===========================================================================


class TestSamplePackCuratorFull:
    """Tests for the SamplePackCurator full implementation."""

    MOCK_RESPONSE = json.dumps({
        "packs": [
            {
                "name": "Drums",
                "description": "Drum samples",
                "tags": ["drums", "percussion"],
                "sample_hashes": ["abc123"],
                "confidence": 0.92,
            }
        ],
        "uncategorized": [],
    })

    def test_returns_populated_result(self, gate):
        client = _make_client(self.MOCK_RESPONSE)

        from src.features.implementations.sample_pack_curator import SamplePackCurator
        feature = SamplePackCurator(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        assert isinstance(result, FeatureResult)
        assert isinstance(result.data, PackCurationResult)
        assert len(result.data.packs) == 1
        assert result.data.packs[0].name == "Drums"
        assert result.data.packs[0].description == "Drum samples"
        assert "drums" in result.data.packs[0].tags
        assert result.data.packs[0].sample_hashes == ["abc123"]
        assert result.data.packs[0].confidence == 0.92
        assert result.data.uncategorized == []
        assert client.chat.call_count == 1

    def test_empty_llm_response_returns_empty_packs(self, gate):
        client = _make_client("{}")
        from src.features.implementations.sample_pack_curator import SamplePackCurator
        feature = SamplePackCurator(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        assert isinstance(result.data, PackCurationResult)
        assert result.data.packs == []
        assert result.data.uncategorized == []

    def test_invalid_json_returns_parse_error_result(self, gate):
        client = _make_client("not valid json at all")
        from src.features.implementations.sample_pack_curator import SamplePackCurator
        feature = SamplePackCurator(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        assert isinstance(result.data, PackCurationResult)
        # Parse error now surfaces raw text in a visible field
        assert len(result.data.packs) >= 1

    def test_multiple_packs_returned(self, gate):
        response = json.dumps({
            "packs": [
                {
                    "name": "Kicks",
                    "description": "Kick drums",
                    "tags": ["kick"],
                    "sample_hashes": ["abc123"],
                    "confidence": 0.9,
                },
                {
                    "name": "Snares",
                    "description": "Snare drums",
                    "tags": ["snare"],
                    "sample_hashes": ["def456"],
                    "confidence": 0.88,
                },
            ],
            "uncategorized": ["ghi789"],
        })
        client = _make_client(response)
        from src.features.implementations.sample_pack_curator import SamplePackCurator
        feature = SamplePackCurator(client=client, gate=gate)
        result = feature.execute([make_mock_result("abc123"), make_mock_result("def456")])

        assert len(result.data.packs) == 2
        assert result.data.packs[0].name == "Kicks"
        assert result.data.packs[1].name == "Snares"
        assert result.data.uncategorized == ["ghi789"]


# ===========================================================================
# Feature 2: NaturalLanguageSearch
# ===========================================================================


class TestNaturalLanguageSearchFull:
    """Tests for the NaturalLanguageSearch full implementation."""

    MOCK_RESPONSE = json.dumps({
        "matches": [
            {
                "sample_hash": "abc123",
                "relevance_score": 0.9,
                "explanation": "matches query for punchy kick",
                "matched_attributes": ["drum_type", "characteristics"],
            }
        ],
    })

    def test_returns_matches(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.natural_language_search import NaturalLanguageSearch
        feature = NaturalLanguageSearch(client=client, gate=gate)
        result = feature.execute([make_mock_result()], query="punchy kick drum")

        assert isinstance(result.data, SearchResult)
        assert result.data.query == "punchy kick drum"
        assert len(result.data.matches) == 1
        assert result.data.matches[0].sample_hash == "abc123"
        assert result.data.matches[0].relevance_score == 0.9
        assert result.data.matches[0].explanation == "matches query for punchy kick"
        assert "drum_type" in result.data.matches[0].matched_attributes
        assert result.data.total_searched == 1

    def test_empty_query_returns_empty_result(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.natural_language_search import NaturalLanguageSearch
        feature = NaturalLanguageSearch(client=client, gate=gate)
        result = feature.execute([make_mock_result()], query="")

        assert isinstance(result.data, SearchResult)
        assert result.data.matches == []
        assert result.data.total_searched == 1
        # No LLM call should be made for empty query
        assert client.chat.call_count == 0

    def test_results_sorted_by_relevance(self, gate):
        response = json.dumps({
            "matches": [
                {"sample_hash": "low", "relevance_score": 0.3, "explanation": "weak", "matched_attributes": []},
                {"sample_hash": "high", "relevance_score": 0.95, "explanation": "strong", "matched_attributes": []},
                {"sample_hash": "mid", "relevance_score": 0.6, "explanation": "moderate", "matched_attributes": []},
            ],
        })
        client = _make_client(response)
        from src.features.implementations.natural_language_search import NaturalLanguageSearch
        feature = NaturalLanguageSearch(client=client, gate=gate)
        result = feature.execute(
            [make_mock_result("low"), make_mock_result("high"), make_mock_result("mid")],
            query="some query",
        )

        scores = [m.relevance_score for m in result.data.matches]
        assert scores == sorted(scores, reverse=True)
        assert result.data.matches[0].sample_hash == "high"


# ===========================================================================
# Feature 3: DAWContextSuggester
# ===========================================================================


class TestDAWContextSuggesterFull:
    """Tests for the DAWContextSuggester full implementation."""

    MOCK_RESPONSE = json.dumps({
        "suggestions": [
            {
                "sample_hash": "abc123",
                "fit_score": 0.8,
                "reason": "good fit for the project key and tempo",
                "conflicts": [],
                "layer_with": [],
            }
        ],
        "layer_groups": [],
    })

    def test_returns_suggestion_result(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.daw_suggestions import DAWContextSuggester
        feature = DAWContextSuggester(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        assert isinstance(result.data, SuggestionResult)
        assert len(result.data.suggestions) == 1
        assert result.data.suggestions[0].sample_hash == "abc123"
        assert result.data.suggestions[0].fit_score == 0.8
        assert result.data.suggestions[0].reason == "good fit for the project key and tempo"

    def test_custom_daw_context_is_used(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.daw_suggestions import DAWContextSuggester
        feature = DAWContextSuggester(client=client, gate=gate)

        custom_context = DAWContext(
            bpm=140.0, key="A minor", genre="drum and bass", mood="dark",
            notes="heavy bass needed"
        )
        result = feature.execute([make_mock_result()], context=custom_context)

        assert result.data.context == custom_context
        assert result.data.context.bpm == 140.0
        assert result.data.context.key == "A minor"
        assert result.data.context.genre == "drum and bass"
        # Verify the context was passed to the LLM prompt
        call_args = client.chat.call_args
        assert "140.0" in call_args[1]["user_prompt"] or "140.0" in str(call_args)

    def test_default_context_when_none_provided(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.daw_suggestions import DAWContextSuggester
        feature = DAWContextSuggester(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        # Default context should be electronic, 120 bpm, C major
        assert result.data.context.bpm == 120.0
        assert result.data.context.key == "C major"

    def test_layer_groups_returned(self, gate):
        response = json.dumps({
            "suggestions": [
                {"sample_hash": "abc123", "fit_score": 0.8, "reason": "good", "conflicts": [], "layer_with": ["def456"]},
                {"sample_hash": "def456", "fit_score": 0.7, "reason": "ok", "conflicts": [], "layer_with": ["abc123"]},
            ],
            "layer_groups": [["abc123", "def456"]],
        })
        client = _make_client(response)
        from src.features.implementations.daw_suggestions import DAWContextSuggester
        feature = DAWContextSuggester(client=client, gate=gate)
        result = feature.execute([make_mock_result("abc123"), make_mock_result("def456")])

        assert len(result.data.layer_groups) == 1
        assert result.data.layer_groups[0] == ["abc123", "def456"]


# ===========================================================================
# Feature 4: BatchRenamer
# ===========================================================================


class TestBatchRenamerFull:
    """Tests for the BatchRenamer full implementation."""

    MOCK_RESPONSE = json.dumps({
        "entries": [
            {
                "sample_hash": "abc123",
                "new_name": "Kick_Punchy_C4",
                "confidence": 0.9,
            }
        ],
        "conflicts": [],
    })

    def test_returns_rename_entries(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.batch_rename import BatchRenamer
        feature = BatchRenamer(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        assert isinstance(result.data, RenameResult)
        assert len(result.data.entries) == 1
        assert result.data.entries[0].new_name == "Kick_Punchy_C4"
        assert result.data.entries[0].confidence == 0.9
        assert result.data.conflicts == []

    def test_dry_run_flag_preserved(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.batch_rename import BatchRenamer
        feature = BatchRenamer(client=client, gate=gate)

        result_dry = feature.execute([make_mock_result()], dry_run=True)
        assert result_dry.data.is_dry_run is True

        result_live = feature.execute([make_mock_result()], dry_run=False)
        assert result_live.data.is_dry_run is False

    def test_custom_template_preserved(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.batch_rename import BatchRenamer
        feature = BatchRenamer(client=client, gate=gate)
        result = feature.execute([make_mock_result()], template="BPM_Key_Type")

        assert result.data.naming_template == "BPM_Key_Type"

    def test_naming_conflicts_reported(self, gate):
        response = json.dumps({
            "entries": [
                {"sample_hash": "abc123", "new_name": "Kick_01", "confidence": 0.9},
                {"sample_hash": "def456", "new_name": "Kick_01", "confidence": 0.85},
            ],
            "conflicts": ["Kick_01 used by two samples"],
        })
        client = _make_client(response)
        from src.features.implementations.batch_rename import BatchRenamer
        feature = BatchRenamer(client=client, gate=gate)
        result = feature.execute([make_mock_result("abc123"), make_mock_result("def456")])

        assert len(result.data.conflicts) == 1
        assert "Kick_01" in result.data.conflicts[0]


# ===========================================================================
# Feature 5: ProductionNotesGenerator
# ===========================================================================


class TestProductionNotesGeneratorFull:
    """Tests for the ProductionNotesGenerator full implementation."""

    MOCK_RESPONSE = json.dumps({
        "eq_suggestions": ["Cut 200Hz by 2dB to reduce muddiness"],
        "layering_advice": ["Layer with a sub bass for low-end weight"],
        "mixing_tips": ["Pan slightly left for stereo width"],
        "arrangement_placement": ["Use in drops and choruses"],
        "processing_chain": ["1. HPF at 80Hz", "2. Compression 3:1"],
        "compatible_genres": ["house", "techno"],
    })

    def test_returns_populated_notes(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.production_notes import ProductionNotesGenerator
        feature = ProductionNotesGenerator(client=client, gate=gate)
        # ProductionNotes is a "single" feature, pass a single result not a list
        result = feature.execute(make_mock_result())

        assert isinstance(result.data, ProductionNotesResult)
        assert result.data.sample_hash == "abc123"
        assert len(result.data.eq_suggestions) == 1
        assert "200Hz" in result.data.eq_suggestions[0]
        assert len(result.data.layering_advice) == 1
        assert len(result.data.mixing_tips) == 1
        assert len(result.data.arrangement_placement) == 1
        assert len(result.data.processing_chain) == 2
        assert "house" in result.data.compatible_genres
        assert "techno" in result.data.compatible_genres

    def test_single_result_not_list(self, gate):
        """ProductionNotes is a 'single' feature; passing a list should raise ValueError."""
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.production_notes import ProductionNotesGenerator
        feature = ProductionNotesGenerator(client=client, gate=gate)

        with pytest.raises(ValueError, match="expects a single"):
            feature.execute([make_mock_result()])

    def test_empty_llm_response_returns_empty_lists(self, gate):
        client = _make_client("{}")
        from src.features.implementations.production_notes import ProductionNotesGenerator
        feature = ProductionNotesGenerator(client=client, gate=gate)
        result = feature.execute(make_mock_result())

        assert isinstance(result.data, ProductionNotesResult)
        assert result.data.eq_suggestions == []
        assert result.data.layering_advice == []
        assert result.data.mixing_tips == []
        assert result.data.compatible_genres == []


# ===========================================================================
# Feature 6: SpokenContentAnalyzer
# ===========================================================================


class TestSpokenContentAnalyzerFull:
    """Tests for the SpokenContentAnalyzer full implementation."""

    MOCK_RESPONSE = json.dumps({
        "sentiment": "positive",
        "sentiment_score": 0.8,
        "tone": "energetic",
        "language_register": "informal",
        "genre_fit": ["hip-hop", "EDM vocal chop"],
        "content_warnings": [],
        "licensing_notes": "Clear for use",
    })

    def _make_speech_result(self, hash_val="abc123"):
        """Build a mock result with speech detected."""
        result = make_mock_result(hash_val)
        result.llm_analysis.contains_speech = True
        result.llm_analysis.transcription = "Yeah, let's go!"
        return result

    def test_speech_sample_returns_populated_result(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.speech_deep_analyzer import SpokenContentAnalyzer
        feature = SpokenContentAnalyzer(client=client, gate=gate)
        result = feature.execute(self._make_speech_result())

        assert isinstance(result.data, SpokenContentResult)
        assert result.data.sample_hash == "abc123"
        assert result.data.sentiment == "positive"
        assert result.data.sentiment_score == 0.8
        assert result.data.tone == "energetic"
        assert result.data.language_register == "informal"
        assert "hip-hop" in result.data.genre_fit
        assert result.data.content_warnings == []
        assert result.data.licensing_notes == "Clear for use"
        assert client.chat.call_count == 1

    def test_non_speech_sample_returns_neutral_defaults(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.speech_deep_analyzer import SpokenContentAnalyzer
        feature = SpokenContentAnalyzer(client=client, gate=gate)
        # Use a result with no speech detected
        result = feature.execute(make_mock_result())

        assert isinstance(result.data, SpokenContentResult)
        assert result.data.sentiment == "neutral"
        assert result.data.sentiment_score == 0.0
        assert result.data.tone == "neutral"
        assert result.data.language_register == "unknown"
        assert result.data.genre_fit == []
        assert result.data.licensing_notes == "No speech detected"
        # No LLM call should be made
        assert client.chat.call_count == 0

    def test_single_feature_rejects_list(self, gate):
        """SpokenContentAnalyzer is a 'single' feature; list input should raise."""
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.speech_deep_analyzer import SpokenContentAnalyzer
        feature = SpokenContentAnalyzer(client=client, gate=gate)

        with pytest.raises(ValueError, match="expects a single"):
            feature.execute([self._make_speech_result()])


# ===========================================================================
# Feature 7: SimilarSampleFinder
# ===========================================================================


class TestSimilarSampleFinderFull:
    """Tests for the SimilarSampleFinder full implementation."""

    MOCK_RESPONSE = json.dumps({
        "matches": [
            {
                "sample_hash": "def456",
                "similarity_score": 0.85,
                "explanation": "similar timbre and drum type",
                "shared_attributes": {"drum_type": "kick", "key": "C major"},
            }
        ],
        "weighting_strategy": "balanced",
    })

    def test_returns_similarity_result(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.similar_sample_finder import SimilarSampleFinder
        feature = SimilarSampleFinder(client=client, gate=gate)
        result = feature.execute(
            [make_mock_result("abc123"), make_mock_result("def456")],
            reference_hash="abc123",
        )

        assert isinstance(result.data, SimilarityResult)
        assert result.data.reference_hash == "abc123"
        assert len(result.data.matches) == 1
        assert result.data.matches[0].sample_hash == "def456"
        assert result.data.matches[0].similarity_score == 0.85
        assert result.data.matches[0].explanation == "similar timbre and drum type"
        assert result.data.matches[0].shared_attributes["drum_type"] == "kick"
        assert result.data.weighting_strategy == "balanced"

    def test_defaults_reference_to_first_sample(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.similar_sample_finder import SimilarSampleFinder
        feature = SimilarSampleFinder(client=client, gate=gate)
        result = feature.execute([make_mock_result("first"), make_mock_result("second")])

        assert result.data.reference_hash == "first"

    def test_single_result_mode(self, gate):
        """SimilarSampleFinder supports single+batch, so single input should work."""
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.similar_sample_finder import SimilarSampleFinder
        feature = SimilarSampleFinder(client=client, gate=gate)
        result = feature.execute(make_mock_result("abc123"))

        assert isinstance(result.data, SimilarityResult)
        assert result.data.reference_hash == "abc123"


# ===========================================================================
# Feature 8: SampleChainSuggester
# ===========================================================================


class TestSampleChainSuggesterFull:
    """Tests for the SampleChainSuggester full implementation."""

    MOCK_RESPONSE = json.dumps({
        "ordered_hashes": ["abc123", "def456"],
        "transitions": [
            {
                "from_hash": "abc123",
                "to_hash": "def456",
                "transition_note": "energy builds with rising pitch",
                "compatibility_score": 0.9,
            }
        ],
        "energy_arc": "ascending",
        "overall_score": 0.85,
    })

    def test_returns_chain_result(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.sample_chain import SampleChainSuggester
        feature = SampleChainSuggester(client=client, gate=gate)
        result = feature.execute([make_mock_result("abc123"), make_mock_result("def456")])

        assert isinstance(result.data, ChainResult)
        assert result.data.ordered_hashes == ["abc123", "def456"]
        assert len(result.data.transitions) == 1
        assert result.data.transitions[0].from_hash == "abc123"
        assert result.data.transitions[0].to_hash == "def456"
        assert result.data.transitions[0].compatibility_score == 0.9
        assert "energy builds" in result.data.transitions[0].transition_note
        assert result.data.energy_arc == "ascending"
        assert result.data.overall_score == 0.85

    def test_energy_preference_passed(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.sample_chain import SampleChainSuggester
        feature = SampleChainSuggester(client=client, gate=gate)
        feature.execute(
            [make_mock_result("abc123"), make_mock_result("def456")],
            energy_preference="descending",
        )

        call_args = client.chat.call_args
        prompt_text = str(call_args)
        assert "descending" in prompt_text

    def test_invalid_json_falls_back_gracefully(self, gate):
        client = _make_client("totally broken json {{{")
        from src.features.implementations.sample_chain import SampleChainSuggester
        feature = SampleChainSuggester(client=client, gate=gate)
        result = feature.execute([make_mock_result("abc123")])

        assert isinstance(result.data, ChainResult)
        # parse_json_response now returns _parse_error dict; feature surfaces raw text
        assert result.data.overall_score == 0.0
        assert "parse error" in (result.data.energy_arc or "").lower() or result.data.ordered_hashes == []


# ===========================================================================
# Feature 9: MarketplaceDescriptionGen
# ===========================================================================


class TestMarketplaceDescriptionGenFull:
    """Tests for the MarketplaceDescriptionGen full implementation."""

    MOCK_RESPONSE = json.dumps({
        "headline": "Epic Drums - Professional Drum Kit",
        "description": "A collection of punchy, professional-grade drum samples.",
        "tags": ["drums", "punchy", "electronic", "kick", "snare"],
        "genres": ["electronic", "house", "techno"],
        "stats": {"total_samples": 2, "one_shots": 2, "loops": 0},
    })

    def test_returns_marketplace_result(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.marketplace_description import MarketplaceDescriptionGen
        feature = MarketplaceDescriptionGen(client=client, gate=gate)
        result = feature.execute(
            [make_mock_result("abc123"), make_mock_result("def456")],
            pack_name="Epic Drums",
        )

        assert isinstance(result.data, MarketplaceResult)
        assert result.data.pack_name == "Epic Drums"
        assert result.data.headline == "Epic Drums - Professional Drum Kit"
        assert "punchy" in result.data.description
        assert len(result.data.tags) == 5
        assert "electronic" in result.data.genres
        assert result.data.stats["total_samples"] == 2

    def test_pack_name_kwarg_preserved(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.marketplace_description import MarketplaceDescriptionGen
        feature = MarketplaceDescriptionGen(client=client, gate=gate)

        result = feature.execute(
            [make_mock_result()],
            pack_name="My Custom Pack",
        )
        assert result.data.pack_name == "My Custom Pack"

    def test_default_pack_name(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.marketplace_description import MarketplaceDescriptionGen
        feature = MarketplaceDescriptionGen(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        assert result.data.pack_name == "Untitled Pack"

    def test_target_platforms_from_kwargs(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.marketplace_description import MarketplaceDescriptionGen
        feature = MarketplaceDescriptionGen(client=client, gate=gate)
        result = feature.execute(
            [make_mock_result()],
            platforms=["splice", "loopmasters"],
        )

        assert result.data.target_platforms == ["splice", "loopmasters"]


# ===========================================================================
# Feature 10: AnomalyReporter
# ===========================================================================


class TestAnomalyReporterFull:
    """Tests for the AnomalyReporter full implementation."""

    MOCK_RESPONSE = json.dumps({
        "flags": [
            {
                "sample_hash": "abc123",
                "flag_type": "clipping",
                "severity": "high",
                "description": "Peak exceeds 0dB, detected clipping artifacts",
                "recommendation": "Reduce gain by 3dB and re-export",
            }
        ],
        "duplicate_groups": [],
        "overall_quality_score": 0.7,
        "summary": "One clipping issue found in the batch",
    })

    def test_returns_anomaly_report_with_flags(self, gate):
        client = _make_client(self.MOCK_RESPONSE)
        from src.features.implementations.anomaly_reporter import AnomalyReporter
        feature = AnomalyReporter(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        assert isinstance(result.data, AnomalyReport)
        assert len(result.data.flags) == 1
        assert result.data.flags[0].sample_hash == "abc123"
        assert result.data.flags[0].flag_type == "clipping"
        assert result.data.flags[0].severity == "high"
        assert "Peak exceeds 0dB" in result.data.flags[0].description
        assert "Reduce gain" in result.data.flags[0].recommendation
        assert result.data.duplicate_groups == []
        assert result.data.overall_quality_score == 0.7
        assert "clipping" in result.data.summary

    def test_clean_batch_returns_perfect_score(self, gate):
        clean_response = json.dumps({
            "flags": [],
            "duplicate_groups": [],
            "overall_quality_score": 1.0,
            "summary": "No anomalies detected",
        })
        client = _make_client(clean_response)
        from src.features.implementations.anomaly_reporter import AnomalyReporter
        feature = AnomalyReporter(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        assert isinstance(result.data, AnomalyReport)
        assert result.data.flags == []
        assert result.data.overall_quality_score == 1.0
        assert result.data.summary == "No anomalies detected"

    def test_duplicate_groups_populated(self, gate):
        response = json.dumps({
            "flags": [],
            "duplicate_groups": [
                {
                    "hashes": ["abc123", "def456"],
                    "similarity_reason": "Nearly identical waveform and spectral profile",
                }
            ],
            "overall_quality_score": 0.8,
            "summary": "One potential duplicate group found",
        })
        client = _make_client(response)
        from src.features.implementations.anomaly_reporter import AnomalyReporter
        feature = AnomalyReporter(client=client, gate=gate)
        result = feature.execute([make_mock_result("abc123"), make_mock_result("def456")])

        assert len(result.data.duplicate_groups) == 1
        assert result.data.duplicate_groups[0].hashes == ["abc123", "def456"]
        assert "identical" in result.data.duplicate_groups[0].similarity_reason


# ===========================================================================
# Cross-cutting: Gate & Template Method tests
# ===========================================================================


class TestFeatureGateIntegration:
    """Tests verifying that the gate check is enforced through execute()."""

    def test_disabled_feature_raises(self):
        """Disabled feature should raise FeatureDisabledError through execute()."""
        from src.utils.errors import FeatureDisabledError
        from src.features.implementations.sample_pack_curator import SamplePackCurator

        disabled_config = {
            "user_id": "test",
            "sample_pack_curator": {"enabled": False},
        }
        gate = FeatureGate(config=disabled_config, entitlement=AlwaysEntitled())
        client = _make_client("{}")
        feature = SamplePackCurator(client=client, gate=gate)

        with pytest.raises(FeatureDisabledError):
            feature.execute([make_mock_result()])

    def test_feature_result_metadata(self, gate):
        """FeatureResult wrapper should contain correct metadata."""
        response = json.dumps({"packs": [], "uncategorized": []})
        client = _make_client(response)
        from src.features.implementations.sample_pack_curator import SamplePackCurator
        feature = SamplePackCurator(client=client, gate=gate)
        result = feature.execute([make_mock_result()])

        assert isinstance(result, FeatureResult)
        assert result.feature_id == "sample_pack_curator"
        assert result.model_used == "mock/test-model"
        assert result.processing_time >= 0.0
        assert isinstance(result.timestamp, datetime)
