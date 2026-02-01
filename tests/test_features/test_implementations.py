"""Tests for all 10 LLM feature stub implementations."""

import pytest

from src.features.implementations import ALL_FEATURE_CLASSES
from src.features.models import (
    AnomalyReport,
    ChainResult,
    CostEstimate,
    FeatureResult,
    MarketplaceResult,
    PackCurationResult,
    ProductionNotesResult,
    RenameResult,
    SearchResult,
    SimilarityResult,
    SpokenContentResult,
    SuggestionResult,
)
from src.features.protocols import BaseLLMFeature


# ---------------------------------------------------------------------------
# Feature registration
# ---------------------------------------------------------------------------


class TestFeatureRegistration:
    def test_all_feature_classes_count(self):
        assert len(ALL_FEATURE_CLASSES) == 10

    def test_all_extend_base(self):
        for cls in ALL_FEATURE_CLASSES:
            assert issubclass(cls, BaseLLMFeature)

    def test_unique_feature_ids(self, mock_client, all_enabled_gate):
        ids = set()
        for cls in ALL_FEATURE_CLASSES:
            feature = cls(client=mock_client, gate=all_enabled_gate)
            assert feature.feature_id not in ids
            ids.add(feature.feature_id)

    def test_all_have_version(self, mock_client, all_enabled_gate):
        for cls in ALL_FEATURE_CLASSES:
            feature = cls(client=mock_client, gate=all_enabled_gate)
            assert feature.version


# ---------------------------------------------------------------------------
# Feature 1: Smart Sample Pack Curator
# ---------------------------------------------------------------------------


class TestSamplePackCurator:
    def test_execute_returns_pack_curation(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        from src.features.implementations import SamplePackCurator

        feature = SamplePackCurator(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_results)

        assert isinstance(result, FeatureResult)
        assert isinstance(result.data, PackCurationResult)
        assert result.feature_id == "sample_pack_curator"

    def test_feature_type_is_batch(self, mock_client, all_enabled_gate):
        from src.features.implementations import SamplePackCurator

        feature = SamplePackCurator(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "batch"

    def test_estimate_cost(self, mock_client, all_enabled_gate, mock_analysis_results):
        from src.features.implementations import SamplePackCurator

        feature = SamplePackCurator(client=mock_client, gate=all_enabled_gate)
        estimate = feature.estimate_cost(mock_analysis_results)
        assert isinstance(estimate, CostEstimate)
        assert estimate.sample_count == 3


# ---------------------------------------------------------------------------
# Feature 2: Natural Language Search
# ---------------------------------------------------------------------------


class TestNaturalLanguageSearch:
    def test_execute_returns_search_result(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        from src.features.implementations import NaturalLanguageSearch

        feature = NaturalLanguageSearch(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_results, query="warm pad")

        assert isinstance(result.data, SearchResult)
        assert result.data.query == "warm pad"

    def test_feature_type_is_single_plus_batch(self, mock_client, all_enabled_gate):
        from src.features.implementations import NaturalLanguageSearch

        feature = NaturalLanguageSearch(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "single+batch"


# ---------------------------------------------------------------------------
# Feature 3: DAW-Contextualized Suggestions
# ---------------------------------------------------------------------------


class TestDAWContextSuggester:
    def test_execute_returns_suggestion_result(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        from src.features.implementations import DAWContextSuggester

        feature = DAWContextSuggester(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_results)

        assert isinstance(result.data, SuggestionResult)

    def test_feature_type_is_batch(self, mock_client, all_enabled_gate):
        from src.features.implementations import DAWContextSuggester

        feature = DAWContextSuggester(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "batch"


# ---------------------------------------------------------------------------
# Feature 4: Automatic Batch Rename
# ---------------------------------------------------------------------------


class TestBatchRenamer:
    def test_execute_returns_rename_result(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        from src.features.implementations import BatchRenamer

        feature = BatchRenamer(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_results)

        assert isinstance(result.data, RenameResult)
        assert result.data.is_dry_run is True

    def test_feature_type_is_batch(self, mock_client, all_enabled_gate):
        from src.features.implementations import BatchRenamer

        feature = BatchRenamer(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "batch"


# ---------------------------------------------------------------------------
# Feature 5: Production Notes
# ---------------------------------------------------------------------------


class TestProductionNotesGenerator:
    def test_execute_returns_production_notes(
        self, mock_client, all_enabled_gate, mock_analysis_result
    ):
        from src.features.implementations import ProductionNotesGenerator

        feature = ProductionNotesGenerator(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_result)

        assert isinstance(result.data, ProductionNotesResult)
        assert result.data.sample_hash == "abc123hash"

    def test_feature_type_is_single(self, mock_client, all_enabled_gate):
        from src.features.implementations import ProductionNotesGenerator

        feature = ProductionNotesGenerator(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "single"

    def test_rejects_list_input(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        from src.features.implementations import ProductionNotesGenerator

        feature = ProductionNotesGenerator(client=mock_client, gate=all_enabled_gate)
        with pytest.raises(ValueError, match="single"):
            feature.execute(mock_analysis_results)


# ---------------------------------------------------------------------------
# Feature 6: Spoken Content Deep Analyzer
# ---------------------------------------------------------------------------


class TestSpokenContentAnalyzer:
    def test_execute_returns_spoken_content(
        self, mock_client, all_enabled_gate, mock_analysis_result
    ):
        from src.features.implementations import SpokenContentAnalyzer

        feature = SpokenContentAnalyzer(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_result)

        assert isinstance(result.data, SpokenContentResult)
        assert result.data.sentiment == "neutral"

    def test_feature_type_is_single(self, mock_client, all_enabled_gate):
        from src.features.implementations import SpokenContentAnalyzer

        feature = SpokenContentAnalyzer(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "single"


# ---------------------------------------------------------------------------
# Feature 7: Similar Sample Finder
# ---------------------------------------------------------------------------


class TestSimilarSampleFinder:
    def test_execute_single_returns_similarity(
        self, mock_client, all_enabled_gate, mock_analysis_result
    ):
        from src.features.implementations import SimilarSampleFinder

        feature = SimilarSampleFinder(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_result, reference_hash="ref123")

        assert isinstance(result.data, SimilarityResult)
        assert result.data.reference_hash == "ref123"

    def test_execute_batch_returns_similarity(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        from src.features.implementations import SimilarSampleFinder

        feature = SimilarSampleFinder(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_results, reference_hash="ref456")

        assert isinstance(result.data, SimilarityResult)

    def test_feature_type_is_single_plus_batch(self, mock_client, all_enabled_gate):
        from src.features.implementations import SimilarSampleFinder

        feature = SimilarSampleFinder(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "single+batch"


# ---------------------------------------------------------------------------
# Feature 8: Sample Chain
# ---------------------------------------------------------------------------


class TestSampleChainSuggester:
    def test_execute_returns_chain_result(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        from src.features.implementations import SampleChainSuggester

        feature = SampleChainSuggester(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_results)

        assert isinstance(result.data, ChainResult)

    def test_feature_type_is_batch(self, mock_client, all_enabled_gate):
        from src.features.implementations import SampleChainSuggester

        feature = SampleChainSuggester(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "batch"


# ---------------------------------------------------------------------------
# Feature 9: Marketplace Description Generator
# ---------------------------------------------------------------------------


class TestMarketplaceDescriptionGen:
    def test_execute_returns_marketplace_result(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        from src.features.implementations import MarketplaceDescriptionGen

        feature = MarketplaceDescriptionGen(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_results, pack_name="Test Pack")

        assert isinstance(result.data, MarketplaceResult)
        assert result.data.pack_name == "Test Pack"

    def test_feature_type_is_batch(self, mock_client, all_enabled_gate):
        from src.features.implementations import MarketplaceDescriptionGen

        feature = MarketplaceDescriptionGen(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "batch"


# ---------------------------------------------------------------------------
# Feature 10: Anomaly Reporter
# ---------------------------------------------------------------------------


class TestAnomalyReporter:
    def test_execute_returns_anomaly_report(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        from src.features.implementations import AnomalyReporter

        feature = AnomalyReporter(client=mock_client, gate=all_enabled_gate)
        result = feature.execute(mock_analysis_results)

        assert isinstance(result.data, AnomalyReport)
        assert result.data.overall_quality_score == 1.0

    def test_feature_type_is_batch(self, mock_client, all_enabled_gate):
        from src.features.implementations import AnomalyReporter

        feature = AnomalyReporter(client=mock_client, gate=all_enabled_gate)
        assert feature.feature_type == "batch"
