"""
Individual LLM feature implementations.

Each feature extends BaseLLMFeature and provides:
- A unique feature_id for toggle/entitlement gating
- An _execute_impl() with the actual LLM prompt logic
- An estimate_cost() for pre-execution cost/time estimation
"""

from src.features.implementations.sample_pack_curator import SamplePackCurator
from src.features.implementations.natural_language_search import NaturalLanguageSearch
from src.features.implementations.daw_suggestions import DAWContextSuggester
from src.features.implementations.batch_rename import BatchRenamer
from src.features.implementations.production_notes import ProductionNotesGenerator
from src.features.implementations.speech_deep_analyzer import SpokenContentAnalyzer
from src.features.implementations.similar_sample_finder import SimilarSampleFinder
from src.features.implementations.sample_chain import SampleChainSuggester
from src.features.implementations.marketplace_description import MarketplaceDescriptionGen
from src.features.implementations.anomaly_reporter import AnomalyReporter

ALL_FEATURE_CLASSES = [
    SamplePackCurator,
    NaturalLanguageSearch,
    DAWContextSuggester,
    BatchRenamer,
    ProductionNotesGenerator,
    SpokenContentAnalyzer,
    SimilarSampleFinder,
    SampleChainSuggester,
    MarketplaceDescriptionGen,
    AnomalyReporter,
]

__all__ = [
    "SamplePackCurator",
    "NaturalLanguageSearch",
    "DAWContextSuggester",
    "BatchRenamer",
    "ProductionNotesGenerator",
    "SpokenContentAnalyzer",
    "SimilarSampleFinder",
    "SampleChainSuggester",
    "MarketplaceDescriptionGen",
    "AnomalyReporter",
    "ALL_FEATURE_CLASSES",
]
