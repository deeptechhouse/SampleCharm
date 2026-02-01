"""
Core module containing data models, audio processing, and analysis engine.

Uses lazy imports for modules with heavy dependencies (librosa, tensorflow).
"""

# Models are lightweight - import directly
from src.core.models import (
    AudioSample,
    FeatureCache,
    SourceClassification,
    SpatialInfo,
    MusicalAnalysis,
    PercussiveAnalysis,
    RhythmicAnalysis,
    LLMAnalysis,
    AnalysisResult,
    validate_confidence,
    validate_drum_type,
)

__all__ = [
    # Models (always available)
    "AudioSample",
    "FeatureCache",
    "SourceClassification",
    "SpatialInfo",
    "MusicalAnalysis",
    "PercussiveAnalysis",
    "RhythmicAnalysis",
    "LLMAnalysis",
    "AnalysisResult",
    "validate_confidence",
    "validate_drum_type",
    # Heavy modules (lazy loaded)
    "AudioLoader",
    "create_audio_loader",
    "FeatureExtractor",
    "Analyzer",
    "BaseAnalyzer",
    "MultiStrategyAnalyzer",
    "AudioAnalysisEngine",
    "create_analysis_engine",
    "CacheManager",
    "create_cache_manager",
    # Batch processing
    "BatchProcessor",
    "BatchResult",
    "ResultWriter",
    "TextResultWriter",
    "JSONResultWriter",
    "create_result_writer",
]


def __getattr__(name: str):
    """Lazy load modules with heavy dependencies."""
    if name in ("AudioLoader", "create_audio_loader"):
        from src.core.loader import AudioLoader, create_audio_loader
        return AudioLoader if name == "AudioLoader" else create_audio_loader
    elif name == "FeatureExtractor":
        from src.core.features import FeatureExtractor
        return FeatureExtractor
    elif name in ("Analyzer", "BaseAnalyzer"):
        from src.core.analyzer_base import Analyzer, BaseAnalyzer
        return Analyzer if name == "Analyzer" else BaseAnalyzer
    elif name == "MultiStrategyAnalyzer":
        from src.core.multi_strategy import MultiStrategyAnalyzer
        return MultiStrategyAnalyzer
    elif name in ("AudioAnalysisEngine", "create_analysis_engine"):
        from src.core.engine import AudioAnalysisEngine, create_analysis_engine
        return AudioAnalysisEngine if name == "AudioAnalysisEngine" else create_analysis_engine
    elif name in ("CacheManager", "create_cache_manager"):
        from src.core.cache import CacheManager, create_cache_manager
        return CacheManager if name == "CacheManager" else create_cache_manager
    elif name in ("BatchProcessor", "BatchResult"):
        from src.core.batch_processor import BatchProcessor, BatchResult
        return BatchProcessor if name == "BatchProcessor" else BatchResult
    elif name in ("ResultWriter", "TextResultWriter", "JSONResultWriter", "create_result_writer"):
        from src.core.result_writer import ResultWriter, TextResultWriter, JSONResultWriter, create_result_writer
        if name == "ResultWriter":
            return ResultWriter
        elif name == "TextResultWriter":
            return TextResultWriter
        elif name == "JSONResultWriter":
            return JSONResultWriter
        else:
            return create_result_writer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
