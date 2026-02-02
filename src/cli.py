"""
SampleCharm - Audio Sample Analysis CLI

This module provides the command-line interface for the Audio Sample Analysis Application.
It can be invoked as 'SampleCharm' from anywhere after installation.

Example usage:
    # Single file analysis
    SampleCharm path/to/audio.wav
    SampleCharm --output results.json path/to/audio.wav

    # Batch processing
    SampleCharm --batch path/to/directory/
    SampleCharm --batch --recursive path/to/directory/
    SampleCharm --batch file1.wav file2.wav file3.wav
    SampleCharm --batch --output-file results.txt path/to/directory/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.core.engine import create_analysis_engine
from src.core.models import AnalysisResult
from src.features.models import FeatureResult
from src.utils.config import load_config
from src.utils.logging import setup_logging


def print_single_result(file_path: Path, result: AnalysisResult) -> None:
    """Print analysis results for a single file to console."""
    print("\n" + "=" * 60)
    print("SAMPLECHARM ANALYSIS RESULTS")
    print("=" * 60)
    print(f"File: {file_path.name}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print(f"Quality: {result.quality_metadata.get('quality_tier', 'Unknown')}")
    print("-" * 60)
    print(result.get_summary())
    print("-" * 60)

    if result.source_classification:
        print(f"\nSource Classification:")
        print(f"  Type: {result.source_classification.source_type}")
        print(f"  Confidence: {result.source_classification.confidence:.2%}")
        if result.source_classification.characteristics:
            print(f"  Characteristics: {', '.join(result.source_classification.characteristics)}")

    if result.musical_analysis:
        print(f"\nMusical Analysis:")
        print(f"  Has Pitch: {result.musical_analysis.has_pitch}")
        if result.musical_analysis.has_pitch:
            print(f"  Note: {result.musical_analysis.note_name}")
            print(f"  Frequency: {result.musical_analysis.fundamental_frequency:.1f} Hz")
        if result.musical_analysis.estimated_key:
            print(f"  Key: {result.musical_analysis.estimated_key}")

    if result.percussive_analysis:
        print(f"\nPercussive Analysis:")
        print(f"  Drum Type: {result.percussive_analysis.drum_type}")
        print(f"  Confidence: {result.percussive_analysis.confidence:.2%}")
        print(f"  Synthesized: {result.percussive_analysis.is_synthesized}")

    if result.rhythmic_analysis:
        print(f"\nRhythmic Analysis:")
        print(f"  One-Shot: {result.rhythmic_analysis.is_one_shot}")
        if result.rhythmic_analysis.has_rhythm:
            print(f"  Tempo: {result.rhythmic_analysis.tempo_bpm:.1f} BPM")
            print(f"  Beats: {result.rhythmic_analysis.num_beats:.1f}")

    if result.llm_analysis:
        print(f"\nLLM Analysis:")
        print(f"  Suggested Name: {result.llm_analysis.suggested_name}")
        print(f"  Description: {result.llm_analysis.description}")
        if result.llm_analysis.contains_speech:
            print(f"  Contains Speech: Yes")
            if result.llm_analysis.transcription:
                print(f"  Transcription: {result.llm_analysis.transcription}")
            if result.llm_analysis.detected_words:
                print(f"  Detected Words: {', '.join(result.llm_analysis.detected_words)}")
            if result.llm_analysis.speech_language:
                print(f"  Language: {result.llm_analysis.speech_language}")
            if result.llm_analysis.speech_confidence is not None:
                print(f"  Speech Confidence: {result.llm_analysis.speech_confidence:.2%}")
        else:
            print(f"  Contains Speech: No")
        if result.llm_analysis.tags:
            print(f"  Tags: {', '.join(result.llm_analysis.tags)}")
        print(f"  Model: {result.llm_analysis.model_used}")


def print_feature_result(feature_id: str, result: FeatureResult) -> None:
    """Print a FeatureResult to the console in a readable format."""
    print("\n" + "=" * 60)
    print(f"FEATURE RESULT: {feature_id}")
    print("=" * 60)
    print(f"  Model: {result.model_used}")
    print(f"  Processing Time: {result.processing_time:.3f}s")
    print(f"  Timestamp: {result.timestamp.isoformat()}")
    print("-" * 60)

    data = result.data
    if hasattr(data, "__dataclass_fields__"):
        from dataclasses import fields as dc_fields
        for f in dc_fields(data):
            value = getattr(data, f.name)
            if isinstance(value, list) and len(value) > 5:
                print(f"  {f.name}: [{len(value)} items]")
                for item in value[:5]:
                    print(f"    - {item}")
                print(f"    ... and {len(value) - 5} more")
            else:
                print(f"  {f.name}: {value}")
    else:
        print(f"  {data}")
    print("-" * 60)


def list_features(config: dict) -> None:
    """List all available LLM features and their status."""
    engine = create_analysis_engine(config)
    try:
        if engine.feature_manager is None:
            print("No LLM features configured.")
            print("Add an 'llm_features' section to your config to enable features.")
            return

        features = engine.feature_manager.list_features()
        if not features:
            print("No features registered.")
            return

        print("\n" + "=" * 70)
        print("AVAILABLE LLM FEATURES")
        print("=" * 70)
        print(f"{'ID':<20} {'Name':<25} {'Type':<12} {'Status':<10}")
        print("-" * 70)
        for feat in features:
            status = "enabled" if feat.get("enabled") else "disabled"
            print(f"{feat['id']:<20} {feat['name']:<25} {feat['type']:<12} {status:<10}")
        print("-" * 70)
        print(f"{len(features)} feature(s) total")
    finally:
        engine.shutdown()


def analyze_single_file(
    audio_file: Path,
    config: dict,
    output_json: Optional[Path] = None,
    output_txt: Optional[Path] = None,
    verbose: bool = False,
    feature_id: Optional[str] = None,
    estimate_only: bool = False
) -> int:
    """
    Analyze a single audio file.

    Args:
        audio_file: Path to audio file
        config: Configuration dictionary
        output_json: Optional path for JSON output
        output_txt: Optional path for text output
        verbose: Enable verbose error output
        feature_id: Optional LLM feature to run after analysis
        estimate_only: If True, show cost estimate without running feature

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        return 1

    print(f"Analyzing: {audio_file}")
    engine = create_analysis_engine(config)

    try:
        result = engine.analyze(audio_file)

        # Print to console
        print_single_result(audio_file, result)

        # Save JSON output if requested
        if output_json:
            with open(output_json, 'w') as f:
                f.write(result.to_json(indent=2))
            print(f"\nJSON results saved to: {output_json}")

        # Save text output if requested
        if output_txt:
            from src.core.result_writer import TextResultWriter
            writer = TextResultWriter()
            writer.write({audio_file: result}, output_txt)
            print(f"Text results saved to: {output_txt}")

        # Run LLM feature if requested
        if feature_id:
            if engine.feature_manager is None:
                print("\nError: No LLM features configured. Add 'llm_features' to config.")
                return 1

            if estimate_only:
                estimate = engine.feature_manager.estimate(feature_id, result)
                print(f"\n--- Cost Estimate for '{feature_id}' ---")
                print(f"  Estimated Tokens: {estimate.estimated_tokens}")
                print(f"  Estimated Cost: ${estimate.estimated_cost_usd:.4f}")
                print(f"  Estimated Time: {estimate.estimated_time_seconds:.1f}s")
                print(f"  Sample Count: {estimate.sample_count}")
                if estimate.warning:
                    print(f"  Warning: {estimate.warning}")
                return 0

            feature_result = engine.run_feature(feature_id, result)
            print_feature_result(feature_id, feature_result)

        return 0

    except Exception as e:
        print(f"Error during analysis: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        engine.shutdown()


def analyze_batch(
    inputs: List[Path],
    config: dict,
    recursive: bool = False,
    output_txt: Optional[Path] = None,
    output_json: Optional[Path] = None,
    verbose: bool = False,
    feature_id: Optional[str] = None,
    estimate_only: bool = False
) -> int:
    """
    Analyze multiple audio files in batch mode.

    Args:
        inputs: List of paths (files or directories)
        config: Configuration dictionary
        recursive: Search directories recursively
        output_txt: Optional path for text output
        output_json: Optional path for JSON output
        verbose: Enable verbose error output
        feature_id: Optional LLM feature to run after analysis
        estimate_only: If True, show cost estimate without running feature

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    from src.core.batch_processor import BatchProcessor
    from src.core.result_writer import TextResultWriter, JSONResultWriter

    engine = create_analysis_engine(config)

    def progress_callback(current: int, total: int, file_path: Path) -> None:
        """Print progress updates."""
        print(f"[{current}/{total}] Processing: {file_path.name}")

    try:
        # Create batch processor
        processor = BatchProcessor(
            engine=engine,
            progress_callback=progress_callback
        )

        # Process files
        batch_result = processor.process(inputs, recursive=recursive)

        # Print summary
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total Files: {batch_result.total_files}")
        print(f"Successful: {batch_result.success_count}")
        print(f"Failed: {batch_result.failure_count}")
        print(f"Success Rate: {batch_result.success_rate:.1f}%")
        print(f"Total Time: {batch_result.total_time:.2f}s")

        if batch_result.failed:
            print("\nFailed Files:")
            for path, error in batch_result.failed.items():
                print(f"  {path.name}: {error}")

        # Write text output (default if no output specified)
        if output_txt or (not output_json and batch_result.successful):
            txt_path = output_txt or Path(f"samplecharm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            writer = TextResultWriter()
            writer.write(batch_result.successful, txt_path)
            print(f"\nText results saved to: {txt_path}")

        # Write JSON output if requested
        if output_json:
            writer = JSONResultWriter()
            writer.write(batch_result.successful, output_json)
            print(f"JSON results saved to: {output_json}")

        # Run LLM feature if requested
        if feature_id and batch_result.successful:
            if engine.feature_manager is None:
                print("\nError: No LLM features configured. Add 'llm_features' to config.")
                return 1

            results_list = list(batch_result.successful.values())

            if estimate_only:
                estimate = engine.feature_manager.estimate(feature_id, results_list)
                print(f"\n--- Cost Estimate for '{feature_id}' ---")
                print(f"  Estimated Tokens: {estimate.estimated_tokens}")
                print(f"  Estimated Cost: ${estimate.estimated_cost_usd:.4f}")
                print(f"  Estimated Time: {estimate.estimated_time_seconds:.1f}s")
                print(f"  Sample Count: {estimate.sample_count}")
                if estimate.warning:
                    print(f"  Warning: {estimate.warning}")
                return 0

            # Single-type features can't accept a list — run per result
            feature_obj = engine.feature_manager.get_feature(feature_id)
            if feature_obj and feature_obj.feature_type == "single":
                for i, single_result in enumerate(results_list):
                    feature_result = engine.run_feature(feature_id, single_result)
                    print(f"\n--- Feature '{feature_id}' result for file {i+1}/{len(results_list)} ---")
                    print_feature_result(feature_id, feature_result)
            else:
                feature_result = engine.run_feature(feature_id, results_list)
                print_feature_result(feature_id, feature_result)

        return 0 if batch_result.failure_count == 0 else 1

    except Exception as e:
        print(f"Error during batch processing: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        engine.shutdown()


def launch_gui():
    """Launch the GUI application."""
    try:
        import tkinter as tk
    except ImportError:
        print("Error: tkinter is not available.")
        print("Please install tkinter to use the GUI.")
        print("On Linux: sudo apt-get install python3-tk")
        sys.exit(1)
    
    try:
        # Import GUI from project root
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from gui import SampleCharmGUI
        root = tk.Tk()
        app = SampleCharmGUI(root)
        root.mainloop()
    except ImportError as e:
        print(f"Error: Failed to import GUI module: {e}")
        print("Make sure gui.py is in the project root directory.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for SampleCharm audio analysis."""
    parser = argparse.ArgumentParser(
        prog="SampleCharm",
        description="Analyze audio files using hybrid ML approach with LLM-powered naming and description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  GUI mode:
    SampleCharm gui

  List LLM features:
    SampleCharm features

  Single file:
    SampleCharm audio.wav
    SampleCharm --output results.json audio.wav
    SampleCharm --output-file results.txt audio.wav

  Batch processing:
    SampleCharm --batch samples/
    SampleCharm --batch --recursive samples/
    SampleCharm --batch file1.wav file2.wav file3.wav
    SampleCharm --batch --output-file results.txt samples/

  LLM features:
    SampleCharm --feature production_notes audio.wav
    SampleCharm --feature production_notes --estimate-only audio.wav
    SampleCharm --batch --feature pack_curator samples/
        """
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="SampleCharm 1.1.0"
    )

    # Check if first argument is "features"
    if len(sys.argv) > 1 and sys.argv[1] == "features":
        args = parser.parse_args(sys.argv[2:] if len(sys.argv) > 2 else [])
        config_path = str(args.config) if args.config else None
        config = load_config(config_path)
        list_features(config)
        return

    # Check if first argument is "gui"
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        # Launch GUI mode
        launch_gui()
        return

    # Otherwise, use original CLI mode
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Audio file(s) or directory to analyze"
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Enable batch processing mode for multiple files or directories"
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search directories recursively (only with --batch)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON output (single file mode)"
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=None,
        help="Path to save text results file (.txt)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to save JSON results file (batch mode)"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Run a specific LLM feature after analysis (e.g. 'production_notes')"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Show cost estimate for the feature without running it (requires --feature)"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = str(args.config) if args.config else None
    config = load_config(config_path)

    # Setup logging
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    setup_logging(
        level=log_level,
        log_format="text",
        colored=True,
        console_enabled=True
    )

    # Determine mode: batch or single file
    is_batch = args.batch or len(args.inputs) > 1 or args.inputs[0].is_dir()

    if is_batch:
        # Batch processing mode
        exit_code = analyze_batch(
            inputs=args.inputs,
            config=config,
            recursive=args.recursive,
            output_txt=args.output_file,
            output_json=args.output_json,
            verbose=args.verbose,
            feature_id=args.feature,
            estimate_only=args.estimate_only
        )
    else:
        # Single file mode
        exit_code = analyze_single_file(
            audio_file=args.inputs[0],
            config=config,
            output_json=args.output,
            output_txt=args.output_file,
            verbose=args.verbose,
            feature_id=args.feature,
            estimate_only=args.estimate_only
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
