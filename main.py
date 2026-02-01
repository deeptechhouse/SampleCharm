"""
Audio Sample Analysis Application - Main Entry Point

Example usage:
    python main.py path/to/audio.wav
    python main.py --config config/config.yaml path/to/audio.wav
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Any

from src.core.engine import create_analysis_engine
from src.core.loader import create_audio_loader
from src.core.queue_manager import LongFileQueue
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.prompts import LongFileAction, prompt_long_file_action, prompt_queued_file_action


def main():
    """Main entry point for audio analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze audio files using hybrid ML approach"
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file to analyze"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON output"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
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

    # Validate input file
    if not args.audio_file.exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Create loader to check file duration
    loader = create_audio_loader(config.get('audio', {}))
    queue = LongFileQueue()
    
    # Create analysis engine
    engine = create_analysis_engine(config)

    try:
        # Process files (initial file + queue)
        files_to_process = [args.audio_file]
        
        while files_to_process or not queue.is_empty():
            # Get next file to process
            if files_to_process:
                current_file = files_to_process.pop(0)
                is_queued = False
            else:
                # Process queue
                current_file = queue.get_next()
                if current_file is None:
                    break
                is_queued = True
            
            # Check if file is long
            if loader.is_long_file(current_file):
                duration = loader.get_duration(current_file)
                max_duration = loader.max_duration
                
                if is_queued:
                    # File reached top of queue - ask again
                    action = prompt_queued_file_action(current_file, duration, max_duration)
                else:
                    # New long file detected - ask user
                    action = prompt_long_file_action(current_file, duration, max_duration)
                
                if action == LongFileAction.PROCESS_NOW:
                    # Process immediately
                    print(f"\nProcessing long file: {current_file.name}")
                    result = _process_file(engine, current_file, args.output, args.verbose)
                    if result:
                        _print_results(result, current_file)
                elif action == LongFileAction.QUEUE:
                    # Add to queue
                    queue.add(current_file)
                    print(f"\nAdded to queue: {current_file.name} (queue size: {queue.size()})")
                    continue
                elif action == LongFileAction.SKIP:
                    # Skip file
                    print(f"\nSkipping: {current_file.name}")
                    continue
            else:
                # Normal file - process immediately
                print(f"\nAnalyzing: {current_file.name}")
                result = _process_file(engine, current_file, args.output, args.verbose)
                if result:
                    _print_results(result, current_file)
        
        # Check if queue still has files
        if not queue.is_empty():
            print(f"\n⚠️  Queue still contains {queue.size()} file(s):")
            for queued_file in queue.list_files():
                print(f"  - {queued_file.name}")
            print("\nRun the program again to process queued files.")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        if not queue.is_empty():
            print(f"Queue contains {queue.size()} file(s) that were not processed.")
        sys.exit(0)

        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(f"File: {args.audio_file.name}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        print(f"Quality: {result.quality_metadata.get('quality_tier', 'Unknown')}")
        print("-" * 60)
        print(result.get_summary())
        print("-" * 60)

        # Detailed results
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

    finally:
        engine.shutdown()


def _process_file(engine, file_path: Path, output_path: Optional[Path] = None, verbose: bool = False) -> Optional[Any]:
    """
    Process a single audio file.
    
    Args:
        engine: Analysis engine
        file_path: Path to audio file
        output_path: Optional path to save JSON output
        verbose: Enable verbose error output
        
    Returns:
        AnalysisResult or None if error
    """
    try:
        result = engine.analyze(file_path)
        
        # Save output if requested
        if output_path:
            # If output path is a directory, create filename based on input
            if output_path.is_dir() or output_path.suffix == '':
                output_file = output_path / f"{file_path.stem}_analysis.json"
            else:
                output_file = output_path
            
            with open(output_file, 'w') as f:
                f.write(result.to_json(indent=2))
            print(f"Results saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"Error during analysis of {file_path.name}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def _print_results(result: Any, file_path: Path) -> None:
    """
    Print analysis results.
    
    Args:
        result: AnalysisResult
        file_path: Path to analyzed file
    """
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"File: {file_path.name}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print(f"Quality: {result.quality_metadata.get('quality_tier', 'Unknown')}")
    print("-" * 60)
    print(result.get_summary())
    print("-" * 60)

    # Detailed results
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
            if result.llm_analysis.detected_words:
                print(f"  Detected Words: {', '.join(result.llm_analysis.detected_words)}")
            if result.llm_analysis.speech_language:
                print(f"  Language: {result.llm_analysis.speech_language}")
        else:
            print(f"  Contains Speech: No")
        if result.llm_analysis.tags:
            print(f"  Tags: {', '.join(result.llm_analysis.tags)}")
        print(f"  Model: {result.llm_analysis.model_used}")


if __name__ == "__main__":
    main()
