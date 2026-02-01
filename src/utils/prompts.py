"""
Interactive prompts for user decisions.

Handles user input for long file processing decisions.
"""

import sys
from enum import Enum
from pathlib import Path
from typing import Optional


class LongFileAction(Enum):
    """Actions user can take for long files."""
    PROCESS_NOW = "process_now"
    QUEUE = "queue"
    SKIP = "skip"


def prompt_long_file_action(file_path: Path, duration: float, max_duration: float) -> LongFileAction:
    """
    Prompt user for action on long audio file.
    
    Args:
        file_path: Path to the audio file
        duration: Actual duration of the file
        max_duration: Maximum allowed duration
        
    Returns:
        LongFileAction: User's choice
    """
    print("\n" + "=" * 60)
    print("⚠️  LONG AUDIO FILE DETECTED")
    print("=" * 60)
    print(f"File: {file_path.name}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Maximum: {max_duration:.1f} seconds")
    print(f"Exceeds limit by: {duration - max_duration:.1f} seconds")
    print("-" * 60)
    print("How would you like to handle this file?")
    print()
    print("  1. Process immediately (may take longer)")
    print("  2. Add to end of queue (process after other files)")
    print("  3. Skip this file")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1/2/3): ").strip()
            
            if choice == "1":
                return LongFileAction.PROCESS_NOW
            elif choice == "2":
                return LongFileAction.QUEUE
            elif choice == "3":
                return LongFileAction.SKIP
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled. Skipping file.")
            return LongFileAction.SKIP


def prompt_queued_file_action(file_path: Path, duration: float, max_duration: float) -> LongFileAction:
    """
    Prompt user for action when queued file reaches top of queue.
    
    Args:
        file_path: Path to the audio file
        duration: Actual duration of the file
        max_duration: Maximum allowed duration
        
    Returns:
        LongFileAction: User's choice
    """
    print("\n" + "=" * 60)
    print("📋 QUEUED FILE READY FOR PROCESSING")
    print("=" * 60)
    print(f"File: {file_path.name}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Maximum: {max_duration:.1f} seconds")
    print("-" * 60)
    print("How would you like to handle this queued file?")
    print()
    print("  1. Process now")
    print("  2. Move to end of queue (process later)")
    print("  3. Remove from queue (skip)")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1/2/3): ").strip()
            
            if choice == "1":
                return LongFileAction.PROCESS_NOW
            elif choice == "2":
                return LongFileAction.QUEUE
            elif choice == "3":
                return LongFileAction.SKIP
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled. Removing from queue.")
            return LongFileAction.SKIP
