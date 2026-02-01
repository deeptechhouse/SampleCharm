"""
Queue manager for handling long audio files.

Manages a queue of files that exceed the duration limit, allowing users
to process them later or skip them.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Deque, Optional

logger = logging.getLogger(__name__)


class LongFileQueue:
    """
    Queue manager for long audio files.
    
    Files that exceed max_duration are queued for later processing.
    Users can choose to process immediately, queue, or skip.
    """

    def __init__(self):
        """Initialize empty queue."""
        self.queue: Deque[Path] = deque()
        self.logger = logging.getLogger('queue')

    def add(self, file_path: Path) -> None:
        """
        Add file to end of queue.
        
        Args:
            file_path: Path to audio file
        """
        self.queue.append(file_path)
        self.logger.info(f"Added to queue: {file_path} (queue size: {len(self.queue)})")

    def get_next(self) -> Optional[Path]:
        """
        Get next file from queue (FIFO).
        
        Returns:
            Path to next file, or None if queue is empty
        """
        if self.queue:
            file_path = self.queue.popleft()
            self.logger.info(f"Removed from queue: {file_path} (queue size: {len(self.queue)})")
            return file_path
        return None

    def remove(self, file_path: Path) -> bool:
        """
        Remove specific file from queue.
        
        Args:
            file_path: Path to file to remove
            
        Returns:
            True if file was found and removed, False otherwise
        """
        try:
            self.queue.remove(file_path)
            self.logger.info(f"Removed from queue: {file_path} (queue size: {len(self.queue)})")
            return True
        except ValueError:
            return False

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0

    def size(self) -> int:
        """Get current queue size."""
        return len(self.queue)

    def clear(self) -> None:
        """Clear all files from queue."""
        count = len(self.queue)
        self.queue.clear()
        self.logger.info(f"Cleared queue ({count} files removed)")

    def list_files(self) -> list[Path]:
        """Get list of all files in queue (in order)."""
        return list(self.queue)
