"""
Cache manager for the Audio Sample Analysis Application.

Provides in-memory LRU caching for analysis results.
"""

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from src.core.models import AnalysisResult


class CacheManager:
    """
    Thread-safe in-memory LRU cache for analysis results.

    Features:
    - LRU (Least Recently Used) eviction
    - Time-to-live (TTL) expiration
    - Thread-safe operations
    - Size-limited storage
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600
    ):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of cached items
            ttl: Time to live in seconds (default 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.RLock()
        self.logger = logging.getLogger("cache")

        # Statistics
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[AnalysisResult]:
        """
        Get cached result by key.

        Args:
            key: Cache key (typically file hash)

        Returns:
            AnalysisResult if found and not expired, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl:
                # Expired - remove and return None
                del self._cache[key]
                self._misses += 1
                self.logger.debug(f"Cache expired: {key[:8]}...")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            self.logger.debug(f"Cache hit: {key[:8]}...")

            return value

    def set(self, key: str, value: AnalysisResult) -> None:
        """
        Store result in cache.

        Args:
            key: Cache key (typically file hash)
            value: AnalysisResult to cache
        """
        with self._lock:
            # Remove if exists (to update timestamp)
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self.logger.debug(f"Evicted: {oldest_key[:8]}...")

            # Store with current timestamp
            self._cache[key] = (value, time.time())
            self.logger.debug(f"Cached: {key[:8]}...")

    def delete(self, key: str) -> bool:
        """
        Remove item from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self.logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, size, and hit ratio
        """
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = self._hits / total if total > 0 else 0.0

            return {
                'hits': self._hits,
                'misses': self._misses,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_ratio': hit_ratio,
                'ttl': self.ttl
            }

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        removed = 0

        with self._lock:
            keys_to_remove = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self.ttl
            ]

            for key in keys_to_remove:
                del self._cache[key]
                removed += 1

        if removed > 0:
            self.logger.info(f"Cleaned up {removed} expired entries")

        return removed

    def __len__(self) -> int:
        """Return number of cached items."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache (doesn't update LRU order)."""
        with self._lock:
            if key not in self._cache:
                return False

            _, timestamp = self._cache[key]
            return time.time() - timestamp <= self.ttl


def create_cache_manager(config: Optional[Dict[str, Any]] = None) -> CacheManager:
    """
    Factory function to create CacheManager with configuration.

    Args:
        config: Optional configuration dict

    Returns:
        CacheManager: Configured cache instance
    """
    if config is None:
        config = {}

    return CacheManager(
        max_size=config.get('max_size', 1000),
        ttl=config.get('ttl', 3600)
    )
