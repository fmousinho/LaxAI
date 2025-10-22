"""
Rolling frame cache with prefetching for video frame navigation.

This module implements a smart cache that:
- Keeps the last 5 frames in memory (for backward navigation)
- Prefetches the next 5 frames in background (for forward navigation)
- Uses JPEG encoding by default for speed
- Thread-safe with minimal locking
"""
import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class RollingFrameCache:
    """LRU cache with intelligent prefetching for video frames.
    
    Features:
    - Rolling window: keeps previous N and prefetches next N frames
    - Thread-safe operations with minimal locking
    - Background prefetching to avoid blocking main thread
    - Automatic eviction of old frames (LRU policy)
    - Memory-bounded with configurable limits
    
    Attributes:
        window_size (int): Number of frames to keep behind and prefetch ahead
        max_cache_size (int): Maximum total frames to keep in memory
    """
    
    def __init__(
        self,
        window_size: int = 5,
        max_workers: int = 2,
        format: str = "jpeg",
        quality: int = 85,
        frame_skip_interval: int = 1
    ):
        """Initialize rolling frame cache.
        
        Args:
            window_size: Number of frames to cache before/after current frame
            max_workers: Number of background threads for prefetching
            format: Default image format ('jpeg' or 'png')
            quality: Default JPEG quality (1-100)
            frame_skip_interval: Interval between frames to cache (for skipping frames)
        """
        self.window_size = window_size
        self.max_cache_size = window_size * 2  # Keep window_size before + window_size after
        self.format = format
        self.quality = quality
        
        # Thread-safe cache: frame_id -> (image_bytes, timestamp)
        self._cache: OrderedDict[int, Tuple[bytes, float]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Prefetch management
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="frame-prefetch")
        self._prefetch_futures = {}  # frame_id -> Future
        self.frame_skip_interval = frame_skip_interval
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "prefetch_hits": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def get(
        self,
        frame_id: int,
        encode_func: Callable[[int, str, int], bytes],
        total_frames: int,
        trigger_prefetch: bool = True
    ) -> bytes:
        """Get frame from cache or encode it, then trigger prefetch.
        
        Args:
            frame_id: Frame index to retrieve
            encode_func: Function(frame_id, format, quality) -> bytes to encode frame
            total_frames: Total number of frames in video (for boundary checks)
            trigger_prefetch: Whether to trigger prefetching of next frames
            
        Returns:
            bytes: Encoded image data
        """
        start_time = time.time()
        
        with self._lock:
            self._stats["total_requests"] += 1
            
            # Check cache first
            if frame_id in self._cache:
                self._stats["hits"] += 1
                image_bytes, _ = self._cache[frame_id]
                # Move to end (mark as recently used)
                self._cache.move_to_end(frame_id)
                
                logger.debug(f"Cache HIT for frame {frame_id} ({time.time() - start_time:.3f}s)")
                
                # Trigger prefetch for next window
                if trigger_prefetch:
                    self._trigger_prefetch(frame_id, encode_func, total_frames)
                
                return image_bytes
            else:
                self._stats["misses"] += 1
        
        # Cache miss - encode synchronously
        logger.debug(f"Cache MISS for frame {frame_id}, encoding...")
        image_bytes = encode_func(frame_id, self.format, self.quality)
        encode_time = time.time() - start_time
        
        # Store in cache
        with self._lock:
            self._cache[frame_id] = (image_bytes, time.time())
            self._cache.move_to_end(frame_id)
            self._evict_if_needed()
        
        logger.debug(f"Encoded frame {frame_id} in {encode_time:.3f}s")
        
        # Trigger prefetch for next window
        if trigger_prefetch:
            self._trigger_prefetch(frame_id, encode_func, total_frames)
        
        return image_bytes
    
    def _trigger_prefetch(
        self,
        current_frame: int,
        encode_func: Callable[[int, str, int], bytes],
        total_frames: int
    ):
        """Trigger background prefetching for next N frames.
        
        Prefetches frames [current+1, current+window_size] in background.
        Frames fetched in self.frame_skip_interval steps.
        Skips frames already cached or being prefetched.
        """
        frames_to_prefetch = []
        
        with self._lock:
            for offset in range(1, self.window_size + 1):
                next_frame = current_frame + offset * self.frame_skip_interval
                
                # Stop at video boundary
                if next_frame >= total_frames:
                    break
                
                # Skip if already cached or being prefetched
                if next_frame in self._cache or next_frame in self._prefetch_futures:
                    continue
                
                frames_to_prefetch.append(next_frame)
        
        # Submit prefetch jobs (outside lock to avoid blocking)
        for frame_id in frames_to_prefetch:
            future = self._executor.submit(
                self._prefetch_frame,
                frame_id,
                encode_func
            )
            self._prefetch_futures[frame_id] = future
            logger.debug(f"Prefetching frame {frame_id}")
    
    def _prefetch_frame(
        self,
        frame_id: int,
        encode_func: Callable[[int, str, int], bytes]
    ):
        """Background worker to prefetch and cache a frame.
        
        Args:
            frame_id: Frame index to prefetch
            encode_func: Function to encode frame
        """
        try:
            start_time = time.time()
            
            # Check if already cached (race condition)
            with self._lock:
                if frame_id in self._cache:
                    return
            
            # Encode frame
            image_bytes = encode_func(frame_id, self.format, self.quality)
            
            # Store in cache
            with self._lock:
                # Double-check not cached (another thread might have added it)
                if frame_id not in self._cache:
                    self._cache[frame_id] = (image_bytes, time.time())
                    self._cache.move_to_end(frame_id)
                    self._evict_if_needed()
                    self._stats["prefetch_hits"] += 1
            
            logger.debug(f"Prefetched frame {frame_id} in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error prefetching frame {frame_id}: {e}")
        
        finally:
            # Remove from pending futures
            with self._lock:
                self._prefetch_futures.pop(frame_id, None)
    
    def _evict_if_needed(self):
        """Evict oldest frames if cache exceeds max size (LRU policy).
        
        Must be called with lock held.
        """
        while len(self._cache) > self.max_cache_size:
            # Remove oldest (first in OrderedDict)
            evicted_frame, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug(f"Evicted frame {evicted_frame} (cache size: {len(self._cache)})")
    
    def invalidate(self, frame_id: Optional[int] = None):
        """Invalidate cache entries.
        
        Args:
            frame_id: Specific frame to invalidate, or None to clear all
        """
        with self._lock:
            if frame_id is None:
                self._cache.clear()
                logger.info("Cache cleared")
            elif frame_id in self._cache:
                del self._cache[frame_id]
                logger.debug(f"Invalidated frame {frame_id}")
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            dict: Cache performance metrics
        """
        with self._lock:
            total = self._stats["total_requests"]
            hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
            
            return {
                "total_requests": total,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "prefetch_hits": self._stats["prefetch_hits"],
                "hit_rate_percent": round(hit_rate, 2),
                "cache_size": len(self._cache),
                "evictions": self._stats["evictions"],
                "pending_prefetches": len(self._prefetch_futures)
            }
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        with self._lock:
            self._cache.clear()
            self._executor.shutdown(wait=False)
            logger.info("Frame cache shut down")
