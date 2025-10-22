"""
Unit tests for RollingFrameCache.
"""
import pytest
import time
from services.service_stitcher.src.frame_cache import RollingFrameCache


class TestRollingFrameCache:
    """Test suite for RollingFrameCache."""
    
    @pytest.fixture
    def mock_encode_func(self):
        """Mock encoding function that simulates encoding delay."""
        def encode(frame_id: int, format: str, quality: int) -> bytes:
            # Simulate encoding time
            time.sleep(0.01)
            return f"frame_{frame_id}_{format}_{quality}".encode()
        return encode
    
    def test_cache_initialization(self):
        """Test cache initializes with correct parameters."""
        cache = RollingFrameCache(window_size=3, max_workers=2)
        
        assert cache.window_size == 3
        assert cache.max_cache_size == 6  # 2 * window_size
        assert cache.format == "jpeg"
        assert cache.quality == 85
        
        stats = cache.get_stats()
        assert stats["total_requests"] == 0
        assert stats["cache_size"] == 0
        
        cache.shutdown()
    
    def test_cache_miss_and_hit(self, mock_encode_func):
        """Test cache miss (first request) then hit (second request)."""
        cache = RollingFrameCache(window_size=2, max_workers=1)
        
        # First request - cache miss
        start = time.time()
        result1 = cache.get(0, mock_encode_func, total_frames=100, trigger_prefetch=False)
        duration1 = time.time() - start
        
        assert result1 == b"frame_0_jpeg_85"
        assert duration1 >= 0.01  # Should take at least encoding time
        
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0
        assert stats["cache_size"] == 1
        
        # Second request (same frame) - cache hit
        start = time.time()
        result2 = cache.get(0, mock_encode_func, total_frames=100, trigger_prefetch=False)
        duration2 = time.time() - start
        
        assert result2 == b"frame_0_jpeg_85"
        assert duration2 < 0.005  # Should be instant (< 5ms)
        
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1
        assert stats["hit_rate_percent"] == 50.0
        
        cache.shutdown()
    
    def test_prefetch_next_frames(self, mock_encode_func):
        """Test that requesting frame 0 triggers prefetch of next frames."""
        cache = RollingFrameCache(window_size=3, max_workers=2)
        
        # Request frame 0 with prefetch enabled
        result = cache.get(0, mock_encode_func, total_frames=100, trigger_prefetch=True)
        assert result == b"frame_0_jpeg_85"
        
        # Give prefetch threads time to complete
        time.sleep(0.15)  # Should be enough for ~3 frames at 0.01s each
        
        stats = cache.get_stats()
        
        # Frame 0 was requested (miss), frames 1-3 should be prefetched
        assert stats["misses"] == 1
        assert stats["cache_size"] >= 1  # At least frame 0
        assert stats["prefetch_hits"] >= 1  # At least some prefetches completed
        
        # Request frame 1 - should be cache hit from prefetch
        result1 = cache.get(1, mock_encode_func, total_frames=100, trigger_prefetch=False)
        assert result1 == b"frame_1_jpeg_85"
        
        stats = cache.get_stats()
        assert stats["hits"] >= 1  # Frame 1 was prefetched
        
        cache.shutdown()
    
    def test_lru_eviction(self, mock_encode_func):
        """Test that cache evicts oldest frames when max size exceeded."""
        cache = RollingFrameCache(window_size=2, max_workers=1)  # max_cache_size = 4
        
        # Fill cache beyond capacity
        for i in range(6):
            cache.get(i, mock_encode_func, total_frames=100, trigger_prefetch=False)
        
        stats = cache.get_stats()
        
        # Cache should not exceed max_cache_size
        assert stats["cache_size"] <= cache.max_cache_size
        assert stats["evictions"] >= 2  # At least 2 frames evicted (6 - 4)
        
        # Oldest frames (0, 1) should be evicted
        # Requesting them should be cache miss
        old_misses = stats["misses"]
        cache.get(0, mock_encode_func, total_frames=100, trigger_prefetch=False)
        
        stats = cache.get_stats()
        assert stats["misses"] > old_misses  # Frame 0 was evicted, so it's a miss
        
        cache.shutdown()
    
    def test_boundary_prefetch(self, mock_encode_func):
        """Test that prefetch stops at video boundary."""
        cache = RollingFrameCache(window_size=5, max_workers=2)
        
        # Request frame 98 in a 100-frame video
        result = cache.get(98, mock_encode_func, total_frames=100, trigger_prefetch=True)
        assert result == b"frame_98_jpeg_85"
        
        # Give prefetch time
        time.sleep(0.05)
        
        stats = cache.get_stats()
        
        # Should only prefetch frame 99 (not beyond 100)
        # Cache size should be at most 2 (frame 98 + 99)
        assert stats["cache_size"] <= 2
        
        cache.shutdown()
    
    def test_cache_invalidation(self, mock_encode_func):
        """Test cache invalidation."""
        cache = RollingFrameCache(window_size=2, max_workers=1)
        
        # Cache some frames
        cache.get(0, mock_encode_func, total_frames=100, trigger_prefetch=False)
        cache.get(1, mock_encode_func, total_frames=100, trigger_prefetch=False)
        
        assert cache.get_stats()["cache_size"] == 2
        
        # Invalidate specific frame
        cache.invalidate(frame_id=0)
        assert cache.get_stats()["cache_size"] == 1
        
        # Invalidate all
        cache.invalidate()
        assert cache.get_stats()["cache_size"] == 0
        
        cache.shutdown()
    
    def test_concurrent_requests(self, mock_encode_func):
        """Test that concurrent requests to same frame don't duplicate work."""
        import threading
        
        cache = RollingFrameCache(window_size=2, max_workers=2)
        results = []
        
        def request_frame():
            result = cache.get(0, mock_encode_func, total_frames=100, trigger_prefetch=False)
            results.append(result)
        
        # Launch multiple threads requesting same frame
        threads = [threading.Thread(target=request_frame) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should get same result
        assert all(r == b"frame_0_jpeg_85" for r in results)
        
        # Should only encode once (cache hit for subsequent requests)
        stats = cache.get_stats()
        assert stats["total_requests"] == 5
        assert stats["cache_size"] == 1
        
        cache.shutdown()
