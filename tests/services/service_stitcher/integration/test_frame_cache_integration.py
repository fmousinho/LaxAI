"""
Integration tests for frame cache with real GCS video.
"""
import pytest
import time
from services.service_stitcher.src.video_manager import VideoManager


@pytest.mark.integration
class TestFrameCacheIntegration:
    """Integration tests for frame cache with real video from GCS."""
    
    @pytest.fixture
    def video_manager(self):
        """Create a VideoManager with a real video."""
        manager = VideoManager(tenant_id="test_tenant")
        
        # Load a real video from GCS
        # Using the standard test video path
        result = manager.load_video(
            "test_tenant/process/test_unit_test_video_service/imported/test_video.mp4"
        )
        
        yield manager
        
        # Cleanup
        if manager.cap:
            manager.cap.__exit__(None, None, None)
    
    def test_cache_basic_functionality(self, video_manager):
        """Test that cache works with real video frames."""
        # Get initial cache stats
        initial_stats = video_manager.frame_cache.get_stats()
        assert initial_stats["total_requests"] == 0
        assert initial_stats["cache_size"] == 0
        
        # Request first frame - should be cache miss
        start_time = time.time()
        image_bytes_1 = video_manager.get_raw_frame_image(0, format="jpeg", quality=85)
        first_request_time = time.time() - start_time
        
        assert len(image_bytes_1) > 0
        assert image_bytes_1.startswith(b'\xff\xd8')  # JPEG magic bytes
        
        stats = video_manager.frame_cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0
        assert stats["cache_size"] >= 1
        
        print(f"\nFirst request (cache miss): {first_request_time:.3f}s")
        print(f"Image size: {len(image_bytes_1) / 1024 / 1024:.2f} MB")
        
        # Request same frame again - should be cache hit
        start_time = time.time()
        image_bytes_2 = video_manager.get_raw_frame_image(0, format="jpeg", quality=85)
        second_request_time = time.time() - start_time
        
        assert image_bytes_1 == image_bytes_2
        
        stats = video_manager.frame_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate_percent"] == 50.0
        
        print(f"Second request (cache hit): {second_request_time:.3f}s")
        print(f"Speedup: {first_request_time / second_request_time:.1f}x faster")
        
        # Cache hit should be significantly faster
        assert second_request_time < first_request_time / 5, "Cache hit should be at least 5x faster"
    
    def test_prefetch_effectiveness(self, video_manager):
        """Test that prefetching improves navigation performance."""
        # Request frame 0 (will trigger prefetch of frames 1-5)
        frame_0 = video_manager.get_raw_frame_image(0, format="jpeg", quality=85)
        assert len(frame_0) > 0
        
        # Wait for prefetch to complete
        time.sleep(2.0)  # Give time for background threads to encode
        
        stats = video_manager.frame_cache.get_stats()
        print(f"\nAfter frame 0 request:")
        print(f"  Cache size: {stats['cache_size']}")
        print(f"  Prefetch hits: {stats['prefetch_hits']}")
        print(f"  Pending prefetches: {stats['pending_prefetches']}")
        
        # Request frame 1 - should be fast (from prefetch)
        start_time = time.time()
        frame_1 = video_manager.get_raw_frame_image(1, format="jpeg", quality=85)
        frame_1_time = time.time() - start_time
        
        assert len(frame_1) > 0
        
        stats_after = video_manager.frame_cache.get_stats()
        print(f"\nFrame 1 request time: {frame_1_time:.3f}s")
        print(f"Frame 1 was cache hit: {stats_after['hits'] > stats['hits']}")
        
        # Frame 1 should be very fast (< 100ms) since it was prefetched
        assert frame_1_time < 0.1, f"Prefetched frame should be instant, took {frame_1_time:.3f}s"
    
    def test_sequential_navigation_performance(self, video_manager):
        """Test performance of sequential forward navigation."""
        times = []
        
        # Navigate through first 10 frames
        for frame_id in range(10):
            start = time.time()
            image = video_manager.get_raw_frame_image(frame_id, format="jpeg", quality=85)
            duration = time.time() - start
            times.append(duration)
            
            assert len(image) > 0
            
            # Give prefetch a moment to work (except on last frame)
            if frame_id < 9:
                time.sleep(0.5)
        
        stats = video_manager.frame_cache.get_stats()
        
        print(f"\nSequential navigation (10 frames):")
        print(f"  Average time: {sum(times) / len(times):.3f}s")
        print(f"  First frame: {times[0]:.3f}s (cold)")
        print(f"  Frame 2-10 avg: {sum(times[1:]) / len(times[1:]):.3f}s (with prefetch)")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Cache hits: {stats['hits']}")
        print(f"  Hit rate: {stats['hit_rate_percent']:.1f}%")
        print(f"  Prefetch hits: {stats['prefetch_hits']}")
        
        # With prefetching, later frames should be much faster
        avg_prefetch_time = sum(times[2:]) / len(times[2:])  # Skip first 2
        assert avg_prefetch_time < 0.2, f"Prefetched frames should be fast, avg was {avg_prefetch_time:.3f}s"
        
        # Should have good hit rate
        assert stats["hit_rate_percent"] > 50, "Should have >50% hit rate with prefetch"
    
    def test_backward_navigation(self, video_manager):
        """Test that backward navigation uses cached frames."""
        # Navigate forward to frame 5
        for frame_id in range(6):
            video_manager.get_raw_frame_image(frame_id, format="jpeg", quality=85)
            time.sleep(0.3)  # Give prefetch time
        
        stats_before = video_manager.frame_cache.get_stats()
        
        # Now go back to frame 3 - should be in cache
        start = time.time()
        frame_3 = video_manager.get_raw_frame_image(3, format="jpeg", quality=85)
        backward_time = time.time() - start
        
        assert len(frame_3) > 0
        
        stats_after = video_manager.frame_cache.get_stats()
        
        print(f"\nBackward navigation to frame 3:")
        print(f"  Time: {backward_time:.3f}s")
        print(f"  Was cache hit: {stats_after['hits'] > stats_before['hits']}")
        print(f"  Cache size: {stats_after['cache_size']}")
        
        # Should be instant (cache hit)
        assert backward_time < 0.1, f"Backward navigation should be instant, took {backward_time:.3f}s"
        assert stats_after['hits'] > stats_before['hits'], "Should be cache hit"
    
    def test_cache_stats_endpoint_integration(self, video_manager):
        """Test that cache stats are accurate and useful."""
        # Do some operations
        video_manager.get_raw_frame_image(0, format="jpeg", quality=85)
        time.sleep(1.0)  # Let prefetch work
        video_manager.get_raw_frame_image(1, format="jpeg", quality=85)
        video_manager.get_raw_frame_image(1, format="jpeg", quality=85)  # Duplicate
        
        stats = video_manager.frame_cache.get_stats()
        
        print(f"\nCache statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Verify stats make sense
        assert stats["total_requests"] == 3
        assert stats["misses"] >= 1
        assert stats["hits"] >= 1
        assert stats["cache_size"] > 0
        assert 0 <= stats["hit_rate_percent"] <= 100
