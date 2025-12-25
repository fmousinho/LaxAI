import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Adjust path to import modules
import sys
import os
sys.path.append(os.path.abspath("services/service_tracking/src"))
sys.path.append(os.path.abspath("."))

from schemas.tracking import TrackingParams
from tracker.byte_tracker import BYTETracker, STrack
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class MockReIdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
    
    def forward(self, x):
        # Return random feature vector of size (Batch, 128)
        return torch.randn(x.size(0), 128).to(self.device)
    
    def parameters(self):
        yield torch.tensor(0.0, device=self.device)

class TestEmbeddingFlow(unittest.TestCase):
    def setUp(self):
        self.params = TrackingParams(
            embedding_update_frequency=30,
            embedding_quality_threshold=0.8,
            embedding_min_detection_confidence=0.6,
            track_activation_threshold=0.6 # Ensure high confidence dets are kept
        )
        self.reid_model = MockReIdModel()
        self.tracker = BYTETracker(self.params, reid_model=self.reid_model)
        # Mock frame (H, W, C)
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_embedding_init_and_frequency(self):
        # Frame 1: High confidence detection
        dets = np.array([[100, 100, 200, 200, 0.9]])
        self.tracker.frame_id = 0 # Explicitly set
        # Assign assigns current frame and calls update
        output_tracks = self.tracker.assign_tracks_to_detections(dets, None, None, self.frame)
        
        # On first frame, track is typically created and put into pending
        all_tracks = self.tracker.tracked_stracks + self.tracker.pending_stracks
        self.assertTrue(len(all_tracks) > 0)
        track = all_tracks[0]
        
        # Check features initialized
        self.assertIsNotNone(track.features)
        self.assertEqual(track.features_count, 1)
        initial_features = track.features.clone()
        
        # Frame 2-29: Updates. Should NOT update features (frequency 30)
        # Note: track needs to become activated to stay in tracked_stracks
        # Force activation for test simplicity if needed, or just run sequence
        if not track.is_activated:
             track.is_activated = True
             if track in self.tracker.pending_stracks:
                 self.tracker.pending_stracks.remove(track)
                 self.tracker.tracked_stracks.append(track)

        for i in range(1, 30):
            self.tracker.assign_tracks_to_detections(dets, None, None, self.frame)
            self.assertTrue(torch.equal(track.features, initial_features), f"Features changed at frame {i}")
            
        # Frame 30: Update. Should update features.
        # Ensure no overlap
        self.tracker.assign_tracks_to_detections(dets, None, None, self.frame)
        self.assertFalse(torch.equal(track.features, initial_features), "Features did NOT update at frame 30")
        self.assertEqual(track.features_count, 2)
        
    def test_overlap_skip(self):
        # Create a track
        dets = np.array([[100, 100, 200, 200, 0.9]])
        self.tracker.assign_tracks_to_detections(dets, None, None, self.frame)
        track = self.tracker.tracked_stracks[0] if self.tracker.tracked_stracks else self.tracker.pending_stracks[0]
        # Force activate
        track.is_activated = True
        if track in self.tracker.pending_stracks:
             self.tracker.pending_stracks.remove(track)
             self.tracker.tracked_stracks.append(track)
        
        # Fast forward close to update time
        track.tracklet_len = 29
        
        # Frame 30 with overlap from another detection
        # Dets: [Track's new pos, Overlapping det]
        dets_overlap = np.array([
            [100, 100, 200, 200, 0.9], # Track match
            [110, 110, 210, 210, 0.8]  # Overlap
        ])
        
        # We need to ensure reid features logic sees both detections.
        # update_track_reid_features takes 'all_detections'.
        
        features_before = track.features.clone()
        self.tracker.assign_tracks_to_detections(dets_overlap, None, None, self.frame)
        
        # Should SKIP update due to overlap
        self.assertTrue(torch.equal(track.features, features_before), "Features updated despite detection overlap")
        
    def test_track_overlap_skip(self):
        # Initialize TWO tracks
        # Frame 0
        dets = np.array([
            [100, 100, 200, 200, 0.9], # Track 1
            [500, 500, 600, 600, 0.9]  # Track 2
        ])
        self.tracker.assign_tracks_to_detections(dets, None, None, self.frame)
        all_active = self.tracker.tracked_stracks + self.tracker.pending_stracks
        self.assertEqual(len(all_active), 2)
        
        # Force activate both
        for t in all_active:
             t.is_activated = True
             if t in self.tracker.pending_stracks:
                 self.tracker.pending_stracks.remove(t)
                 self.tracker.tracked_stracks.append(t)

        t1 = self.tracker.tracked_stracks[0]
        
        # Fast forward
        t1.tracklet_len = 29
        
        # Frame 30: T2 moves to overlap T1
        # Dets for this frame
        dets_overlap = np.array([
            [100, 100, 200, 200, 0.9], # T1
            [120, 120, 220, 220, 0.9]  # T2 (overlapping T1)
        ])
        
        features_before = t1.features.clone()
        self.tracker.assign_tracks_to_detections(dets_overlap, None, None, self.frame)
        
        # Should SKIP update due to track overlap
        self.assertTrue(torch.equal(t1.features, features_before), "Features updated despite track overlap")

if __name__ == '__main__':
    unittest.main()
