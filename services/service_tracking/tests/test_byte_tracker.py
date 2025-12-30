import unittest
import numpy as np
import sys
import os

# Add the service_tracking directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
service_tracking_dir = os.path.dirname(current_dir)
sys.path.insert(0, service_tracking_dir)

from tracker.byte_tracker import BYTETracker, STrack
from tracker.basetrack import TrackState
from tracker.config import TrackingParams

class TestBYTETracker(unittest.TestCase):
    def test_byte_tracker_update_with_cam_motion(self):
        params = TrackingParams()
        tracker = BYTETracker(params)
        
        # 1. First Update: Initialize a track
        # Detections: Nx5 array (x1, y1, x2, y2, score)
        # Create a detection at [100, 100, 150, 150] with high score
        det1 = np.array([[100, 100, 150, 150, 0.9]])
        
        # Call update without camera motion (None, None)
        output_stracks = tracker.update(det1, None, None)
        
        self.assertEqual(len(output_stracks), 1)
        track = output_stracks[0]
        self.assertEqual(track.track_id, 1)
        # track state should be initialized.
        # mean is (x, y, a, h, vx, vy, va, vh)
        # x, y are center. 100, 100, 150, 150 -> w=50, h=50. center=125, 125
        self.assertTrue(abs(track.mean[0] - 125.0) < 1.0)
        self.assertTrue(abs(track.mean[1] - 125.0) < 1.0)
        
        # 2. Second Update: Apply Camera Motion Compensation
        # Simulate camera moved +10 in X and +10 in Y.
        # So objects should shift -10 in X and -10 in Y?
        # WAIT: The implementation in tracker.update calls STrack.compensate_cam_motion(S, T)
        # BUT STrack.compensate_cam_motion is an INSTANCE method.
        # Calling it as STrack.compensate_cam_motion(S, T) implies it's static?
        # Let's see if it crashes.
        
        S_array = np.ones(8)
        T_array = np.zeros(8)
        T_array[0] = 10.0 # Shift X by 10
        T_array[1] = 10.0 # Shift Y by 10
        
        # Detections: Assume object moved to 135, 135 naturally? 
        # Or let's just provide a valid detection to keep it tracked.
        det2 = np.array([[135, 135, 185, 185, 0.9]])
        
        try:
            tracker.update(det2, S_array, T_array)
        except TypeError as e:
            self.fail(f"BYTETracker.update failed with TypeError, possibly due to incorrect method call: {e}")
        except Exception as e:
            self.fail(f"BYTETracker.update failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
