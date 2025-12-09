import unittest
import numpy as np
import cv2
import sys
import os

# Add the service_tracking directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
service_tracking_dir = os.path.dirname(current_dir)
sys.path.insert(0, service_tracking_dir)

from tracker.cam_mvmt import calculate_transform

class TestCamMvmt(unittest.TestCase):
    def test_calculate_transform_basic(self):
        # Create a blank image
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw some features (white circles)
        for i in range(10, 90, 20):
            for j in range(10, 90, 20):
                cv2.circle(frame1, (i, j), 2, (255, 255, 255), -1)

        # Create frame2 identically (no movement)
        frame2 = frame1.copy()

        # Run the function
        try:
            M, T = calculate_transform(frame1, frame2)
            print("M:", M)
            print("T:", T)
        except Exception as e:
            self.fail(f"calculate_transform failed with exception: {e}")

    def test_calculate_transform_translation(self):
        # Create a blank image
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw some features
        points = []
        for i in range(20, 80, 20):
            for j in range(20, 80, 20):
                points.append((i, j))
                cv2.circle(frame1, (i, j), 2, (255, 255, 255), -1)
        
        # Create frame2 shifted by (5, 5)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        for (x, y) in points:
             cv2.circle(frame2, (x + 5, y + 5), 2, (255, 255, 255), -1)

        try:
            M, T = calculate_transform(frame1, frame2)
            print("Translation M:", M)
            print("Translation T:", T)
            
            # Assertions
            self.assertEqual(M.shape, (8, 8))
            self.assertEqual(T.shape, (8,))
            
            # Translation should be present in X/Y (indices 0, 1)
            self.assertTrue(abs(T[0] - 5) < 1, f"Expected T[0] ~ 5, got {T[0]}")
            self.assertTrue(abs(T[1] - 5) < 1, f"Expected T[1] ~ 5, got {T[1]}")
            
            # Translation should NOT be present in Velocities (indices 4, 5)
            # Camera translation does not add to object velocity directly
            self.assertEqual(T[4], 0)
            self.assertEqual(T[5], 0)
            
            # Scale should be Identity for pure translation
            self.assertTrue(np.allclose(M, np.eye(8)))
            
        except Exception as e:
            self.fail(f"calculate_transform failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
