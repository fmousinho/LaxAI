import logging
logger = logging.getLogger(__name__)

import numpy as np
import cv2

N_CORNERS = 20


def calculate_transform(
        frame1: np.ndarray, frame2: np.ndarray
    ) -> (np.ndarray, np.ndarray):
    """
    Calculate the transform matrix and translation vector between two frames.

    Args:
        frame1: The first frame, in RGB.
        frame2: The second frame, in RGB. 
    
    Returns:
        A tuple of the transform and translation vector.
        Transform S is a 8x1 array with the scale changes on the diagonal.
        Translation T is a 8x1 vector with the translation changes.
        Indices are ordered as: x, y, a, h, vx, vy, va, vh
        Use as: transform @ [x, y, a, h, vx, vy, va, vh] + translation
    """

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    pts1 = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=N_CORNERS, qualityLevel=0.01, minDistance=30)
    
    # If no features found, return identity
    if pts1 is None:
        return np.ones(8), np.zeros(8)

    pts2, status, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, pts1, None)

    # Select good points
    good1 = pts1[status == 1]
    good2 = pts2[status == 1]
    
    if len(good1) < 3:
        return np.ones(8), np.zeros(8)
    
    S = np.ones(8)
    T = np.zeros(8)
    
    # Translation
    dists = good2 - good1
    t = dists.mean(axis=0)
    if np.linalg.norm(t) > 1:
        T[:2] = t

    # Scale change (h, or y axis)
    # Estimate scale by comparing standard deviation of point distribution in Y
    y1_std = np.std(good1[:, 1])
    y2_std = np.std(good2[:, 1])
    
    y_scale = 1.0
    if y1_std > 1e-4:
        y_scale = y2_std / y1_std
        
    if abs(y_scale-1) > 0.03:
        max_scale = 1.2
        min_scale = .8
        S[3] = np.clip(y_scale, min_scale, max_scale)
        S[7] = np.clip(y_scale, min_scale, max_scale)

    # Aspect ratio change (a)
    # Estimate scale in X
    x1_std = np.std(good1[:, 0])
    x2_std = np.std(good2[:, 0])
    
    x_scale = 1.0
    if x1_std > 1e-4:
        x_scale = x2_std / x1_std
        
    # Scale of aspect ratio is x_scale / y_scale
    if y_scale > 1e-4:
        a_scale = x_scale/y_scale
    else:
        a_scale = 1.0
        
    if abs(a_scale-1) > 0.03:
        max_scale = 1.2
        min_scale = .8
        S[2] = np.clip(a_scale, min_scale, max_scale)
        S[6] = np.clip(a_scale, min_scale, max_scale)

    return S, T 


    

  


