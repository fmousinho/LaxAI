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
        A tuple of the transform and translation vector as numpy arrays.
        Transform Matrix M is a 8x8 array with the scale changes on the diagonal.
        Translation Vector T is a 8x1 vector with the translation changes.
        Indices are ordered as: x, y, a, h, vx, vy, va, vh
        Use as: transform @ [x, y, a, h, vx, vy, va, vh] + translation
    """

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    pts1 = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=N_CORNERS, qualityLevel=0.01, minDistance=30)
    
    # If no features found, return identity
    if pts1 is None:
        return np.eye(8), np.zeros(8)

    pts2, status, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, pts1, None)

    # Select good points
    good1 = pts1[status == 1]
    good2 = pts2[status == 1]
    
    if len(good1) < 3:
        return np.eye(8), np.zeros(8)
    
    S = np.eye(8)
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

    affine_ret = cv2.estimateAffinePartial2D(good1, good2, method=cv2.RANSAC)
    affine_matrix = affine_ret[0]
    
    if affine_matrix is None:
        return np.eye(8), np.zeros(8)

    M = np.eye(8)
    M[:2, :2] = affine_matrix[:, :2]
    M[4:6, 4:6] = affine_matrix[:, :2]

    s = np.linalg.norm(affine_matrix[0, :2])
    s = np.clip(s, 1e-4, 1.2)
    M[3,3] = s
    M[7,7] = s

    T[:2] = affine_matrix[:, 2]
    
    return M, T
    
    



    

  


