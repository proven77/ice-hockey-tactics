import numpy as np
import cv2

class Homography:
  def __init__(self, source: np.ndarray, target: np.ndarray, ransac_threshold=None, min_inliers=4):
    if source.shape != target.shape:
      raise ValueError("Source and target must have the same shape")
    
    if source.ndim != 2 or source.shape[1] != 2 or len(source) < 4:
      raise ValueError("Homography needs at least four pairs of 2D points")
    
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    
    if ransac_threshold is None:
      self.m, mask = cv2.findHomography(source, target)
    else:
      self.m, mask = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold)

    if self.m is None or not np.isfinite(self.m).all() or np.linalg.matrix_rank(self.m) < 3:
      raise ValueError("Degenerate homography")
    self.inliers = int(np.count_nonzero(mask))
    if self.inliers < min_inliers:
      raise ValueError("Not enough homography inliers")
    
  def transform_points(self, points):
    if points.size == 0:
      return points
  
    if points.shape[1] != 2:
      raise ValueError("Points must be 2D arrays")
    
    points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(points, self.m)
    transformed_points = transformed_points.reshape(-1, 2)
    denominator = points[:, 0] @ self.m[2, :2] + self.m[2, 2]
    transformed_points[np.abs(denominator) < 1e-7] = np.nan
    return transformed_points
