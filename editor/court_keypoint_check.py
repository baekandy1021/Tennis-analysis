import cv2
import numpy as np
from mini_court import MiniCourt


class CourtKeypointCheck():
    """Fallback court-view check used when no trained view classifier exists:
    predicted court keypoints must span a large part of the frame and map onto the
    canonical court layout with a consistent homography."""

    def __init__(self, court_line_detector, reference_frame,
                 min_hull_area_ratio=0.15,
                 min_inliers=10,
                 max_reprojection_error_ratio=0.05):
        self.court_line_detector = court_line_detector
        self.min_hull_area_ratio = min_hull_area_ratio
        self.min_inliers = min_inliers
        self.max_reprojection_error_ratio = max_reprojection_error_ratio

        mini_court = MiniCourt(reference_frame)
        self.canonical_keypoints = np.array(mini_court.get_court_drawing_keypoints(),
                                             dtype=np.float32).reshape(-1, 2)
        self.canonical_court_width = mini_court.get_width_of_mini_court()

    def classify_keypoints(self, keypoints, frame_shape):
        points = np.array(keypoints, dtype=np.float32).reshape(-1, 2)

        frame_area = frame_shape[0] * frame_shape[1]
        hull_area = cv2.contourArea(cv2.convexHull(points))
        if hull_area < self.min_hull_area_ratio * frame_area:
            return False

        homography, mask = cv2.findHomography(points, self.canonical_keypoints, method=cv2.RANSAC)
        if homography is None or mask is None:
            return False
        if int(mask.sum()) < self.min_inliers:
            return False

        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - self.canonical_keypoints, axis=1)
        mean_error = errors[mask.ravel().astype(bool)].mean()
        return mean_error < self.max_reprojection_error_ratio * self.canonical_court_width

    def is_court_frame(self, frame):
        return self.classify_keypoints(self.court_line_detector.predict(frame), frame.shape)
