import cv2
import numpy as np
from mini_court import MiniCourt


class PointClassifier():
    def __init__(self, court_line_detector, reference_frame,
                 min_hull_area_ratio=0.15,
                 min_inliers=10,
                 max_reprojection_error_ratio=0.05,
                 samples_per_shot=5,
                 shot_edge_skip=5,
                 majority_ratio=0.6):
        self.court_line_detector = court_line_detector
        self.min_hull_area_ratio = min_hull_area_ratio
        self.min_inliers = min_inliers
        self.max_reprojection_error_ratio = max_reprojection_error_ratio
        self.samples_per_shot = samples_per_shot
        self.shot_edge_skip = shot_edge_skip
        self.majority_ratio = majority_ratio

        mini_court = MiniCourt(reference_frame)
        self.canonical_keypoints = np.array(mini_court.get_court_drawing_keypoints(),
                                             dtype=np.float32).reshape(-1, 2)
        self.canonical_court_width = mini_court.get_width_of_mini_court()

    def predict_keypoints(self, frame):
        return self.court_line_detector.predict(frame)

    def classify_keypoints(self, keypoints, frame_shape):
        points = np.array(keypoints, dtype=np.float32).reshape(-1, 2)

        frame_area = frame_shape[0] * frame_shape[1]
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        if hull_area < self.min_hull_area_ratio * frame_area:
            return False

        homography, mask = cv2.findHomography(points, self.canonical_keypoints, method=cv2.RANSAC)
        if homography is None or mask is None:
            return False

        inlier_count = int(mask.sum())
        if inlier_count < self.min_inliers:
            return False

        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - self.canonical_keypoints, axis=1)
        inlier_errors = errors[mask.ravel().astype(bool)]
        mean_error = inlier_errors.mean()

        return mean_error < self.max_reprojection_error_ratio * self.canonical_court_width

    def is_court_frame(self, frame):
        keypoints = self.predict_keypoints(frame)
        return self.classify_keypoints(keypoints, frame.shape)

    def is_point_shot(self, frames, start, end):
        lo = start + self.shot_edge_skip
        hi = end - self.shot_edge_skip
        if hi <= lo:
            lo, hi = start, end

        sample_indices = np.linspace(lo, hi - 1, num=min(self.samples_per_shot, hi - lo), dtype=int)
        sample_indices = sorted(set(sample_indices.tolist()))

        votes = [self.is_court_frame(frames[i]) for i in sample_indices]
        return sum(votes) / len(votes) >= self.majority_ratio
