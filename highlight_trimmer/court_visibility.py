import os
import pickle
import numpy as np


def _keypoint_jitter(keypoints_a, keypoints_b):
    points_a = np.array(keypoints_a, dtype=np.float32).reshape(-1, 2)
    points_b = np.array(keypoints_b, dtype=np.float32).reshape(-1, 2)
    return np.linalg.norm(points_a - points_b, axis=1).mean()


def detect_court_visible_frames(video_frames, classifier, stub_path=None, read_from_stub=False,
                                 jitter_threshold_ratio=0.03):
    """A frame counts as court-visible only if its predicted keypoints pass the
    homography/reprojection check AND are stable (low frame-to-frame jitter)
    relative to a geometrically-valid neighbor. The keypoint regressor can't say
    "no court here" -- on non-court frames it still emits a plausible-looking
    but hallucinated layout that can pass the homography check alone. Since the
    camera is static during real court shots, genuine court frames have near-
    identical keypoints frame-to-frame, while hallucinated guesses on unrelated
    content (crowd/graphics/replays) jump around erratically -- jitter catches
    that where the homography check alone can't."""
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)

    keypoints_list = [classifier.predict_keypoints(frame) for frame in video_frames]
    geometry_valid = [classifier.classify_keypoints(kp, video_frames[i].shape)
                       for i, kp in enumerate(keypoints_list)]

    jitter_threshold = jitter_threshold_ratio * classifier.canonical_court_width

    n = len(video_frames)
    court_visible = []
    for i in range(n):
        if not geometry_valid[i]:
            court_visible.append(False)
            continue

        stable_with_prev = (i > 0 and geometry_valid[i - 1] and
                             _keypoint_jitter(keypoints_list[i], keypoints_list[i - 1]) <= jitter_threshold)
        stable_with_next = (i < n - 1 and geometry_valid[i + 1] and
                             _keypoint_jitter(keypoints_list[i], keypoints_list[i + 1]) <= jitter_threshold)

        court_visible.append(stable_with_prev or stable_with_next)

    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(court_visible, f)

    return court_visible
