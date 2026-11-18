import cv2


class ShotSegmenter():
    """Finds camera cuts from HSV-histogram jumps between consecutive sampled frames.
    Works incrementally: call update(frame) per sampled frame, then shots()."""

    def __init__(self, hist_diff_threshold=0.4, min_shot_len=10):
        self.hist_diff_threshold = hist_diff_threshold
        self.min_shot_len = min_shot_len  # in sampled frames
        self._prev_hist = None
        self._cut_indices = [0]
        self._count = 0

    def _histogram(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def update(self, frame):
        hist = self._histogram(frame)
        if self._prev_hist is not None:
            diff = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > self.hist_diff_threshold:
                self._cut_indices.append(self._count)
        self._prev_hist = hist
        self._count += 1

    def shots(self):
        """Returns [(start, end)] in sampled-frame indices, end exclusive."""
        if self._count == 0:
            return []
        cuts = self._cut_indices + [self._count]
        shots = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            if end - start >= self.min_shot_len or not shots:
                shots.append((start, end))
            else:
                # fold short flicker segments into the previous shot
                shots[-1] = (shots[-1][0], end)
        return shots
