import cv2


class ShotSegmenter():
    def __init__(self, hist_diff_threshold=0.4, min_shot_len=10):
        self.hist_diff_threshold = hist_diff_threshold
        self.min_shot_len = min_shot_len

    def _histogram(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def detect_shots(self, frames):
        if len(frames) == 0:
            return []

        cut_frames = [0]
        prev_hist = self._histogram(frames[0])
        for i in range(1, len(frames)):
            hist = self._histogram(frames[i])
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > self.hist_diff_threshold:
                cut_frames.append(i)
            prev_hist = hist
        cut_frames.append(len(frames))

        shots = []
        for start, end in zip(cut_frames[:-1], cut_frames[1:]):
            if end - start >= self.min_shot_len:
                shots.append((start, end))
            elif shots:
                # fold short flicker segments into the previous shot
                prev_start, _ = shots[-1]
                shots[-1] = (prev_start, end)

        return shots
