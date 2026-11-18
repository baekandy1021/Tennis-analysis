import numpy as np


def _merge(intervals, max_gap=0.0):
    merged = []
    for start, end, conf in sorted(intervals):
        if merged and start - merged[-1][1] <= max_gap:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], end), max(prev[2], conf))
        else:
            merged.append((start, end, conf))
    return merged


class RallyDetector():
    """Turns per-sample ball positions (+ optional audio hit times) inside court-view
    intervals into live-rally segments. All thresholds are in seconds or in fractions
    of the frame height per second, so they don't depend on fps or resolution."""

    def __init__(self,
                 min_ball_speed=0.5,         # frame-heights/s for a sample to count as "ball in play"
                 activity_window_s=1.0,
                 min_activity_ratio=0.3,
                 min_active_s=1.0,
                 hit_persist_s=0.5,          # y-direction must hold this long after a reversal
                 max_hit_gap_s=4.0,          # hits further apart start a new rally
                 pre_roll_s=1.5,
                 post_roll_s=2.0,
                 merge_gap_s=1.0,
                 min_rally_s=2.0):
        self.min_ball_speed = min_ball_speed
        self.activity_window_s = activity_window_s
        self.min_activity_ratio = min_activity_ratio
        self.min_active_s = min_active_s
        self.hit_persist_s = hit_persist_s
        self.max_hit_gap_s = max_hit_gap_s
        self.pre_roll_s = pre_roll_s
        self.post_roll_s = post_roll_s
        self.merge_gap_s = merge_gap_s
        self.min_rally_s = min_rally_s

    def ball_activity(self, times, ball_y, ball_x):
        """Boolean per sample: ball detected and moving fast, smoothed over a window."""
        times = np.asarray(times)
        moving = np.zeros(len(times), dtype=bool)
        last = None
        for i, (x, y) in enumerate(zip(ball_x, ball_y)):
            if np.isnan(y):
                continue
            if last is not None and times[i] - times[last] <= 0.5:
                dist = np.hypot(x - ball_x[last], y - ball_y[last])
                moving[i] = dist / (times[i] - times[last]) >= self.min_ball_speed
            last = i

        active = np.zeros(len(times), dtype=bool)
        half = self.activity_window_s / 2
        for i, t in enumerate(times):
            window = (times >= t - half) & (times <= t + half)
            active[i] = moving[window].mean() >= self.min_activity_ratio
        return active

    def ball_hit_times(self, times, ball_y):
        """Vertical direction reversals of the (interpolated, smoothed) ball track that
        persist for hit_persist_s: the ball changed ends of the court, i.e. it was hit."""
        times = np.asarray(times)
        y = np.asarray(ball_y, dtype=float)
        valid = ~np.isnan(y)
        if valid.sum() < 5:
            return []
        y = np.interp(times, times[valid], y[valid])
        dt = np.median(np.diff(times)) if len(times) > 1 else 0.1
        k = max(1, int(round(0.2 / dt)))
        y = np.convolve(y, np.ones(k) / k, mode="same")
        direction = np.sign(np.diff(y))
        persist = max(2, int(round(self.hit_persist_s / dt)))

        hits = []
        for i in range(1, len(direction) - persist):
            if direction[i] != 0 and direction[i - 1] != 0 and direction[i] != direction[i - 1]:
                if np.all(direction[i:i + persist] == direction[i]):
                    # only trust reversals where the ball was actually seen nearby
                    if valid[max(0, i - persist):i + persist].mean() >= 0.3:
                        hits.append(float(times[i]))
        return hits

    def detect(self, view_intervals, times, ball_x, ball_y, audio_hits=()):
        """view_intervals: [(start_s, end_s)] court-view spans. Returns [(start, end, conf)]."""
        times = np.asarray(times)
        ball_x = np.asarray(ball_x, dtype=float)
        ball_y = np.asarray(ball_y, dtype=float)
        rallies = []

        for v_start, v_end in view_intervals:
            in_view = (times >= v_start) & (times < v_end)
            if in_view.sum() < 3:
                continue
            t, bx, by = times[in_view], ball_x[in_view], ball_y[in_view]

            candidates = []
            # 1) sustained ball activity
            active = self.ball_activity(t, by, bx)
            run_start = None
            for i, a in enumerate(np.append(active, False)):
                if a and run_start is None:
                    run_start = i
                elif not a and run_start is not None:
                    s, e = t[run_start], t[min(i, len(t) - 1)]
                    if e - s >= self.min_active_s:
                        candidates.append((s - self.pre_roll_s, e + self.post_roll_s / 2, 0.6))
                    run_start = None

            # 2) groups of hits (ball reversals + audio impacts)
            ball_hits = self.ball_hit_times(t, by)
            audio_in_view = [h for h in audio_hits if v_start <= h < v_end]
            events = sorted([(h, "ball") for h in ball_hits] + [(h, "audio") for h in audio_in_view])
            group = []
            for event in events + [(float("inf"), None)]:
                if group and event[0] - group[-1][0] > self.max_hit_gap_s:
                    kinds = {k for _, k in group}
                    # a lone audio spike is too weak on its own (crowd, commentary)
                    if len(group) >= 2 or "ball" in kinds:
                        conf = min(1.0, 0.4 + 0.15 * len(group) + (0.2 if len(kinds) == 2 else 0))
                        candidates.append((group[0][0] - self.pre_roll_s, group[-1][0] + self.post_roll_s, conf))
                    group = []
                group.append(event)

            for s, e, conf in _merge(candidates, self.merge_gap_s):
                s, e = max(s, v_start), min(e, v_end)
                if e - s >= self.min_rally_s:
                    rallies.append((round(s, 2), round(e, 2), round(conf, 2)))

        return _merge(rallies, self.merge_gap_s)
