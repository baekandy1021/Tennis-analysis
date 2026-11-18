"""Cut a tennis video down to live rallies.

    python -m editor.cli input_videos/input_video_small.mp4 -o output_videos/rallies.mp4
    python -m editor.cli my_match.mp4 --source own          # single static camera, skip view gate
    python -m editor.cli clip.mp4 --cuts-only               # just write the cut list
"""
import argparse
import os
import pickle

import numpy as np
import torch
from ultralytics import YOLO

from utils import iter_frames
from editor.shot_segmenter import ShotSegmenter
from editor.view_gate import load_view_checker, vote_shots
from editor.rally_detector import RallyDetector
from editor.audio_hits import detect_hit_times
from editor.export import write_cuts, render

CACHE_DIR = "editor_cache"


def analyze(video_path, source, fps, max_height, view_model, court_model, ball_model, check_every_s):
    """One streaming pass. Returns per-sample signals; no frames are kept in memory."""
    gate = source == "broadcast"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ball_detector = YOLO(ball_model)
    segmenter = ShotSegmenter(min_shot_len=max(2, int(fps)))  # >= ~1 s
    checker = None
    check_step = max(1, int(round(check_every_s * fps)))

    times, ball_x, ball_y, view_labels = [], [], [], {}
    latest_view = True
    for i, (_, t, frame) in enumerate(iter_frames(video_path, target_fps=fps, max_height=max_height)):
        times.append(t)
        h = frame.shape[0]
        if gate:
            segmenter.update(frame)
            if checker is None:
                checker = load_view_checker(view_model, court_model, frame)
            if i % check_step == 0:
                latest_view = bool(checker.is_court_frame(frame))
                view_labels[i] = latest_view

        x = y = np.nan
        if latest_view:  # don't spend ball inference on closeups/replays
            boxes = ball_detector.predict(frame, conf=0.15, verbose=False, device=device)[0].boxes
            if len(boxes):
                x1, y1, x2, y2 = boxes.xyxy[int(boxes.conf.argmax())].tolist()
                x, y = (x1 + x2) / 2 / h, (y1 + y2) / 2 / h  # normalized by frame height
        ball_x.append(x)
        ball_y.append(y)
        if i % 200 == 0:
            print(f"  analyzed {t:7.1f}s", flush=True)

    shots = segmenter.shots() if gate else [(0, len(times))]
    return {"times": times, "ball_x": ball_x, "ball_y": ball_y,
            "shots": shots, "view_labels": view_labels, "gate": gate}


def view_intervals(signals):
    times = signals["times"]
    dt = times[1] - times[0] if len(times) > 1 else 0.1
    shots = signals["shots"]
    keep = vote_shots(shots, signals["view_labels"]) if signals["gate"] else [True] * len(shots)
    return [(times[s], times[e - 1] + dt) for (s, e), k in zip(shots, keep) if k]


def main():
    parser = argparse.ArgumentParser(description="Cut a tennis video down to live rallies.")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", help="Rendered video path (default: output_videos/<name>_rallies.mp4)")
    parser.add_argument("--source", choices=["broadcast", "own"], default="broadcast",
                        help="broadcast = has camera cuts/replays (runs the view gate); own = single static camera")
    parser.add_argument("--fps", type=float, default=10, help="Analysis frame rate")
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--check-every", type=float, default=0.5, help="Seconds between view-gate checks")
    parser.add_argument("--view-model", default="models/view_classifier.pt")
    parser.add_argument("--court-model", default="models/keypoints_model.pth")
    parser.add_argument("--ball-model", default="models/yolo8_best.pt")
    parser.add_argument("--no-audio", action="store_true", help="Ignore the audio track")
    parser.add_argument("--cuts-only", action="store_true", help="Write cuts JSON, skip rendering")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached analysis")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("output_videos", exist_ok=True)
    view_tag = "clf" if os.path.exists(args.view_model) else "kp"
    cache_path = os.path.join(CACHE_DIR, f"{name}_{args.source}_{view_tag}_{args.fps:g}fps_{args.max_height}p.pkl")

    if os.path.exists(cache_path) and not args.refresh:
        with open(cache_path, "rb") as f:
            signals = pickle.load(f)
        print(f"Loaded cached analysis {cache_path}")
    else:
        print(f"Analyzing {args.input} at {args.fps:g} fps ...")
        signals = analyze(args.input, args.source, args.fps, args.max_height,
                          args.view_model, args.court_model, args.ball_model, args.check_every)
        with open(cache_path, "wb") as f:
            pickle.dump(signals, f)

    intervals = view_intervals(signals)
    audio_hits = [] if args.no_audio else detect_hit_times(args.input)
    cuts = RallyDetector().detect(intervals, signals["times"], signals["ball_x"], signals["ball_y"], audio_hits)

    duration = signals["times"][-1] if signals["times"] else 0
    kept = sum(e - s for s, e, _ in cuts)
    print(f"Court-view spans: {len(intervals)}  audio hits: {len(audio_hits)}  rallies: {len(cuts)}  "
          f"kept {kept:.1f}s of {duration:.1f}s")
    for s, e, c in cuts:
        print(f"  {s:8.2f} -> {e:8.2f}  ({e - s:5.1f}s, conf {c})")

    cuts_path = os.path.join("output_videos", f"{name}_cuts.json")
    write_cuts(cuts, cuts_path, args.input)
    print(f"Wrote {cuts_path}")

    if not args.cuts_only and cuts:
        output = args.output or os.path.join("output_videos", f"{name}_rallies.mp4")
        render(args.input, cuts, output)
        print(f"Rendered {output}")


if __name__ == "__main__":
    main()
