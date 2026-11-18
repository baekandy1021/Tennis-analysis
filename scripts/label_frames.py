"""Dump evenly spaced frames from videos for hand-sorting into court_view/ and other/.

    python scripts/label_frames.py input_videos/*.mp4 --every 1.0
Then move each image in labels/unsorted/ into labels/court_view/ or labels/other/
(main wide camera = court_view; closeups, replays, crowd, graphics, side angles = other),
and run scripts/train_view_classifier.py.
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import iter_frames

parser = argparse.ArgumentParser()
parser.add_argument("videos", nargs="+")
parser.add_argument("--every", type=float, default=1.0, help="Seconds between dumped frames")
parser.add_argument("--out", default="labels")
args = parser.parse_args()

for sub in ("unsorted", "court_view", "other"):
    os.makedirs(os.path.join(args.out, sub), exist_ok=True)

total = 0
for video in args.videos:
    name = os.path.splitext(os.path.basename(video))[0]
    for idx, t, frame in iter_frames(video, target_fps=1.0 / args.every, max_height=360):
        cv2.imwrite(os.path.join(args.out, "unsorted", f"{name}_{t:08.2f}.jpg"), frame)
        total += 1
print(f"Wrote {total} frames to {os.path.join(args.out, 'unsorted')}")
