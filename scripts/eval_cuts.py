"""Score predicted rally cuts against hand labels.

Ground truth: a JSON list of [start_s, end_s] rallies, e.g. labels/input_video_small_rallies.json
    python scripts/eval_cuts.py output_videos/input_video_small_cuts.json labels/input_video_small_rallies.json
"""
import argparse
import json
import statistics


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


parser = argparse.ArgumentParser()
parser.add_argument("predicted")
parser.add_argument("truth")
parser.add_argument("--iou", type=float, default=0.5)
args = parser.parse_args()

with open(args.predicted) as f:
    pred = [(s["start"], s["end"]) for s in json.load(f)["segments"]]
with open(args.truth) as f:
    truth = [tuple(x) for x in json.load(f)]

matched, start_err, end_err, used = 0, [], [], set()
for t in truth:
    best = max(((iou(p, t), j) for j, p in enumerate(pred) if j not in used), default=(0, None))
    if best[1] is not None and best[0] >= args.iou:
        used.add(best[1])
        matched += 1
        start_err.append(abs(pred[best[1]][0] - t[0]))
        end_err.append(abs(pred[best[1]][1] - t[1]))

recall = matched / len(truth) if truth else 0
precision = matched / len(pred) if pred else 0
print(f"rallies: truth {len(truth)}  predicted {len(pred)}  matched {matched} (IoU >= {args.iou})")
print(f"recall {recall:.2f}  precision {precision:.2f}")
if start_err:
    print(f"median start error {statistics.median(start_err):.2f}s  median end error {statistics.median(end_err):.2f}s")
