"""Train the court-view gate from hand-sorted frames (see scripts/label_frames.py).

    python scripts/train_view_classifier.py --labels labels --out models/view_classifier.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from editor.view_gate import train_view_classifier

parser = argparse.ArgumentParser()
parser.add_argument("--labels", default="labels")
parser.add_argument("--out", default="models/view_classifier.pt")
args = parser.parse_args()

train_view_classifier(os.path.join(args.labels, "court_view"), os.path.join(args.labels, "other"), args.out)
