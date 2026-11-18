import argparse

from utils import read_video, save_video, get_video_fps
from court_line_detector import CourtLineDetector
from highlight_trimmer import ShotSegmenter, PointClassifier


def trim_highlights(input_video_path, output_video_path, court_model_path="models/keypoints_model.pth"):
    frames = read_video(input_video_path)
    fps = get_video_fps(input_video_path)

    segmenter = ShotSegmenter()
    shots = segmenter.detect_shots(frames)

    court_line_detector = CourtLineDetector(court_model_path)
    classifier = PointClassifier(court_line_detector, reference_frame=frames[0])

    point_frames = []
    for start, end in shots:
        if classifier.is_point_shot(frames, start, end):
            point_frames.extend(frames[start:end])

    save_video(point_frames, output_video_path, fps=fps)
    print(f"Kept {len(point_frames)}/{len(frames)} frames across {len(shots)} detected shots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim a highlight reel down to just the points being played.")
    parser.add_argument("input_video", help="Path to the input highlight reel video.")
    parser.add_argument("output_video", help="Path to write the trimmed output video to.")
    parser.add_argument("--court-model", default="models/keypoints_model.pth", help="Path to the court keypoints model.")
    args = parser.parse_args()

    trim_highlights(args.input_video, args.output_video, args.court_model)
