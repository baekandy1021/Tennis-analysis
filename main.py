import argparse
from utils import iter_frames, save_video, get_video_fps
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from editor.export import read_cuts
from mini_court import MiniCourt
import cv2


def main():
    parser = argparse.ArgumentParser(description="Player/ball tracking + mini court on a clip.")
    parser.add_argument("input", nargs="?", default="input_videos/input_video1.mp4")
    parser.add_argument("--cuts", help="Rally cut list from `python -m editor.cli` -- only analyze those spans")
    parser.add_argument("--use-stubs", action="store_true",
                        help="Reuse saved player/ball detections (only valid for the same video + cuts)")
    args = parser.parse_args()

    # Read only the frames we'll analyze: rally segments from the editor's cut list, or the whole clip
    input_video_path = args.input
    fps = get_video_fps(input_video_path)
    segments = read_cuts(args.cuts) if args.cuts else None
    kept_indices, court_frames, segment_ids = [], [], []
    for frame_idx, t, frame in iter_frames(input_video_path):
        segment_id = 0 if segments is None else next(
            (j for j, (s, e, _) in enumerate(segments) if s <= t < e), None)
        if segment_id is not None:
            kept_indices.append(frame_idx)
            court_frames.append(frame)
            segment_ids.append(segment_id)
    print(f"Analyzing {len(court_frames)} frames.")
    if not court_frames:
        print("No frames to analyze.")
        return

    # Frame ranges (in the kept-frame list) belonging to each rally segment / camera shot
    segment_ranges = {}
    for i, segment_id in enumerate(segment_ids):
        start, _ = segment_ranges.get(segment_id, (i, i))
        segment_ranges[segment_id] = (start, i + 1)

    # Court keypoints per frame, smoothed within each segment: the broadcast camera
    # reframes between shots and can zoom/pan during a rally
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = []
    for segment_id in sorted(segment_ranges):
        start, end = segment_ranges[segment_id]
        court_keypoints.extend(court_line_detector.predict_smoothed(court_frames[start:end]))

    # Stage 2: run player + ball detection only on the retained frames
    player_tracker = PlayerTracker(model_path='yolov8l')
    ball_tracker = BallTracker(model_path="models/yolo8_best.pt")
    player_detections = player_tracker.detect_frames(court_frames,
                                                    read_from_stub=args.use_stubs,
                                                    stub_path="tracker_stubs/player_detection.pkl")

    ball_detections = ball_tracker.detect_frames(court_frames,
                                                    read_from_stub=args.use_stubs,
                                                    stub_path="tracker_stubs/ball_detection.pkl")

    # Interpolate within each segment only -- never bridge a gap across a camera cut
    interpolated = []
    for segment_id in sorted(segment_ranges):
        start, end = segment_ranges[segment_id]
        interpolated.extend(ball_tracker.interpolate_ball_positions(ball_detections[start:end]))
    ball_detections = interpolated

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(court_frames[0])  # only uses the frame size

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                            ball_detections,
                                                                                                            court_keypoints)

    # Draw output on the retained (court-visible) frames only
    output_video_frames = player_tracker.draw_bboxes(court_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))

    ## Draw original frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {kept_indices[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "output_videos/output_video.avi", fps=fps)

if __name__ == "__main__":
    main()
