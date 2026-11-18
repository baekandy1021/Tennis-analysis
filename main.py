from utils import read_video, save_video, get_video_fps
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from highlight_trimmer import PointClassifier, detect_court_visible_frames
from mini_court import MiniCourt
import cv2


def main():
    # Read video
    input_video_path = "input_videos/input_video1.mp4"
    video_frames = read_video(input_video_path)

    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Stage 1: keep only frames where the full court is actually visible
    classifier = PointClassifier(court_line_detector, reference_frame=video_frames[0])
    court_visible = detect_court_visible_frames(video_frames, classifier,
                                                stub_path="tracker_stubs/court_visible.pkl",
                                                read_from_stub=False)

    kept_indices = [i for i, visible in enumerate(court_visible) if visible]
    print(f"Kept {len(kept_indices)}/{len(video_frames)} frames with the court visible.")
    if not kept_indices:
        print("No court-visible frames found, nothing to write.")
        return

    court_frames = [video_frames[i] for i in kept_indices]

    # Stage 2: run player + ball detection only on the retained frames
    player_tracker = PlayerTracker(model_path='yolov8l')
    ball_tracker = BallTracker(model_path="models/yolo8_best.pt")
    player_detections = player_tracker.detect_frames(court_frames,
                                                    read_from_stub=False,
                                                    stub_path="tracker_stubs/player_detection.pkl")

    ball_detections = ball_tracker.detect_frames(court_frames,
                                                    read_from_stub=False,
                                                    stub_path="tracker_stubs/ball_detection.pkl")

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0])

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

    fps = get_video_fps(input_video_path)
    save_video(output_video_frames, "output_videos/output_video.avi", fps=fps)

if __name__ == "__main__":
    main()
