from utils import read_video, save_video, get_video_fps
from court_line_detector import CourtLineDetector
from highlight_trimmer import PointClassifier, detect_court_visible_frames
import cv2


def main():
    input_video_path = "input_videos/input_video_small.mp4"
    video_frames = read_video(input_video_path)

    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)

    classifier = PointClassifier(court_line_detector, reference_frame=video_frames[0])
    court_visible = detect_court_visible_frames(video_frames, classifier,
                                                stub_path="tracker_stubs/court_visible.pkl",
                                                read_from_stub=True)

    kept_indices = [i for i, visible in enumerate(court_visible) if visible]
    print(f"Kept {len(kept_indices)}/{len(video_frames)} frames with the court visible.")
    if not kept_indices:
        print("No court-visible frames found, nothing to write.")
        return

    output_video_frames = [video_frames[i] for i in kept_indices]
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {kept_indices[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    fps = get_video_fps(input_video_path)
    save_video(output_video_frames, "output_videos/court_check.avi", fps=fps)

if __name__ == "__main__":
    main()
