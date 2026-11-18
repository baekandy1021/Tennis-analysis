import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def save_video(output_video_frames, output_video_path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()



def iter_frames(video_path, target_fps=None, max_height=None):
    """Stream (frame_idx, timestamp_s, frame) without loading the whole video.
    Subsamples to ~target_fps (grab() skips decoding of dropped frames) and
    downscales so height <= max_height. Timestamps are in source-time seconds."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / target_fps)) if target_fps else 1
    frame_idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if max_height and frame.shape[0] > max_height:
                    scale = max_height / frame.shape[0]
                    frame = cv2.resize(frame, (int(frame.shape[1] * scale), max_height), interpolation=cv2.INTER_AREA)
                yield frame_idx, frame_idx / src_fps, frame
            frame_idx += 1
    finally:
        cap.release()
