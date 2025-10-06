from ultralytics import YOLO # type: ignore

model = YOLO('yolov8m')

model.predict('input_videos/input_video.mp4',save = True)