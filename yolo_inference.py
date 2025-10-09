from ultralytics import YOLO # type: ignore

# model = YOLO('models/yolo5_best.pt')
model = YOLO('yolov8l')

result = model.track('input_videos/input_video.mp4',conf=0.2, save = True)

