from pathlib import Path 

import cv2 
from ultralytics import YOLO 

INPUT_VIDEO_PATH = Path("data/input/sample.mp4")
OUTPUT_VIDEO_PATH = Path("data/output/annotated_video.mp4")

MODEL_PATH = "yolo26n.pt"
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_SIZE=960
PERSON_CLASS_ID = 0

if not INPUT_VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {INPUT_VIDEO_PATH.resolve()}"
    )
    
capture = cv2.VideoCapture(str(INPUT_VIDEO_PATH))

if not capture.isOpened():
    raise RuntimeError(
        f"OpenCV could not open: {INPUT_VIDEO_PATH.resolve()}"
    )
    
fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration_seconds = frame_count / fps if fps > 0 else 0.0

print(f"Input video: {INPUT_VIDEO_PATH.resolve()}")
print(f"Resolution: {frame_width} x {frame_height}")
print(f"Frame rate: {fps:.2f} FPS")
print(f"Frame count: {frame_count}")
print(f"Duration: {duration_seconds:.2f} seconds")

OUTPUT_VIDEO_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

model = YOLO(MODEL_PATH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO_PATH),
    fourcc,
    fps,
    (frame_width, frame_height)
)

if not writer.isOpened():
    capture.release()
    raise RuntimeError(
        f"OpenCV could not create: {OUTPUT_VIDEO_PATH.resolve()}"
    )
    
processed_frames = 0

while True:
    read_successfully, frame = capture.read()
    
    if not read_successfully:
        break 
    
    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_SIZE,
        classes=[PERSON_CLASS_ID],
        verbose=False
    )
    
    annotated_frame = results[0].plot()
    writer.write(annotated_frame)
    
    processed_frames += 1
    
    if processed_frames % 25 == 0:
        print(
            f"Processed {processed_frames}/{frame_count} frames"
        )
        
capture.release()
writer.release()

print(f"Procssed frames: {processed_frames}")
print(f"Output video: {OUTPUT_VIDEO_PATH.resolve()}")