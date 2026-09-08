from pathlib import Path 

from geofence_monitor.ingestion import iter_video_frames
from geofence_monitor.inference import infer_frame, load_detector , infer_video

video_path = Path("/home/fishe/code/cv-geofence-monitoring-mvp/data/input/sample.mp4")

confidence_threshold = 0.50

model, weights = load_detector()
categories = weights.meta["categories"]

results = infer_video(
    model,
    video_path,
    categories,
    confidence_threshold=0.50,
    allowed_classes={"person", "car", "truck"},
)

for result in results[:5]:
    print(result)