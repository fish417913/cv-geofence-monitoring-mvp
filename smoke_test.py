import torch
from ultralytics import YOLO

# Load a small pretrained object-detection model
model = YOLO("yolo26n.pt")

# Run inference on an official Ultralytics test image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True
)

# One image produces one result object
result = results[0]

print(f"\nImage size: {result.orig_shape}")
print(f"Number of detections: {len(result.boxes)}")
print(f"CUDA available: {torch.cuda.is_available()}\n")

# Examine each individual detection
for box in result.boxes:
    class_id = int(box.cls.item())
    class_name = result.names[class_id]
    confidence = float(box.conf.item())
    coordinates = box.xyxy[0].tolist()

    print(
        f"Class: {class_name:10s} "
        f"Confidence: {confidence:.3f} "
        f"Box: {[round(value, 1) for value in coordinates]}"
    )
