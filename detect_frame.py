from pathlib import Path

import cv2
from ultralytics import YOLO

def calculate_iou(box_a, box_b):
    """Calculate Intersection over Union for two xyxy bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    box_a_area = (ax2 - ax1) * (ay2 - ay1)
    box_b_area = (bx2 - bx1) * (by2 - by1)

    union_area = box_a_area + box_b_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

def suppress_duplicate_detections(
    boxes,
    confidences,
    class_ids,
    iou_threshold=0.80
):
    """
    Retain high-confidence detections and suppress highly overlapping
    detections of the same object class.
    """

    # Process the highest-confidence detections first.
    confidence_order = sorted(
        range(len(boxes)),
        key=lambda index: confidences[index],
        reverse=True
    )

    kept_indices = []
    suppressed_detections = []

    for candidate_index in confidence_order:
        duplicate_found = False

        for kept_index in kept_indices:
            # Only compare detections belonging to the same class
            if class_ids[candidate_index] != class_ids[kept_index]:
                continue

            overlap = calculate_iou(
                boxes[candidate_index],
                boxes[kept_index]
            )

            if overlap >= iou_threshold:
                duplicate_found = True

                suppressed_detections.append(
                    {
                        "suppressed_index": candidate_index,
                        "kept_index": kept_index,
                        "iou": overlap
                    }
                )

                break

        if not duplicate_found:
            kept_indices.append(candidate_index)

    return kept_indices, suppressed_detections


IMAGE_PATH = Path("data/output/first_frame.jpg")
CLEANED_IMAGE_PATH = Path("data/output/first_frame_cleaned.jpg")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH.resolve()}"
    )

annotated_frame = cv2.imread(str(IMAGE_PATH))

if annotated_frame is None:
    raise RuntimeError(
        f"OpenCV could not read the image: {IMAGE_PATH.resolve()}"
    )

# Load the pretrained detector
model = YOLO("yolo26n.pt")

# Run inference on the extracted video frame
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    imgsz=960,
    save=True,
    project=str(Path("data/output").resolve()),
    name="frame_detection_960"
)

result = results[0]

print(f"\nFrame: {IMAGE_PATH.resolve()}")
print(f"Original frame shape: {result.orig_shape}")
print(f"Number of detections: {len(result.boxes)}\n")

for detection_id, box in enumerate(result.boxes, start=1):
    class_id = int(box.cls.item())
    class_name = result.names[class_id]
    confidence = float(box.conf.item())

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    box_width = x2 - x1
    box_height = y2 - y1

    print(
        f"Detection #{detection_id}\n"
        f"Class: {class_name:12s} "
        f"Confidence: {confidence:.3f} "
        f"Center ({center_x:.1f}, {center_y:.1f})"
        f"Box:        "
        f"({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f}))\n"
        f"  Size:    {box_width:.1f} x {box_height:.1f}\n"
    )

boxes = result.boxes.xyxy.cpu().tolist()
confidences = result.boxes.conf.cpu().tolist()
class_ids = result.boxes.cls.int().cpu().tolist()

kept_indices, suppressed_detections = suppress_duplicate_detections(
    boxes=boxes,
    confidences=confidences,
    class_ids=class_ids,
    iou_threshold=0.80
)

print("\nSuspected duplicate pairs:")

duplicate_found = False

for first_index in range(len(boxes)):
    for second_index in range(first_index + 1, len(boxes)):
        overlap = calculate_iou(
            boxes[first_index],
            boxes[second_index]
        )

        if overlap >= 0.80:
            duplicate_found = True

            print(
                f"  Detection #{first_index + 1} and "
                f"Detection #{second_index + 1}: "
                f"IoU = {overlap:.3f}"
            )

if not duplicate_found:
    print("  None")

print("\nDuplicate suppression results:")
print(f"  Raw detections:    {len(boxes)}")
print(f"  Retained detections:    {len(kept_indices)}")
print(f"  Suppressed detections:    {len(suppressed_detections)}")

for suppression in suppressed_detections:
    suppressed_number = suppression["suppressed_index"] + 1
    kept_number = suppression["kept_index"] + 1
    overlap = suppression["iou"]

    print(
        f"  Suppressed Detection #{suppressed_number} "
        f"in favor of Detection #{kept_number} "
        f"(IoU = {overlap:.3f})"

    )

print("\nFinal retained detections:")

for original_index in kept_indices:
    class_id = class_ids[original_index]
    class_name = result.names[class_id]
    confidence = confidences[original_index]

    x1, y1, x2, y2 = boxes[original_index]

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    print(
        f"  Original Detection #{original_index + 1}: "
        f"{class_name}, "
        f"confidence={confidence:.3f}, "
        f"center=({center_x:.1f}, {center_y:.1f})"
    )

for original_index in kept_indices:
    class_id = class_ids[original_index]
    class_name = result.names[class_id]
    confidence = confidences[original_index]

    x1, y1, x2, y2 = boxes[original_index]

    # OpenCV drawing functions require integer pixel coordinates
    x1_int = int(round(x1))
    y1_int = int(round(y1))
    x2_int = int(round(x2))
    y2_int = int(round(y2))

    center_x = int(round((x1 + x2) / 2))
    center_y = int(round(y1 + y2) / 2)

    label = f"{class_name} {confidence:.2f}"

    # Draw the bounding box
    cv2.rectangle(
        annotated_frame,
        (x1_int, y1_int),
        (x2_int, y2_int),
        (255, 0, 0),
        thickness=3
    )

    # Draw the center point
    cv2.circle(
        annotated_frame,
        (center_x, center_y),
        radius=6,
        color=(0, 0, 255),
        thickness=-1
    )

    # Keep the label inside the image
    label_y = max(y1_int - 10, 25)

    cv2.putText(
        annotated_frame,
        label,
        (x1_int, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA
    )

CLEANED_IMAGE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

saved_successfully = cv2.imwrite(
    str(CLEANED_IMAGE_PATH),
    annotated_frame
)

if not saved_successfully:
    raise RuntimeError(
        "OpenCV could not save the cleaned annotated image."
    )

print(
    f"\nCleaned annotated image saved to: "
    f"{CLEANED_IMAGE_PATH.resolve()}"
)