from pathlib import Path

import cv2

VIDEO_PATH = Path("data/input/sample.mp4")
OUTPUT_FRAME_PATH = Path("data/output/first_frame.jpg")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH.resolve()}\n"
        "Place an MP4 video at data/input/sample.mp4."
    )

# VideoCapture represents the connection to the video file
capture = cv2.VideoCapture(str(VIDEO_PATH))

if not capture.isOpened():
    raise RuntimeError(
        f"OpenCV could not open the video: {VIDEO_PATH.resolve()}"
    )

# Read basic video metadata
fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Duration can be estimated from frame count and frame rate
duration_seconds = frame_count / fps if fps > 0 else 0.0

# Read one frame from the video
read_successfully, first_frame = capture.read()

# Always release the video resource when finished
capture.release()

if not read_successfully or first_frame is None:
    raise RuntimeError("OpenCV opened the video but could not read its first frame.")

OUTPUT_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)

saved_successfully = cv2.imwrite(
    str(OUTPUT_FRAME_PATH),
    first_frame
)

if not saved_successfully:
    raise RuntimeError(
        f"OpenCV could not save the frame to {OUTPUT_FRAME_PATH.resolve()}"
    )

print(f"Video: {VIDEO_PATH.resolve()}")
print(f"Resolution: {frame_width} x {frame_height}")
print(f"Frames per second: {fps:.2f}")
print(f"Total frames: {frame_count}")
print(f"Approximate duration: {duration_seconds:.2f} seconds")
print(f"First frame shape: {first_frame.shape}")
print(f"Saved first frame: {OUTPUT_FRAME_PATH.resolve()}")
