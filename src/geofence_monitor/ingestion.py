from dataclasses import dataclass 
from pathlib import Path 
from collections.abc import Iterator
import cv2, numpy as np

@dataclass(frozen=True)
class VideoFrame:
    frame: np.ndarray
    frame_number: int 
    timestamp_seconds: float 
    source_video: Path 
    
def iter_video_frames(video_path: Path) -> Iterator[VideoFrame]:
    """Yield decoded video frames with ingestion metadata."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError("Video reports an invalid frame rate.")
    
    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success: break 
            yield VideoFrame(
                frame=frame,
                frame_number=frame_number,
                timestamp_seconds=frame_number / fps,
                source_video=video_path
            )
            frame_number += 1
            
    finally:
        capture.release()