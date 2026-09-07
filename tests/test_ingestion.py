import pytest 
from pathlib import Path 


import cv2 
import numpy as np 

from geofence_monitor.ingestion import iter_video_frames

def create_test_video(video_path: Path) -> None:
    """Create a three-frame, 10 FPS video for testing."""
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64,48)
    )
    
    for intensity in (0, 100, 200):
        frame = np.full(
            (48, 64, 3),
            intensity, 
            dtype=np.uint8
        )
        writer.write(frame)
        
    writer.release()
    
def test_iter_video_frames_reads_expected_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "test_video.avi"
    create_test_video(video_path)
    
    frames = list(iter_video_frames(video_path))
    
    assert len(frames) == 3 
    
    assert frames[0].frame_number == 0 
    assert frames[1].frame_number == 1
    assert frames[2].frame_number == 2
    
    assert frames[0].timestamp_seconds == 0.0
    assert frames[1].timestamp_seconds == 0.1
    assert frames[2].timestamp_seconds == 0.2 
    
def test_iter_video_frames_rejects_missing_file(tmp_path: Path) -> None:
    video_path = tmp_path / "missing_video.avi"
    
    with pytest.raises(
        FileNotFoundError,
        match="Video not found"
    ):
        list(iter_video_frames(video_path))