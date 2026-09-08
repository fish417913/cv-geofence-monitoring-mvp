import numpy as np 
import torch
from pathlib import Path 
import cv2 
import torch 
import pytest 

from geofence_monitor.inference import (
    frame_to_tensor, run_inference, infer_frame, infer_video, filter_detections
)

def test_frame_to_tensor_converts_shape_and_dtype():
    frame = np.zeros((48,64,3), dtype=np.uint8)
    
    tensor = frame_to_tensor(frame)
    
    assert tensor.shape == (3,48,64)
    assert tensor.dtype == torch.float32 
    assert tensor.min().item() == 0.0 
    assert tensor.max().item() == 0.0 
    

class DummyDetector(torch.nn.Module):
    def forward(self, images):
        image = images[0]
        device = image.device 
        categories = {
                        1: "person"
                    }
        
        return [{
            "boxes": torch.tensor(
                [[10.0, 8.0, 40.0, 36.0]],
                device=device
            ),
            "labels": torch.tensor(
                [1],
                dtype=torch.int64,
                device=device
            ),
            
            "scores": torch.tensor(
                [0.90],
                device=device 
            )
        }]
    
def test_run_inference_returns_detection():
    model = DummyDetector()
    tensor = torch.ones((3,48,64))
    
    output = run_inference(model, tensor)
    
    assert "boxes" in output 
    assert "labels" in output 
    assert "scores" in output 
    assert output["boxes"].shape == (1, 4)
    assert output["labels"].item() == 1
    assert output["scores"].item() == pytest.approx(0.90) 
    
def test_infer_frame_processes_opencv_frame():
    model = DummyDetector()
    frame = np.full((48,64,3), 255, dtype=np.uint8)
    
    output = infer_frame(model, frame)
    
    assert output["boxes"].shape == (1, 4)
    assert output["labels"].item() == 1
    assert output["scores"].item() == pytest.approx(0.90)
    
def create_test_video(video_path: Path) -> None:
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48)
    )
    
    for value in (0, 127, 255):
        frame = np.full((48, 64, 3), value, dtype=np.uint8)
        writer.write(frame)
        
    writer.release()
    
def test_infer_video_returns_one_output_per_frame(tmp_path):
    video_path = tmp_path / "test_video.avi"
    create_test_video(video_path)
    
    model = DummyDetector()
    categories = {1: "person"}
    
    results = infer_video(model, 
                          video_path,
                          categories,
                          confidence_threshold=0.50,
                          allowed_classes={"person"}
                          )
    
    assert len(results) == 3
    
    assert results[0]["frame_number"] == 0
    assert results[0]["timestamp_seconds"] == pytest.approx(0.0)
    
    assert len(results[0]["detections"]) == 1
    assert results[0]["detections"][0]["class_name"] == "person"
    assert results[0]["detections"][0]["confidence"] == pytest.approx(0.90)
    
def test_filter_detections_applies_confidence_and_class_filters():
    output = {
        "boxes": torch.tensor([
            [10.0, 10.0, 30.0, 40.0],
            [20.0, 20.0, 50.0, 60.0],
            [30.0, 30.0, 70.0, 80.0]
        ]),
        "labels": torch.tensor([1, 3, 13]),
        "scores": torch.tensor([0.90, 0.80, 0.95])
    }
    
    categories = {
        1: "person",
        3: "car",
        13: "bench"
    }
    
    detections = filter_detections(
        output,
        categories,
        confidence_threshold=0.85,
        allowed_classes={"person", "car"}
    )
    
    assert len(detections) == 1
    assert detections[0]["class_name"] == "person"
    assert detections[0]["confidence"] == pytest.approx(0.90)
    assert detections[0]["bbox_xyxy"] == [10.0, 10.0, 30.0, 40.0]

def test_filter_detections_keeps_multiple_valid_detections():
    output = {
        "boxes": torch.tensor([
            [10.0, 10.0, 30.0, 40.0],
            [20.0, 20.0, 50.0, 60.0],
        ]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.90, 0.85]),
    }

    categories = {1: "person"}

    detections = filter_detections(
        output,
        categories,
        confidence_threshold=0.50,
        allowed_classes={"person"},
    )

    assert len(detections) == 2
    
def test_filter_detections_returns_empty_list_when_none_pass():
    output = {
        "boxes": torch.tensor([[10.0, 10.0, 30.0, 40.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.20]),
    }

    categories = {1: "person"}

    detections = filter_detections(
        output,
        categories,
        confidence_threshold=0.50,
        allowed_classes={"person"},
    )

    assert detections == []