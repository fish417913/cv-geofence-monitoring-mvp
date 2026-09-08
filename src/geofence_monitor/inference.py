import cv2 
import torch 

from geofence_monitor.ingestion import iter_video_frames

from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn
)

def load_detector():
    """Load a pretrained Faster R-CNN object detector."""
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights
    )
    
    return model, weights 

def frame_to_tensor(frame):
    """Convert an OpenCV BGR frame into a PyTorch RGB tensor."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    tensor = torch.from_numpy(rgb_frame)
    tensor = tensor.permute(2,0,1)
    tensor = tensor.float() / 255.0
    
    return tensor 

def run_inference(model, tensor, device="cpu"):
    """Run a preprocessed tensor through a PyTorch model."""
    model = model.to(device)
    tensor = tensor.to(device)
    
    model.eval()
    
    with torch.inference_mode():
        output = model([tensor])[0]
        
    return output 

def infer_frame(model, frame, device="cpu"):
    """Preprocess one OpenCV frame and run model inference."""
    tensor = frame_to_tensor(frame)
    output = run_inference(model, tensor, device=device)
    return output 

def infer_video(
    model, 
    video_path, 
    categories,
    confidence_threshold=0.50,
    allowed_classes=None,
    device="cpu"
    ):
    
    """Run filtered object detection over every frame in a video."""
    results = []
    
    model = model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for video_frame in iter_video_frames(video_path):
            tensor = frame_to_tensor(video_frame.frame)
            tensor = tensor.to(device)
            
            
            output = model([tensor])[0]
            
            detections = filter_detections(
                output,
                categories,
                confidence_threshold=confidence_threshold,
                allowed_classes=allowed_classes
            )
            
            results.append(
                {
                    "frame_number": video_frame.frame_number,
                    "timestamp_seconds": video_frame.timestamp_seconds,
                    "detections": detections
                }
            )
        
    return results 

def filter_detections(
    output,
    categories,
    confidence_threshold=0.50,
    allowed_classes=None
    
):
    """Keep detections that meet confidence and class requirements."""
    detections = []
    
    for label, score, box in zip(
        output["labels"],
        output["scores"],
        output["boxes"]
    ):
        class_name = categories[label.item()]
        
        if score.item() < confidence_threshold:
            continue 
        
        if allowed_classes and class_name not in allowed_classes:
            continue 
        
        detections.append(
            {
                "class_name": class_name, 
                "confidence": score.item(),
                "bbox_xyxy": box.tolist()
            }
        )
        
    return detections 