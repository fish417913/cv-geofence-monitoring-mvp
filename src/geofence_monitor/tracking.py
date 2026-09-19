import numpy as np 
import supervision as sv 

from geofence_monitor.models import BoundingBox, TrackObservation

def build_track_observation(
    track_id,
    detection,
    frame_number,
    timestamp_seconds
):
    """Convert one tracked detection into a TrackObservation"""
    
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    
    bounding_box = BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2
    )
    
    return TrackObservation(
        track_id=track_id,
        class_id=detection["class_id"],
        class_name=detection["class_name"],
        confidence=detection["confidence"],
        bounding_box=bounding_box,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds
    )
    
def detections_to_supervision(detections):
    """Convert filtered detections into Supervision's detection format"""
    
    xyxy = np.array(
        [detection["bbox_xyxy"] for detection in detections],
        dtype=float 
    ).reshape(-1, 4)
    
    confidence = np.array(
        [detection["confidence"] for detection in detections],
        dtype=float 
    )
    
    class_id = np.array(
        [detection["class_id"] for detection in detections],
        dtype=int
    )
    
    class_name = np.array(
        [detection["class_name"] for detection in detections]
    )
    
    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={
            "class_name": class_name 
        }
    )
    
def tracked_detections_to_observations(
    tracked,
    frame_number,
    timestamp_seconds
):
    """Convert confirmed ByteTrack detections into TrackObservation objects"""
    
    observations = []
    
    for index, track_id in enumerate(tracked.tracker_id):
        if track_id < 0:
            continue 
        
        detection = {
            "class_id": int(tracked.class_id[index]),
            "class_name": str(tracked.data["class_name"][index]),
            "confidence": float(tracked.confidence[index]),
            "bbox_xyxy": tracked.xyxy[index].tolist()
        }
        
        observation = build_track_observation(
            track_id=int(track_id),
            detection=detection,
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds
        )
        
        observations.append(observation)
        
    return observations