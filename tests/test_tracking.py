from trackers import ByteTrackTracker

from geofence_monitor.models import TrackObservation

from geofence_monitor.tracking import (
    build_track_observation,
    detections_to_supervision,
    tracked_detections_to_observations
)

def test_build_track_observation_creates_domain_model():
    detection = {
        "class_id": 1,
        "class_name": "person",
        "confidence": 0.91,
        "bbox_xyxy": [10.0, 20.0, 50.0, 90.0]
    }
    
    observation = build_track_observation(
        track_id=7,
        detection=detection,
        frame_number=12,
        timestamp_seconds=0.4
    )
    
    assert observation.track_id == 7
    assert observation.class_name == "person"
    assert observation.bounding_box.x1 == 10.0
    assert observation.bounding_box.y2 == 90.0
    
def test_detections_to_supervision_preserves_detection_data():
    detections = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.91,
            "bbox_xyxy": [10.0, 20.0, 50.0, 90.0]
        },
        {
            "class_id": 3,
            "class_name": "car",
            "confidence": 0.82,
            "bbox_xyxy": [100.0, 120.0, 180.0, 200.0]
        }
    ]
    
    converted = detections_to_supervision(detections)
    
    assert converted.xyxy.shape == (2, 4)
    assert converted.xyxy[0].tolist() == [10.0, 20.0, 50.0, 90.0]
    assert converted.confidence.tolist() == [0.91, 0.82]
    assert converted.class_id.tolist() == [1, 3]
    
def test_detections_to_supervision_handles_empty_frame():
    converted = detections_to_supervision([])
    
    assert converted.xyxy.shape == (0, 4)
    assert converted.confidence.shape == (0,)
    assert converted.class_id.shape == (0,)
    
def test_bytetrack_preserves_identity_across_consecutive_frames():
    tracker = ByteTrackTracker(
        minimum_consecutive_frames=1
    )
    
    frame_1 = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.91,
            "bbox_xyxy": [10.0, 20.0, 50.0, 90.0]
        }
    ]
    
    frame_2 = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.90,
            "bbox_xyxy": [12.0, 21.0, 52.0, 91.0]
        }
    ]
    
    frame_3 = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.89,
            "bbox_xyxy": [14.0, 22.0, 54.0, 92.0]
        }
    ]
    
    tracked_1 = tracker.update(
        detections_to_supervision(frame_1)
    )
    
    tracked_2 = tracker.update(
        detections_to_supervision(frame_2)
    )
    
    tracked_3 = tracker.update(
        detections_to_supervision(frame_3)
    )
    
    assert tracked_1.tracker_id[0] == -1
    assert tracked_2.tracker_id[0] >= 0 
    assert tracked_3.tracker_id[0] == tracked_2.tracker_id[0]
    
def test_detections_to_supervision_preserves_class_name():
    detections = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.91,
            "bbox_xyxy": [10.0, 20.0, 50.0, 90.0]
        }
    ]
    
    converted = detections_to_supervision(detections)
    
    assert converted.data["class_name"][0] == "person"
    
def test_bytetrack_preserves_class_name_metadata():
    tracker = ByteTrackTracker(
        minimum_consecutive_frames=1
    )
    
    frame_1 = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.91,
            "bbox_xyxy": [10.0, 20.0, 50.0, 90.0]
        }
    ]
    
    frame_2 = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.90,
            "bbox_xyxy": [12.0, 21.0, 52.0, 91.0]
        }
    ]
    
    tracker.update(
        detections_to_supervision(frame_1)
    )
    
    tracked = tracker.update(
        detections_to_supervision(frame_2)
    )
    
    assert tracked.data["class_name"][0] == "person"
    
def test_tracked_detections_to_observations_builds_confirmed_tracks():
    tracker = ByteTrackTracker(
        minimum_consecutive_frames=1
    )
    
    frame_1 = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.91,
            "bbox_xyxy": [10.0, 20.0, 50.0, 90.0]
        }
    ]
    
    frame_2 = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.90,
            "bbox_xyxy": [12.0, 21.0, 52.0, 91.0]
        }
    ]
    
    tracker.update(
        detections_to_supervision(frame_1)
    )
    
    tracked = tracker.update(
        detections_to_supervision(frame_2)
    )
    
    observations = tracked_detections_to_observations(
        tracked,
        frame_number=1,
        timestamp_seconds=0.1
    )
    
    assert len(observations) == 1
    
    observation = observations[0]
    
    assert isinstance(observation, TrackObservation)
    assert observation.track_id >= 0 
    assert observation.class_id == 1 
    assert observation.class_name == "person"
    assert observation.frame_number == 1 
    assert observation.timestamp_seconds == 0.1 
    
def test_tracked_detections_to_observations_skips_tentative_tracks():
    tracker = ByteTrackTracker(
        minimum_consecutive_frames=1
    )
    
    frame_1 = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.91,
            "bbox_xyxy": [10.0, 20.0,50.0, 90.0]
        }
    ]
    
    tracked = tracker.update(
        detections_to_supervision(frame_1)
    )
    
    observations = tracked_detections_to_observations(
        tracked,
        frame_number=0,
        timestamp_seconds=0.0
    )
    
    assert observations == []