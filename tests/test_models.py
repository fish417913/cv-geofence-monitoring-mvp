import pytest 

from geofence_monitor.models import (
    BoundingBox,
    CrossingDirection,
    CrossingEvent,
    Geofence,
    Point,
    TrackObservation
)

def make_valid_observation() -> TrackObservation:
    """Create a valid observation for use in multiple tests"""
    
    return TrackObservation(
        track_id=7,
        class_id=0,
        class_name="person",
        confidence=0.84,
        bounding_box=BoundingBox(
            x1=100.0,
            y1=200.0,
            x2=300.0,
            y2=500.0
        ),
        frame_number=50,
        timestamp_seconds=2.0
    )
    
def test_track_observation_stores_expected_values() -> None:
    observation = make_valid_observation()
    
    assert observation.track_id == 7
    assert observation.class_id == 0 
    assert observation.class_name == "person"
    assert observation.confidence == 0.84
    assert observation.frame_number == 50
    assert observation.timestamp_seconds == 2.0
    
def test_track_observation_rejects_negative_track_id() -> None:
    with pytest.raises(
        ValueError,
        match="track_id must be zero or greater"
    ):
        TrackObservation(
            track_id=-1,
            class_id=0,
            class_name="person",
            confidence=0.84,
            bounding_box=BoundingBox(
                x1=100.0,
                y1=200.0,
                x2=300.0,
                y2=500.0
            ),
            frame_number=50,
            timestamp_seconds=2.0
        )
        
def test_track_observation_rejects_empty_class_name() -> None:
    with pytest.raises(
        ValueError,
        match="class_name must not be empty"
    ):
        TrackObservation(
            track_id=7,
            class_id=0,
            class_name="  ",
            confidence=0.84,
            bounding_box=BoundingBox(
                x1=100.0,
                y1=200.0,
                x2=300.0,
                y2=500.0
            ),
            frame_number=50,
            timestamp_seconds=2.0
        )
        
def test_track_observation_rejects_invalid_confidence() -> None:
    with pytest.raises(
        ValueError,
        match="confidence must be between 0.0 and 1.0"
    ):
        TrackObservation(
            track_id=7,
            class_id=0,
            class_name="person",
            confidence=1.25,
            bounding_box=BoundingBox(
                x1=100.0,
                y1=200.0,
                x2=300.0,
                y2=500.0
            ),
            frame_number=50,
            timestamp_seconds=2.0
        )
        
def test_point_stores_expected_coordinates() -> None:
    point = Point(
        x=200.0,
        y=500.0
    )
    
    assert point.x == 200.0
    assert point.y == 500.0
    
def test_point_accepts_origin() -> None:
    point = Point(
        x=0.0,
        y=0.0
    )
    
    assert point == Point(
        x=0.0,
        y=0.0
    )
    
def test_point_rejects_negative_x_coordinate() -> None:
    with pytest.raises(
        ValueError,
        match="x must be zero or greater"
    ):
        Point(
            x=-1.0,
            y=100.0
        )
        
def test_point_rejects_negative_y_coordinate() -> None:
    with pytest.raises(
        ValueError,
        match="y must be zero or greater"
    ):
        Point(
            x=100.0,
            y=-1.0
        )
        
def make_valid_geofence() -> Geofence:
    """Create a valid polygon geofence for use in tests"""
    
    return Geofence(
        geofence_id="sidewalk_zone",
        name="Sidewalk Zone",
        points=(
            Point(x=100.0, y=800.0),
            Point(x=900.0, y=800.0),
            Point(x=1000.0, y=1800.0),
            Point(x=50.0, y=1800.0)
        ),
        frame_width=1080,
        frame_height=1920
    )
    
def test_geofence_stores_expected_values() -> None:
    geofence = make_valid_geofence()
    
    assert geofence.geofence_id == "sidewalk_zone"
    assert geofence.name == "Sidewalk Zone"
    assert len(geofence.points) == 4
    assert geofence.frame_width == 1080
    assert geofence.frame_height == 1920
    
def test_geofence_requires_at_least_three_points() -> None:
    with pytest.raises(
        ValueError,
        match="requires at least three points"
    ):
        Geofence(
            geofence_id="invalid_zone",
            name="Invalid Zone",
            points=(
                Point(x=100.0, y=100.0),
                Point(x=500.0, y=500.0)
            ),
            frame_width=1080,
            frame_height=1920
        )
        
def test_geofence_rejects_point_beyond_frame_width() -> None:
    with pytest.raises(
        ValueError,
        match="x coordinate must be inside the frame"
    ):
        Geofence(
            geofence_id="invalid_zone",
            name="Invalid Zone",
            points=(
                Point(x=100.0, y=100.0),
                Point(x=1080.0, y=100.0),
                Point(x=500.0, y=500.0)
            ),
            frame_width=1080,
            frame_height=1920
        )
        
def test_geofence_rejects_point_beyond_frame_height() -> None:
    with pytest.raises(
        ValueError,
        match="y coordinate must be inside the frame"
    ):
        Geofence(
            geofence_id="invalid_zone",
            name="Invalid Zone",
            points=(
                Point(x=100.0, y=100.0),
                Point(x=500.0, y=1920.0),
                Point(x=500.0, y=500.0)
            ),
            frame_width=1080,
            frame_height=1920
        )
        
def test_geofence_rejects_non_positive_frame_width() -> None:
    with pytest.raises(
        ValueError,
        match="frame_width must be greater than zero"
    ):
        Geofence(
            geofence_id="invalid_zone",
            name="Invalid Zone",
            points=(
                Point(x=0.0, y=0.0),
                Point(x=0.0, y=100.0),
                Point(x=100.0, y=100.0)
            ),
            frame_width=0,
            frame_height=1920
        )
        
def make_valid_crossing_event() -> CrossingEvent:
    """Create a valid crossing event for use in tests"""
    
    return CrossingEvent(
        event_id="test_zone:7:25:entry",
        geofence_id="test_zone",
        track_id=7,
        object_class="person",
        direction=CrossingDirection.ENTRY,
        frame_number=25,
        timestamp_seconds=1.0,
        confidence=0.90,
        bounding_box=BoundingBox(
            x1=250.0,
            y1=100.0,
            x2=350.0,
            y2=300.0
        ),
        anchor_point=Point(
            x=300.0,
            y=300.0
        )
    )
    
def test_crossing_event_stores_expected_values() -> None:
    event = make_valid_crossing_event()
    
    assert event.event_id == "test_zone:7:25:entry"
    assert event.geofence_id == "test_zone"
    assert event.track_id == 7
    assert event.object_class == "person"
    assert event.direction is CrossingDirection.ENTRY
    assert event.frame_number == 25 
    assert event.timestamp_seconds == 1.0 
    assert event.confidence == 0.90
    assert event.anchor_point == Point(
        x=300.0,
        y=300.0
    )
    
def test_crossing_event_rejects_empty_event_id() -> None:
    with pytest.raises(
        ValueError,
        match="event_id must not be empty"
    ):
        CrossingEvent(
            event_id="  ",
            geofence_id="test_zone",
            track_id=7,
            object_class="person",
            direction=CrossingDirection.ENTRY,
            frame_number=25,
            timestamp_seconds=1.0,
            confidence=0.90,
            bounding_box=BoundingBox(
                x1=250.0,
                y1=100.0,
                x2=350.0,
                y2=300.0
            ),
            anchor_point=Point(
                x=300.0,
                y=300.0
            )
        )
        
def test_crossing_event_rejects_negative_frame_number() -> None:
    with pytest.raises(
        ValueError,
        match="frame_number must be zero or greater"
    ):
        CrossingEvent(
                    event_id="test_zone:7:-1:entry",
                    geofence_id="test_zone",
                    track_id=7,
                    object_class="person",
                    direction=CrossingDirection.ENTRY,
                    frame_number=-1,
                    timestamp_seconds=1.0,
                    confidence=0.90,
                    bounding_box=BoundingBox(
                        x1=250.0,
                        y1=100.0,
                        x2=350.0,
                        y2=300.0
                    ),
                    anchor_point=Point(
                        x=300.0,
                        y=300.0
                    )
                )
        
def test_crossing_event_rejects_invalid_confidence() -> None:
    with pytest.raises(
        ValueError,
        match="confidence must be between 0.0 and 1.0"
    ):
        CrossingEvent(
                    event_id="test_zone:7:25:entry",
                    geofence_id="test_zone",
                    track_id=7,
                    object_class="person",
                    direction=CrossingDirection.ENTRY,
                    frame_number=25,
                    timestamp_seconds=1.0,
                    confidence=-0.10,
                    bounding_box=BoundingBox(
                        x1=250.0,
                        y1=100.0,
                        x2=350.0,
                        y2=300.0
                    ),
                    anchor_point=Point(
                        x=300.0,
                        y=300.0
                    )
                )
        