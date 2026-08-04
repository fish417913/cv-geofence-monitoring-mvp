import pytest 

from geofence_monitor.crossing import ( 
    CrossingEngine,
    classify_crossing,
    update_track_geofence_state
)
from geofence_monitor.models import (
    BoundingBox,
    CrossingDirection,
    CrossingEvent,
    Geofence,
    Point,
    PointLocation,
    TrackGeofenceState,
    TrackObservation
)

def test_outside_to_inside_is_entry() -> None:
    result = classify_crossing(
        previous_location=PointLocation.OUTSIDE,
        current_location=PointLocation.INSIDE
    )
    
    assert result is CrossingDirection.ENTRY 
    
def test_inside_to_outside_is_exit() -> None:
    result = classify_crossing(
        previous_location=PointLocation.INSIDE,
        current_location=PointLocation.OUTSIDE
    )
    
    assert result is CrossingDirection.EXIT 
    
def test_remaining_outside_is_not_crossing() -> None:
    result = classify_crossing(
        previous_location=PointLocation.OUTSIDE,
        current_location=PointLocation.OUTSIDE
    )
    
    assert result is None 
    
def test_remaining_inside_is_not_crossing() -> None:
    result = classify_crossing(
        previous_location=PointLocation.INSIDE,
        current_location=PointLocation.INSIDE
    )
    
    assert result is None 
    
def test_moving_to_boundary_is_not_confirmed_crossing() -> None:
    result = classify_crossing(
        previous_location=PointLocation.OUTSIDE,
        current_location=PointLocation.BOUNDARY
    )
    
    assert result is None 
    
def test_moving_from_boundary_is_not_direct_crossing() -> None:
    result = classify_crossing(
        previous_location=PointLocation.BOUNDARY,
        current_location=PointLocation.INSIDE
    )
    
    assert result is None 
    
def test_first_observation_initializes_state_without_event() -> None:
    state = TrackGeofenceState()
    
    result = update_track_geofence_state(
        state=state,
        current_location=PointLocation.OUTSIDE
    )
    
    assert result is None 
    assert state.last_confirmed_location is PointLocation.OUTSIDE
    
def test_outside_boundary_inside_sequence_is_entry() -> None:
    state = TrackGeofenceState()
    
    first_result = update_track_geofence_state(
        state=state,
        current_location=PointLocation.OUTSIDE
    )
    
    boundary_result = update_track_geofence_state(
        state=state,
        current_location=PointLocation.BOUNDARY
    )
    
    entry_result = update_track_geofence_state(
        state=state,
        current_location=PointLocation.INSIDE
    )
    
    assert first_result is None 
    assert boundary_result is None 
    assert entry_result is CrossingDirection.ENTRY
    assert state.last_confirmed_location is PointLocation.INSIDE 
    
def test_inside_boundary_outside_sequence_is_exit() -> None:
    state = TrackGeofenceState()
    
    update_track_geofence_state(
        state=state,
        current_location=PointLocation.INSIDE
    )
    
    update_track_geofence_state(
        state=state,
        current_location=PointLocation.BOUNDARY
    )
    
    result = update_track_geofence_state(
        state=state,
        current_location=PointLocation.OUTSIDE
    )
    
    assert result is CrossingDirection.EXIT 
    assert state.last_confirmed_location is PointLocation.OUTSIDE 
    
def test_boundary_does_not_replace_confirmed_location() -> None:
    state = TrackGeofenceState(
        last_confirmed_location=PointLocation.OUTSIDE
    )
    
    result = update_track_geofence_state(
        state=state,
        current_location=PointLocation.BOUNDARY
    )
    
    assert result is None 
    assert state.last_confirmed_location is PointLocation.OUTSIDE
    
def test_repeated_inside_observations_do_not_repeat_entry() -> None:
    state = TrackGeofenceState(
        last_confirmed_location=PointLocation.OUTSIDE
    )
    
    first_result = update_track_geofence_state(
        state=state,
        current_location=PointLocation.INSIDE
    )
    
    second_result = update_track_geofence_state(
        state=state,
        current_location=PointLocation.INSIDE
    )
    
    third_result = update_track_geofence_state(
        state=state,
        current_location=PointLocation.INSIDE
    )
    
    assert first_result is CrossingDirection.ENTRY
    assert second_result is None 
    assert third_result is None 
    
def test_crossing_engine_initializes_new_track() -> None:
    engine = CrossingEngine()
    
    result = engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE
    )
    
    state = engine.get_state(7)
    
    assert result is None 
    assert state is not None 
    assert state.last_confirmed_location is PointLocation.OUTSIDE 
    assert engine.active_track_count == 1
    
def test_crossing_engine_detects_entry_for_one_track() -> None:
    engine = CrossingEngine()
    
    engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE
    )
    
    result = engine.update(
        track_id=7,
        current_location=PointLocation.INSIDE
    )
    
    assert result is CrossingDirection.ENTRY 
    
def test_crossing_engine_keeps_tracks_independent() -> None:
    engine = CrossingEngine()
    
    engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE
    )
    
    engine.update(
        track_id=12,
        current_location=PointLocation.INSIDE
    )
    
    track_7_result = engine.update(
        track_id=7,
        current_location=PointLocation.INSIDE
    )
    
    track_12_result = engine.update(
        track_id=12,
        current_location=PointLocation.INSIDE
    )
    
    assert track_7_result is CrossingDirection.ENTRY 
    assert track_12_result is None 
    
    assert (
        engine.get_state(7).last_confirmed_location
        is PointLocation.INSIDE
    )
    
    assert (
        engine.get_state(12).last_confirmed_location
        is PointLocation.INSIDE 
    )
    
    assert engine.active_track_count == 2
    
def test_crossing_engine_rejects_negative_track_id() -> None:
    engine = CrossingEngine()
    
    with pytest.raises(
        ValueError,
        match="track_id must be zero or greater"
    ):
        engine.update(
            track_id=-1,
            current_location=PointLocation.OUTSIDE
        )
        
def make_test_geofence() -> Geofence:
    """Create a square geofence for crossing tests"""
    
    return Geofence(
        geofence_id="test_zone",
        name="Test Zone",
        points=(
            Point(x=100.0, y=100.0),
            Point(x=500.0, y=100.0),
            Point(x=500.0, y=500.0),
            Point(x=100.0, y=500.0)
        ),
        frame_width=640,
        frame_height=640
    )
    
def make_observation(
    track_id: int,
    bounding_box: BoundingBox,
    frame_number: int
) -> TrackObservation:
    """Create a tracked observation for crossing tests"""
    
    return TrackObservation(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.90,
        bounding_box=bounding_box,
        frame_number=frame_number,
        timestamp_seconds=frame_number / 25.0
    )
    
def test_observation_outside_initializes_track_without_event() -> None:
    engine = CrossingEngine()
    geofence = make_test_geofence()
    
    outside_observation = make_observation(
        track_id=7,
        bounding_box=BoundingBox(
            x1=20.0,
            y1=100.0,
            x2=80.0,
            y2=300.0
        ),
        frame_number=0
    )
    
    result = engine.update_observation(
        observation=outside_observation,
        geofence=geofence 
    )
    
    state = engine.get_state(7)
    
    assert result is None 
    assert state is not None 
    assert state.last_confirmed_location is PointLocation.OUTSIDE 
    
def test_observations_crossing_into_geofence_generate_entry() -> None:
    engine = CrossingEngine()
    geofence = make_test_geofence()
    
    outside_observation = make_observation(
        track_id=7,
        bounding_box=BoundingBox(
                x1=20.0,
                y1=100.0,
                x2=80.0,
                y2=300.0
            ),
            frame_number=0
        )
        
    
    inside_observation = make_observation(
        track_id=7,
        bounding_box=BoundingBox(
            x1=250.0,
            y1=100.0,
            x2=350.0,
            y2=300.0
        ),
        frame_number=1
    )
    
    first_result = engine.update_observation(
        observation=outside_observation,
        geofence=geofence
    )
    
    second_result = engine.update_observation(
        observation=inside_observation,
        geofence=geofence 
    )
    
    assert first_result is None 
    assert second_result is CrossingDirection.ENTRY 
    
def test_observations_crossing_out_of_geofence_generate_exit() -> None:
    engine = CrossingEngine()
    geofence = make_test_geofence()
    
    inside_observation = make_observation(
        track_id=12,
        bounding_box=BoundingBox(
            x1=250.0,
            y1=100.0,
            x2=350.0,
            y2=300.0
        ),
        frame_number=0
    )
    
    outside_observation = make_observation(
        track_id=12,
        bounding_box=BoundingBox(
            x1=520.0,
            y1=100.0,
            x2=580.0,
            y2=300.0
        ),
        frame_number=1
    )
    
    engine.update_observation(
        observation=inside_observation,
        geofence=geofence
    )
    
    result = engine.update_observation(
        observation=outside_observation,
        geofence=geofence
    )
    
    assert result is CrossingDirection.EXIT
    
def test_process_observation_returns_none_without_crossing() -> None:
    engine = CrossingEngine()
    geofence = make_test_geofence()
    
    outside_observation = make_observation(
        track_id=7,
        bounding_box=BoundingBox(
            x1=20.0,
            y1=100.0,
            x2=80.0,
            y2=300.0
        ),
        frame_number=24
    )
    
    result = engine.process_observation(
        observation=outside_observation,
        geofence=geofence
    )
    
    assert result is None 
    
def test_process_observation_creates_entry_event() -> None:
    engine = CrossingEngine()
    geofence = make_test_geofence()
    
    outside_observation = make_observation(
        track_id=7,
        bounding_box=BoundingBox(
            x1=20.0,
            y1=100.0,
            x2=80.0,
            y2=300.0
        ),
        frame_number=24
    )
    
    inside_observation = make_observation(
        track_id=7,
        bounding_box=BoundingBox(
            x1=250.0,
            y1=100.0,
            x2=350.0,
            y2=300.0
        ),
        frame_number=25
    )
    
    engine.process_observation(
        observation=outside_observation,
        geofence=geofence
    )
    
    event = engine.process_observation(
        observation=inside_observation,
        geofence=geofence
    )
    
    assert isinstance(event, CrossingEvent)
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
    
def test_repeated_inside_observation_does_not_create_second_event() -> None:
    engine = CrossingEngine()
    geofence = make_test_geofence()
    
    outside_observation = make_observation(
        track_id=7,
        bounding_box=BoundingBox(
            x1=20.0,
            y1=100.0,
            x2=80.0,
            y2=300.0
        ),
        frame_number=24
    )
    
    first_inside_observation = make_observation(
        track_id=7,
        bounding_box=BoundingBox(
            x1=250.0,
            y1=100.0,
            x2=350.0,
            y2=300.0
        ),
        frame_number=25
    )
    
    second_inside_observation = make_observation(
        track_id=7,
        bounding_box=BoundingBox(
            x1=260.0,
            y1=100.0,
            x2=360.0,
            y2=300.0
        ),
        frame_number=26
    )
    
    engine.process_observation(
        observation=outside_observation,
        geofence=geofence
    )
    
    first_event = engine.process_observation(
        observation=first_inside_observation,
        geofence=geofence
    )
    
    second_event = engine.process_observation(
        observation=second_inside_observation,
        geofence=geofence
    )
    
    assert first_event is not None 
    assert first_event.direction is CrossingDirection.ENTRY
    assert second_event is None 
    
def test_debouncing_requires_multiple_inside_observations() -> None:
    engine = CrossingEngine(
        stable_frames=3
    )
    
    engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE
    )
    
    first_result = engine.update(
        track_id=7,
        current_location=PointLocation.INSIDE
    )
    
    second_result = engine.update(
        track_id=7,
        current_location=PointLocation.INSIDE
    )
    
    third_result = engine.update(
        track_id=7,
        current_location=PointLocation.INSIDE
    )
    
    assert first_result is None 
    assert second_result is None 
    assert third_result is CrossingDirection.ENTRY
    
def test_jitter_back_to_confirmed_side_cancels_candidate() -> None:
    engine = CrossingEngine(
        stable_frames=3
    )
    
    engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE
    )
    
    first_inside_result = engine.update(
        track_id=7,
        current_location=PointLocation.INSIDE
    )
    
    second_inside_result = engine.update(
        track_id=7,
        current_location=PointLocation.INSIDE
    )
    
    outside_result = engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE
    )
    
    state = engine.get_state(7)
    
    assert first_inside_result is None 
    assert second_inside_result is None 
    assert outside_result is None 
    
    assert state is not None 
    assert state.last_confirmed_location is PointLocation.OUTSIDE
    assert state.candidate_location is None 
    assert state.candidate_frame_count == 0
    
def test_alternating_locations_do_not_confirm_crossing() -> None:
    engine = CrossingEngine(
        stable_frames=3
    )
    
    engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE
    )
    
    results = [
        engine.update(
            track_id=7,
            current_location=PointLocation.INSIDE
        ),
        engine.update(
            track_id=7,
            current_location=PointLocation.OUTSIDE
        ),
        engine.update(
            track_id=7,
            current_location=PointLocation.INSIDE
        ),
        engine.update(
            track_id=7,
            current_location=PointLocation.OUTSIDE
        ),
        engine.update(
            track_id=7,
            current_location=PointLocation.INSIDE
        )
    ]
    
    assert all(result is None for result in results)
    
def test_debounced_exit_requires_multiple_outside_observations() -> None:
    engine = CrossingEngine(
        stable_frames=2
    )
    
    engine.update(
        track_id=12,
        current_location=PointLocation.INSIDE
    )
    
    first_result = engine.update(
        track_id=12,
        current_location=PointLocation.OUTSIDE
    )
    
    second_result = engine.update(
        track_id=12,
        current_location=PointLocation.OUTSIDE
    )
    
    assert first_result is None 
    assert second_result is CrossingDirection.EXIT 
    
def test_crossing_engine_rejects_invalid_stable_frames() -> None:
    with pytest.raises(
        ValueError,
        match="stable_frames must be at least 1"
    ):
        CrossingEngine(
            stable_frames=0
        )
        
def test_engine_records_last_seen_frame() -> None:
    engine = CrossingEngine()
    
    engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE,
        frame_number=100
    )
    
    state = engine.get_state(7)
    
    assert state is not None 
    assert state.last_seen_frame == 100
    
def test_remove_stale_tracks_removes_old_track() -> None:
    engine = CrossingEngine()
    
    engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE,
        frame_number=100
    )
    
    removed_track_ids = engine.remove_stale_tracks(
        current_frame=176,
        max_age_frames=75
    )
    
    assert removed_track_ids == [7]
    assert engine.get_state(7) is None 
    assert engine.active_track_count == 0
    
def test_remove_stale_tracks_keeps_recent_track() -> None:
    engine = CrossingEngine()
    
    engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE,
        frame_number=100
    )
    
    removed_track_ids = engine.remove_stale_tracks(
        current_frame=175,
        max_age_frames=75
    )
    
    assert removed_track_ids == []
    assert engine.get_state(7) is not None
    assert engine.active_track_count == 1
    
def test_remove_stale_tracks_only_removes_expired_tracks() -> None:
    engine = CrossingEngine()
    
    engine.update(
        track_id=7,
        current_location=PointLocation.OUTSIDE,
        frame_number=100
    )
    
    engine.update(
        track_id=12,
        current_location=PointLocation.INSIDE,
        frame_number=160
    )
    
    removed_track_ids = engine.remove_stale_tracks(
        current_frame=176,
        max_age_frames=75
    )
    
    assert removed_track_ids == [7]
    assert engine.get_state(7) is None 
    assert engine.get_state(12) is not None 
    assert engine.active_track_count == 1
    
def test_remove_stale_tracks_rejects_negative_current_frame() -> None:
    engine = CrossingEngine()
    
    with pytest.raises(
        ValueError,
        match="current_frame must be zero or greater"
    ):
        engine.remove_stale_tracks(
            current_frame=-1,
            max_age_frames=75
        )
        
def test_remove_stale_tracks_rejects_negative_max_age() -> None:
    engine = CrossingEngine()
    
    with pytest.raises(
        ValueError,
        match="max_age_frames must be zero or greater"
    ):
        engine.remove_stale_tracks(
            current_frame=100,
            max_age_frames=-1
        )