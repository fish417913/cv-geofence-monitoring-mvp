from geofence_monitor.geometry import (
    box_bottom_center,
    locate_point_in_geofence
)

from geofence_monitor.models import (
    CrossingDirection,
    CrossingEvent,
    Geofence,
    PointLocation,
    TrackGeofenceState,
    TrackObservation
)

def classify_crossing(
    previous_location: PointLocation,
    current_location: PointLocation
) -> CrossingDirection | None:
    """Classify a direct transition across a geofence boundary"""
    
    if (
        previous_location is PointLocation.OUTSIDE
        and current_location is PointLocation.INSIDE
    ):
        return CrossingDirection.ENTRY
    
    if (
        previous_location is PointLocation.INSIDE
        and current_location is PointLocation.OUTSIDE
    ):
        return CrossingDirection.EXIT
    
    return None 

def update_track_geofence_state(
    state: TrackGeofenceState,
    current_location: PointLocation,
    stable_frames: int = 1
) -> CrossingDirection | None:
    """Update one track's geofence state and return a confirmed crossing"""
    
    if stable_frames < 1:
        raise ValueError(
            "stable_frames must be at least 1."
        )
    
    # A boundary observation does not establish either side
    if current_location is PointLocation.BOUNDARY:
        return None 
    
    # The first clear observation establishes the baseline
    if state.last_confirmed_location is None:
        state.last_confirmed_location = current_location
        state.candidate_location = None 
        state.candidate_frame_count = 0
        return None 
    
    # The object remains on its confirmed side
    if current_location is state.last_confirmed_location:
        state.candidate_location = None 
        state.candidate_frame_count = 0
        return None 
    
    # Start counting a possible new location
    if current_location is not state.candidate_location:
        state.candidate_location = current_location
        state.candidate_frame_count = 1
    else:
        state.candidate_frame_count += 1
        
    # The changed location has not persisted long enough
    if state.candidate_frame_count < stable_frames:
        return None 
    
    previous_location = state.last_confirmed_location
    
    # Confirm the candidate location
    state.last_confirmed_location = current_location
    state.candidate_location = None 
    state.candidate_frame_count = 0
 
    return classify_crossing(
        previous_location=previous_location,
        current_location=current_location
    )
    
class CrossingEngine:
    """Maintain geofence state independently for multiple tracks"""
    
    def __init__(
        self,
        stable_frames: int = 1
        ) -> None:
        if stable_frames < 1:
            raise ValueError(
                "stable_frames must be at least 1."
            )
            
        self._stable_frames = stable_frames
        self._track_states: dict[int, TrackGeofenceState] = {}
        
    def update(
        self,
        track_id: int,
        current_location: PointLocation,
        frame_number: int | None = None 
    ) -> CrossingDirection | None:
        """Update one track and return any detected crossing"""
        
        if track_id < 0:
            raise ValueError(
                "track_id must be zero or greater."
            )
            
        if frame_number is not None and frame_number < 0:
            raise ValueError(
                "frame_number must be zero or greater."
            )
            
        if track_id not in self._track_states:
            self._track_states[track_id] = TrackGeofenceState()
            
        state = self._track_states[track_id]
        
        if frame_number is not None:
            state.last_seen_frame = frame_number
        
        return update_track_geofence_state(
            state=state,
            current_location=current_location,
            stable_frames=self._stable_frames
        )
        
    def update_observation(
        self,
        observation:TrackObservation,
        geofence: Geofence
    ) -> CrossingDirection | None:
        """Evaluate a tracked observation against a geofence"""
        
        anchor_point = box_bottom_center(
            observation.bounding_box
        )
        
        current_location = locate_point_in_geofence(
            point=anchor_point,
            geofence=geofence
        )
        
        return self.update(
            track_id=observation.track_id,
            current_location=current_location,
            frame_number=observation.frame_number
        )
        
    def get_state(
        self,
        track_id: int
    ) -> TrackGeofenceState | None:
        """Return the state for a track, if one exists"""
        
        return self._track_states.get(track_id)
    
    def process_observation(
        self,
        observation: TrackObservation,
        geofence: Geofence
    ) -> CrossingEvent | None:
        """Process an observation and create an event for a crossing."""
        
        anchor_point = box_bottom_center(
            observation.bounding_box
        )
        
        current_location = locate_point_in_geofence(
            point=anchor_point,
            geofence=geofence
        )
        
        direction = self.update(
            track_id=observation.track_id,
            current_location=current_location,
            frame_number=observation.frame_number
        )
        
        if direction is None:
            return None 
        
        event_id = (
            f"{geofence.geofence_id}:"
            f"{observation.track_id}:"
            f"{observation.frame_number}:"
            f"{direction.value}"
        )
        
        return CrossingEvent(
            event_id=event_id,
            geofence_id=geofence.geofence_id,
            track_id=observation.track_id,
            object_class=observation.class_name,
            direction=direction,
            frame_number=observation.frame_number,
            timestamp_seconds=observation.timestamp_seconds,
            confidence=observation.confidence,
            bounding_box=observation.bounding_box,
            anchor_point=anchor_point
        )
        
    def remove_stale_tracks(
        self,
        current_frame: int,
        max_age_frames: int
    ) -> list[int]:
        """Remove tracks not observed within the permitted frame age"""
        
        if current_frame < 0:
            raise ValueError(
                "current_frame must be zero or greater."
            )
            
        if max_age_frames < 0:
            raise ValueError(
                "max_age_frames must be zero or greater."
            )
            
        stale_track_ids = []
        
        for track_id, state in self._track_states.items():
            if state.last_seen_frame is None:
                continue 
            
            track_age = current_frame - state.last_seen_frame
            
            if track_age > max_age_frames:
                stale_track_ids.append(track_id)
                
        for track_id in stale_track_ids:
            del self._track_states[track_id]
            
        return stale_track_ids
    
    @property
    def active_track_count(self) -> int:
        """Return the number of tracks currently stored"""
        
        return len(self._track_states)