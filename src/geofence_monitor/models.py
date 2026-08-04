from dataclasses import dataclass 
from enum import Enum 

class PointLocation(str, Enum):
    """A point's position relative to a polygon geofence"""
    
    OUTSIDE = "outside"
    BOUNDARY = "boundary"
    INSIDE = "inside"
    
class CrossingDirection(str, Enum):
    """The direction of an observed geofence crossing"""
    
    ENTRY = "entry"
    EXIT = "exit"
    
@dataclass(frozen=True)
class Point:
    """A two-dimensional point in image coordinates"""
    
    x: float 
    y: float 
    
    def __post_init__(self) -> None:
        """Validate that both coordinates are non-negative"""
        
        if self.x < 0.0:
            raise ValueError(
                "x must be zero or greater."
            )
            
        if self.y < 0.0:
            raise ValueError(
                "y must be zero or greater."
            )
    
@dataclass
class TrackGeofenceState:
    """Geofence state maintained for one tracked object"""
    
    last_confirmed_location: PointLocation | None = None 
    candidate_location: Point | None = None 
    candidate_frame_count: int = 0
    last_seen_frame: int | None = None 

@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned bounding box in xyxy coordinate format."""
    
    x1: float 
    y1: float 
    x2: float 
    y2: float 
    
    def __post_init__(self) -> None:
        """Validate that the bounding-box coordinates are ordered correctly"""
        
        if self.x2 <= self.x1:
            raise ValueError(
                "x2 must be greater than x1."
            )
            
        if self.y2 <= self.y1:
            raise ValueError(
                "y2 must be greater than y1."
            )
            
@dataclass(frozen=True)
class TrackObservation:
    """A tracked object observed in one video frame."""
    
    track_id: int 
    class_id: int 
    class_name: int 
    confidence: float 
    bounding_box: BoundingBox 
    frame_number: int 
    timestamp_seconds: float 
    
    def __post_init__(self) -> None:
        """Validate the tracked observation"""
        
        if self.track_id < 0:
            raise ValueError(
                "track_id must be zero or greater."
            )
            
        if self.class_id < 0:
            raise ValueError(
                "class_id must be zero or greater."
            )
            
        if not self.class_name.strip():
            raise ValueError(
                "class_name must not be empty"
            )
            
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )
            
        if self.frame_number < 0:
            raise ValueError(
                "frame_number must be zero or greater."
            )
            
        if self.timestamp_seconds < 0.0:
            raise ValueError(
                "timestamp_seconds must be zero or greater."
            )
            
            
@dataclass(frozen=True)
class Geofence:
    """A static polygon geofence defined in image coordinates"""
    
    geofence_id: str 
    name: str 
    points: tuple[Point, ...]
    frame_width: int 
    frame_height: int 
    
    def __post_init__(self) -> None:
        """Validate the geofence definition"""
        
        if not self.geofence_id.strip():
            raise ValueError(
                "geofence_id must not be empty."
            )
            
        if not self.name.strip():
            raise ValueError(
                "name must not be empty."
            )
            
        if len(self.points) < 3:
            raise ValueError(
                "A polygone geofence requires at least three points."
            )
            
        if self.frame_width <= 0:
            raise ValueError(
                "frame_width must be greater than zero."
            )
            
        if self.frame_height <= 0:
            raise ValueError(
                "frame_height must be greater than zero."
            )
            
        for point in self.points:
            if point.x >= self.frame_width:
                raise ValueError(
                    "Geofence point x coordinate must be inside the frame."
                )
                
            if point.y >= self.frame_height:
                raise ValueError(
                    "Geofence point y coordinate must be inside the frame."
                )
                
@dataclass(frozen=True)
class CrossingEvent:
    """A confirmed geofence-crossing event"""
    
    event_id: str 
    geofence_id: str 
    track_id: int 
    object_class: str 
    direction: CrossingDirection
    frame_number: int 
    timestamp_seconds: float 
    confidence: float 
    bounding_box: BoundingBox
    anchor_point: Point 
    
    def __post_init__(self) -> None:
        """Validate the crossing event"""
        
        if not self.event_id.strip():
            raise ValueError(
                "event_id must not be empty."
            )
            
        if not self.geofence_id.strip():
            raise ValueError(
                "geofence_id must not be empty."
            )
            
        if self.track_id < 0:
            raise ValueError(
                "track_id must be zero or greater."
            )
            
        if not self.object_class.strip():
            raise ValueError(
                "object_class must not be empty."
            )
            
        if self.frame_number < 0:
            raise ValueError(
                "frame_number must be zero or greater."
            )
            
        if self.timestamp_seconds < 0.0:
            raise ValueError(
                "timestamp_seconds must be zero or greater."
            )
            
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )