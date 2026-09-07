from dataclasses import dataclass 
from geofence_monitor.geometry import box_bottom_center, box_center, box_height, box_width
from geofence_monitor.models import BoundingBox

@dataclass(frozen=True)
class SpatialFeatures:
    width: float 
    height: float 
    area: float 
    aspect_ratio: float 
    center_x: float 
    center_y: float 
    anchor_x: float 
    anchor_y: float 
    
def extract_spatial_features(box: BoundingBox) -> SpatialFeatures:
    """Convert one detection box into reusable spatial features."""
    width = box_width(box)
    height = box_height(box)
    center = box_center(box)
    anchor = box_bottom_center(box)
    return SpatialFeatures(
        width=width, height=height, area=width * height, 
        aspect_ratio=width / height, 
        center_x=center.x, center_y=center.y, 
        anchor_x=anchor.x, anchor_y=anchor.y
    )