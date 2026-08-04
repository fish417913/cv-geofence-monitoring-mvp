import cv2 
import numpy as np 

from geofence_monitor.models import (
    BoundingBox,
    Geofence,
    Point,
    PointLocation
)

def box_width(box: BoundingBox) -> float:
    """Return the width of a bounding box"""
    
    return box.x2 - box.x1 

def box_height(box: BoundingBox) -> float:
    """Return the height of a bounding box"""
    
    return box.y2 - box.y1 

def box_center(box: BoundingBox) -> Point:
    """Return the geometric center of a bounding box"""
    
    return Point(
        x=(box.x1 + box.x2) / 2,
        y=(box.y1 + box.y2) / 2
    )

def box_bottom_center(box: BoundingBox) -> Point:
    """Return the bottom-center point of a bounding box"""
    
    return Point(
        x=(box.x1 + box.x2) / 2,
        y=box.y2
    )

def calculate_iou(
    box_a: BoundingBox,
    box_b: BoundingBox
) -> float:
    """Calculate Intersection over Union for two bounding boxes"""
    
    intersection_x1 = max(box_a.x1, box_b.x1)
    intersection_y1 = max(box_a.y1, box_b.y1)
    intersection_x2 = min(box_a.x2, box_b.x2)
    intersection_y2 = min(box_a.y2, box_b.y2)
    
    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1
    )
    
    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1
    )
    
    intersection_area = (
        intersection_width * intersection_height
    )
    
    box_a_area = box_width(box_a) * box_height(box_a)
    box_b_area = box_width(box_b) * box_height(box_b)
    
    union_area = (
        box_a_area
        + box_b_area
        - intersection_area
    )
    
    if union_area <= 0:
        return 0.0 
    
    return intersection_area / union_area

def locate_point_in_geofence(
    point: Point,
    geofence: Geofence
) -> PointLocation:
    """Determine a point's position relative to a polygon geofence"""
    
    contour = np.array(
        [
            [vertex.x, vertex.y]
            for vertex in geofence.points
        ],
        dtype=np.float32
    )
    
    result= cv2.pointPolygonTest(
        contour,
        (point.x, point.y),
        measureDist=False
    )
    
    if result > 0:
        return PointLocation.INSIDE
    
    if result < 0:
        return PointLocation.OUTSIDE
    
    return PointLocation.BOUNDARY