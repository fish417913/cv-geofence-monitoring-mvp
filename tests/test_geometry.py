import pytest 

from geofence_monitor.geometry import (
    box_bottom_center,
    box_center,
    box_height,
    box_width,
    calculate_iou,
    locate_point_in_geofence
)

from geofence_monitor.models import (
BoundingBox,
Geofence,
Point,
PointLocation
)

def test_bounding_box_dimensions() -> None:
    box = BoundingBox(
        x1=100.0,
        y1=200.0,
        x2=300.0,
        y2=500.0
    )
    
    assert box_width(box) == 200.0
    assert box_height(box) == 300.0
    
def test_bounding_box_center_points() -> None:
    box = BoundingBox(
        x1=100.0,
        y1=200.0,
        x2=300.0,
        y2=500.0
    )
    
    assert box_center(box) == Point(
        x=200.0,
        y=350.0
    )
    assert box_bottom_center(box) == Point(
        x=200.0,
        y=500.0
    )
    
def test_identical_boxes_have_iou_of_one() -> None:
    box_a = BoundingBox(
        x1=0.0,
        y1=0.0,
        x2=100.0,
        y2=100.0
    )
    
    box_b = BoundingBox(
        x1=0.0,
        y1=0.0,
        x2=100.0,
        y2=100.0
    )
    
    assert calculate_iou(box_a, box_b) == 1.0
    
def test_partially_overlapping_boxes_have_expected_iou() -> None:
    box_a = BoundingBox(
        x1=0.0,
        y1=0.0,
        x2=100.0,
        y2=100.0
    )
    
    box_b = BoundingBox(
        x1=50.0,
        y1=50.0,
        x2=150.0,
        y2=150.0
    )
    
    expected_iou = 2500.0 / 17500.0
    
    assert calculate_iou(box_a, box_b) == pytest.approx(
        expected_iou
    )
    
def test_non_overlapping_boxes_have_iou_of_zero() -> None:
    box_a = BoundingBox(
        x1=0.0,
        y1=0.0,
        x2=100.0,
        y2=100.0
    )
    
    box_b = BoundingBox(
        x1=200.0,
        y1=200.0,
        x2=300.0,
        y2=300.0
    )
    
    assert calculate_iou(box_a, box_b) == 0.0
    
def test_bounding_box_rejects_reversed_x_coordinates() -> None:
    with pytest.raises(
        ValueError,
        match="x2 must be greater than x1"
    ):
        BoundingBox(
            x1=300.0,
            y1=200.0,
            x2=100.0,
            y2=500.0
        )
        
def test_bounding_box_rejects_reversed_y_coordinates() -> None:
    with pytest.raises(
        ValueError,
        match="y2 must be greater than y1"
    ):
        BoundingBox(
            x1=100.0,
            y1=500.0,
            x2=300.0,
            y2=200.0
        )
        
def make_square_geofence() -> Geofence:
    """Create a square geofence for geometry tests."""
    
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
    
def test_point_inside_geofence() -> None:
    geofence = make_square_geofence()
    point = Point(x=300.0, y=300.0)
    
    result = locate_point_in_geofence(
        point,
        geofence
    )
    
    assert result is PointLocation.INSIDE
    
def test_point_outside_geofence() -> None:
    geofence = make_square_geofence()
    point = Point(x=600.0, y=300.0)
    
    result = locate_point_in_geofence(
        point,
        geofence
    )
    
    assert result is PointLocation.OUTSIDE 
    
def test_point_on_geofence_edge() -> None:
    geofence = make_square_geofence()
    point = Point(x=100.0, y=300.0)
    
    result = locate_point_in_geofence(
        point,
        geofence 
    )
    
    assert result is PointLocation.BOUNDARY 
    
def test_point_on_geofence_vertex() -> None:
    geofence = make_square_geofence()
    point = Point(x=100.0, y=100.0)
    
    result = locate_point_in_geofence(
        point,
        geofence
    )
    
    assert result is PointLocation.BOUNDARY