from geofence_monitor.features import extract_spatial_features 
from geofence_monitor.models import BoundingBox 

def test_extract_spatial_features() -> None: 
    box = BoundingBox(
        x1=100.0,
        y1=200.0,
        x2=300.0,
        y2=600.0
    )
    
    features = extract_spatial_features(box)
    
    assert features.width == 200.0 
    assert features.height == 400.0 
    assert features.area == 80000.0
    assert features.aspect_ratio == 0.5
    assert features.center_x == 200.0
    assert features.center_y == 400.0 
    assert features.anchor_x == 200.0
    assert features.anchor_y == 600.0