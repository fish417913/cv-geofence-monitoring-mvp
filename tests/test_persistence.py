from geofence_monitor.models import Geofence, Point 
from geofence_monitor.persistence import (
    load_geofence,
    save_geofence,
)
import pytest 

def test_geofence_round_trip(tmp_path):
    geofence = Geofence(
        geofence_id="fence_01",
        name="Main Geofence",
        points=(
            Point(x=100.0, y=200.0),
            Point(x=400.0, y=200.0),
            Point(x=400.0, y=600.0),
            Point(x=100.0, y=600.0)
        ),
        frame_width=1080,
        frame_height=1920
    )
    
    path = tmp_path / "geofence.json"
    
    save_geofence(geofence, path)
    loaded_geofence = load_geofence(path)
    
    assert loaded_geofence == geofence 
    
def test_load_geofence_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing_geofence.json"
    
    with pytest.raises(FileNotFoundError):
        load_geofence(missing_path)