import json 
from pathlib import Path 

from geofence_monitor.models import Geofence, Point 

def save_geofence(
    geofence: Geofence,
    path: Path
) -> None:
    """Save a geofence definition to JSON."""
    
    data = {
        "geofence_id": geofence.geofence_id,
        "name": geofence.name, 
        "frame_width": geofence.frame_width,
        "frame_height": geofence.frame_height,
        "points": [
            {
                "x": point.x,
                "y": point.y
            }
            for point in geofence.points 
        ]
    }
    
    path.write_text(
        json.dumps(data, indent=2)
    )
    
def load_geofence(
    path: Path
) -> Geofence:
    """Load a geofence definition from JSON."""
    
    data = json.loads(
        path.read_text()
    )
    
    points = tuple(
        Point(
            x=point['x'],
            y=point['y']
        )
        for point in data['points']
    )
    
    return Geofence(
        geofence_id=data['geofence_id'],
        name=data['name'],
        points=points,
        frame_width=data['frame_width'],
        frame_height=data['frame_height']
    )