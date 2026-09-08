import numpy as np 

from geofence_monitor.inference import load_detector, infer_frame


def test_load_detector():
    model, weights = load_detector()

    print(type(model))
    print(weights.meta["categories"][:5])

    assert model is not None
    
def test_real_detector_returns_detection_structure():
    model, weights = load_detector()
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    output = infer_frame(model, frame)
    
    print(output.keys())
    print("boxes:", output["boxes"].shape)
    print("labels:", output["labels"].shape)
    print("scores:", output["scores"].shape)
    
    assert "boxes" in output 
    assert "labels" in output 
    assert "scores" in output 
    assert output["boxes"].shape[1] == 4