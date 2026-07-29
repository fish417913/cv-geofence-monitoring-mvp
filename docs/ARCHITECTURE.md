# System Architecture

## Computer Vision Geofence Monitoring MVP

This document describes the current and planned architecture for the Computer Vision Geofence Monitoring MVP.

The architecture is intentionally incremental. Each component should be implemented and tested independently before being integrated into the complete workflow.

---

## 1. Architecture Goals

The MVP architecture should support the following end-to-end workflow:

1. Accept a prerecorded video.
2. Extract and process the video frame by frame.
3. Detect selected object classes.
4. Assign temporary tracking IDs.
5. compare tracked-object positions with a static geofence.
6. Detect entry and exit transitions.
7. Generate reviewable event records.
8. Present results through a simple local interface.

The architecture prioritizes:

- Understandable Python code
- Incremental development
- Component-level testing
- Reproducible experiments
- Local execution
- Clear separation of responsibilities
- Human review of generated events
- Easy replacement of individual components

---

## 2. MVP Scope

### Included

The initial MVP will support:

- One prerecorded video
- One camera viewpoint
- One static geofence
- People and selected vehicle classes
- Pretrained object detection
- Temporary object tracking IDs
- Entry and exit event generation
- Local evidence-image storage
- Local structured event storage
- A simple Streamlit or similar interface

### Deferred

The initial MVP will not include:

- Live multi-camera processing
- Cross-camera identity matching
- Facial recognition
- Biometric identification
- Gait recognition
- Demographic estimation
- Pattern-of-life analysis
- Dynamic or moving geofences
- Automated model retraining
- Distributed processing
- Cloud deployment
- Operational alert integrations
- Multi-modal sensor fusion

---

## 3. High-Level Architecture

```text
+-----------------------+
|   Prerecorded Video   |
+-----------+-----------+
            |
            v
+-----------------------+
|    Video Ingestion    |
|       OpenCV          |
+-----------+-----------+
            |
            v
+-----------------------+
|   Frame Processing    |
| Resize / timestamps   |
+-----------+-----------+
            |
            v
+-----------------------+
|   Object Detection    |
|  Ultralytics YOLO     |
+-----------+-----------+
            |
            v
+-----------------------+
| Detection Filtering   |
| Class / confidence /  |
| duplicate suppression |
+-----------+-----------+
            |
            v
+-----------------------+
|    Object Tracking    |
|      ByteTrack        |
+-----------+-----------+
            |
            v
+-----------------------+
|   Geofence Analysis   |
| Position and crossing |
+-----------+-----------+
            |
            v
+-----------------------+
|   Event Generation    |
| Entry / exit records  |
+-----------+-----------+
            |
            +-------------------+
            |                   |
            v                   v
+-----------------------+  +-----------------------+
|   Evidence Storage    |  | Structured Metadata   |
| Images / video clips  |  | JSON / CSV / SQLite   |
+-----------+-----------+  +-----------+-----------+
            |                          |
            +-------------+------------+
                          |
                          v
              +-----------------------+
              |    Review Interface   |
              |      Streamlit        |
              +-----------------------+
```

---

## 4. Current Implemented Architecture

The current prototype implements only the first portion of the planned system.

```text
Prerecorded video
        |
        v
OpenCV video inspection
        |
        v
First-frame extraction
        |
        v
YOLO person detection
        |
        v
Bounding-box extraction
        |
        v
IoU duplicate analysis
        |
        v
Class-aware suppression
        |
        v
Cleaned annotated image
```

Current scripts:

```text
smoke_test.py
inspect_video.py
detect_frame.py
```

The current implementation operates on one extracted frame rather than the complete video.

---

## 5. Component Responsibilities

## 5.1 Video Ingestion

### Purpose

Open a local prerecorded video and provide frames to the processing pipeline.

### Current implementation

Implemented in:

```text
inspect_video.py
```

### Responsibilities

- Validate that the video file exists.
- Open the video with `cv2.VideoCapture`.
- Verify that OpenCV can decode the file.
- Read video metadata.
- Read individual frames.
- Release the video resource after processing.

### Current metadata

The inspection script reads:

- Frame width
- Frame height
- Frames per second
- Total frame count
- Approximate duration
- NumPy frame shape

### Planned extension

The video-ingestion component will eventually yield frames sequentially:

```python
frame_number = 0

while True:
    success, frame = capture.read()

    if not success:
        break

    timestamp_seconds = frame_number / fps

    # Send frame to the next pipeline stage.

    frame_number += 1
```

### Output contract

Each frame should eventually be associated with:

```text
frame
frame_number
timestamp_seconds
source_video
```

---

## 5.2 Frame Processing

### Purpose

Prepare each decoded video frame for model inference and downstream analysis.

### Responsibilities

- Preserve the original frame.
- Track original dimensions.
- Apply model-input resizing through the inference API.
- Maintain frame number and timestamp.
- Optionally create a copy for annotation.
- Avoid permanently altering the source frame.

### Current decision

Ultralytics currently handles model-input resizing through:

```python
imgsz=960
```

The detector returns bounding-box coordinates mapped back to the original frame dimensions.

### Planned considerations

Future frame processing may include:

- Optional frame skipping
- Resolution configuration
- Region-of-interest cropping
- Color-space conversion
- Image-quality checks
- Blur detection
- Timestamp overlays

Frame skipping should only be introduced after baseline correctness is established because skipped frames may reduce tracking stability or miss rapid crossings.

---

## 5.3 Object Detection

### Purpose

Locate and classify relevant objects in each frame.

### Current implementation

Implemented in:

```text
smoke_test.py
detect_frame.py
```

### Current model

```text
yolo26n.pt
```

### Current inference settings

```text
Confidence threshold: 0.25
Inference image size: 960
Device: CPU
```

### Current detected class

The current primary class is:

```text
person
```

### Planned MVP classes

The initial configurable class set may include:

- Person
- Car
- Truck
- Bus
- Bicycle
- Motorcycle

The class list should remain limited during the MVP to reduce unnecessary detections and simplify evaluation.

### Detection output

Each raw YOLO detection currently provides:

```text
class_id
class_name
confidence
x1
y1
x2
y2
```

Derived values include:

```text
box_width
box_height
center_x
center_y
```

A future normalized detection record may resemble:

```python
{
    "frame_number": 0,
    "timestamp_seconds": 0.0,
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.837,
    "bbox": {
        "x1": 560.0,
        "y1": 758.4,
        "x2": 715.7,
        "y2": 1059.7,
    },
    "center": {
        "x": 637.8,
        "y": 909.1,
    },
}
```

---

## 5.4 Detection Filtering and Post-Processing

### Purpose

Convert raw model predictions into a cleaner set of detections for tracking and geofence analysis.

### Current filtering stages

The current prototype applies:

1. YOLO confidence filtering
2. Same-class comparison
3. Intersection over Union calculation
4. High-overlap duplicate suppression

### Confidence filtering

The current threshold is:

```text
0.25
```

Predictions below the threshold are not returned by the inference call.

### Intersection over Union

IoU compares two bounding boxes:

```text
IoU = intersection area / union area
```

The current duplicate pair had:

```text
IoU = 0.934
```

### Duplicate suppression rule

The current experimental rule is:

```text
When two detections have the same class
and an IoU of at least 0.80,
retain the higher-confidence detection.
```

### Processing order

Detections are processed from highest confidence to lowest confidence.

```text
Sort detections by confidence
        |
        v
Keep highest-confidence detection
        |
        v
Compare next candidate with retained detections
        |
        +---- IoU below threshold ----> retain
        |
        +---- IoU at or above threshold ----> suppress
```

### Important limitation

The current duplicate-suppression threshold has been validated on only one primary frame.

It must be tested across multiple frames before it is treated as a stable configuration.

### Planned refactoring

The reusable logic should eventually move from `detect_frame.py` into a module such as:

```text
src/geofence_monitoring/detection.py
src/geofence_monitoring/geometry.py
```

---

## 5.5 Object-Location Representation

### Current representation

The current prototype uses the geometric center of each bounding box:

```text
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

### Planned alternative

For ground-plane crossing analysis, the bottom-center point may be more appropriate:

```text
bottom_center_x = (x1 + x2) / 2
bottom_center_y = y2
```

### Comparison

```text
Bounding-box center
        |
        +-- Often represents the torso
        +-- Stable when the full object is visible
        +-- Easy to calculate

Bottom-center point
        |
        +-- Approximates ground contact
        +-- Often better for line-crossing logic
        +-- May be affected by truncated boxes
```

Both approaches should be evaluated against the sample video before selecting the default geofence point.

---

## 5.6 Object Tracking

### Status

Not yet implemented.

### Planned tracker

The preferred initial tracker is:

```text
ByteTrack
```

### Purpose

Detection answers:

> What objects are visible in this frame?

Tracking answers:

> Which current detection corresponds to the same object seen in previous frames?

### Responsibilities

The tracking component will:

- Receive cleaned detections for each frame.
- Assign temporary tracking IDs.
- Match detections across consecutive frames.
- Preserve identities during short interruptions.
- Maintain recent object locations.
- Provide trajectories for visualization.
- Remove expired tracks.

### Planned track record

A track may contain:

```python
{
    "track_id": 12,
    "class_name": "person",
    "last_confidence": 0.84,
    "bbox": {
        "x1": 560.0,
        "y1": 758.4,
        "x2": 715.7,
        "y2": 1059.7,
    },
    "position": {
        "x": 637.8,
        "y": 909.1,
    },
    "first_frame": 15,
    "last_frame": 42,
    "trajectory": [
        [612.4, 905.0],
        [620.2, 906.8],
        [628.5, 907.9],
        [637.8, 909.1],
    ],
}
```

### Important limitation

A tracking ID is not a permanent identity.

It means only:

> The tracker believes these detections belong to the same visible object during this video sequence.

It does not identify a person by name or across unrelated videos.

---

## 5.7 Geofence Definition

### Status

Not yet implemented.

### Initial geofence type

The first geofence should be a static line.

A line is simpler to implement and reason about than an arbitrary polygon.

### Planned representation

```python
geofence = {
    "geofence_id": "main_boundary",
    "type": "line",
    "start": {
        "x": 400,
        "y": 300,
    },
    "end": {
        "x": 400,
        "y": 1600,
    },
}
```

### Later polygon representation

```python
geofence = {
    "geofence_id": "restricted_area",
    "type": "polygon",
    "points": [
        [300, 400],
        [850, 400],
        [900, 1300],
        [250, 1300],
    ],
}
```

### Coordinate system

Geofence coordinates will be stored in original-frame pixel coordinates.

For example:

```text
Origin: upper-left corner
Positive x direction: right
Positive y direction: down
```

```text
(0, 0) --------------------> x
  |
  |
  |
  |
  v
  y
```

### Planned interface

A representative video frame will be displayed in a local interface.

The user will select:

- Two points for a line, or
- Multiple points for a polygon

The selected coordinates will be saved and reused for every frame from the same static camera.

---

## 5.8 Geofence Position Analysis

### Status

Not yet implemented.

### Line-geofence concept

For each tracked-object position, the system will determine which side of the line contains the object.

Given a line from:

```text
A = (x1, y1)
B = (x2, y2)
```

and an object point:

```text
P = (px, py)
```

the sign of a two-dimensional cross product can indicate the side of the line:

```text
side = (x2 - x1)(py - y1) - (y2 - y1)(px - x1)
```

Conceptually:

```text
side > 0  -> one side of the line
side < 0  -> the opposite side
side = 0  -> directly on the line
```

The application should assign meaningful names to the two sides, such as:

```text
outside
inside
```

The meaning depends on the direction in which the user defines the geofence.

### Polygon concept

For a polygon geofence, the system can use a point-in-polygon operation such as:

```python
cv2.pointPolygonTest(...)
```

The result can indicate whether the selected object point is:

- Outside
- On the boundary
- Inside

---

## 5.9 Crossing Detection

### Status

Not yet implemented.

### Purpose

Generate an event only when a tracked object changes geofence state.

### Required state

For every active track, the system must remember:

```text
previous geofence state
current geofence state
```

Possible states may include:

```text
outside
inside
on_boundary
unknown
```

### Entry transition

```text
Previous state: outside
Current state:  inside
Result:         entry event
```

### Exit transition

```text
Previous state: inside
Current state:  outside
Result:         exit event
```

### No event

```text
outside -> outside
inside  -> inside
```

An object that remains inside must not generate a new event on every frame.

### Planned track-state structure

```python
track_geofence_state = {
    12: {
        "previous_state": "outside",
        "current_state": "inside",
        "last_event_frame": 145,
    }
}
```

### Boundary stabilization

Video detections may move slightly between frames.

A tracked point near the boundary could alternate rapidly:

```text
outside
inside
outside
inside
```

Potential stabilization strategies include:

- Require a minimum movement distance.
- Require the new state to persist for several frames.
- Use a narrow neutral boundary zone.
- Add a per-track event cooldown.
- Use line-segment intersection with the trajectory.
- Compare smoothed tracking positions.

Only the simplest reliable approach should be included in the MVP.

---

## 5.10 Event Generation

### Status

Not yet implemented.

### Purpose

Create a structured record whenever a valid geofence transition occurs.

### Planned event schema

```python
{
    "event_id": "event-000001",
    "event_type": "geofence_crossing",
    "direction": "entry",
    "geofence_id": "main_boundary",
    "track_id": 12,
    "object_class": "person",
    "confidence": 0.84,
    "frame_number": 145,
    "timestamp_seconds": 5.80,
    "position": {
        "x": 637.8,
        "y": 909.1,
    },
    "bbox": {
        "x1": 560.0,
        "y1": 758.4,
        "x2": 715.7,
        "y2": 1059.7,
    },
    "evidence_image": "events/event-000001.jpg",
    "review_status": "unreviewed",
}
```

### Minimum required fields

Each MVP event should include:

- Event identifier
- Timestamp
- Frame number
- Object class
- Tracking ID
- Direction
- Geofence identifier
- Confidence score
- Evidence-image path
- Review status

### Event identifier

Event identifiers should be unique within the processing run.

A simple MVP format may be:

```text
event-000001
event-000002
event-000003
```

A later implementation may use UUID values.

---

## 5.11 Evidence Generation

### Status

Single-frame evidence generation is currently implemented.

### Current output

```text
data/output/first_frame_cleaned.jpg
```

### Planned event evidence

For each crossing event, the system should save:

- The triggering frame
- The tracked-object bounding box
- The tracking ID
- The geofence
- Entry or exit label
- Event timestamp

A future filename may resemble:

```text
events/event-000001_entry_track-12.jpg
```

### Optional later extension

The system may save a short event clip containing:

- Several frames before the crossing
- The crossing frame
- Several frames after the crossing

This is useful but not required for the first MVP.

---

## 5.12 Structured Storage

### Status

Not yet implemented.

### Initial options

Possible MVP storage formats include:

```text
JSON
CSV
SQLite
```

### Recommended development sequence

```text
Step 1: JSON or CSV for transparent inspection
Step 2: SQLite when filtering and review-state updates are needed
```

### Why SQLite may eventually be appropriate

SQLite supports:

- Local storage
- No separate database server
- Structured queries
- Event filtering
- Review-status updates
- Streamlit integration

### Planned logical tables

```text
processing_runs
geofences
detections
tracks
events
```

The first MVP may use only an `events` table.

---

## 5.13 Review Interface

### Status

Not yet implemented.

### Preferred framework

```text
Streamlit
```

### Planned interface responsibilities

The interface should support:

- Video upload
- Video metadata display
- Representative-frame selection
- Geofence definition
- Processing configuration
- Processing initiation
- Progress display
- Annotated-video playback
- Event-table display
- Evidence-image review
- Event filtering
- Valid-event marking
- False-alarm marking

### Planned pages or sections

```text
1. Video and configuration
2. Geofence definition
3. Processing status
4. Event review
```

### Human-in-the-loop principle

Model-generated events should be treated as candidates for review.

The interface should allow a reviewer to assign:

```text
unreviewed
valid
false_alarm
```

---

## 6. Planned Data Flow

```text
User selects video
        |
        v
Video metadata is read
        |
        v
Representative frame is displayed
        |
        v
User defines geofence
        |
        v
Video frames are decoded sequentially
        |
        v
YOLO generates raw detections
        |
        v
Detections are filtered by class and confidence
        |
        v
Duplicate detections are suppressed
        |
        v
Tracker assigns or updates tracking IDs
        |
        v
Object position is calculated
        |
        v
Object geofence state is determined
        |
        v
Previous and current states are compared
        |
        +---- no transition ----> continue processing
        |
        +---- entry or exit ----> generate event
                                      |
                                      v
                              Save evidence image
                                      |
                                      v
                              Save event metadata
                                      |
                                      v
                              Display in review UI
```

---

## 7. Proposed Future Project Structure

The current scripts are intentionally simple. As reusable logic grows, the project may be reorganized.

```text
cv-geofence-monitoring-mvp/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── configs/
│   └── default.yaml
├── data/
│   ├── input/
│   │   └── .gitkeep
│   ├── output/
│   │   └── .gitkeep
│   └── events/
│       └── .gitkeep
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENT_LOG.md
│   └── ROADMAP.md
├── scripts/
│   ├── smoke_test.py
│   ├── inspect_video.py
│   └── detect_frame.py
├── src/
│   └── geofence_monitoring/
│       ├── __init__.py
│       ├── config.py
│       ├── detection.py
│       ├── events.py
│       ├── geofence.py
│       ├── geometry.py
│       ├── storage.py
│       ├── tracking.py
│       ├── video.py
│       └── visualization.py
└── tests/
    ├── test_duplicate_suppression.py
    ├── test_events.py
    ├── test_geofence.py
    └── test_geometry.py
```

This refactoring should occur only when the current scripts contain enough stable, reusable logic to justify it.

---

## 8. Configuration Strategy

The current prototype contains hard-coded settings.

Examples include:

```text
Video path
Model name
Confidence threshold
Inference image size
Duplicate IoU threshold
Output path
```

A later configuration object may contain:

```python
config = {
    "video_path": "data/input/sample.mp4",
    "model_name": "yolo26n.pt",
    "allowed_classes": ["person"],
    "confidence_threshold": 0.25,
    "image_size": 960,
    "duplicate_iou_threshold": 0.80,
    "tracking_enabled": True,
    "tracker": "bytetrack",
}
```

Configuration should first be implemented as Python constants or a dataclass.

A YAML or JSON configuration file may be introduced later if it adds real value.

---

## 9. Error Handling

Each component should fail with a specific, understandable error.

### Video errors

Examples:

- File does not exist
- Unsupported codec
- OpenCV cannot open the file
- Video contains zero readable frames
- Invalid frame rate
- Invalid resolution

### Model errors

Examples:

- Model weights cannot be downloaded
- Model file is corrupt
- PyTorch cannot initialize
- Inference returns an invalid result

### Output errors

Examples:

- Output directory cannot be created
- Annotated image cannot be saved
- Output video writer cannot initialize
- Event metadata cannot be written

### Geofence errors

Examples:

- Fewer than two line points
- Invalid polygon
- Coordinates outside the frame
- Missing geofence configuration

The interface should present useful messages rather than exposing only raw stack traces.

---

## 10. Performance Considerations

The current prototype runs on CPU.

Observed single-frame inference time has varied between runs, but the higher-resolution configuration remains suitable for offline experimentation.

### Main performance factors

- Model size
- Input image size
- Video resolution
- Number of frames
- Number of detections
- Tracking overhead
- Annotation overhead
- Video encoding
- Hardware acceleration

### MVP priority

The initial priority order is:

```text
1. Correctness
2. Understandability
3. Reproducibility
4. Processing speed
```

The prerecorded-video MVP does not need to process in real time.

### Planned measurements

Full-video processing should eventually record:

- Total processing time
- Frames processed
- Average frames per second
- Average detection time
- Average tracking time
- Number of crossing events
- Peak memory use if practical

---

## 11. Test Strategy

Testing should occur at multiple levels.

### Unit tests

Planned unit-test targets include:

- IoU calculation
- Duplicate suppression
- Center-point calculation
- Bottom-center calculation
- Line-side calculation
- Point-in-polygon logic
- State-transition logic
- Event generation

### Component tests

Examples:

- Open a known video
- Decode a known number of frames
- Run detection on a known image
- Track one object across a short clip
- Detect one known line crossing

### Integration tests

Examples:

- Process a short prerecorded video
- Detect and track a person
- Detect one crossing
- Save one event
- Load the event in the review interface

### Manual visual evaluation

Computer vision outputs should also be visually inspected for:

- Missed objects
- Duplicate boxes
- Incorrect classes
- Tracking-ID switches
- Incorrect geofence states
- Duplicate events
- Incorrect direction labels
- Misaligned evidence images

---

## 12. Architecture Risks

### Detection misses

Small, blurred, dark, or heavily occluded objects may not be detected.

### Duplicate detections

The detector may produce multiple boxes for one object.

### False positives

Background patterns may be classified as relevant objects.

### Tracking-ID switches

A tracker may assign a new identity after occlusion or overlap.

### Boundary jitter

Small changes in the detected position may produce repeated line-side changes.

### Camera motion

The MVP assumes a static camera. Camera movement can make a static pixel-coordinate geofence invalid.

### Perspective distortion

Objects farther from the camera appear smaller, and equal pixel movement does not represent equal real-world distance.

### Video-codec compatibility

OpenCV behavior may vary depending on local codec support.

### Privacy risk

Source footage and event evidence may contain identifiable people or sensitive location information.

### Configuration overfitting

Settings selected from one frame may not generalize to the rest of the video.

---

## 13. Architectural Decisions

### Decision 1: Begin with prerecorded video

Reason:

- Easier debugging
- Repeatable experiments
- No live-stream timing pressure
- Stable expected input
- Suitable for learning

### Decision 2: Use one static camera

Reason:

- Pixel-coordinate geofences remain stable
- Reduces geometric complexity
- Avoids camera-motion compensation

### Decision 3: Use pretrained YOLO

Reason:

- Avoids premature custom training
- Supports common object classes
- Provides a practical baseline
- Integrates with Python and OpenCV

### Decision 4: Use the nano model initially

Reason:

- CPU-friendly
- Fast experimentation
- Small download
- Appropriate for early development

### Decision 5: Test detection before tracking

Reason:

Tracking quality depends on detection quality. Detection problems should be understood before adding another source of complexity.

### Decision 6: Test one frame before full-video processing

Reason:

Single-frame testing isolates model behavior from video-loop, encoding, tracking, and state-management problems.

### Decision 7: Use class-aware duplicate suppression

Reason:

Highly overlapping boxes of the same class may represent duplicate predictions, while overlapping boxes of different classes may both be valid.

### Decision 8: Begin with a line geofence

Reason:

A line provides the simplest useful entry and exit demonstration.

### Decision 9: Maintain per-track geofence state

Reason:

Crossing events are state transitions, not merely detections near a boundary.

### Decision 10: Keep human review in the workflow

Reason:

Computer vision predictions are imperfect. Event records should remain reviewable and correctable.

---

## 14. Definition of Done for the MVP

The MVP architecture is successfully implemented when the system can demonstrate this sequence:

1. A user provides a prerecorded video.
2. A representative frame is displayed.
3. The user defines one static geofence.
4. The system processes the video.
5. Relevant objects are detected.
6. Objects receive reasonably stable tracking IDs.
7. At least one tracked object crosses the geofence.
8. The system determines whether the crossing is an entry or exit.
9. A structured event record is generated.
10. A visual evidence image is saved.
11. The event appears in a local review interface.
12. The user can mark the event as valid or a false alarm.

---

## 15. Current Architecture Milestone

The following components are complete:

- Video-file validation
- Video metadata inspection
- First-frame extraction
- Pretrained YOLO inference
- Bounding-box extraction
- Confidence extraction
- Center-point calculation
- Pairwise IoU calculation
- Same-class duplicate identification
- Higher-confidence duplicate retention
- Custom OpenCV annotation
- Cleaned evidence-image generation

The next architectural increment is:

> Apply the existing detection and post-processing logic to multiple representative frames before building the full video-processing loop.