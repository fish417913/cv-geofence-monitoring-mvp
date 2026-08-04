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
5. Compare tracked-object positions with a static geofence.
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

The project has progressed beyond the original single-frame-only architecture.

The current implementation contains two layers:

1. The original single-frame detection prototype in `detect_frame.py`
2. A tested internal package for geometry, polygon geofencing, per-track state, and crossing-event generation

```text
YOLO detection output
        |
        v
Raw xyxy coordinates
        |
        v
BoundingBox adapter
        |
        v
Reusable geometry functions
        |
        +----------------------+
        |                      |
        v                      v
IoU calculation        Bottom-center anchor
                               |
                               v
                    Point-in-polygon analysis
                               |
                               v
                    Per-track geofence state
                               |
                               v
                 Debounced entry/exit detection
                               |
                               v
                       CrossingEvent
```

Current top-level scripts:

```text
smoke_test.py
inspect_video.py
detect_frame.py
```

Current internal package:

```text
src/geofence_monitor/
├── __init__.py
├── models.py
├── geometry.py
└── crossing.py
```

Current automated tests:

```text
tests/
├── test_models.py
├── test_geometry.py
└── test_crossing.py
```

The current implementation still operates on a single extracted frame for YOLO inference. Full-video iteration and ByteTrack integration are the next major implementation slices.

The current test suite contains:

```text
60 passing tests
```

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

The current single-frame prototype applies:

1. YOLO confidence filtering
2. Same-class comparison
3. Intersection over Union calculation
4. Experimental high-overlap duplicate suppression

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

### Implemented refactoring

Reusable bounding-box geometry now lives in:

```text
src/geofence_monitor/geometry.py
```

The original `detect_frame.py` script no longer maintains its own IoU implementation. Raw Ultralytics coordinate lists are converted into validated `BoundingBox` objects before being passed to the shared `calculate_iou()` function.

```text
Ultralytics xyxy list
        |
        v
BoundingBox adapter
        |
        v
calculate_iou()
```

### Important limitation

Custom duplicate suppression remains experimental.

A high IoU between two same-class boxes may represent a duplicate prediction, but in crowded scenes it may also represent two distinct, heavily overlapping people. The suppression stage should therefore remain optional until it is evaluated across more frames and scenes.

## 5.5 Object-Location Representation

### Current decision

The internal geofence logic uses the bottom-center point of each bounding box:

```text
bottom_center_x = (x1 + x2) / 2
bottom_center_y = y2
```

The reusable implementation is located in:

```text
src/geofence_monitor/geometry.py
```

and returns a validated `Point` object.

### Why bottom-center was selected

The bottom-center point better approximates where a person’s feet or a vehicle’s tires contact the ground.

```text
Bounding-box center
        |
        +-- Often represents the torso
        +-- Useful for general object location
        +-- May cross a ground-level boundary later than the object’s feet

Bottom-center point
        |
        +-- Approximates ground contact
        +-- Better aligned with ground-plane geofences
        +-- Used as the current crossing anchor
```

### Current models

The implementation now includes:

```text
BoundingBox
Point
TrackObservation
```

A `TrackObservation` stores the bounding box, class, confidence, track ID, frame number, and timestamp. The anchor point is derived from the bounding box when geofence analysis occurs.

A confirmed `CrossingEvent` stores both the bounding box and the exact anchor point used for the crossing decision.

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

Implemented as a validated static polygon model.

### Current geofence type

The current internal model uses one static polygon geofence.

```python
Geofence(
    geofence_id="sidewalk_zone",
    name="Sidewalk Zone",
    points=(
        Point(x=100.0, y=800.0),
        Point(x=900.0, y=800.0),
        Point(x=1000.0, y=1800.0),
        Point(x=50.0, y=1800.0),
    ),
    frame_width=1080,
    frame_height=1920,
)
```

### Current validation

A polygon geofence must have:

- A non-empty geofence identifier
- A non-empty display name
- At least three ordered points
- Positive frame width and height
- Points that fall inside the original frame dimensions

Valid coordinates satisfy:

```text
0 <= x < frame_width
0 <= y < frame_height
```

### Coordinate system

Geofence coordinates are stored in original-frame pixel coordinates.

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

### Architectural change from the original plan

The earlier design proposed beginning with a two-point line geofence. During implementation, the project moved directly to a polygon because the MVP’s core user workflow is area entry and exit.

A line-crossing mode may still be added later, but it is not the current primary representation.

### Planned interface

A representative video frame will be displayed in a local interface. The user will select multiple ordered points, and those points will be saved and reused for every frame from the same static camera.

## 5.8 Geofence Position Analysis

### Status

Implemented and covered by automated tests.

### Current approach

The system uses OpenCV’s polygon test:

```python
cv2.pointPolygonTest(...)
```

The tracked object’s bottom-center anchor is classified as:

```text
PointLocation.OUTSIDE
PointLocation.BOUNDARY
PointLocation.INSIDE
```

The reusable function is:

```text
locate_point_in_geofence()
```

in:

```text
src/geofence_monitor/geometry.py
```

### OpenCV result mapping

```text
Positive result -> inside
Zero            -> boundary
Negative result -> outside
```

The current implementation uses:

```python
measureDist=False
```

because the MVP presently needs categorical location rather than signed distance from the boundary.

### Boundary behavior

A boundary point is preserved as a separate state. It is not automatically treated as inside or outside.

This distinction allows the crossing engine to process sequences such as:

```text
OUTSIDE -> BOUNDARY -> INSIDE
```

without losing the last confirmed side of the geofence.

## 5.9 Crossing Detection

### Status

Implemented as a tested, stateful, multi-track crossing engine.

### Basic transition rules

```text
OUTSIDE -> INSIDE = ENTRY
INSIDE -> OUTSIDE = EXIT
```

No event is generated for:

```text
OUTSIDE -> OUTSIDE
INSIDE  -> INSIDE
```

### First-observation rule

The first clear observation establishes a baseline but does not generate an event.

```text
UNKNOWN -> OUTSIDE = initialize only
UNKNOWN -> INSIDE  = initialize only
```

This prevents the system from claiming an entry or exit that occurred before the object was first observed.

### Boundary handling

Boundary observations do not replace the last confirmed location.

```text
Last confirmed state: OUTSIDE
Current observation:  BOUNDARY
Stored confirmed state remains: OUTSIDE
```

This allows:

```text
OUTSIDE -> BOUNDARY -> INSIDE
```

to generate one entry event.

### Per-track state

The `CrossingEngine` maintains a separate `TrackGeofenceState` for each tracking ID.

Current state fields include:

```text
last_confirmed_location
candidate_location
candidate_frame_count
last_seen_frame
```

### Debouncing

The engine supports configurable crossing stabilization through:

```text
stable_frames
```

For example, with `stable_frames=3`:

```text
Confirmed state: OUTSIDE

INSIDE candidate: frame 1
INSIDE candidate: frame 2
INSIDE candidate: frame 3
                  |
                  v
             Confirm ENTRY
```

Returning to the confirmed side before the required count is reached clears the candidate.

### Duplicate-event prevention

After an entry is confirmed, repeated inside observations do not generate additional entry events.

```text
OUTSIDE -> INSIDE = one ENTRY event
INSIDE  -> INSIDE = no event
INSIDE  -> INSIDE = no event
```

### Multiple tracks

The engine stores each tracking ID independently.

```text
Track 7  -> OUTSIDE
Track 12 -> INSIDE
Track 19 -> OUTSIDE
```

A transition for one track does not modify another track’s state.

### Stale-track cleanup

Each track may store its most recent frame number.

```text
track_age = current_frame - last_seen_frame
```

A track is removed when:

```text
track_age > max_age_frames
```

The cleanup method returns the removed tracking IDs for logging, testing, and future interface statistics.

## 5.10 Event Generation

### Status

Structured crossing-event generation is implemented.

### Current event model

A confirmed transition produces a `CrossingEvent` containing:

- Event identifier
- Geofence identifier
- Tracking ID
- Object class
- Entry or exit direction
- Frame number
- Timestamp
- Detection confidence
- Bounding box
- Bottom-center anchor point

### Current event identifier

The event ID is deterministic:

```text
<geofence_id>:<track_id>:<frame_number>:<direction>
```

Example:

```text
test_zone:7:25:entry
```

A deterministic identifier makes repeated processing easier to debug and can later support a unique constraint in SQLite.

### Current event flow

```text
TrackObservation
        |
        v
Bottom-center anchor
        |
        v
Point-in-polygon classification
        |
        v
CrossingEngine state update
        |
        +---- no transition ----> None
        |
        +---- confirmed transition
                      |
                      v
                CrossingEvent
```

### Current limitation

The event model does not yet include:

- Evidence-image path
- Review status
- Processing-run identifier
- Source-video identifier

Those fields will be added when evidence generation and persistence are implemented.

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

### Selected MVP storage

SQLite is the preferred event repository for the MVP.

### Why SQLite

SQLite provides:

- Local storage
- No separate database server
- Structured queries
- Unique event identifiers
- Event filtering
- Review-status updates
- Straightforward Streamlit integration

### Planned first table

The first persistence slice may use only an `events` table.

Later tables may include:

```text
processing_runs
geofences
tracks
events
```

Evidence images will remain separate files. The database will store their file paths rather than binary image data.

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

## 7. Current and Planned Project Structure

The project has begun moving from exploratory scripts into a modular package.

```text
cv-geofence-monitoring-mvp/
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── smoke_test.py
├── inspect_video.py
├── detect_frame.py
├── data/
│   ├── input/
│   │   └── .gitkeep
│   └── output/
│       └── .gitkeep
├── docs/
│   └── architecture.md
├── src/
│   └── geofence_monitor/
│       ├── __init__.py
│       ├── models.py
│       ├── geometry.py
│       └── crossing.py
└── tests/
    ├── test_models.py
    ├── test_geometry.py
    └── test_crossing.py
```

### Current module responsibilities

```text
models.py
    Validated domain models and enums

geometry.py
    Bounding-box geometry, anchor calculation,
    IoU, and polygon location

crossing.py
    Transition rules, per-track state,
    debouncing, stale cleanup, and event creation
```

### Planned modules

As the next slices are implemented, the package may add:

```text
video.py
tracking.py
evidence.py
storage.py
pipeline.py
config.py
```

The existing scripts should remain as focused learning and regression tools until the modular replacements are proven.

## 8. Configuration Strategy

The current prototype still contains several hard-coded settings.

Examples include:

```text
Video path
Model name
Confidence threshold
Inference image size
Duplicate IoU threshold
Output path
```

The crossing engine has introduced two explicit reliability settings:

```text
stable_frames
max_age_frames
```

A future configuration dataclass may contain:

```python
config = {
    "video_path": "data/input/sample.mp4",
    "model_name": "yolo26n.pt",
    "allowed_classes": ["person"],
    "confidence_threshold": 0.25,
    "image_size": 960,
    "custom_duplicate_suppression": False,
    "duplicate_iou_threshold": 0.80,
    "tracker": "bytetrack.yaml",
    "stable_frames": 3,
    "max_age_frames": 75,
}
```

Configuration should first be implemented as Python constants or a dataclass.

A YAML or JSON configuration file should be introduced only when it adds practical value.

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

Testing is being implemented from the smallest deterministic behavior outward.

### Current automated suite

The project currently contains:

```text
60 passing tests
```

Run the complete suite with:

```bash
PYTHONPATH=src python -m pytest tests -v
```

Development dependencies can be installed with:

```bash
python -m pip install -r requirements-dev.txt
```

### Current unit-test coverage

The test suite currently covers:

- Bounding-box validation
- Point validation
- Geofence validation
- Bounding-box width and height
- Geometric center calculation
- Bottom-center anchor calculation
- Identical-box IoU
- Partial-overlap IoU
- Non-overlap IoU
- Point inside polygon
- Point outside polygon
- Point on polygon edge
- Point on polygon vertex
- Track-observation validation
- Direct entry transition
- Direct exit transition
- Boundary-frame handling
- First-observation initialization
- Prevention of duplicate events while remaining inside
- Independent state for multiple tracking IDs
- Observation-to-crossing integration
- Crossing-event creation and validation
- Debounced entry and exit
- Candidate cancellation during jitter
- Last-seen frame recording
- Stale-track cleanup

### Regression testing

The original `detect_frame.py` script remains a manual regression test for:

- YOLO inference
- Shared IoU behavior
- Experimental duplicate suppression
- OpenCV annotation
- Cleaned image output

### Next component tests

The next testing slice should cover:

- Video metadata validation
- Sequential frame delivery
- Frame numbering
- Timestamp calculation
- Safe `VideoCapture` release
- Empty or unreadable video handling

### Later integration tests

Examples:

- Process a short prerecorded clip
- Maintain at least one stable ByteTrack ID
- Detect one known polygon crossing
- Save one evidence image
- Persist one event
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

### Decision 7: Keep custom duplicate suppression experimental

Reason:

A high-IoU same-class pair may be a duplicate, but crowded scenes can contain two valid overlapping people. The feature should remain optional until evaluated across more frames.

### Decision 8: Use a static polygon geofence

Reason:

- Directly represents a monitored area
- Supports inside, outside, and boundary states
- Matches the MVP’s entry-and-exit workflow
- Works with OpenCV point-in-polygon operations

### Decision 9: Use the bounding-box bottom-center as the anchor

Reason:

It better approximates ground contact than the geometric center for people and vehicles crossing a ground-level boundary.

### Decision 10: Maintain per-track geofence state

Reason:

Crossing events are state transitions, not merely detections near a boundary.

### Decision 11: Preserve boundary as a distinct location

Reason:

A boundary observation should not erase the last confirmed side. This supports sequences such as `OUTSIDE -> BOUNDARY -> INSIDE`.

### Decision 12: Support configurable debouncing

Reason:

Bounding-box jitter near the boundary can otherwise generate false entry and exit events.

### Decision 13: Remove stale track state by frame age

Reason:

Trackers may stop reporting objects after occlusion, scene exit, or ID reassignment. Expired state should not remain indefinitely.

### Decision 14: Use deterministic crossing-event identifiers

Reason:

Deterministic IDs improve debugging and can help persistence reject duplicate event records.

### Decision 15: Use a modular monolith

Reason:

The MVP benefits from clear module boundaries without the operational complexity of microservices.

### Decision 16: Keep human review in the workflow

Reason:

Computer vision predictions are imperfect. Event records should remain reviewable and correctable.

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

The first modular architecture slice is complete.

### Implemented models and enums

- `BoundingBox`
- `Point`
- `Geofence`
- `TrackObservation`
- `TrackGeofenceState`
- `CrossingEvent`
- `PointLocation`
- `CrossingDirection`

### Implemented geometry

- Bounding-box width and height
- Geometric center
- Bottom-center anchor
- Intersection over Union
- Point-in-polygon classification
- Explicit inside, outside, and boundary results

### Implemented crossing behavior

- Direct entry and exit classification
- First-observation baseline initialization
- Boundary-state preservation
- Independent per-track state
- Multiple active tracking IDs
- Duplicate-event prevention
- Configurable debouncing
- Candidate-state cancellation during jitter
- Last-seen frame recording
- Stale-track cleanup
- Structured crossing-event creation
- Deterministic event identifiers

### Implemented refactoring

- Shared IoU logic moved into `src/geofence_monitor/geometry.py`
- `detect_frame.py` now converts raw YOLO coordinate lists into `BoundingBox` objects
- Duplicate geometry implementations were removed
- The original detection script still passes its manual regression run

### Automated verification

```text
60 tests passed
```

### Next architecture increment

The next implementation slice is:

> Build a reusable full-video reader that validates video metadata and yields frame number, timestamp, and image data one frame at a time.

That slice will prepare the system for YOLO and ByteTrack processing across the complete prerecorded video.
