# Computer Vision Geofence Monitoring MVP

A learning-focused computer vision project that will detect, track, and review objects crossing a user-defined virtual geofence in prerecorded video.

The project is being developed incrementally so that each computer vision component can be understood, tested, evaluated, and documented before it is integrated into the complete MVP.

---

## Project Goal

The completed MVP will support the following workflow:

1. Upload a prerecorded video.
2. Select or draw a static geofence.
3. Detect people or vehicles in each video frame.
4. Assign temporary tracking IDs to detected objects.
5. Determine when a tracked object crosses the geofence.
6. Classify the crossing as an entry or exit.
7. Save an event record with visual evidence.
8. Review generated events through a simple local interface.

The initial MVP is intentionally limited to:

- One prerecorded video
- One camera
- One static geofence
- A limited set of object classes
- A simple local interface
- Local event storage and review

Advanced capabilities such as biometric identification, gait recognition, cross-camera tracking, demographic estimation, and dynamic geofences are outside the initial scope.

---

Current Status

Current milestone: Models, geometry, and geofence-crossing foundation completed.

The project now includes a modular Python package under:

src/geofence_monitor/

The current implementation can:

Open and inspect a prerecorded video with OpenCV.
Extract a representative frame.
Run pretrained YOLO object detection.
Extract classes, confidence scores, and bounding boxes.
Convert raw YOLO bounding-box arrays into validated application models.
Calculate bounding-box dimensions and center points.
Calculate bottom-center anchor points for ground-level geofence analysis.
Calculate Intersection over Union between bounding boxes.
Identify and suppress highly overlapping same-class predictions.
Define and validate a static polygon geofence.
Determine whether a point is inside, outside, or on the geofence boundary.
Maintain independent geofence state for multiple tracking IDs.
Detect entry and exit transitions.
Preserve confirmed state across boundary observations.
Prevent repeated events while an object remains inside or outside.
Debounce unstable location changes near the geofence boundary.
Record the last frame in which each track was observed.
Remove stale tracking states.
Generate structured crossing-event records.
Draw cleaned bounding boxes, labels, and center points.
Save a cleaned annotated evidence image.
Verify the models, geometry, and crossing logic through automated tests.

The current implementation does not yet:

Process the entire video through the modular pipeline.
Assign persistent tracking IDs with ByteTrack.
Connect real tracked video observations to the crossing engine.
Allow the user to draw the polygon through an interface.
Generate event-specific evidence images.
Store crossing events in SQLite.
Produce an annotated output video.
Provide a Streamlit event-review interface.
---

Current Implemented Pipeline

The current single-frame prototype follows this workflow:

Prerecorded video
        |
        v
OpenCV video inspection
        |
        v
Single-frame extraction
        |
        v
YOLO object detection
        |
        v
Raw Ultralytics bounding boxes
        |
        v
BoundingBox adapter
        |
        v
Shared geometry functions
        |
        v
IoU duplicate analysis
        |
        v
Class-aware duplicate suppression
        |
        v
Cleaned annotated image

The tested geofence-processing foundation follows this workflow:

TrackObservation
        |
        v
Bounding-box bottom-center
        |
        v
Point-in-polygon classification
        |
        v
Per-track geofence state
        |
        v
Debounced transition detection
        |
        v
ENTRY / EXIT / no event
        |
        v
CrossingEvent

The next implementation milestone will connect these two workflows through full-video reading and ByteTrack object tracking.

---

## Implemented Feature Prototype

The first implemented MVP feature is:

> Detect and classify people in a frame extracted from a prerecorded video.

The prototype represents the first functional slice of the larger system:

```text
Video ingestion
      |
      v
Frame extraction
      |
      v
Object detection
      |
      v
Detection post-processing
      |
      v
Visual evidence
```

This is a working computer vision prototype rather than pseudocode or an architecture-only design.

---

## Prototype Results

The initial test used a portrait-oriented video containing seven people walking closely together on a sidewalk.

The scene was intentionally useful because it included:

- Multiple people
- Significant person-to-person overlap
- Partial occlusion
- A person partially cut off by the image boundary
- A moving crowd
- A vertically oriented video frame

These conditions exposed both the strengths and limitations of the detector.

### Baseline Experiment

Configuration:

```text
Model:                yolo26n.pt
Confidence threshold: 0.25
Inference image size: 640
Inference device:     CPU
```

Result:

```text
Visible people counted manually: 7
People detected:                3
People missed:                  4
Approximate recall:             42.9%
```

The detector correctly identified three people but missed four others.

The likely causes included:

- Heavy occlusion
- Several people standing very close together
- Loss of visual detail during resizing
- Partial visibility near the edge of the frame
- Use of the smallest YOLO model variant

### Higher-Resolution Experiment

One variable was changed:

```text
Inference image size: 640 -> 960
```

All other major settings remained the same.

Configuration:

```text
Model:                yolo26n.pt
Confidence threshold: 0.25
Inference image size: 960
Inference device:     CPU
```

Raw result:

```text
Visible people counted manually: 7
Raw person detections:           8
```

Increasing the inference image size preserved more visual detail and substantially improved detection recall.

However, the model produced one duplicate prediction.

### Duplicate Analysis

Two predictions had nearly identical locations and dimensions:

```text
Detection #5 confidence: 0.714
Detection #7 confidence: 0.501
Intersection over Union: 0.934
```

An IoU of `0.934` indicates that the two bounding boxes overlapped almost completely.

The two detections were therefore treated as predictions of the same person.

The lower-confidence detection was suppressed.

Final result:

```text
Raw detections:        8
Retained detections:   7
Suppressed detections: 1
```

The final retained count matched the manual count of seven visible people.

---

## Approximate Evaluation

Treating the manual count of seven people as temporary ground truth:

### Baseline Run

```text
True positives:  3
False positives: 0
False negatives: 4
```

Approximate metrics:

```text
Precision: 100.0%
Recall:     42.9%
```

### Higher-Resolution Run Before Duplicate Suppression

```text
Correct person detections: 7
Duplicate detections:      1
Missed people:             0
```

Approximate metrics:

```text
Precision: 87.5%
Recall:    100.0%
```

### Higher-Resolution Run After Duplicate Suppression

```text
Retained person detections: 7
Duplicate detections:       0
Missed people:              0
```

For this single frame, the cleaned output matched the manual count.

These values are preliminary and should not be interpreted as formal model-performance measurements. The prototype has not yet been evaluated against a labeled dataset or across the full video.

---

## Technical Decisions

### Pretrained YOLO Detector

The prototype uses a pretrained Ultralytics YOLO model rather than training a detector from scratch.

This allows the project to focus first on:

- Computer vision inference
- Video processing
- Bounding-box geometry
- Detection post-processing
- Object tracking
- Geofence-crossing logic
- Event generation
- Event review

Custom training may be considered later if the pretrained detector performs poorly in the intended operating environment.

### Nano Model

The current model is:

```text
yolo26n.pt
```

The nano model was selected because it is:

- Small
- Fast to download
- Practical for CPU-based development
- Appropriate for initial experimentation
- Suitable for learning the end-to-end workflow

Larger models may improve difficult detections but will require more processing time and memory.

### Confidence Threshold

The current confidence threshold is:

```text
0.25
```

Predictions below this threshold are excluded.

The threshold has not yet been formally tuned. Increasing it may reduce false positives but can also remove correct detections of partially visible or heavily occluded people.

### Increased Inference Resolution

The inference image size was increased from `640` to `960`.

This change improved recall in the crowded test frame by preserving more visual information.

The tradeoff is increased computational cost.

This setting must still be tested across:

- Multiple frames
- Less crowded scenes
- Different object sizes
- Different lighting conditions
- Different video resolutions

### Custom Duplicate Suppression

The prototype includes class-aware duplicate suppression based on Intersection over Union.

The current experimental rule is:

```text
If two detections have the same class
and their IoU is at least 0.80,
retain the higher-confidence detection.
```

For the initial frame:

```text
Duplicate-pair IoU: 0.934
Suppression threshold: 0.80
```

The lower-confidence detection was successfully removed.

The `0.80` threshold worked for the initial frame but has not yet been validated across the full video or a broader evaluation set.

### Same-Class Comparison

Duplicate suppression is only applied when detections belong to the same class.

This prevents valid overlapping detections of different classes from being removed.

For example, a person may legitimately overlap with:

- A bicycle
- A motorcycle
- A chair
- A vehicle

### Object Center Points

The current prototype calculates the geometric center of each bounding box:

```text
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

The center point provides a simple representation of object location.

It can later be compared with a line or polygon geofence.

For ground-level geofence crossing, a future version may instead use the bottom-center point:

```text
bottom_center_x = (x1 + x2) / 2
bottom_center_y = y2
```

The bottom-center point may better approximate where a person or vehicle touches the ground.

---

## Computer Vision Concepts Practiced

The prototype introduced and applied the following concepts.

### Image Dimensions

OpenCV represents image-array shape as:

```text
height, width, channels
```

For example:

```text
(1920, 1080, 3)
```

means:

- Height: 1920 pixels
- Width: 1080 pixels
- Color channels: 3

### BGR Color Ordering

OpenCV uses BGR color ordering:

```text
Blue, Green, Red
```

This differs from the RGB ordering commonly used in image descriptions and other visualization libraries.

### Video Metadata

The video-inspection component reads:

- Frame width
- Frame height
- Frames per second
- Total frame count
- Approximate duration

Approximate duration is calculated as:

```text
duration = total frames / frames per second
```

### Inference

Inference is the process of passing new visual data through a trained model to obtain predictions.

The current prototype performs inference only. It does not train or fine-tune the YOLO model.

### Bounding Boxes

YOLO returns bounding boxes in the following format:

```text
x1, y1, x2, y2
```

Where:

- `x1, y1` represent the upper-left corner
- `x2, y2` represent the lower-right corner

### Confidence Scores

Each prediction includes a confidence score indicating the model's strength of belief in that detection.

Confidence is used for filtering and ranking predictions.

It should not automatically be interpreted as a perfectly calibrated probability.

### Occlusion

Occlusion occurs when one object partially blocks another.

The initial test frame included significant occlusion because seven people were walking close together.

Occlusion made it harder for the model to:

- Separate individuals
- Identify partially visible bodies
- Produce one unique box per person

### Precision

Precision measures how many produced detections were correct.

Conceptually:

```text
precision = correct detections / all produced detections
```

### Recall

Recall measures how many actual objects were successfully detected.

Conceptually:

```text
recall = detected objects / all actual objects
```

The initial experiment demonstrated that a detector can have high precision while still having poor recall.

### Intersection over Union

Intersection over Union measures how strongly two bounding boxes overlap.

Conceptually:

```text
IoU = intersection area / union area
```

IoU values range from `0.0` to `1.0`.

```text
0.0 = no overlap
1.0 = identical boxes
```

The duplicate pair in the test frame had an IoU of approximately `0.934`.

### Duplicate Suppression

Duplicate suppression processes detections from highest confidence to lowest confidence.

For each candidate detection:

1. Compare it with already retained detections.
2. Confirm that the class labels match.
3. Calculate IoU.
4. Suppress the candidate if the IoU exceeds the configured threshold.
5. Otherwise, retain it.

---

Project Structure
cv-geofence-monitoring-mvp/
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── smoke_test.py
├── inspect_video.py
├── detect_frame.py
├── docs/
│   └── architecture.md
├── src/
│   └── geofence_monitor/
│       ├── __init__.py
│       ├── models.py
│       ├── geometry.py
│       └── crossing.py
├── tests/
│   ├── test_models.py
│   ├── test_geometry.py
│   └── test_crossing.py
├── data/
│   ├── input/
│   │   └── .gitkeep
│   └── output/
│       └── .gitkeep
└── runs/

Generated videos, images, model weights, Python bytecode, test caches, and local environment files are excluded from version control.

Package Responsibilities
src/geofence_monitor/models.py

Defines validated application data structures and enums, including:

BoundingBox
Point
Geofence
TrackObservation
TrackGeofenceState
CrossingEvent
PointLocation
CrossingDirection
src/geofence_monitor/geometry.py

Provides reusable geometric operations, including:

Bounding-box width and height
Bounding-box center
Bottom-center anchor
Intersection over Union
Polygon point-location classification
src/geofence_monitor/crossing.py

Provides geofence-state and crossing logic, including:

Direct transition classification
Boundary-state handling
Per-track state management
Multiple-track isolation
Entry and exit detection
Debouncing
Duplicate-event prevention
Last-seen frame recording
Stale-track cleanup
Structured crossing-event generation
detect_frame.py

Runs the original single-frame YOLO experiment.

The script now imports the shared BoundingBox model and calculate_iou() function instead of maintaining a separate IoU implementation.
---
Automated Testing

The project uses pytest for automated testing.

Install the development dependencies with:

python -m pip install -r requirements-dev.txt

Run the complete test suite from the repository root:

PYTHONPATH=src python -m pytest tests -v

Current result:

60 passed

The test suite currently covers:

Bounding-box construction and validation
Point construction and validation
Bounding-box dimensions
Center and bottom-center calculations
Intersection over Union
Polygon-geofence validation
Inside, outside, edge, and vertex classifications
Entry and exit transition rules
Boundary-frame handling
Multiple independent tracking IDs
Observation-to-crossing integration
Structured event generation
Duplicate-event prevention
Crossing debouncing
Candidate-state cancellation
Last-seen frame recording
Stale-track cleanup
Invalid configuration handling
---

## Script Responsibilities

### `smoke_test.py`

Loads the pretrained YOLO model and runs inference on a known test image.

Primary purposes:

- Verify the Python environment
- Verify Ultralytics installation
- Verify PyTorch installation
- Verify OpenCV installation
- Verify model-weight download
- Verify object detection
- Inspect YOLO result objects
- Print detected classes, confidence scores, and bounding boxes

### `inspect_video.py`

Opens a prerecorded video with OpenCV and extracts its first frame.

Primary purposes:

- Verify that OpenCV can open the video
- Read video metadata
- Understand frame dimensions
- Calculate approximate duration
- Decode the first frame
- Save a representative frame for testing

### `detect_frame.py`

Runs person detection on an extracted video frame and performs custom post-processing.

Primary purposes:

- Run YOLO inference
- Extract classes
- Extract confidence scores
- Extract bounding boxes
- Calculate bounding-box dimensions
- Calculate center points
- Calculate IoU
- Identify suspected duplicate pairs
- Suppress lower-confidence duplicates
- Print retained detections
- Draw cleaned bounding boxes
- Draw confidence labels
- Draw center points
- Save a cleaned annotated image

---

## Environment

The current development environment uses:

```text
Python:           3.13.2
Ultralytics:      8.4.110
OpenCV:           5.0.0.93
PyTorch:          2.13.0
Inference device: CPU
```

The prototype was developed in WSL2 Ubuntu.

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd cv-geofence-monitoring-mvp
```

The final repository URL will be added after the initial GitHub push.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### 3. Upgrade `pip`

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The YOLO model weights are not committed to Git.

Ultralytics will download the required weights during the first model run.

---

## Usage

### Run the Environment Smoke Test

```bash
python smoke_test.py
```

This verifies that the main computer vision dependencies are functioning.

The first run may download:

```text
yolo26n.pt
```

It may also download the Ultralytics sample image used by the script.

### Add a Local Video

Place an MP4 video at:

```text
data/input/sample.mp4
```

Videos are intentionally excluded from Git because they:

- Can be large
- May contain identifiable people
- May have licensing restrictions
- Are environment-specific test inputs

### Inspect the Video

```bash
python inspect_video.py
```

The script prints:

- Video path
- Resolution
- Frames per second
- Total frames
- Approximate duration
- Extracted frame shape

It saves the first frame to:

```text
data/output/first_frame.jpg
```

### Run Single-Frame Detection

```bash
python detect_frame.py
```

The script:

1. Loads the extracted frame.
2. Loads the pretrained YOLO model.
3. Runs person detection.
4. Prints raw detections.
5. Calculates bounding-box centers.
6. Calculates pairwise IoU.
7. Reports suspected duplicate pairs.
8. Suppresses high-overlap duplicate predictions.
9. Prints final retained detections.
10. Draws cleaned annotations.
11. Saves the cleaned image.

The cleaned output is saved to:

```text
data/output/first_frame_cleaned.jpg
```

---

## Local Files Excluded from Git

The following types of files are intentionally excluded from version control:

### Python Environments and Caches

```text
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

### Model Weights

```text
*.pt
```

### Ultralytics Outputs

```text
runs/
```

### Videos

```text
data/input/*
```

The `.gitkeep` file remains tracked so that the directory exists after cloning.

### Generated Images and Evidence

```text
data/output/*
```

The `.gitkeep` file remains tracked so that the directory exists after cloning.

### Downloaded Smoke-Test Image

```text
bus.jpg
```

### Editor and Operating-System Files

```text
.vscode/
.idea/
.DS_Store
Thumbs.db
```

---

## Data, Privacy, and Responsible Use

This repository does not include the sample video used during development.

Anyone reproducing the project should:

- Use footage they are legally permitted to process.
- Review the license of downloaded stock footage.
- Obtain permission before recording or publishing identifiable people.
- Avoid committing private or sensitive footage.
- Avoid committing generated evidence images containing identifiable people.
- Protect event records that may reveal locations, behaviors, or timestamps.
- Clearly distinguish experimental detections from verified events.
- Maintain human review for consequential decisions.

The MVP is designed for object detection, tracking, geofence-crossing detection, and event review.

The initial scope explicitly excludes:

- Facial recognition
- Biometric identification
- Gait recognition
- Demographic estimation
- Cross-camera identity matching
- Pattern-of-life analysis
- Automated identity correlation

---

## Current Limitations

The prototype has been evaluated primarily on one frame from one video.

Current limitations include:

- No full-video inference loop
- No persistent tracking IDs
- No object trajectory history
- No geofence drawing interface
- No saved geofence configuration
- No entry detection
- No exit detection
- No crossing-event generation
- No event database
- No event-review dashboard
- No automated evaluation dataset
- No formal performance benchmark
- No GPU acceleration
- No command-line configuration
- Hard-coded input and output paths
- Hard-coded model choice
- Hard-coded confidence threshold
- Hard-coded inference image size
- Experimental duplicate-suppression threshold
- Overlapping labels in crowded scenes
- Potential missed detections under severe occlusion
- Potential duplicate detections in difficult scenes
- Potential frame-to-frame instability
- No automated tests for IoU or duplicate suppression
- No structured detection-metadata export
- No exception-reporting framework
- No production security controls

The current results demonstrate feasibility and learning progress. They do not demonstrate production readiness.

---

## Development Roadmap

### Phase 1: Detection Foundation

- [x] Create an isolated Python environment
- [x] Install OpenCV
- [x] Install PyTorch
- [x] Install Ultralytics
- [x] Load a pretrained YOLO model
- [x] Run an object-detection smoke test
- [x] Inspect a prerecorded video
- [x] Read video metadata
- [x] Extract a video frame
- [x] Detect people in the extracted frame
- [x] Extract confidence scores
- [x] Extract bounding-box coordinates
- [x] Calculate bounding-box center points
- [x] Increase inference resolution experimentally
- [x] Implement IoU calculation
- [x] Identify a duplicate prediction
- [x] Implement class-aware duplicate suppression
- [x] Save a cleaned annotated image
- [ ] Refactor detection settings into configuration values
- [ ] Test the detector on multiple frames
- [ ] Compare center points with bottom-center points
- [ ] Add automated tests for IoU
- [ ] Add automated tests for duplicate suppression
- [ ] Export structured detection metadata
- [ ] Process an entire video
- [ ] Save an annotated output video
- [ ] Measure average processing speed

### Phase 2: Object Tracking

- [ ] Integrate an object tracker such as ByteTrack
- [ ] Assign persistent tracking IDs
- [ ] Maintain IDs across consecutive frames
- [ ] Record object trajectories
- [ ] Draw tracking IDs
- [ ] Draw movement paths
- [ ] Evaluate ID stability during overlap and occlusion
- [ ] Associate detection metadata with tracking IDs

### Phase 3: Geofence Definition

- [ ] Display a representative video frame
- [ ] Allow the user to draw a line geofence
- [ ] Save geofence coordinates
- [ ] Draw the geofence on video frames
- [ ] Determine which side of the line contains each object
- [ ] Evaluate center-point and bottom-center approaches
- [ ] Extend the prototype to polygon geofences if needed

### Phase 4: Crossing Detection

- [ ] Store each tracked object's previous geofence state
- [ ] Store each tracked object's current geofence state
- [ ] Detect outside-to-inside transitions
- [ ] Detect inside-to-outside transitions
- [ ] Classify transitions as entry or exit
- [ ] Prevent duplicate events while an object remains inside
- [ ] Add event-cooldown or state-transition logic
- [ ] Test crossing behavior with multiple objects

### Phase 5: Event Generation

- [ ] Create a structured event schema
- [ ] Record event timestamps
- [ ] Record frame numbers
- [ ] Record object classes
- [ ] Record tracking IDs
- [ ] Record confidence scores
- [ ] Record crossing direction
- [ ] Record geofence identifiers
- [ ] Save evidence images
- [ ] Save detection and track metadata
- [ ] Export events to JSON or CSV
- [ ] Consider SQLite for local event persistence

### Phase 6: Event Review Interface

- [ ] Build a simple Streamlit interface
- [ ] Add video upload
- [ ] Add representative-frame display
- [ ] Add interactive geofence drawing
- [ ] Add processing controls
- [ ] Display generated event records
- [ ] Display evidence images
- [ ] Filter by object class
- [ ] Filter by direction
- [ ] Filter by time
- [ ] Mark events as valid
- [ ] Mark events as false alarms
- [ ] Display basic processing statistics

---

## Planned End-to-End Demonstration

The completed MVP should demonstrate the following sequence:

1. A user uploads a prerecorded video.
2. The system displays a representative frame.
3. The user draws a static virtual geofence.
4. The system processes the video.
5. YOLO detects people or vehicles.
6. ByteTrack or a similar tracker assigns tracking IDs.
7. A tracked object moves across the geofence.
8. The system determines the crossing direction.
9. The system creates an entry or exit event.
10. The system saves a visual evidence image.
11. The user reviews the event in a local interface.

---

## Learning Objectives

This project is also a structured computer vision learning exercise.

The learning objectives include:

- Understand how digital images are represented as arrays.
- Understand image width, height, and color channels.
- Understand OpenCV's BGR convention.
- Read and interpret video metadata.
- Extract frames from video.
- Load and run a pretrained object detector.
- Interpret bounding boxes and confidence scores.
- Understand the effects of inference resolution.
- Recognize the impact of occlusion.
- Distinguish precision from recall.
- Calculate object-center coordinates.
- Calculate Intersection over Union.
- Implement custom post-processing.
- Understand duplicate detections.
- Compare CPU inference performance.
- Build experiments by changing one variable at a time.
- Process videos frame by frame.
- Integrate object tracking.
- Understand temporary tracking identities.
- Represent motion with trajectories.
- Define a virtual line or polygon.
- Determine object position relative to a boundary.
- Detect state transitions.
- Generate structured computer vision events.
- Build a human-review workflow.
- Evaluate model limitations honestly.

---

## Key Lessons Learned So Far

### Larger Input Images Can Improve Detection

Increasing the inference image size from `640` to `960` substantially improved detection in the crowded test scene.

More input detail helped the model distinguish partially overlapping people.

### Better Recall Can Introduce More False Positives

The higher-resolution run detected all seven visible people but also produced one duplicate prediction.

Improving one metric can introduce new errors elsewhere.

### Confidence Thresholds Are Not a Complete Solution

Raising the confidence threshold might remove a lower-confidence duplicate, but it could also remove correct detections of partially visible people.

Confidence thresholds must be tuned carefully.

### Bounding-Box Overlap Provides Useful Evidence

The duplicate pair had an IoU of `0.934`.

This provided strong geometric evidence that both predictions represented the same object.

### Center Distance Alone Is Not Sufficient

People in crowded scenes may have nearby box centers.

IoU is more informative because it considers the full shape and overlap of the bounding boxes.

### Detection and Tracking Are Different Problems

The current prototype detects objects independently in one frame.

Tracking will require associating detections across time and maintaining stable identities.

### A Correct Single Frame Does Not Prove Video Reliability

The current result is promising, but detector behavior must be tested across multiple frames and changing conditions.

---

## Future Documentation

The project will maintain additional documents under `docs/`.

Planned documents include:

```text
docs/
├── ARCHITECTURE.md
├── EXPERIMENT_LOG.md
└── ROADMAP.md
```

### `ARCHITECTURE.md`

Will describe:

- Major system components
- Data flow
- Detection pipeline
- Tracking pipeline
- Geofence state management
- Event generation
- Storage
- Interface responsibilities
- Design boundaries

### `EXPERIMENT_LOG.md`

Will record:

- Experiment date
- Input video or frame
- Model version
- Detection settings
- Processing time
- Detection counts
- Manual observations
- Failures
- Decisions
- Next experiments

### `ROADMAP.md`

Will maintain:

- Current milestone
- Completed work
- Near-term tasks
- Deferred features
- Risks
- Definition of done

---

## Contributing

This repository is currently a personal learning and MVP-development project.

Changes should follow these principles:

1. Build one component at a time.
2. Test each component before integration.
3. Change one experimental variable at a time when practical.
4. Document important observations.
5. Avoid overstating model performance.
6. Preserve reproducibility.
7. Keep private video and generated evidence out of Git.
8. Prefer understandable code over premature optimization.
9. Add automated tests as reusable logic is introduced.
10. Keep deferred features outside the initial MVP scope.

---

## License

A project license has not yet been selected.

Until an explicit license is added:

- Do not assume the source code is licensed for unrestricted redistribution.
- Do not assume locally used videos may be redistributed.
- Review the license of any stock footage used for testing.
- Do not commit third-party model files unless their licensing permits it.