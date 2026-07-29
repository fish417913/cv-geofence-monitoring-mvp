# Development Roadmap

## Computer Vision Geofence Monitoring MVP

This document tracks the development status, next steps, deferred work, risks, and definition of done for the Computer Vision Geofence Monitoring MVP.

It should be updated whenever a meaningful development increment is completed.

---

## Project Goal

Build and demonstrate a functional local MVP that:

1. Accepts a prerecorded video.
2. Allows a user to define one static geofence.
3. Detects selected objects.
4. Assigns temporary tracking IDs.
5. Detects when a tracked object crosses the geofence.
6. Classifies the crossing as an entry or exit.
7. Generates a structured event with visual evidence.
8. Allows the user to review the generated event.

---

## Current Status

**Current phase:** Detection foundation

**Current milestone:** Single-frame person-detection prototype completed

**Next milestone:** Validate the current detector configuration on multiple representative video frames

---

## Progress Summary

The project currently supports:

- Prerecorded-video inspection
- Video metadata extraction
- Single-frame extraction
- Pretrained YOLO inference
- Person detection
- Confidence-score extraction
- Bounding-box extraction
- Center-point calculation
- Intersection over Union calculation
- Same-class duplicate diagnosis
- Higher-confidence duplicate retention
- Custom OpenCV annotation
- Cleaned evidence-image generation

The project does not yet support:

- Full-video processing
- Object tracking
- Geofence creation
- Crossing detection
- Entry or exit classification
- Event storage
- Event review

---

# Phase 1: Detection Foundation

## Objective

Develop a reliable and understandable object-detection pipeline before adding tracking or geofence logic.

## Completed

- [x] Create a dedicated project directory
- [x] Create an isolated Python virtual environment
- [x] Install Ultralytics
- [x] Install OpenCV
- [x] Install PyTorch
- [x] Verify the environment
- [x] Load a pretrained YOLO model
- [x] Run a known-image smoke test
- [x] Extract classes
- [x] Extract confidence scores
- [x] Extract bounding-box coordinates
- [x] Open a prerecorded MP4 with OpenCV
- [x] Read video metadata
- [x] Extract the first frame
- [x] Run detection on the extracted frame
- [x] Calculate bounding-box center points
- [x] Compare inference image sizes
- [x] Improve crowded-scene recall with `imgsz=960`
- [x] Implement Intersection over Union
- [x] Identify a duplicate prediction
- [x] Implement same-class duplicate suppression
- [x] Retain the higher-confidence duplicate candidate
- [x] Draw custom bounding boxes
- [x] Draw confidence labels
- [x] Draw object center points
- [x] Save a cleaned annotated image
- [x] Document the current prototype
- [x] Document the planned architecture
- [x] Create an experiment log
- [x] Create a development roadmap

## Next Tasks

- [ ] Extract frames near 25%, 50%, and 75% of the video
- [ ] Run the current detector configuration on each frame
- [ ] Manually compare visible people with retained detections
- [ ] Record detection counts and confidence scores
- [ ] Evaluate the `0.80` duplicate-IoU threshold
- [ ] Compare geometric center with bottom-center location
- [ ] Decide whether the current detector configuration is suitable for full-video testing

## Later Tasks in This Phase

- [ ] Refactor repeated detection logic into reusable functions
- [ ] Filter detections to selected object classes
- [ ] Export detection metadata to JSON or CSV
- [ ] Add automated tests for IoU
- [ ] Add automated tests for duplicate suppression
- [ ] Process every frame in the video
- [ ] Draw detections on every processed frame
- [ ] Save an annotated output video
- [ ] Measure processing time and effective frames per second
- [ ] Document full-video detection limitations

## Phase 1 Exit Criteria

Phase 1 is complete when the system can:

- Open a prerecorded video
- Process every readable frame
- Detect only configured object classes
- Apply detection post-processing consistently
- Save structured detection metadata
- Produce an annotated output video
- Report basic processing statistics

---

# Phase 2: Object Tracking

## Objective

Maintain a temporary identity for each detected object as it moves between frames.

## Planned Tasks

- [ ] Research ByteTrack integration through Ultralytics
- [ ] Run tracking on a short video segment
- [ ] Assign persistent tracking IDs
- [ ] Draw tracking IDs on frames
- [ ] Store each track’s recent positions
- [ ] Draw object trajectories
- [ ] Measure track duration
- [ ] Handle tracks entering and leaving the frame
- [ ] Evaluate tracking during person-to-person overlap
- [ ] Record tracking-ID switches
- [ ] Associate detection metadata with tracking IDs
- [ ] Save an annotated tracking video
- [ ] Document tracking limitations

## Phase 2 Exit Criteria

Phase 2 is complete when:

- Objects receive temporary tracking IDs
- IDs remain reasonably stable across consecutive frames
- Track histories are stored
- Tracking paths can be visualized
- Tracking metadata can be associated with frame-level detections

---

# Phase 3: Geofence Definition

## Objective

Allow a user to define and save one static virtual boundary.

## Planned Tasks

- [ ] Display a representative frame
- [ ] Implement a simple line geofence
- [ ] Allow selection of two line endpoints
- [ ] Save geofence pixel coordinates
- [ ] Draw the geofence on processed frames
- [ ] Define meaningful names for both sides of the line
- [ ] Validate coordinates against frame dimensions
- [ ] Load a saved geofence
- [ ] Consider polygon support after line-based crossing works

## Phase 3 Exit Criteria

Phase 3 is complete when:

- A user can define one line geofence
- The coordinates can be saved and loaded
- The geofence remains fixed across the video
- The geofence appears correctly on annotated frames

---

# Phase 4: Geofence Crossing Detection

## Objective

Determine when a tracked object moves from one side of the geofence to the other.

## Planned Tasks

- [ ] Calculate each tracked object’s geofence position
- [ ] Store previous geofence state by tracking ID
- [ ] Store current geofence state by tracking ID
- [ ] Detect outside-to-inside transitions
- [ ] Detect inside-to-outside transitions
- [ ] Label transitions as entry or exit
- [ ] Prevent repeated events while an object remains inside
- [ ] Test behavior when an object moves along the boundary
- [ ] Test behavior when detections jitter near the boundary
- [ ] Add simple boundary-stabilization logic
- [ ] Compare center-point and bottom-center crossing behavior
- [ ] Validate multiple simultaneous tracks

## Phase 4 Exit Criteria

Phase 4 is complete when:

- A tracked object can be classified as being on either side of the line
- A side change produces one crossing event
- Direction can be classified as entry or exit
- Remaining on one side does not produce repeated events
- Boundary jitter does not create obvious event spam

---

# Phase 5: Event Generation and Storage

## Objective

Create a reviewable record for each valid geofence crossing.

## Planned Tasks

- [ ] Define an event schema
- [ ] Generate a unique event ID
- [ ] Record frame number
- [ ] Record video timestamp
- [ ] Record tracking ID
- [ ] Record object class
- [ ] Record detection confidence
- [ ] Record geofence ID
- [ ] Record crossing direction
- [ ] Record bounding-box coordinates
- [ ] Record object location
- [ ] Save a crossing evidence image
- [ ] Export events to JSON
- [ ] Evaluate CSV or SQLite storage
- [ ] Prevent duplicate event records
- [ ] Validate event records against the annotated video

## Minimum Event Fields

Each event should contain:

```text
event_id
event_type
direction
geofence_id
track_id
object_class
confidence
frame_number
timestamp_seconds
bounding_box
object_position
evidence_image
review_status
```

## Phase 5 Exit Criteria

Phase 5 is complete when:

- Every valid crossing produces one event
- Each event is connected to the correct tracking ID
- Each event contains a timestamp and direction
- Each event contains visual evidence
- Events can be loaded from local storage

---

# Phase 6: Event Review Interface

## Objective

Provide a simple local interface for the complete MVP workflow.

## Preferred Framework

```text
Streamlit
```

## Planned Tasks

- [ ] Add video upload
- [ ] Display video metadata
- [ ] Display a representative frame
- [ ] Add interactive geofence definition
- [ ] Add model configuration controls
- [ ] Add a processing button
- [ ] Display processing progress
- [ ] Display the annotated output video
- [ ] Display a structured event table
- [ ] Display evidence images
- [ ] Filter events by object class
- [ ] Filter events by direction
- [ ] Filter events by time
- [ ] Mark an event as valid
- [ ] Mark an event as a false alarm
- [ ] Persist review status
- [ ] Display basic processing statistics
- [ ] Add clear user-facing error messages

## Phase 6 Exit Criteria

Phase 6 is complete when a user can:

- Select or upload a video
- Define a geofence
- Start processing
- View an annotated result
- Review generated crossing events
- Filter the event list
- Mark events as valid or false alarms

---

# End-to-End MVP Definition of Done

The MVP is complete when the following demonstration succeeds:

1. A user provides a prerecorded video.
2. A representative frame is displayed.
3. The user defines one static geofence.
4. The system processes the video.
5. A relevant object is detected.
6. The object receives a tracking ID.
7. The tracked object crosses the geofence.
8. The system determines the direction of travel.
9. One entry or exit event is generated.
10. An evidence image is saved.
11. The event appears in the review interface.
12. The user marks the event as valid or a false alarm.

---

# Deferred Features

The following features are intentionally outside the initial MVP scope:

- Live operational video streams
- Multiple simultaneous cameras
- Cross-camera identity matching
- Facial recognition
- Biometric identification
- Gait recognition
- Demographic estimation
- Pattern-of-life analysis
- Dynamic geofences
- Moving-camera geofences
- Three-dimensional scene reconstruction
- Camera calibration for real-world distance
- Multi-modal sensor integration
- Automatic model retraining
- Cloud-scale distributed processing
- Real-time external alert integrations
- Permanent person identification

These features should not be added until the core workflow has been validated.

---

# Current Experimental Configuration

```text
Model:                   yolo26n.pt
Target class:            person
Confidence threshold:    0.25
Inference image size:    960
Duplicate IoU threshold: 0.80
Inference device:        CPU
Input type:              prerecorded MP4
Camera assumption:       static
```

These settings remain experimental.

They must be tested across multiple frames before they are treated as project defaults.

---

# Current Risks

## Detection Generalization

The current settings were selected using one primary frame and may not generalize to the rest of the video.

## Occlusion

Overlapping people may cause missed detections, merged detections, or duplicate detections.

## Tracking Instability

Tracking IDs may switch when objects overlap or temporarily disappear.

## Boundary Jitter

Small frame-to-frame position changes may cause false crossing events near the geofence.

## Camera Motion

A static pixel-coordinate geofence will not remain valid if the camera moves.

## Performance

The higher inference resolution increases computational cost.

## Privacy

Input videos and generated evidence may contain identifiable people or sensitive location information.

## Scope Expansion

Advanced capabilities could distract from completing the core MVP workflow.

---

# Development Principles

The project should continue to follow these principles:

1. Build one component at a time.
2. Test each component before integration.
3. Change one important experimental variable at a time.
4. Preserve raw observations.
5. Document decisions and limitations.
6. Avoid overstating performance.
7. Keep private videos and evidence out of Git.
8. Prefer readable code over premature optimization.
9. Add automated tests as reusable logic grows.
10. Keep deferred features outside the initial MVP.

---

# Immediate Next Action

The next development task is:

> Extract and evaluate three representative frames from later points in the sample video using the current detection and duplicate-suppression configuration.

This will determine whether the current single-frame result is repeatable before full-video inference is introduced.