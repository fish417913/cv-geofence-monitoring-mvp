# Experiment Log

## Computer Vision Geofence Monitoring MVP

This document records experiments performed during development of the Computer Vision Geofence Monitoring MVP.

The goal is to preserve:

- What was tested
- Why it was tested
- Which variables were changed
- What results were observed
- What conclusions were supported
- What remains uncertain
- Which technical decisions were made

This is a living document and should be updated as new experiments are completed.

---

## Experiment Logging Principles

Each experiment should follow these principles:

1. State the question being investigated.
2. Record the input data or scene.
3. Record the model and software configuration.
4. Change one important variable at a time when practical.
5. Preserve raw observations.
6. Distinguish manual judgment from measured results.
7. Document failures and unexpected behavior.
8. Avoid generalizing from a single image or video.
9. Record the resulting technical decision.
10. Identify the next experiment.

---

# Experiment 001: YOLO Environment Smoke Test

## Date

```text
2026-07-29
```

## Objective

Verify that the local computer vision environment can:

- Import OpenCV
- Import PyTorch
- Import Ultralytics
- Load a pretrained YOLO model
- Download required model weights
- Run object detection
- Return bounding boxes
- Return class labels
- Return confidence scores
- Save an annotated image

## Environment

```text
Operating environment: WSL2 Ubuntu
Python:                3.13.2
Ultralytics:           8.4.110
OpenCV:                5.0.0.93
PyTorch:               2.13.0
CUDA available:        False
Inference device:      CPU
```

## Model

```text
yolo26n.pt
```

## Input

Ultralytics sample image:

```text
https://ultralytics.com/images/bus.jpg
```

The file was downloaded locally as:

```text
bus.jpg
```

## Configuration

```text
Confidence threshold: 0.25
Default image size:    640
Save annotations:     True
```

## Raw Result

```text
Image size:          1080 x 810
Number of detections: 5
CUDA available:       False
```

Detected objects:

| Class | Confidence |
|---|---:|
| Bus | 0.924 |
| Person | 0.913 |
| Person | 0.905 |
| Person | 0.870 |
| Person | 0.534 |

## Performance

```text
Preprocessing:  5.1 ms
Inference:     54.6 ms
Postprocessing: 0.2 ms
```

## Observations

- The model loaded successfully.
- The required weights were available.
- Inference completed successfully on CPU.
- YOLO detected one bus and four people.
- Bounding-box coordinates were mapped back to the original image dimensions.
- The annotated output image was saved successfully.
- CUDA was unavailable, but CPU inference was fast enough for initial development.

## Conclusion

The computer vision environment passed the smoke test.

The development environment was suitable for continuing with prerecorded-video experiments.

## Decision

Proceed to OpenCV video ingestion and frame extraction.

---

# Experiment 002: OpenCV Video Inspection

## Date

```text
2026-07-29
```

## Objective

Verify that OpenCV can:

- Open the selected MP4 video
- Read video metadata
- Decode at least one frame
- Save an extracted frame as an image

## Input Video

Local path:

```text
data/input/sample.mp4
```

The video is excluded from version control.

## Video Metadata

```text
Resolution:           1080 x 1920
Frames per second:    25.00
Total frames:         461
Approximate duration: 18.44 seconds
Frame shape:          (1920, 1080, 3)
```

## Interpretation

OpenCV reports resolution as:

```text
width x height
```

The NumPy frame shape is reported as:

```text
height, width, channels
```

Therefore, both values describe the same portrait-oriented video:

```text
Width:    1080 pixels
Height:   1920 pixels
Channels: 3
```

## Output

The first decoded frame was saved to:

```text
data/output/first_frame.jpg
```

## Observations

- OpenCV opened the MP4 successfully.
- Video metadata was readable.
- The first frame decoded successfully.
- The extracted frame contained visible people.
- The video was vertically oriented.
- The first frame was suitable for initial person-detection testing.

## Conclusion

The video-ingestion component passed its initial test.

## Decision

Use the extracted first frame as the initial detection-development image.

---

# Experiment 003: Baseline Person Detection at Image Size 640

## Date

```text
2026-07-29
```

## Research Question

Can the pretrained YOLO nano model detect all visible people in the extracted video frame using its initial inference image size?

## Hypothesis

The pretrained model should detect clearly visible people, but it may miss individuals who are:

- Partially hidden
- Closely grouped
- Near the image boundary
- Reduced substantially during model-input resizing

## Input

```text
data/output/first_frame.jpg
```

## Scene Description

The frame contained seven people clustered closely together while walking on a sidewalk.

Important scene characteristics included:

- Seven manually counted people
- Heavy overlap between people
- Partial person-to-person occlusion
- Motion softness
- One or more people near the frame edge
- Portrait-oriented source video

## Model Configuration

```text
Model:                yolo26n.pt
Target class:         person
Confidence threshold: 0.25
Inference image size: 640
Inference device:     CPU
```

## Model Input Shape

```text
640 x 384
```

The original image was substantially reduced before inference.

## Raw Detection Result

```text
Manual person count: 7
Detected people:     3
Missed people:       4
Extra detections:    0
```

Detected confidences:

| Detection | Class | Confidence |
|---:|---|---:|
| 1 | Person | 0.842 |
| 2 | Person | 0.784 |
| 3 | Person | 0.479 |

## Performance

```text
Preprocessing:   4.3 ms
Inference:     106.0 ms
Postprocessing:  1.1 ms
```

## Approximate Evaluation

Treating the manual count as temporary ground truth:

```text
True positives:  3
False positives: 0
False negatives: 4
```

Approximate precision:

```text
3 / (3 + 0) = 1.000
```

```text
Precision: 100.0%
```

Approximate recall:

```text
3 / (3 + 4) = 0.429
```

```text
Recall: 42.9%
```

## Observations

- The three detections appeared to correspond to real people.
- Four visible people were not assigned separate bounding boxes.
- The detector was precise when it returned a box.
- Recall was poor for the crowded scene.
- The model struggled to separate overlapping people.
- The reduction from the original frame to a `640 x 384` input likely removed useful visual detail.
- Partial visibility and occlusion likely contributed to missed detections.

## Conclusion

The baseline configuration was functional but insufficient for the crowded frame.

The problem was primarily missed detections rather than incorrect detections.

## Decision

Keep the same:

- Model
- Confidence threshold
- Input frame
- Hardware

Change only:

```text
Inference image size: 640 -> 960
```

This isolates the effect of preserving more image detail.

---

# Experiment 004: Higher-Resolution Person Detection at Image Size 960

## Date

```text
2026-07-29
```

## Research Question

Will increasing the inference image size improve person detection in the crowded frame?

## Hypothesis

Increasing the inference image size should preserve more visual detail and help the same YOLO model distinguish overlapping people.

The larger input may require more processing time.

## Controlled Variables

The following remained unchanged:

```text
Model:                yolo26n.pt
Target class:         person
Confidence threshold: 0.25
Input frame:          first_frame.jpg
Inference device:     CPU
```

## Independent Variable

```text
Inference image size: 640 -> 960
```

## Model Input Shape

```text
960 x 544
```

## Raw Result

```text
Manual person count:  7
Raw person detections: 8
```

Raw detections:

| Detection | Confidence | Center |
|---:|---:|---|
| 1 | 0.837 | `(637.8, 909.1)` |
| 2 | 0.816 | `(854.8, 891.1)` |
| 3 | 0.739 | `(906.8, 892.2)` |
| 4 | 0.719 | `(1007.5, 870.8)` |
| 5 | 0.714 | `(750.1, 885.0)` |
| 6 | 0.521 | `(583.5, 863.8)` |
| 7 | 0.501 | `(753.9, 886.9)` |
| 8 | 0.477 | `(708.5, 867.2)` |

## Performance Observations

One run reported:

```text
Preprocessing:  4.2 ms
Inference:     74.4 ms
Postprocessing: 0.5 ms
```

A later run reported:

```text
Preprocessing:   6.0 ms
Inference:     119.6 ms
Postprocessing:  1.1 ms
```

## Runtime Interpretation

The inference time varied between executions.

Possible causes include:

- CPU scheduling
- Model warm-up
- Background system activity
- Memory state
- Library initialization
- Measurement variability

No reliable performance conclusion should be based on only one or two runs.

A future benchmark should:

- Process multiple frames
- Exclude or separately report warm-up frames
- Record mean processing time
- Record median processing time
- Record variability
- Record effective frames per second

## Visual Observation

The higher-resolution run appeared to detect all seven visible people.

However, it returned eight boxes.

Two detections had nearly identical center points:

```text
Detection #5 center: (750.1, 885.0)
Detection #7 center: (753.9, 886.9)
```

This suggested that one person may have been detected twice.

## Preliminary Approximate Evaluation

Before duplicate suppression:

```text
Correct person detections: 7
Duplicate predictions:     1
Missed people:             0
```

Approximate precision:

```text
7 / 8 = 0.875
```

```text
Precision: 87.5%
```

Approximate recall:

```text
7 / 7 = 1.000
```

```text
Recall: 100.0%
```

These values are based on manual visual interpretation of one frame.

## Conclusion

Increasing the inference image size substantially improved recall.

The new configuration recovered people missed at image size 640.

The improved result also introduced or exposed a duplicate prediction.

## Decision

Retain `imgsz=960` as the current experimental setting.

Investigate the suspected duplicate geometrically using Intersection over Union rather than changing the confidence threshold immediately.

---

# Experiment 005: Bounding-Box Duplicate Diagnosis

## Date

```text
2026-07-29
```

## Research Question

Do Detection #5 and Detection #7 represent the same person?

## Hypothesis

If the two boxes represent the same visual object, they should have:

- Very similar coordinates
- Similar dimensions
- Very close center points
- Extremely high Intersection over Union

## Relevant Bounding Boxes

### Detection #5

```text
Confidence: 0.714
Box:        (679.6, 700.9) to (820.6, 1069.1)
Size:       141.0 x 368.3
Center:     (750.1, 885.0)
```

### Detection #7

```text
Confidence: 0.501
Box:        (686.8, 705.4) to (821.0, 1068.5)
Size:       134.2 x 363.2
Center:     (753.9, 886.9)
```

## Center-Point Difference

```text
Horizontal difference: approximately 3.8 pixels
Vertical difference:   approximately 1.9 pixels
```

The centers were extremely close.

However, center distance alone was not considered sufficient because different people in a crowd can have nearby centers.

## Intersection over Union

A custom IoU function was implemented.

Conceptually:

```text
IoU = intersection area / union area
```

Pairwise comparison reported:

```text
Detection #5 and Detection #7: IoU = 0.934
```

## Interpretation

An IoU of `0.934` means that the boxes overlap almost completely.

This is substantially greater than ordinary overlap between adjacent people in the scene.

The evidence strongly supports the interpretation that both detections represent the same person.

## Comparison with Another Nearby Detection

Detection #8 also appeared close to the duplicate region, but its overlap was much lower.

Approximate comparisons:

```text
Detection #5 versus Detection #8: IoU approximately 0.293
Detection #7 versus Detection #8: IoU approximately 0.268
```

These values were low enough to support retaining Detection #8 as a separate person.

## Conclusion

Detection #5 and Detection #7 were confirmed as a suspected duplicate pair.

## Decision

Implement class-aware duplicate suppression using an experimental IoU threshold of:

```text
0.80
```

When two boxes:

- Have the same class
- Have IoU greater than or equal to 0.80

retain the higher-confidence detection.

---

# Experiment 006: Class-Aware Duplicate Suppression

## Date

```text
2026-07-29
```

## Objective

Automatically suppress the lower-confidence member of a highly overlapping same-class detection pair.

## Suppression Strategy

1. Sort all detections from highest confidence to lowest confidence.
2. Process one candidate at a time.
3. Compare the candidate with already retained detections.
4. Compare only detections belonging to the same class.
5. Calculate IoU.
6. Suppress the candidate if its IoU with a retained box is at least `0.80`.
7. Otherwise, retain the candidate.

## Configuration

```text
Duplicate IoU threshold: 0.80
Class-aware comparison:  Enabled
Confidence ordering:     Descending
```

## Result

```text
Raw detections:        8
Retained detections:   7
Suppressed detections: 1
```

Suppression decision:

```text
Suppressed Detection #7
Retained Detection #5
IoU: 0.934
```

Relevant confidence scores:

```text
Detection #5 confidence: 0.714
Detection #7 confidence: 0.501
```

The higher-confidence detection was retained.

## Final Retained Detections

| Original Detection | Class | Confidence | Center |
|---:|---|---:|---|
| 1 | Person | 0.837 | `(637.8, 909.1)` |
| 2 | Person | 0.816 | `(854.8, 891.1)` |
| 3 | Person | 0.739 | `(906.8, 892.2)` |
| 4 | Person | 0.719 | `(1007.5, 870.8)` |
| 5 | Person | 0.714 | `(750.1, 885.0)` |
| 6 | Person | 0.521 | `(583.5, 863.8)` |
| 8 | Person | 0.477 | `(708.5, 867.2)` |

## Approximate Final Evaluation

```text
Manual person count:   7
Retained detections:   7
Suspected duplicates:  0
Suspected missed people: 0
```

For this single frame, the cleaned result matched the manual person count.

## Important Qualification

Matching the total count does not prove that every bounding box is perfectly correct.

A formal evaluation would require:

- Ground-truth bounding boxes
- Matching criteria
- IoU thresholds for true-positive assignment
- Multiple labeled frames
- Multiple scenes
- Repeatable evaluation code

The current evaluation is manual and exploratory.

## Conclusion

The custom class-aware duplicate-suppression function behaved as intended on the test frame.

## Decision

Keep the current experimental rule:

```text
Same class
and IoU >= 0.80
-> suppress the lower-confidence detection
```

Do not treat the threshold as validated until it is tested on multiple frames.

---

# Experiment 007: Cleaned OpenCV Annotation

## Date

```text
2026-07-29
```

## Objective

Create a new annotated image containing only the retained detections after duplicate suppression.

## Motivation

The image automatically saved by Ultralytics contained all eight raw detections.

Although the detection data had been cleaned, the saved visualization still included the duplicate box.

A custom annotation step was therefore required.

## Annotation Elements

For every retained detection, OpenCV draws:

- Bounding box
- Class label
- Confidence score
- Center point

Current visual convention:

```text
Bounding boxes: Blue
Labels:         Blue
Center points:  Red
```

## Output

```text
data/output/first_frame_cleaned.jpg
```

## Result

The cleaned image contained:

```text
Seven retained person boxes
Seven center points
No near-identical duplicate box
```

## Visual Observations

- The final retained count matched the seven visible people.
- The duplicate box was no longer present.
- Center points appeared in the expected geometric locations.
- Boxes remained heavily overlapping because the people were physically close together.
- Labels overlapped and were difficult to read in the crowded area.
- Label overlap was a visualization issue rather than a detection-count failure.

## Conclusion

The cleaned annotation pipeline worked successfully.

## Decision

Use custom OpenCV annotation for later stages because it allows the project to draw:

- Filtered detections
- Tracking IDs
- Object trajectories
- Geofence lines
- Entry and exit labels
- Event timestamps
- Evidence markers

---

# Summary of Initial Experiments

## Detection Comparison

| Configuration | Raw Detections | Retained Detections | Suspected Misses | Suspected Duplicates |
|---|---:|---:|---:|---:|
| `imgsz=640` | 3 | 3 | 4 | 0 |
| `imgsz=960` before suppression | 8 | 8 | 0 | 1 |
| `imgsz=960` after suppression | 8 | 7 | 0 | 0 |

## Current Experimental Configuration

```text
Model:                    yolo26n.pt
Confidence threshold:     0.25
Inference image size:     960
Duplicate IoU threshold:  0.80
Target class:             person
Inference device:         CPU
```

## Current Technical Decisions

- Continue using the YOLO nano model during early development.
- Use image size 960 as the current experimental setting.
- Keep the confidence threshold at 0.25 for now.
- Do not raise the confidence threshold merely to remove duplicates.
- Use geometric overlap to diagnose duplicate boxes.
- Compare duplicate candidates only when their classes match.
- Retain the higher-confidence box when suppression is required.
- Use custom OpenCV annotations for cleaned visual output.
- Validate all current settings on additional frames.

---

# Lessons Learned

## Increased Resolution Improved Recall

The inference image-size change had a substantial effect.

```text
imgsz=640 -> 3 detected people
imgsz=960 -> 7 apparent unique people
```

The larger model input preserved details that helped separate crowded and partially occluded people.

## Improved Recall Introduced a Precision Tradeoff

The higher-resolution configuration recovered missed people but also produced one duplicate prediction.

This demonstrates that improving one type of error may expose another.

## Confidence Thresholding Was Not Enough

The duplicate had confidence `0.501`.

Another likely valid detection had confidence `0.477`.

Raising the threshold high enough to remove the duplicate could also remove a real person.

The confidence threshold was therefore not a precise duplicate-removal mechanism.

## IoU Provided Stronger Geometric Evidence

The suspected duplicate pair had nearly identical coordinates and an IoU of `0.934`.

This provided stronger evidence than confidence or center distance alone.

## Crowded Scenes Require Careful Interpretation

Several valid boxes overlapped because the people physically overlapped in the image.

Overlap alone does not always indicate a duplicate.

The amount of overlap, object class, confidence ordering, and visual context all matter.

## Runtime Measurements Vary

Inference times differed between repeated runs.

A proper benchmark requires repeated measurements and summary statistics.

## One Frame Is Not a Valid Benchmark

The prototype worked on the selected frame, but this does not prove:

- Stable detection throughout the video
- Reliable performance on other scenes
- Reliable performance on vehicles
- Reliable duplicate suppression
- Production readiness

---

# Known Limitations of the Experiments

The initial experiments were limited by:

- One main video
- One primary frame
- One manually counted scene
- No labeled ground-truth boxes
- No automated detection matching
- No repeated benchmark
- No alternate model comparison
- No alternate confidence-threshold comparison
- No full-video processing
- No tracking
- No geofence
- No crossing events
- No tests under different lighting
- No tests under different camera angles
- No tests with vehicles
- No tests with severe blur
- No tests with distant objects

---

# Next Planned Experiment

## Experiment 008: Multi-Frame Detection Validation

### Research Question

Do the current detection settings produce reasonable results at different points in the same video?

### Proposed Frames

Extract representative frames near:

```text
25% of the video
50% of the video
75% of the video
```

For a video containing 461 frames, approximate frame numbers are:

```text
25%: frame 115
50%: frame 230
75%: frame 346
```

### Variables to Keep Constant

```text
Model:                   yolo26n.pt
Confidence threshold:    0.25
Inference image size:    960
Duplicate IoU threshold: 0.80
```

### Measurements to Record

For each frame:

- Manual person count
- Raw detection count
- Retained detection count
- Suspected missed people
- Suspected duplicate boxes
- Confidence scores
- Inference time
- Visual observations

### Purpose

This experiment will test whether the current settings work beyond the first frame before the project introduces a full-video loop.

---

# Reusable Experiment Template

Copy this section when adding future experiments.

---

## Experiment XXX: Experiment Name

### Date

```text
YYYY-MM-DD
```

### Research Question

What specific question is being investigated?

### Hypothesis

What result is expected, and why?

### Input

Describe:

- Video
- Frame
- Dataset
- Scene
- Object classes

### Environment

```text
Python:
Ultralytics:
OpenCV:
PyTorch:
Device:
```

### Model Configuration

```text
Model:
Confidence threshold:
Image size:
Target classes:
Tracking configuration:
Geofence configuration:
```

### Controlled Variables

List the settings that remained unchanged.

### Independent Variable

Identify the setting that was intentionally changed.

### Raw Results

Record exact output where practical.

### Measurements

Record:

- Detection count
- Retained count
- Confidence values
- Runtime
- Events
- Tracking IDs
- Other relevant values

### Visual Observations

Describe:

- Correct detections
- Missed detections
- False positives
- Duplicate detections
- Tracking-ID switches
- Boundary jitter
- Incorrect events
- Annotation problems

### Limitations

What prevents strong conclusions?

### Conclusion

What does the evidence support?

### Decision

What setting or implementation decision follows from the experiment?

### Next Experiment

What should be tested next?