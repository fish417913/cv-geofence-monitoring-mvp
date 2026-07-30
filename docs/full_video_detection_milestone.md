# Full-Video Person Detection Milestone

**Completed:** July 30, 2026  
**MVP phase:** Detection foundation

## Objective

Extend the single-frame detection prototype into a complete prerecorded-video processing pipeline. The script reads every frame, detects people with a pretrained YOLO model, draws annotations, and writes the processed frames to a new video.

## Implemented Workflow

    Prerecorded MP4 video
            |
            v
    OpenCV VideoCapture
            |
            v
    Read one frame
            |
            v
    YOLO person detection
            |
            v
    Draw bounding boxes and labels
            |
            v
    OpenCV VideoWriter
            |
            v
    Annotated MP4 video

## Configuration

    Input video:          data/input/sample.mp4
    Output video:         data/output/annotated_video.mp4
    Model:                yolo26n.pt
    Target class:         person
    Confidence threshold: 0.25
    Inference image size: 960
    Video codec:          mp4v
    Inference device:     CPU

The input and generated videos remain excluded from version control to avoid committing large binary files or potentially identifiable footage.

## Validation Result

    Resolution:           1080 x 1920
    Orientation:          portrait
    Frame rate:           25.00 FPS
    Reported frame count: 461
    Approximate duration: 18.44 seconds
    Processed frames:     461

The annotated output video:

- Played successfully.
- Preserved portrait orientation.
- Preserved the original playback rate.
- Displayed person bounding boxes throughout the clip.
- Processed every reported source frame.

## Technical Concepts Practiced

- Frame-by-frame video processing
- OpenCV video capture and writing
- Video metadata and playback-rate preservation
- In-memory YOLO inference
- Person-class filtering
- Output-video encoding
- Resource cleanup
- Processing-progress reporting

## Detection Versus Tracking

The bounding boxes appear to follow people because YOLO detects them again in consecutive frames. Persistent object tracking has not yet been implemented.

Current behavior:

    Frame 1: detect person
    Frame 2: detect person again
    Frame 3: detect person again

Required tracking behavior:

    Frame 1: person -> Track ID 4
    Frame 2: person -> Track ID 4
    Frame 3: person -> Track ID 4

Tracking IDs are required before the system can remember an object's previous position and generate a reliable entry or exit event.

## Current MVP Capability

The project can now:

- Inspect prerecorded-video metadata.
- Extract and analyze an individual frame.
- Detect people using a pretrained YOLO model.
- Process an entire prerecorded video.
- Restrict inference to the person class.
- Draw detections on every decoded frame.
- Save a playable annotated output video.
- Confirm that processed and reported frame counts match.

## Next Development Step

Integrate ByteTrack through the Ultralytics tracking interface, assign persistent tracking IDs, and inspect ID stability during movement, overlap, and partial occlusion before implementing geofence-crossing logic.
