import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from trackers import ByteTrackTracker

from geofence_monitor.crossing import CrossingEngine
from geofence_monitor.inference import infer_video, load_detector
from geofence_monitor.ingestion import get_video_frame, iter_video_frames
from geofence_monitor.models import Geofence, Point
from geofence_monitor.persistence import load_geofence, save_geofence


GEOFENCE_PATH = Path("data/geofence.json")

CANVAS_WIDTH = 540
PROCESS_FRAME_LIMIT = 60
CONFIDENCE_THRESHOLD = 0.50
STABLE_FRAMES = 3
ALLOWED_CLASSES = {"person"}


@st.cache_resource
def get_detector():
    """Load and cache the object detector for the Streamlit session."""
    return load_detector()


st.title("Geofence Monitoring MVP")

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"],
)


if uploaded_video is not None:
    st.success(f"Uploaded: {uploaded_video.name}")

    suffix = Path(uploaded_video.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_file.write(uploaded_video.getbuffer())
        temp_path = temp_file.name

    video_path = Path(temp_path)

    try:
        # Read the first frame so the user can define the geofence.
        first_video_frame = next(iter_video_frames(video_path))

        frame_rgb = cv2.cvtColor(
            first_video_frame.frame,
            cv2.COLOR_BGR2RGB,
        )

        frame_height, frame_width = (
            first_video_frame.frame.shape[:2]
        )

        # ---------------------------------------------------------
        # Load an existing geofence
        # ---------------------------------------------------------

        if GEOFENCE_PATH.exists():
            if st.button("Load saved geofence"):
                loaded_geofence = load_geofence(
                    GEOFENCE_PATH
                )

                if (
                    loaded_geofence.frame_width != frame_width
                    or loaded_geofence.frame_height != frame_height
                ):
                    st.error(
                        "Saved geofence dimensions do not match "
                        "the uploaded video."
                    )
                else:
                    st.session_state["geofence"] = (
                        loaded_geofence
                    )
                    st.session_state["geofence_source"] = (
                        "disk"
                    )

        # ---------------------------------------------------------
        # Draw a geofence
        # ---------------------------------------------------------

        canvas_width = CANVAS_WIDTH
        canvas_height = int(
            frame_height * canvas_width / frame_width
        )

        background_image = Image.fromarray(frame_rgb)

        st.write(
            "Draw the geofence polygon on the video frame."
        )

        canvas_result = st_canvas(
            background_image=background_image,
            drawing_mode="polygon",
            stroke_width=3,
            fill_color="rgba(255, 0, 0, 0.15)",
            width=canvas_width,
            height=canvas_height,
            key="geofence_canvas",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get(
                "objects",
                [],
            )

            if objects:
                polygon = objects[-1]
                raw_points = polygon.get("points", [])

                scale_x = frame_width / canvas_width
                scale_y = frame_height / canvas_height

                geofence_points = tuple(
                    Point(
                        x=point["x"] * scale_x,
                        y=point["y"] * scale_y,
                    )
                    for point in raw_points
                )

                geofence = Geofence(
                    geofence_id="fence_01",
                    name="Main Geofence",
                    points=geofence_points,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                if st.button("Save geofence"):
                    st.session_state["geofence"] = (
                        geofence
                    )
                    st.session_state["geofence_source"] = (
                        "drawn"
                    )

        # ---------------------------------------------------------
        # Active geofence controls
        # ---------------------------------------------------------

        if "geofence" in st.session_state:
            saved_geofence = st.session_state[
                "geofence"
            ]

            source = st.session_state.get(
                "geofence_source",
                "unknown",
            )

            if source == "disk":
                st.success("Geofence loaded from disk.")
            elif source == "drawn":
                st.success(
                    "Geofence saved in this session."
                )
            else:
                st.success("Geofence active.")

            if st.button("Save geofence to disk"):
                save_geofence(
                    geofence=saved_geofence,
                    path=GEOFENCE_PATH,
                )

                st.success(
                    f"Geofence saved to {GEOFENCE_PATH}"
                )

            # -----------------------------------------------------
            # Process the video
            # -----------------------------------------------------

            if st.button(
                f"Process first {PROCESS_FRAME_LIMIT} frames"
            ):
                with st.spinner(
                    "Running detection and tracking..."
                ):
                    model, weights = get_detector()

                    tracker = ByteTrackTracker(
                        minimum_consecutive_frames=1
                    )

                    crossing_engine = CrossingEngine(
                        stable_frames=STABLE_FRAMES
                    )

                    results = infer_video(
                        model=model,
                        video_path=video_path,
                        categories=weights.meta[
                            "categories"
                        ],
                        confidence_threshold=(
                            CONFIDENCE_THRESHOLD
                        ),
                        allowed_classes=ALLOWED_CLASSES,
                        tracker=tracker,
                        geofence=saved_geofence,
                        crossing_engine=crossing_engine,
                        max_frames=PROCESS_FRAME_LIMIT,
                    )

                st.success(
                    f"Processed {len(results)} frames."
                )

                events = [
                    event
                    for result in results
                    for event in result[
                        "crossing_events"
                    ]
                ]

                st.write(
                    f"Crossing events detected: "
                    f"{len(events)}"
                )

                if not events:
                    st.info(
                        "No geofence crossings were "
                        "detected."
                    )

                # -------------------------------------------------
                # Event evidence
                # -------------------------------------------------

                for event in events:
                    evidence_frame = get_video_frame(
                        video_path=video_path,
                        frame_number=event.frame_number,
                    )

                    annotated_frame = (
                        evidence_frame.frame.copy()
                    )

                    geofence_contour = np.array(
                        [
                            [point.x, point.y]
                            for point
                            in saved_geofence.points
                        ],
                        dtype=np.int32,
                    )

                    cv2.polylines(
                        annotated_frame,
                        [geofence_contour],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=3,
                    )

                    box = event.bounding_box

                    cv2.rectangle(
                        annotated_frame,
                        (
                            int(box.x1),
                            int(box.y1),
                        ),
                        (
                            int(box.x2),
                            int(box.y2),
                        ),
                        color=(0, 255, 0),
                        thickness=3,
                    )

                    cv2.circle(
                        annotated_frame,
                        (
                            int(event.anchor_point.x),
                            int(event.anchor_point.y),
                        ),
                        radius=8,
                        color=(0, 0, 255),
                        thickness=-1,
                    )

                    evidence_rgb = cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    st.write(
                        f"Track {event.track_id} - "
                        f"{event.direction.value.upper()} - "
                        f"{event.timestamp_seconds:.2f}s"
                    )

                    st.image(
                        evidence_rgb,
                        caption=(
                            f"Evidence frame "
                            f"{event.frame_number}"
                        ),
                        width=CANVAS_WIDTH,
                    )

    finally:
        os.remove(temp_path)