import os 
import tempfile 
from pathlib import Path 

import cv2 
import streamlit as st 

from geofence_monitor.ingestion import iter_video_frames
from geofence_monitor.models import Geofence, Point
from PIL import Image 
from streamlit_drawable_canvas import st_canvas

st.title("Geofence Monitoring MVP")

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:
    st.success(f"Uploaded: {uploaded_video.name}")
    
    suffix = Path(uploaded_video.name).suffix 
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_video.getbuffer())
        temp_path = temp_file.name 
        
    video_path = Path(temp_path)
    
    try:
        first_video_frame = next(iter_video_frames(video_path))
        frame_rgb = cv2.cvtColor(first_video_frame.frame,
                                 cv2.COLOR_BGR2RGB
                                 )
        
        frame_height, frame_width = first_video_frame.frame.shape[:2]
        
        canvas_width = 540
        canvas_height = int(
            frame_height * canvas_width / frame_width
        )
        
        background_image = Image.fromarray(frame_rgb)
        
        st.write("Draw the geofence polygon on the video frame.")
        
        canvas_result = st_canvas(
            background_image=background_image,
            drawing_mode="polygon",
            stroke_width=3,
            fill_color="rgba(255, 0, 0, 0.15)",
            width=canvas_width,
            height=canvas_height,
            key="geofence_canvas"
        )
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            
            st.write(f"Drawn objects: {len(objects)}")
            
            if objects:
                polygon = objects[-1]
                raw_points = polygon.get("points", [])
                
                scale_x = frame_width / canvas_width 
                scale_y = frame_height / canvas_height
                
                geofence_points = tuple(
                    Point(
                        x=point["x"] * scale_x,
                        y=point["y"] * scale_y
                    )
                    for point in raw_points
                )
                
                geofence = Geofence(
                    geofence_id="fence_01",
                    name="Main Geofence",
                    points=geofence_points,
                    frame_width=frame_width,
                    frame_height=frame_height
                )
                
                if st.button("Save geofence"):
                    st.session_state["geofence"] = geofence 
                st.write("Scaled geofence points:")
                st.write(
                    [
                        (round(point.x, 1), round(point.y, 1))
                        for point in geofence.points
                    ]
                )
             
        if "geofence" in st.session_state:
            saved_geofence = st.session_state["geofence"]
            
            st.success("Geofence saved.")
            
            st.write(
                [
                    (round(point.x, 1), round(point.y, 1))
                    for point in saved_geofence.points 
                ]
            )
        
        st.write(
            f"Frame size: "
            f"{first_video_frame.frame.shape[1]} x "
            f"{first_video_frame.frame.shape[0]}"
        )
        
        st.write(
            f"Frame number: {first_video_frame.frame_number}"
        )
        
        st.write(
            f"Timestamp: {first_video_frame.timestamp_seconds:.2f} seconds"
        )
        
    finally:
        os.remove(temp_path)
    
 