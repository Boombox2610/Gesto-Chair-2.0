import streamlit as st
import json
import time
import os
import pandas as pd

# Page config
st.set_page_config(
    page_title="Gesto-Drive CV Dashboard",
    page_icon="📷",
    layout="wide"
)

# Constants
PAYLOAD_FILE = "payloads.jsonl"
FRAME_FILE = "latest_frame.jpg"
REFRESH_RATE = 0.5  # seconds

def load_latest_payload():
    """Read the last line from the JSONL file."""
    if not os.path.exists(PAYLOAD_FILE):
        return None
        
    try:
        # For small files, reading all lines is fine. 
        # The file is rotated at 50MB so it won't grow infinitely.
        with open(PAYLOAD_FILE, "r") as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1].strip())
    except Exception:
        pass
    return None

def main():
    st.title("Gesto-Drive CV Dashboard")
    
    payload = load_latest_payload()
    
    if not payload:
        st.warning(f"No payload found. Is the pipeline running? (Looking for '{PAYLOAD_FILE}')")
        time.sleep(REFRESH_RATE)
        st.rerun()
        return

    # Sidebar
    with st.sidebar:
        st.header("Settings & Info")
        # Assuming the payload might eventually transmit scene mode, we can show it here
        st.subheader("Active Vocabulary")
        with st.expander("View Labels"):
            st.markdown("**Critical Hazards:** Pothole, open manhole, glass door, step edge...")
            st.markdown("**Dynamic Agents:** Person, car, motorcycle, bicycle...")
            st.markdown("**OCR Triggers:** Stop sign, traffic light, exit sign...")

    # Header Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frame ID", payload.get("frame_id", 0))
    with col2:
        fps = payload.get("fps", 0)
        status = payload.get("status", "unknown")
        
        # Color coding
        if status == "ok":
            st.metric("FPS", fps, "OK")
        elif status == "degraded":
            st.metric("FPS", fps, "-Degraded", delta_color="inverse")
        else:
            st.metric("FPS", fps, status, delta_color="off")
    with col3:
        st.metric("Detected Objects", len(payload.get("scene_understanding", [])))
    with col4:
        st.metric("Timestamp", payload.get("timestamp", 0))

    st.divider()

    # Main Row
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("Live Frame")
        if os.path.exists(FRAME_FILE):
            try:
                st.image(FRAME_FILE, width="stretch")
            except Exception as e:
                st.warning("Frame loading...")
        else:
            st.info("No frame image available.")
            
        st.subheader("Errors")
        errors = payload.get("errors", [])
        if errors:
            for err in errors:
                st.error(err)
        else:
            st.success("No active errors.")

    with right_col:
        st.subheader("Scene Understanding")
        objects = payload.get("scene_understanding", [])
        if objects:
            df_objs = pd.DataFrame(objects)
            # Add priority styling if it exists
            if "priority" in df_objs.columns:
                def highlight_priority(val):
                    if val == "critical":
                        return "background-color: #ff4b4b; color: white"
                    elif val == "standard":
                        return "background-color: #ffa421; color: black"
                    return ""
                
                cols = ["object_id", "class", "confidence", "position", "depth_score", "priority"]
                # Only use columns that actually exist in the df (in case of legacy payload)
                cols = [c for c in cols if c in df_objs.columns]
                df_objs = df_objs[cols]
                
                if "priority" in df_objs.columns:
                    st.dataframe(df_objs.style.map(highlight_priority, subset=["priority"]))
                else:
                    st.dataframe(df_objs)
            else:
                cols = ["object_id", "class", "confidence", "position", "depth_score"]
                df_objs = df_objs[cols]
                st.dataframe(df_objs)
            
            # Depth Score Chart
            st.bar_chart(df_objs.set_index("class")["depth_score"])
        else:
            st.info("No objects detected in the current frame.")
            
        st.subheader("Recognized Faces")
        faces = payload.get("recognized_faces", [])
        if faces:
            st.dataframe(pd.DataFrame(faces))
        else:
            st.info("No known faces recognized.")
            
        st.subheader("Environmental Text (OCR)")
        texts = payload.get("environmental_text", [])
        if texts:
            st.dataframe(pd.DataFrame(texts))
        else:
            st.info("No text detected.")

    st.divider()
    
    with st.expander("Raw JSON Payload"):
        st.json(payload)

    # Auto-refresh loop
    time.sleep(REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()
