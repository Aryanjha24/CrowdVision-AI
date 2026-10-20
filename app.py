"""
Enterprise AI-Powered Person Tracking & Crowd Analytics
Module: Streamlit Web Dashboard & Live Interactive Analytics

Author: Aryan Jha / Antigravity Team
Description: Interactive web dashboard for video ingestion, real-time computer
             vision tracking visualization, live SQL metric charting, and dwell
             heatmap generation.
"""

import os
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from tracker import CrowdTracker
from database import CrowdAnalyticsDB

# Page Configuration
st.set_page_config(
    page_title="AI Person Tracking & Crowd Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS Theme
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient & Card Styling */
    .dashboard-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        padding: 24px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .dashboard-title {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 32px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 12px;
        font-weight: 500;
        margin-top: 4px;
    }
    .delta-green { color: #10b981; }
    .delta-cyan  { color: #38bdf8; }
    .delta-amber { color: #f59e0b; }
    .delta-purple{ color: #a855f7; }
    
    /* Video Container Card */
    .video-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    /* Status Badge */
    .badge-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-live::before {
        content: "";
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Database singleton
db = CrowdAnalyticsDB(db_path="crowd_analytics.db")

# Top Header
st.markdown(
    """
    <div class="dashboard-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="dashboard-title">AI-Powered Person Tracking & Crowd Analytics</h1>
                <p class="dashboard-subtitle">Enterprise Edge Computer Vision • BoT-SORT Re-Identification • Spatial Heatmaps • SQLite Audit Logging</p>
            </div>
            <div>
                <span class="badge-live">SYSTEM READY</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.image(
    "https://img.icons8.com/fluency/96/motion-detector.png",
    width=64,
)
st.sidebar.title("Pipeline Controls")

# Video Ingestion
video_source_type = st.sidebar.radio(
    "Video Stream Source",
    options=["Sample Video (Default)", "Upload MP4 Video", "Live Webcam"],
    index=0,
)

target_video_path = None
if video_source_type == "Sample Video (Default)":
    default_sample = "1000296905.mp4"
    if os.path.exists(default_sample):
        target_video_path = default_sample
        st.sidebar.success(f"Loaded: `{default_sample}`")
    else:
        st.sidebar.error("Sample video file not found in workspace.")
elif video_source_type == "Upload MP4 Video":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        target_video_path = tfile.name
        st.sidebar.success("Custom video uploaded successfully!")
else:
    target_video_path = 0
    st.sidebar.info("Using primary webcam device (index 0).")

# Model & Tracker Settings
st.sidebar.subheader("Model Configuration")
model_choice = st.sidebar.selectbox(
    "Inference Engine",
    options=["yolov8n.pt (PyTorch)", "yolov8n.onnx (ONNX Accelerated)"],
    index=0,
)
model_weights = "yolov8n.pt" if "yolov8n.pt" in model_choice else "yolov8n.onnx"

conf_thresh = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.15,
    max_value=0.95,
    value=0.35,
    step=0.05,
)

iou_thresh = st.sidebar.slider(
    "IoU Threshold (NMS)",
    min_value=0.20,
    max_value=0.85,
    value=0.50,
    step=0.05,
)

# Feature Toggles
st.sidebar.subheader("Visualization Layers")
toggle_heatmap = st.sidebar.checkbox("Dwell-Time Heatmap Overlay", value=True)
heatmap_alpha = st.sidebar.slider("Heatmap Intensity / Alpha", 0.1, 0.8, 0.35, 0.05, disabled=not toggle_heatmap)
toggle_traces = st.sidebar.checkbox("Movement Trajectory Tails", value=True)
trace_length = st.sidebar.slider("Trajectory History Frames", 10, 100, 45, 5, disabled=not toggle_traces)

# Virtual Line Coordinate Setting
st.sidebar.subheader("Virtual Line Crossing")
line_y_percent = st.sidebar.slider("Line Vertical Position (%)", 10, 90, 60, 5)

# Database Controls
st.sidebar.subheader("Database Operations")
if st.sidebar.button("Wipe Analytics DB", use_container_width=True):
    db.reset_database()
    st.sidebar.success("Database records reset!")
    st.rerun()

# Real-Time KPI Cards Placeholder
kpi_cols = st.columns(5)
metric_occupancy = kpi_cols[0].empty()
metric_in = kpi_cols[1].empty()
metric_out = kpi_cols[2].empty()
metric_net = kpi_cols[3].empty()
metric_dwell = kpi_cols[4].empty()

def update_kpis(active_occ: int, c_in: int, c_out: int, avg_dwell: float):
    net_flow = c_in - c_out
    metric_occupancy.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Active Occupancy</div>
            <div class="metric-value">{active_occ}</div>
            <div class="metric-delta delta-cyan">Current In-Frame</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_in.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Entries (IN)</div>
            <div class="metric-value">{c_in}</div>
            <div class="metric-delta delta-green">↑ Cumulative Crossings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_out.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Exits (OUT)</div>
            <div class="metric-value">{c_out}</div>
            <div class="metric-delta delta-amber">↓ Cumulative Crossings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_net.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Net Foot Traffic</div>
            <div class="metric-value">{net_flow:+d}</div>
            <div class="metric-delta delta-purple">Δ (In - Out) Balance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_dwell.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg Dwell Time</div>
            <div class="metric-value">{avg_dwell:.1f}s</div>
            <div class="metric-delta delta-cyan">Person Linger Metric</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Initial KPI render
initial_summary = db.get_summary_metrics()
update_kpis(
    active_occ=initial_summary["active_occupancy"],
    c_in=initial_summary["total_in"],
    c_out=initial_summary["total_out"],
    avg_dwell=initial_summary["average_dwell_seconds"],
)

st.markdown("<br>", unsafe_allow_html=True)

# Main Application Tabs
tab_live, tab_analytics, tab_database = st.tabs(
    ["📹 Live Video Stream & Tracking", "📊 Real-Time Analytics & Charts", "🗄️ SQLite Audit Trail"]
)

with tab_live:
    col_video, col_side_info = st.columns([3, 1])

    with col_video:
        video_placeholder = st.empty()
        playback_progress = st.progress(0)
        status_text = st.empty()

    with col_side_info:
        st.markdown("### Streaming Control")
        start_btn = st.button("▶ Start Video Analytics", type="primary", use_container_width=True)
        stop_btn = st.button("⏹ Stop / Pause", use_container_width=True)

        st.markdown("---")
        st.markdown("### Telemetry Status")
        fps_metric = st.empty()
        tracks_metric = st.empty()
        frame_metric = st.empty()

        fps_metric.metric("Processing Speed", "0.0 FPS")
        tracks_metric.metric("Total Unique IDs", f"{initial_summary['total_unique_persons']}")
        frame_metric.metric("Processed Frames", "0")

    if start_btn and target_video_path is not None:
        cap = cv2.VideoCapture(target_video_path)
        if not cap.isOpened():
            st.error(f"Failed to open video source: {target_video_path}")
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0

            # Line Coordinates
            line_y = int(height * (line_y_percent / 100.0))
            line_start = (int(width * 0.05), line_y)
            line_end = (int(width * 0.95), line_y)

            # Initialize tracker
            tracker = CrowdTracker(
                model_path=model_weights,
                tracker_config="custom_botsort.yaml",
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                line_start=line_start,
                line_end=line_end,
                db_path="crowd_analytics.db",
                enable_heatmap=toggle_heatmap,
                heatmap_alpha=heatmap_alpha,
                trace_length=trace_length,
            )

            frame_num = 0
            start_clock = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    status_text.success("Video processing completed!")
                    break

                frame_num += 1
                simulated_ts = start_clock + (frame_num / fps_source)

                # Process Frame through CV Pipeline
                annotated_frame, telemetry = tracker.process_frame(
                    frame=frame,
                    timestamp=simulated_ts,
                )

                # Convert BGR to RGB for Streamlit rendering
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Update Telemetry & KPIs
                progress_val = min(1.0, max(0.0, frame_num / total_frames))
                playback_progress.progress(progress_val)
                status_text.text(f"Processing frame {frame_num}/{total_frames} ({progress_val * 100:.1f}%)")

                fps_metric.metric("Processing Speed", f"{telemetry['fps']} FPS")
                tracks_metric.metric("Total Unique IDs", f"{telemetry['total_tracked']}")
                frame_metric.metric("Processed Frames", f"{frame_num}")

                # Update top KPI cards every 5 frames
                if frame_num % 5 == 0:
                    summary_now = tracker.db.get_summary_metrics()
                    update_kpis(
                        active_occ=telemetry["active_occupancy"],
                        c_in=telemetry["cumulative_in"],
                        c_out=telemetry["cumulative_out"],
                        avg_dwell=summary_now["average_dwell_seconds"],
                    )

            cap.release()

with tab_analytics:
    st.subheader("Crowd Analytics Intelligence Hub")
    
    col_chart1, col_chart2 = st.columns(2)

    # 1. Historical Occupancy Snapshot Line Chart
    with col_chart1:
        st.markdown("#### Real-Time Occupancy & Net Traffic Flow")
        snapshots = db.get_recent_snapshots(limit=50)
        if snapshots:
            df_snap = pd.DataFrame(snapshots)
            df_snap["timestamp"] = pd.to_datetime(df_snap["timestamp"])
            df_snap = df_snap.set_index("timestamp")
            st.line_chart(
                df_snap[["current_occupancy", "net_occupancy", "cumulative_in", "cumulative_out"]],
                use_container_width=True,
            )
        else:
            st.info("No time-series snapshots logged yet. Run the video to generate live charts.")

    # 2. Cumulative Foot Traffic Distribution
    with col_chart2:
        st.markdown("#### Cumulative Ingress vs. Egress")
        metrics = db.get_summary_metrics()
        df_flow = pd.DataFrame(
            {
                "Direction": ["Entered (IN)", "Exited (OUT)"],
                "Count": [metrics["total_in"], metrics["total_out"]],
            }
        ).set_index("Direction")
        st.bar_chart(df_flow, use_container_width=True, color=["#10b981"])

    st.markdown("---")

    # 3. Spatial Heatmap Analysis
    st.subheader("2D Spatial Dwell-Time Heatmap Reconstruction")
    st.markdown(
        "Cumulative spatial dwell coordinates reconstructed from SQLite `DwellLog` persistent centroids."
    )
    heatmap_pts = db.get_heatmap_points()
    if heatmap_pts:
        df_pts = pd.DataFrame(heatmap_pts)
        st.scatter_chart(
            df_pts,
            x="x",
            y="y",
            size="weight",
            color="weight",
            use_container_width=True,
        )
    else:
        st.info("No persistent dwell points logged yet. Process video to accumulate spatial records.")

with tab_database:
    st.subheader("SQLite Real-Time Event Audit Trail")
    st.markdown("Every virtual line-crossing event is logged with sub-second timestamps, track IDs, and confidence scores.")

    col_db_actions1, col_db_actions2 = st.columns([1, 4])
    with col_db_actions1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    events = db.get_recent_events(limit=100)
    if events:
        df_events = pd.DataFrame(events)
        st.dataframe(
            df_events,
            use_container_width=True,
            column_config={
                "id": st.column_config.NumberColumn("Event ID"),
                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                "track_id": st.column_config.NumberColumn("Track ID"),
                "event_type": st.column_config.TextColumn("Action (IN/OUT)"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0),
                "centroid_x": st.column_config.NumberColumn("Centroid X"),
                "centroid_y": st.column_config.NumberColumn("Centroid Y"),
            },
        )
        csv_data = df_events.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Audit Log (CSV)",
            data=csv_data,
            file_name="crowd_analytics_events.csv",
            mime="text/csv",
        )
    else:
        st.info("No crossing events recorded yet. Run video analytics to populate events.")
