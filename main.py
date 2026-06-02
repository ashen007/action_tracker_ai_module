"""
main.py
-------
Streamlit dashboard for the Video Surveillance AI system.
Run: streamlit run main.py
API: Started automatically on port 8000 in a background thread.

Layout:
  [Left 25%] Controls | [Center 50%] Live feed | [Right 25%] Metrics
  [Bottom]  Heatmap | Dwell Chart | Action Breakdown | Export
"""

import io
import json
import os
import sys
import time
import threading
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---- Ensure project root on path ----
sys.path.insert(0, str(Path(__file__).parent))

# ---- Imports ----
from models.detector import PersonDetector
from models.tracker import PersonTracker, FallbackTracker
from models.pose_estimator import PoseEstimator
from models.action_classifier import ActionClassifier
from core.zone_manager import ZoneManager
from core.dwell_tracker import DwellTracker
from core.event_logger import EventLogger
from utils.video_source import VideoSource
from utils.overlay import (
    draw_person, draw_fps_counter, draw_alert_banner,
    draw_stats_overlay, update_heatmap, build_heatmap_image, frame_to_rgb_bytes
)

# ===========================================================================
# Page config
# ===========================================================================

st.set_page_config(
    page_title="Video Surveillance AI",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===========================================================================
# Session state initialization
# ===========================================================================

def init_state():
    defaults = {
        "session_id": None,
        "running": False,
        "paused": False,
        "models_loaded": False,
        "detector": None,
        "tracker": None,
        "pose": None,
        "action_clf": None,
        "zone_mgr": None,
        "dwell_tracker": None,
        "logger": None,
        "source": None,
        "latest_frame": None,
        "frame_idx": 0,
        "fps": 0.0,
        "fps_times": [],
        "heatmap": None,
        "active_tracks": [],
        "violation_active": False,
        "events": [],
        "person_table": [],
        "dwell_threshold": 60.0,
        "conf_threshold": 0.5,
        "show_bbox": True,
        "show_zones": True,
        "show_actions": True,
        "show_dwell": True,
        "stop_flag": False,
        "_proc_thread": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ===========================================================================
# Model loading
# ===========================================================================

@st.cache_resource(show_spinner="Loading AI models…")
def load_models(conf: float = 0.5):
    detector = PersonDetector(conf_threshold=conf)
    try:
        tracker = PersonTracker()
        if not tracker.is_ready():
            raise Exception
    except Exception:
        tracker = FallbackTracker()
    pose = PoseEstimator()
    action_clf = ActionClassifier()
    return {"detector": detector, "tracker": tracker,
            "pose": pose, "action_clf": action_clf}


# ===========================================================================
# Processing functions
# ===========================================================================

def start_processing(models: dict):
    st.session_state.stop_flag = False
    st.session_state.frame_idx = 0
    st.session_state.fps_times = []
    st.session_state.events = []
    st.session_state.person_table = []
    st.session_state.heatmap = None

    thread = threading.Thread(
        target=_proc_loop,
        args=(st.session_state, models),
        daemon=True,
    )
    thread.start()
    st.session_state._proc_thread = thread
    st.session_state.running = True


def _proc_loop(state, models: dict):
    """Background processing — writes results directly to session_state."""
    source = state.source
    detector = models["detector"]  # pulled from cache, not session_state
    tracker = models["tracker"]
    pose = models["pose"]
    action_clf = models["action_clf"]
    zone_mgr = state.zone_mgr
    dwell_tracker = state.dwell_tracker
    logger = state.logger

    heatmap_initialized = False

    while not state.stop_flag:
        if state.paused:
            time.sleep(0.05)
            continue
        if not source or not source.is_open():
            break

        ret, frame = source.read()
        if not ret or frame is None:
            if source.is_file():
                break
            time.sleep(0.01)
            continue

        now = time.time()
        state.frame_idx += 1
        ts = source.get_timestamp() or now

        # FPS
        state.fps_times = [t for t in state.fps_times if now - t <= 1.0]
        state.fps_times.append(now)
        state.fps = len(state.fps_times)

        frame_h, frame_w = frame.shape[:2]

        if not heatmap_initialized:
            state.heatmap = np.zeros((frame_h // 4, frame_w // 4), dtype=np.float32)
            heatmap_initialized = True

        # Detection
        detections = detector.detect(frame) if detector and detector.is_ready() else []
        ds_input = PersonDetector.to_deepsort_format(detections)

        # Tracking
        tracks = tracker.update(ds_input, frame) if tracker else []
        active_ids = [t[0] for t in tracks]
        dwell_tracker.deactivate_missing(active_ids, ts)

        violations_this_frame = []
        person_rows = []

        for track in tracks:
            tid, x1, y1, x2, y2, conf = track
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            # Zone check
            zone_ids = zone_mgr.get_zone_ids(x1, y1, x2, y2) if zone_mgr else []

            # Dwell
            dwell_rec = dwell_tracker.update(tid, ts, zone_ids)
            dwell_secs = dwell_rec.get_total_dwell()

            # Pose
            kps = None
            if pose and pose.is_ready() and (x2 - x1) > 20 and (y2 - y1) > 40:
                crop = frame[y1:y2, x1:x2]
                kps = pose.estimate(crop)

            # Action
            action = action_clf.classify(tid, kps, (x1, y1, x2, y2), ts, dwell_secs) if action_clf else "Unknown"
            dwell_tracker.set_action(tid, action, ts)

            # Violations
            prohibited = zone_mgr.get_prohibited_zones_for_bbox(x1, y1, x2, y2) if zone_mgr else []
            is_violation = len(prohibited) > 0
            if is_violation:
                violations_this_frame.extend(prohibited)
                dwell_tracker.increment_violation(tid)
                for z in prohibited:
                    evt = logger.log_zone_violation(tid, state.frame_idx, ts, z.id, z.name, action, [x1, y1, x2, y2])
                    state.events = state.events[-49:] + [evt]

            is_loitering = dwell_tracker.is_loitering(tid)

            # Draw bbox
            if state.show_bbox:
                draw_person(frame, tid, x1, y1, x2, y2,
                            action=action if state.show_actions else "",
                            dwell_str=dwell_rec.format_dwell() if state.show_dwell else "",
                            is_violation=is_violation,
                            is_loitering=is_loitering,
                            conf=conf)

            # Build person table row
            current_zone_names = []
            if zone_mgr:
                for zid in zone_ids:
                    z = zone_mgr.get_zone_by_id(zid)
                    if z:
                        current_zone_names.append(z.name)
            person_rows.append({
                "Track ID": tid,
                "Action": action,
                "Zone": ", ".join(current_zone_names) or "—",
                "Time in Zone": f"{dwell_secs:.0f}s",
                "Status": "⚠ VIOLATION" if is_violation else ("🟡 Loitering" if is_loitering else "✅ OK"),
            })

            logger.update_person_summary(
                tid, dwell_rec.first_seen, dwell_rec.last_seen,
                dwell_secs, dwell_rec.zones_visited, dwell_rec.violation_count,
                action, dwell_rec.action_history,
            )

        state.person_table = person_rows
        state.active_tracks = tracks
        state.violation_active = len(violations_this_frame) > 0

        # Heatmap
        if tracks and state.heatmap is not None:
            scaled = [(t[0], t[1]//4, t[2]//4, t[3]//4, t[4]//4, t[5]) for t in tracks]
            state.heatmap = update_heatmap(state.heatmap, scaled)

        # Zone overlays
        if state.show_zones and zone_mgr:
            frame = zone_mgr.draw_all_zones(frame)
            for z in violations_this_frame:
                z.draw_violation_flash(frame)

        # HUD
        draw_fps_counter(frame, state.fps, state.frame_idx)
        stats = dwell_tracker.get_stats()
        draw_stats_overlay(frame,
                           total_persons=stats.get("total_tracked", 0),
                           current_persons=stats.get("currently_active", 0),
                           violations=len(violations_this_frame),
                           avg_dwell=stats.get("avg_dwell", 0))
        if violations_this_frame:
            names = ", ".join(z.name for z in violations_this_frame)
            draw_alert_banner(frame, f"ZONE VIOLATION — {names}")

        state.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save heatmap periodically
        if state.frame_idx % 150 == 0 and logger and state.heatmap is not None:
            logger.save_heatmap(state.heatmap)

    state.running = False
    state.stop_flag = False
    if logger:
        logger.flush()
        logger.flush_summary()
    print("[main] Processing loop finished.")


# ===========================================================================
# Header
# ===========================================================================

st.markdown("""
<h1 style='margin-bottom:0'>📷 Video Surveillance AI</h1>
<p style='color:#888;margin-top:4px'>Real-time person detection, tracking, action recognition & zone enforcement</p>
""", unsafe_allow_html=True)
st.divider()

# ===========================================================================
# 3-column layout
# ===========================================================================

left_col, center_col, right_col = st.columns([1, 2, 1])

# ===========================================================================
# LEFT PANEL — Controls
# ===========================================================================

with left_col:
    st.subheader("⚙️ Controls")

    # Video source selection
    source_type = st.radio("Video Source", ["Upload File", "RTSP Stream"], horizontal=True)

    uploaded_file = None
    rtsp_url = ""

    if source_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Drag & drop video", type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed"
        )
    else:
        rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://user:pass@192.168.1.1:554/stream")

    # Zone config
    st.markdown("**Zone Configuration**")
    zones_file = st.file_uploader("Upload zones.json", type=["json"], key="zones_upload")

    # Settings
    st.markdown("**Settings**")
    dwell_threshold = st.slider("Dwell threshold (s)", 10, 300, 60, key="dwell_thresh_slider")
    conf_threshold = st.slider("Detection confidence", 0.3, 0.9, 0.5, step=0.05, key="conf_slider")

    st.markdown("**Overlay Toggles**")
    col1, col2 = st.columns(2)
    with col1:
        show_bbox    = st.checkbox("Bounding Boxes", value=True)
        show_zones   = st.checkbox("Zones", value=True)
    with col2:
        show_actions = st.checkbox("Actions", value=True)
        show_dwell   = st.checkbox("Dwell Time", value=True)

    st.session_state.show_bbox    = show_bbox
    st.session_state.show_zones   = show_zones
    st.session_state.show_actions = show_actions
    st.session_state.show_dwell   = show_dwell

    # Source info
    if st.session_state.source and st.session_state.source.is_open():
        info = st.session_state.source.get_info()
        st.markdown("**Source Info**")
        st.json({
            "Type": info["type"],
            "Resolution": f"{info['width']}×{info['height']}",
            "FPS": info["fps"],
            "Duration": f"{info['duration_seconds']}s" if info["duration_seconds"] > 0 else "Live",
        })
        if info["type"] == "file":
            progress = st.session_state.source.progress()
            st.progress(progress)

    # Start / Pause / Stop buttons
    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        start_clicked = st.button("▶ Start", type="primary", use_container_width=True)
    with btn_col2:
        pause_clicked = st.button("⏸ Pause", use_container_width=True)
    with btn_col3:
        stop_clicked  = st.button("⏹ Stop",  use_container_width=True)

    # Handle button actions
    if start_clicked and not st.session_state.running:
        models = load_models(conf_threshold)  # cached, not copied

        session_id = uuid.uuid4().hex[:8]
        st.session_state.session_id = session_id
        st.session_state.dwell_threshold = dwell_threshold
        st.session_state.conf_threshold = conf_threshold
        st.session_state.dwell_tracker = DwellTracker(dwell_threshold=dwell_threshold)
        st.session_state.logger = EventLogger(session_id)

        # Store zone_mgr and dwell_tracker references only
        if zones_file:
            zone_data = json.loads(zones_file.read())
            zm = ZoneManager()
            zm.load_from_dict(zone_data)
            st.session_state.zone_mgr = zm
        else:
            st.session_state.zone_mgr = ZoneManager("data/zones.json")

        # Open source
        source = VideoSource()
        ok = False
        if source_type == "Upload File" and uploaded_file:
            suffix = Path(uploaded_file.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.flush()
            ok = source.open_file(tmp.name)
        elif source_type == "RTSP Stream" and rtsp_url:
            ok = source.open_rtsp(rtsp_url)

        if ok:
            st.session_state.source = source
            start_processing(models)  # pass models directly
            st.success(f"Session {session_id} started!")
        else:
            st.error("Could not open video source.")

    if pause_clicked and st.session_state.running:
        st.session_state.paused = not st.session_state.paused
        if st.session_state.source:
            if st.session_state.paused:
                st.session_state.source.pause()
            else:
                st.session_state.source.resume()

    if stop_clicked:
        st.session_state.stop_flag = True
        st.session_state.running = False
        if st.session_state.source:
            st.session_state.source.stop()
        st.info("Session stopped.")

# ===========================================================================
# CENTER PANEL — Live Video Feed
# ===========================================================================

with center_col:
    st.subheader("🎬 Live Feed")

    status_badge = "🟢 LIVE" if (st.session_state.running and not st.session_state.paused) else \
                   "⏸ PAUSED" if st.session_state.paused else "⚫ IDLE"
    fps_text = f"FPS: {st.session_state.fps:.1f}" if st.session_state.running else ""
    frame_text = f"Frame: {st.session_state.frame_idx}" if st.session_state.running else ""

    st.markdown(
        f"<div style='display:flex;gap:12px;align-items:center;margin-bottom:6px'>"
        f"<span style='font-size:1.1em'>{status_badge}</span>"
        f"<span style='color:#888'>{fps_text}</span>"
        f"<span style='color:#888'>{frame_text}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Violation alert
    if st.session_state.violation_active:
        st.error("⚠️  ACTIVE ZONE VIOLATION DETECTED")

    # Video display placeholder
    video_placeholder = st.empty()

    if st.session_state.latest_frame is not None:
        video_placeholder.image(
            st.session_state.latest_frame,
            use_container_width=True,
            caption=f"Frame {st.session_state.frame_idx}",
        )
    else:
        video_placeholder.markdown(
            "<div style='height:360px;background:#111;border-radius:8px;"
            "display:flex;align-items:center;justify-content:center;"
            "color:#555;font-size:1.2em'>No video stream — press ▶ Start</div>",
            unsafe_allow_html=True
        )

# ===========================================================================
# RIGHT PANEL — Metrics & Event Log
# ===========================================================================

with right_col:
    st.subheader("📊 Metrics")

    # Live stats
    stats = st.session_state.dwell_tracker.get_stats() if st.session_state.dwell_tracker else {}

    m1, m2 = st.columns(2)
    m1.metric("Total Detected",  stats.get("total_tracked", 0))
    m2.metric("In Frame",        stats.get("currently_active", 0))
    m3, m4 = st.columns(2)
    m3.metric("Violations",      "⚠" if st.session_state.violation_active else stats.get("loitering_count", 0))
    m4.metric("Avg Dwell (s)",   f"{stats.get('avg_dwell', 0):.0f}")

    st.markdown("---")

    # Event log
    st.markdown("**📋 Live Event Log**")
    events = st.session_state.events[-20:]
    if events:
        for evt in reversed(events):
            evt_type = evt.get("event_type", "")
            icon = "🚨" if "violation" in evt_type else "ℹ️"
            t = evt.get("datetime", "")[:19].replace("T", " ")
            st.markdown(
                f"<small>{icon} <b>ID:{evt.get('track_id','?')}</b> "
                f"{evt_type} @ {t}</small>",
                unsafe_allow_html=True
            )
    else:
        st.caption("No events yet.")

    st.markdown("---")

    # Per-person table
    st.markdown("**👤 Person Tracking Table**")
    if st.session_state.person_table:
        df_persons = pd.DataFrame(st.session_state.person_table)
        st.dataframe(df_persons, use_container_width=True, height=220)
    else:
        st.caption("No persons tracked yet.")

# ===========================================================================
# BOTTOM TABS
# ===========================================================================

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Heatmap", "⏱ Dwell Time Chart", "🎭 Action Breakdown", "💾 Export"])

# -- Tab 1: Heatmap --
with tab1:
    st.subheader("Position Heatmap")
    if st.session_state.heatmap is not None and st.session_state.heatmap.sum() > 0:
        h_arr = st.session_state.heatmap
        # Normalize and apply colormap
        norm = cv2.normalize(h_arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        st.image(colored_rgb, use_container_width=True, caption="Person position density (brighter = more activity)")
    else:
        st.info("Heatmap will appear after processing starts.")

# -- Tab 2: Dwell Time Chart --
with tab2:
    st.subheader("Top Dwell Times")
    if st.session_state.dwell_tracker:
        records = st.session_state.dwell_tracker.get_all_records()
        if records:
            data = {
                f"ID:{tid}": round(rec.get_total_dwell(), 1)
                for tid, rec in sorted(
                    records.items(),
                    key=lambda x: x[1].get_total_dwell(),
                    reverse=True
                )[:10]
            }
            df_dwell = pd.DataFrame.from_dict(
                {"Track": list(data.keys()), "Dwell (s)": list(data.values())}
            )
            st.bar_chart(df_dwell.set_index("Track"), use_container_width=True)
        else:
            st.info("No dwell data yet.")
    else:
        st.info("Start processing to see dwell times.")

# -- Tab 3: Action Breakdown --
with tab3:
    st.subheader("Action Distribution")
    if st.session_state.action_clf:
        dist = st.session_state.action_clf.get_action_distribution()
        if dist:
            df_actions = pd.DataFrame.from_dict(
                {"Action": list(dist.keys()), "Count": list(dist.values())}
            )
            st.bar_chart(df_actions.set_index("Action"), use_container_width=True)
        else:
            st.info("No action data yet.")
    else:
        st.info("Start processing to see action distribution.")

# -- Tab 4: Export --
with tab4:
    st.subheader("Export Session Data")
    if st.session_state.logger:
        col_a, col_b = st.columns(2)
        with col_a:
            events_bytes = st.session_state.logger.get_events_json_bytes()
            st.download_button(
                "⬇️  Download events.json",
                data=events_bytes,
                file_name=f"events_{st.session_state.session_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_b:
            summary_bytes = st.session_state.logger.get_summary_csv_bytes()
            st.download_button(
                "⬇️  Download summary.csv",
                data=summary_bytes,
                file_name=f"summary_{st.session_state.session_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if st.session_state.session_id:
            st.caption(f"Session: `{st.session_state.session_id}` | "
                       f"Events: {len(st.session_state.events)} | "
                       f"Frames processed: {st.session_state.frame_idx}")
    else:
        st.info("Start a session to export data.")

# ===========================================================================
# Auto-refresh while running
# ===========================================================================

if st.session_state.running and not st.session_state.paused:
    time.sleep(0.1)
    st.rerun()