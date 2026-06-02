"""
api.py
------
FastAPI REST API for the video surveillance system.
Runs on http://localhost:8000

Endpoints:
  POST /session/start       — start a new analysis session
  POST /session/stop        — stop current session
  POST /video/upload        — upload video file
  POST /video/rtsp          — connect to RTSP stream
  GET  /video/frame         — get latest annotated frame (JPEG bytes)
  GET  /analytics/stats     — live metrics
  GET  /analytics/events    — recent event log
  GET  /analytics/heatmap   — heatmap image (JPEG)
  GET  /export/events       — download events.json
  GET  /export/summary      — download summary.csv
  POST /control/{action}    — pause / resume / stop
  GET  /zones               — list loaded zones
  POST /zones/upload        — upload zones.json
"""

import io
import os
import time
import uuid
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel

# ---- Import project modules ----
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

# ---------------------------------------------------------------------------
app = FastAPI(title="Video Surveillance AI", version="1.0.0")

# ---------------------------------------------------------------------------
# Global session state
# ---------------------------------------------------------------------------

class SessionState:
    def __init__(self):
        self.session_id: Optional[str] = None
        self.running: bool = False
        self.paused: bool = False

        # Models
        self.detector: Optional[PersonDetector] = None
        self.tracker = None
        self.pose: Optional[PoseEstimator] = None
        self.action_clf: Optional[ActionClassifier] = None

        # Core managers
        self.zone_mgr: Optional[ZoneManager] = None
        self.dwell_tracker: Optional[DwellTracker] = None
        self.logger: Optional[EventLogger] = None

        # Video
        self.source: Optional[VideoSource] = None

        # Frame state
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_idx: int = 0
        self.fps: float = 0.0
        self._fps_times: list = []

        # Heatmap
        self.heatmap: Optional[np.ndarray] = None

        # Active tracks this frame
        self.active_tracks: list = []
        self.violation_active: bool = False

        # Config
        self.dwell_threshold: float = 60.0
        self.conf_threshold: float = 0.5
        self.show_overlays: bool = True

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def reset(self):
        self.__init__()


_state = SessionState()


# ---------------------------------------------------------------------------
# Background processing loop
# ---------------------------------------------------------------------------

def _processing_loop():
    """Main frame-by-frame analysis loop running in background thread."""
    state = _state
    prev_time = time.time()

    while state.running and state.source and state.source.is_open():
        if state.paused:
            time.sleep(0.05)
            continue

        ret, frame = state.source.read()
        if not ret or frame is None:
            if state.source.is_file():
                break  # End of file
            time.sleep(0.01)
            continue

        now = time.time()
        state.frame_idx += 1
        ts = state.source.get_timestamp() or now

        # FPS calculation
        state._fps_times.append(now)
        state._fps_times = [t for t in state._fps_times if now - t <= 1.0]
        state.fps = len(state._fps_times)

        frame_h, frame_w = frame.shape[:2]

        # Initialize heatmap on first frame
        if state.heatmap is None:
            state.heatmap = np.zeros((frame_h // 4, frame_w // 4), dtype=np.float32)

        # 1. Detect persons
        detections = []
        if state.detector and state.detector.is_ready():
            detections = state.detector.detect(frame)
        ds_input = PersonDetector.to_deepsort_format(detections)

        # 2. Track
        tracks = []
        if state.tracker and state.tracker.is_ready():
            tracks = state.tracker.update(ds_input, frame)

        active_ids = [t[0] for t in tracks]

        # Mark lost tracks
        if state.dwell_tracker:
            state.dwell_tracker.deactivate_missing(active_ids, ts)

        # 3. Per-person analysis
        violations_this_frame = []

        for track in tracks:
            tid, x1, y1, x2, y2, conf = track
            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            # Zone check
            zone_ids = []
            if state.zone_mgr:
                zone_ids = state.zone_mgr.get_zone_ids(x1, y1, x2, y2)

            # Dwell tracking
            dwell_rec = None
            if state.dwell_tracker:
                dwell_rec = state.dwell_tracker.update(tid, ts, zone_ids)

            dwell_secs = dwell_rec.get_total_dwell() if dwell_rec else 0.0

            # Pose estimation (on crop)
            kps = None
            if state.pose and state.pose.is_ready() and (x2 - x1) > 20 and (y2 - y1) > 40:
                crop = frame[y1:y2, x1:x2]
                kps = state.pose.estimate(crop)

            # Action classification
            action = "Unknown"
            if state.action_clf:
                action = state.action_clf.classify(
                    tid, kps, (x1, y1, x2, y2), ts, dwell_secs
                )
            if state.dwell_tracker:
                state.dwell_tracker.set_action(tid, action, ts)

            # Prohibited zone violation check
            prohibited_zones = []
            if state.zone_mgr:
                prohibited_zones = state.zone_mgr.get_prohibited_zones_for_bbox(x1, y1, x2, y2)

            is_violation = len(prohibited_zones) > 0
            if is_violation:
                violations_this_frame.extend(prohibited_zones)
                if state.dwell_tracker:
                    state.dwell_tracker.increment_violation(tid)
                if state.logger:
                    for z in prohibited_zones:
                        state.logger.log_zone_violation(
                            tid, state.frame_idx, ts, z.id, z.name,
                            action, [x1, y1, x2, y2]
                        )

            is_loitering = state.dwell_tracker.is_loitering(tid) if state.dwell_tracker else False

            # Draw on frame
            if state.show_overlays:
                dwell_str = dwell_rec.format_dwell() if dwell_rec else ""
                draw_person(frame, tid, x1, y1, x2, y2,
                            action=action, dwell_str=dwell_str,
                            is_violation=is_violation, is_loitering=is_loitering,
                            conf=conf)

            # Update person summary
            if state.logger and dwell_rec:
                state.logger.update_person_summary(
                    tid,
                    first_seen=dwell_rec.first_seen,
                    last_seen=dwell_rec.last_seen,
                    dwell_seconds=dwell_rec.get_total_dwell(),
                    zones_visited=dwell_rec.zones_visited,
                    violations=dwell_rec.violation_count,
                    primary_action=action,
                    action_history=dwell_rec.action_history,
                )

        # Update heatmap (downsampled)
        if tracks and state.heatmap is not None:
            scaled = [(t[0], t[1]//4, t[2]//4, t[3]//4, t[4]//4, t[5]) for t in tracks]
            state.heatmap = update_heatmap(state.heatmap, scaled)

        # Draw zones
        if state.show_overlays and state.zone_mgr:
            frame = state.zone_mgr.draw_all_zones(frame)
            # Flash violation zones
            for z in violations_this_frame:
                z.draw_violation_flash(frame)

        # Draw HUD
        if state.show_overlays:
            draw_fps_counter(frame, state.fps, state.frame_idx)
            stats = state.dwell_tracker.get_stats() if state.dwell_tracker else {}
            draw_stats_overlay(
                frame,
                total_persons=stats.get("total_tracked", 0),
                current_persons=stats.get("currently_active", 0),
                violations=len(violations_this_frame),
                avg_dwell=stats.get("avg_dwell", 0),
            )
            if violations_this_frame:
                names = ", ".join(z.name for z in violations_this_frame)
                draw_alert_banner(frame, f"ZONE VIOLATION — {names}")

        state.violation_active = len(violations_this_frame) > 0
        state.active_tracks = tracks

        with state._lock:
            state.latest_frame = frame.copy()

        # Save heatmap periodically
        if state.frame_idx % 150 == 0 and state.logger and state.heatmap is not None:
            state.logger.save_heatmap(state.heatmap)

    state.running = False
    if state.logger:
        state.logger.flush()
        state.logger.flush_summary()
    print("[API] Processing loop ended.")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RTSPRequest(BaseModel):
    url: str
    dwell_threshold: float = 60.0
    conf_threshold: float = 0.5

class ControlRequest(BaseModel):
    pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Video Surveillance AI API"}


@app.post("/session/start")
def start_session(dwell_threshold: float = 60.0, conf_threshold: float = 0.5):
    global _state
    if _state.running:
        return {"status": "already_running", "session_id": _state.session_id}

    session_id = uuid.uuid4().hex[:8]
    _state.session_id = session_id
    _state.dwell_threshold = dwell_threshold
    _state.conf_threshold = conf_threshold

    # Initialize models
    print("[API] Loading models...")
    _state.detector = PersonDetector(conf_threshold=conf_threshold)
    try:
        _state.tracker = PersonTracker()
        if not _state.tracker.is_ready():
            raise Exception("DeepSORT failed")
    except Exception:
        print("[API] Falling back to IoU tracker.")
        _state.tracker = FallbackTracker()

    _state.pose = PoseEstimator()
    _state.action_clf = ActionClassifier()
    _state.zone_mgr = ZoneManager("data/zones.json")
    _state.dwell_tracker = DwellTracker(dwell_threshold=dwell_threshold)
    _state.logger = EventLogger(session_id)

    return {"status": "ready", "session_id": session_id}


@app.post("/session/stop")
def stop_session():
    _state.running = False
    if _state.source:
        _state.source.stop()
    if _state.logger:
        _state.logger.flush()
        _state.logger.flush_summary()
    return {"status": "stopped"}


@app.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    if not _state.session_id:
        raise HTTPException(400, "Start a session first.")

    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp_path = tmp.name

    source = VideoSource()
    if not source.open_file(tmp_path):
        raise HTTPException(400, f"Cannot open video file: {file.filename}")

    _state.source = source
    _state.running = True
    _state.frame_idx = 0

    thread = threading.Thread(target=_processing_loop, daemon=True)
    thread.start()
    _state._thread = thread

    return {"status": "started", "info": source.get_info()}


@app.post("/video/rtsp")
def connect_rtsp(req: RTSPRequest):
    if not _state.session_id:
        raise HTTPException(400, "Start a session first.")

    source = VideoSource()
    if not source.open_rtsp(req.url):
        raise HTTPException(400, f"Cannot connect to RTSP: {req.url}")

    _state.source = source
    _state.running = True
    _state.frame_idx = 0

    thread = threading.Thread(target=_processing_loop, daemon=True)
    thread.start()
    _state._thread = thread

    return {"status": "streaming", "info": source.get_info()}


@app.get("/video/frame")
def get_frame():
    with _state._lock:
        frame = _state.latest_frame
    if frame is None:
        raise HTTPException(404, "No frame available yet.")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/analytics/stats")
def get_stats():
    stats = _state.dwell_tracker.get_stats() if _state.dwell_tracker else {}
    return {
        "session_id": _state.session_id,
        "running": _state.running,
        "frame_idx": _state.frame_idx,
        "fps": round(_state.fps, 1),
        "violation_active": _state.violation_active,
        **stats,
    }


@app.get("/analytics/events")
def get_events(n: int = 20):
    if not _state.logger:
        return {"events": []}
    return {"events": _state.logger.get_recent_events(n)}


@app.get("/analytics/heatmap")
def get_heatmap():
    if _state.heatmap is None or _state.source is None:
        raise HTTPException(404, "No heatmap data.")
    h, w = _state.heatmap.shape
    img = build_heatmap_image(_state.heatmap, (h, w))
    _, buf = cv2.imencode(".jpg", img)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/control/{action}")
def control(action: str):
    if action == "pause":
        _state.paused = True
        if _state.source: _state.source.pause()
        return {"status": "paused"}
    elif action == "resume":
        _state.paused = False
        if _state.source: _state.source.resume()
        return {"status": "resumed"}
    elif action == "stop":
        return stop_session()
    raise HTTPException(400, f"Unknown action: {action}")


@app.get("/export/events")
def export_events():
    if not _state.logger:
        raise HTTPException(404, "No active session.")
    data = _state.logger.get_events_json_bytes()
    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=events_{_state.session_id}.json"},
    )


@app.get("/export/summary")
def export_summary():
    if not _state.logger:
        raise HTTPException(404, "No active session.")
    data = _state.logger.get_summary_csv_bytes()
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=summary_{_state.session_id}.csv"},
    )


@app.get("/zones")
def get_zones():
    if not _state.zone_mgr:
        return {"zones": []}
    return _state.zone_mgr.to_dict()


@app.post("/zones/upload")
async def upload_zones(file: UploadFile = File(...)):
    import json
    content = await file.read()
    data = json.loads(content)
    if not _state.zone_mgr:
        _state.zone_mgr = ZoneManager()
    n = _state.zone_mgr.load_from_dict(data)
    return {"status": "loaded", "zones_count": n}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")