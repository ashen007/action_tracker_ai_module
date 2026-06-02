"""
run.py
------
CLI entry point for the Video Surveillance AI system.

Usage:
  python run.py                  # launch Streamlit dashboard
  python run.py --api            # launch FastAPI server only
  python run.py --demo           # run demo on sample public-domain video
  python run.py --demo --headless  # headless demo (no UI, outputs to files)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

# Sample public-domain video (Blender Foundation, CC)
DEMO_VIDEO_URL = (
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
)
DEMO_VIDEO_LOCAL = "data/demo_video.mp4"


def download_demo_video() -> str:
    """Download sample video if not already present."""
    path = Path(DEMO_VIDEO_LOCAL)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100_000:
        print(f"[Demo] Using cached video: {path}")
        return str(path)

    print(f"[Demo] Downloading sample video from {DEMO_VIDEO_URL} ...")
    try:
        urllib.request.urlretrieve(DEMO_VIDEO_URL, str(path))
        print(f"[Demo] Downloaded to {path} ({path.stat().st_size // 1024} KB)")
        return str(path)
    except Exception as e:
        print(f"[Demo] Download failed: {e}")
        print("[Demo] Please provide your own video file as data/demo_video.mp4")
        return ""


def run_headless_demo(video_path: str, max_frames: int = 300):
    """Run analysis on video without UI, save results to data/sessions/."""
    sys.path.insert(0, str(Path(__file__).parent))

    import uuid
    import cv2
    import numpy as np

    from models.detector import PersonDetector
    from models.tracker import PersonTracker, FallbackTracker
    from models.pose_estimator import PoseEstimator
    from models.action_classifier import ActionClassifier
    from core.zone_manager import ZoneManager
    from core.dwell_tracker import DwellTracker
    from core.event_logger import EventLogger
    from utils.video_source import VideoSource
    from utils.overlay import (
        draw_person, draw_fps_counter, draw_stats_overlay,
        update_heatmap, frame_to_rgb_bytes
    )

    session_id = f"demo_{uuid.uuid4().hex[:6]}"
    print(f"\n[Demo] Starting headless session: {session_id}")
    print(f"[Demo] Video: {video_path}")

    # Load components
    detector = PersonDetector(conf_threshold=0.5)
    try:
        tracker = PersonTracker()
        if not tracker.is_ready():
            raise Exception
    except Exception:
        tracker = FallbackTracker()

    pose = PoseEstimator()
    action_clf = ActionClassifier()
    zone_mgr = ZoneManager("data/zones.json")
    dwell_tracker = DwellTracker(dwell_threshold=60.0)
    logger = EventLogger(session_id)

    source = VideoSource()
    if not source.open_file(video_path):
        print("[Demo] Cannot open video.")
        return

    info = source.get_info()
    print(f"[Demo] Video info: {info['width']}x{info['height']} @ {info['fps']} fps")

    heatmap = None
    frame_count = 0
    t0 = time.time()

    print("[Demo] Processing frames... (press Ctrl+C to stop early)")
    try:
        while frame_count < max_frames:
            ret, frame = source.read()
            if not ret or frame is None:
                break

            frame_count += 1
            ts = source.get_timestamp() or time.time()

            if heatmap is None:
                h, w = frame.shape[:2]
                heatmap = np.zeros((h // 4, w // 4), dtype=np.float32)

            # Detect + track
            detections = detector.detect(frame)
            ds_input = PersonDetector.to_deepsort_format(detections)
            tracks = tracker.update(ds_input, frame)
            active_ids = [t[0] for t in tracks]
            dwell_tracker.deactivate_missing(active_ids, ts)

            for track in tracks:
                tid, x1, y1, x2, y2, conf = track
                x1, y1 = max(0, x1), max(0, y1)
                h_f, w_f = frame.shape[:2]
                x2, y2 = min(w_f, x2), min(h_f, y2)

                zone_ids = zone_mgr.get_zone_ids(x1, y1, x2, y2)
                dwell_rec = dwell_tracker.update(tid, ts, zone_ids)
                dwell_secs = dwell_rec.get_total_dwell()

                kps = None
                if pose.is_ready() and (x2 - x1) > 20 and (y2 - y1) > 40:
                    crop = frame[y1:y2, x1:x2]
                    kps = pose.estimate(crop)

                action = action_clf.classify(tid, kps, (x1, y1, x2, y2), ts, dwell_secs)
                dwell_tracker.set_action(tid, action, ts)

                prohibited = zone_mgr.get_prohibited_zones_for_bbox(x1, y1, x2, y2)
                if prohibited:
                    dwell_tracker.increment_violation(tid)
                    for z in prohibited:
                        logger.log_zone_violation(tid, frame_count, ts, z.id, z.name, action, [x1, y1, x2, y2])

                draw_person(frame, tid, x1, y1, x2, y2, action=action,
                            dwell_str=dwell_rec.format_dwell(),
                            is_violation=len(prohibited) > 0,
                            is_loitering=dwell_tracker.is_loitering(tid))

                logger.update_person_summary(
                    tid, dwell_rec.first_seen, dwell_rec.last_seen,
                    dwell_secs, dwell_rec.zones_visited, dwell_rec.violation_count,
                    action, dwell_rec.action_history,
                )

            if tracks:
                scaled = [(t[0], t[1]//4, t[2]//4, t[3]//4, t[4]//4, t[5]) for t in tracks]
                heatmap = update_heatmap(heatmap, scaled)

            draw_fps_counter(frame, frame_count / max(0.001, time.time() - t0), frame_count)
            stats = dwell_tracker.get_stats()
            draw_stats_overlay(frame, stats.get("total_tracked", 0),
                               stats.get("currently_active", 0), 0, stats.get("avg_dwell", 0))

            if frame_count % 50 == 0:
                elapsed = time.time() - t0
                print(f"[Demo] Frame {frame_count}/{max_frames} | "
                      f"Persons: {stats.get('currently_active', 0)} | "
                      f"Speed: {frame_count/elapsed:.1f} fps")

    except KeyboardInterrupt:
        print("\n[Demo] Interrupted.")

    source.stop()
    elapsed = time.time() - t0
    print(f"\n[Demo] Processed {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} fps)")

    # Save outputs
    logger.flush()
    logger.flush_summary()
    if heatmap is not None:
        logger.save_heatmap(heatmap)

    stats = dwell_tracker.get_stats()
    print(f"\n[Demo] === Results ===")
    print(f"  Total persons tracked: {stats.get('total_tracked', 0)}")
    print(f"  Max dwell time: {stats.get('max_dwell', 0):.1f}s")
    print(f"  Events logged: {len(logger.get_all_events())}")
    print(f"  Output directory: data/sessions/{session_id}/")
    print(f"\n  events.json   → data/sessions/{session_id}/events.json")
    print(f"  summary.csv   → data/sessions/{session_id}/summary.csv")
    print(f"  heatmap.npy   → data/sessions/{session_id}/heatmap.npy")


def launch_streamlit():
    """Launch the Streamlit dashboard."""
    print("[Launcher] Starting Streamlit dashboard on http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "main.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
    ])


def launch_api():
    """Launch FastAPI server."""
    print("[Launcher] Starting FastAPI server on http://localhost:8000")
    subprocess.run([
        sys.executable, "-m", "uvicorn", "api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ])


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Surveillance AI")
    parser.add_argument("--demo", action="store_true", help="Run demo on sample video")
    parser.add_argument("--headless", action="store_true", help="Run demo without UI")
    parser.add_argument("--api", action="store_true", help="Start FastAPI server only")
    parser.add_argument("--frames", type=int, default=300, help="Max frames for demo")
    parser.add_argument("--video", type=str, default="", help="Path to custom demo video")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    if args.api:
        launch_api()
    elif args.demo:
        video_path = args.video or download_demo_video()
        if not video_path:
            sys.exit(1)
        if args.headless:
            run_headless_demo(video_path, max_frames=args.frames)
        else:
            # Set env var so main.py uses the demo video
            os.environ["DEMO_VIDEO"] = video_path
            launch_streamlit()
    else:
        launch_streamlit()