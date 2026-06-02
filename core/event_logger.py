"""
event_logger.py
---------------
Handles all local storage: events.json, summary.csv, heatmap.npy.
Thread-safe writes using a lock.
"""

import json
import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class EventLogger:
    """Logs detection events and summaries to local files."""

    def __init__(self, session_id: str, base_dir: str = "data/sessions"):
        self.session_id = session_id
        self.session_dir = Path(base_dir) / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.session_dir / "events.json"
        self.summary_path = self.session_dir / "summary.csv"
        self.heatmap_path = self.session_dir / "heatmap.npy"

        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        self._summary: Dict[str, Dict] = {}  # track_id -> summary dict

        # Initialize events file
        if not self.events_path.exists():
            self._write_events_file()

        # Initialize summary CSV
        if not self.summary_path.exists():
            self._init_summary_csv()

    # ------------------------------------------------------------------
    # Event Logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        track_id: int,
        frame_idx: int,
        timestamp: float,
        data: Optional[Dict] = None,
    ) -> Dict:
        """Log a single timestamped event."""
        event = {
            "id": len(self._events),
            "session_id": self.session_id,
            "event_type": event_type,  # e.g. "person_detected", "zone_violation", "action_change"
            "track_id": track_id,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            **(data or {}),
        }
        with self._lock:
            self._events.append(event)
            # Batch-write every 50 events to reduce I/O
            if len(self._events) % 50 == 0:
                self._write_events_file()
        return event

    def log_zone_violation(
        self,
        track_id: int,
        frame_idx: int,
        timestamp: float,
        zone_id: str,
        zone_name: str,
        action: str,
        bbox: List[int],
    ):
        return self.log_event(
            "zone_violation",
            track_id,
            frame_idx,
            timestamp,
            {
                "zone_id": zone_id,
                "zone_name": zone_name,
                "action": action,
                "bbox": bbox,
            },
        )

    def log_action_change(
        self,
        track_id: int,
        frame_idx: int,
        timestamp: float,
        old_action: str,
        new_action: str,
    ):
        return self.log_event(
            "action_change",
            track_id,
            frame_idx,
            timestamp,
            {"old_action": old_action, "new_action": new_action},
        )

    def flush(self):
        """Force-write all pending events to disk."""
        with self._lock:
            self._write_events_file()

    def _write_events_file(self):
        with open(self.events_path, "w") as f:
            json.dump(self._events, f, indent=2, default=str)

    def get_recent_events(self, n: int = 20) -> List[Dict]:
        """Return the last n events."""
        return self._events[-n:]

    def get_all_events(self) -> List[Dict]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Summary (per-person CSV)
    # ------------------------------------------------------------------

    def _init_summary_csv(self):
        columns = [
            "track_id", "first_seen", "last_seen",
            "total_dwell_seconds", "zones_visited", "violations",
            "primary_action", "action_history", "status",
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.summary_path, index=False)

    def update_person_summary(
        self,
        track_id: int,
        first_seen: float,
        last_seen: float,
        dwell_seconds: float,
        zones_visited: List[str],
        violations: int,
        primary_action: str,
        action_history: List[str],
        status: str = "active",
    ):
        """Upsert a row in the in-memory summary dict."""
        with self._lock:
            self._summary[str(track_id)] = {
                "track_id": track_id,
                "first_seen": datetime.fromtimestamp(first_seen).isoformat(),
                "last_seen": datetime.fromtimestamp(last_seen).isoformat(),
                "total_dwell_seconds": round(dwell_seconds, 2),
                "zones_visited": "|".join(zones_visited),
                "violations": violations,
                "primary_action": primary_action,
                "action_history": "|".join(action_history[-10:]),  # last 10
                "status": status,
            }

    def flush_summary(self):
        """Write summary dict to CSV."""
        with self._lock:
            if self._summary:
                df = pd.DataFrame(list(self._summary.values()))
                df.to_csv(self.summary_path, index=False)

    def get_summary_df(self) -> pd.DataFrame:
        if self._summary:
            return pd.DataFrame(list(self._summary.values()))
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def save_heatmap(self, heatmap: np.ndarray):
        np.save(str(self.heatmap_path), heatmap)

    def load_heatmap(self) -> Optional[np.ndarray]:
        if self.heatmap_path.exists():
            return np.load(str(self.heatmap_path))
        return None

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def get_events_json_bytes(self) -> bytes:
        return json.dumps(self._events, indent=2, default=str).encode()

    def get_summary_csv_bytes(self) -> bytes:
        self.flush_summary()
        if self.summary_path.exists():
            return self.summary_path.read_bytes()
        return b""