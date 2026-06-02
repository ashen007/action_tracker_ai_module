"""
dwell_tracker.py
----------------
Tracks how long each person (by Track ID) has been present overall
and inside each specific zone.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class PersonDwellRecord:
    """Holds all timing info for a single tracked person."""

    def __init__(self, track_id: int, first_seen: float):
        self.track_id = track_id
        self.first_seen: float = first_seen
        self.last_seen: float = first_seen
        self.total_dwell: float = 0.0  # cumulative seconds in frame

        # Zone timing: zone_id -> (entry_time, cumulative_seconds)
        self.zone_entry: Dict[str, float] = {}       # zone_id -> entry timestamp
        self.zone_dwell: Dict[str, float] = defaultdict(float)  # zone_id -> total seconds
        self.zones_visited: List[str] = []
        self.violation_count: int = 0

        # Action tracking
        self.current_action: str = "Unknown"
        self.action_history: List[str] = []
        self.action_start: float = first_seen

        self.is_active: bool = True

    def update(self, timestamp: float, current_zones: List[str]):
        """
        Called every frame this person is visible.
        current_zones: list of zone IDs the person is currently inside.
        """
        delta = timestamp - self.last_seen
        self.last_seen = timestamp
        self.total_dwell += delta

        # Handle zone exits
        for zone_id in list(self.zone_entry.keys()):
            if zone_id not in current_zones:
                # Person left this zone
                entry_t = self.zone_entry.pop(zone_id)
                self.zone_dwell[zone_id] += timestamp - entry_t

        # Handle zone entries
        for zone_id in current_zones:
            if zone_id not in self.zone_entry:
                self.zone_entry[zone_id] = timestamp
                if zone_id not in self.zones_visited:
                    self.zones_visited.append(zone_id)

    def mark_lost(self, timestamp: float):
        """Called when track is lost — close all open zone timers."""
        for zone_id, entry_t in self.zone_entry.items():
            self.zone_dwell[zone_id] += timestamp - entry_t
        self.zone_entry.clear()
        self.is_active = False
        self.last_seen = timestamp

    def get_zone_dwell(self, zone_id: str) -> float:
        """Total seconds in a zone (including current open interval)."""
        current = 0.0
        if zone_id in self.zone_entry:
            current = time.time() - self.zone_entry[zone_id]
        return self.zone_dwell[zone_id] + current

    def get_total_dwell(self) -> float:
        return self.total_dwell

    def set_action(self, action: str, timestamp: float):
        if action != self.current_action:
            self.action_history.append(self.current_action)
            self.current_action = action
            self.action_start = timestamp

    def format_dwell(self) -> str:
        """Return human-readable dwell time."""
        secs = int(self.total_dwell)
        if secs < 60:
            return f"{secs}s"
        return f"{secs // 60}m{secs % 60:02d}s"


class DwellTracker:
    """Manages PersonDwellRecord for all active and historical tracks."""

    def __init__(self, dwell_threshold: float = 60.0):
        """
        dwell_threshold: seconds before a person is flagged as 'loitering'.
        """
        self.dwell_threshold = dwell_threshold
        self._records: Dict[int, PersonDwellRecord] = {}

    def update(
        self,
        track_id: int,
        timestamp: float,
        current_zones: List[str],
    ) -> PersonDwellRecord:
        """Update or create a record for this track."""
        if track_id not in self._records:
            self._records[track_id] = PersonDwellRecord(track_id, timestamp)
        rec = self._records[track_id]
        rec.is_active = True
        rec.update(timestamp, current_zones)
        return rec

    def mark_lost(self, track_id: int, timestamp: float):
        if track_id in self._records:
            self._records[track_id].mark_lost(timestamp)

    def set_action(self, track_id: int, action: str, timestamp: float):
        if track_id in self._records:
            self._records[track_id].set_action(action, timestamp)

    def increment_violation(self, track_id: int):
        if track_id in self._records:
            self._records[track_id].violation_count += 1

    def get_record(self, track_id: int) -> Optional[PersonDwellRecord]:
        return self._records.get(track_id)

    def get_all_records(self) -> Dict[int, PersonDwellRecord]:
        return dict(self._records)

    def get_active_records(self) -> Dict[int, PersonDwellRecord]:
        return {tid: r for tid, r in self._records.items() if r.is_active}

    def is_loitering(self, track_id: int) -> bool:
        rec = self._records.get(track_id)
        if rec:
            return rec.get_total_dwell() >= self.dwell_threshold
        return False

    def get_stats(self) -> Dict:
        active = self.get_active_records()
        dwells = [r.get_total_dwell() for r in active.values()]
        return {
            "total_tracked": len(self._records),
            "currently_active": len(active),
            "avg_dwell": round(sum(dwells) / len(dwells), 1) if dwells else 0,
            "max_dwell": round(max(dwells), 1) if dwells else 0,
            "loitering_count": sum(1 for r in active.values()
                                   if r.get_total_dwell() >= self.dwell_threshold),
        }

    def deactivate_missing(self, active_ids: List[int], timestamp: float):
        """Mark tracks not seen in this frame as inactive."""
        for tid, rec in self._records.items():
            if rec.is_active and tid not in active_ids:
                self.mark_lost(tid, timestamp)