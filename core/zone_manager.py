"""
zone_manager.py
---------------
Loads zone definitions from zones.json, draws zone overlays,
and checks whether a point/bbox is inside any zone using Shapely.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("[ZoneManager] WARNING: shapely not installed — zone checks disabled.")


class Zone:
    """Represents a single monitoring zone."""

    def __init__(self, config: Dict):
        self.id: str = config["id"]
        self.name: str = config["name"]
        self.gps_coords: List[List[float]] = config.get("gps_coords", [])
        self.pixel_coords: List[List[int]] = config["pixel_coords"]
        self.prohibited: bool = config.get("prohibited", False)
        self.color: Tuple[int, int, int] = tuple(config.get("color", [0, 255, 0]))
        self.description: str = config.get("description", "")

        # Build shapely polygon from pixel coords
        self._polygon: Optional[object] = None
        if SHAPELY_AVAILABLE and len(self.pixel_coords) >= 3:
            self._polygon = Polygon(self.pixel_coords)

        # Numpy array for drawing
        self._pts = np.array(self.pixel_coords, dtype=np.int32)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if pixel point (x, y) is inside this zone."""
        if self._polygon is None:
            # Fallback: use cv2 pointPolygonTest
            result = cv2.pointPolygonTest(self._pts, (float(x), float(y)), False)
            return result >= 0
        return self._polygon.contains(Point(x, y))

    def contains_bbox(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if the centroid of the bbox is inside the zone."""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return self.contains_point(cx, cy)

    def draw(self, frame: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """Draw semi-transparent polygon + border on the frame."""
        overlay = frame.copy()
        # Fill
        cv2.fillPoly(overlay, [self._pts], self.color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Border
        border_color = (
            min(self.color[0] + 60, 255),
            min(self.color[1] + 60, 255),
            min(self.color[2] + 60, 255),
        )
        cv2.polylines(frame, [self._pts], isClosed=True, color=border_color, thickness=2)
        # Label
        label_pt = tuple(self._pts[0])
        cv2.putText(
            frame,
            f"{'⚠ ' if self.prohibited else ''}{self.name}",
            (label_pt[0], label_pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            border_color,
            2,
        )
        return frame

    def draw_violation_flash(self, frame: np.ndarray) -> np.ndarray:
        """Flash a bright red border when a violation is active."""
        cv2.polylines(frame, [self._pts], isClosed=True, color=(0, 0, 255), thickness=4)
        return frame


class ZoneManager:
    """Manages all zones for a session."""

    def __init__(self, zones_path: Optional[str] = None):
        self.zones: List[Zone] = []
        if zones_path:
            self.load(zones_path)

    def load(self, zones_path: str) -> int:
        """Load zones from a JSON file. Returns number of zones loaded."""
        path = Path(zones_path)
        if not path.exists():
            print(f"[ZoneManager] zones file not found: {zones_path}")
            return 0

        with open(path) as f:
            data = json.load(f)

        self.zones = [Zone(z) for z in data.get("zones", [])]
        print(f"[ZoneManager] Loaded {len(self.zones)} zones from {zones_path}")
        return len(self.zones)

    def load_from_dict(self, data: Dict) -> int:
        self.zones = [Zone(z) for z in data.get("zones", [])]
        return len(self.zones)

    def check_point(self, x: float, y: float) -> List[Zone]:
        """Return all zones that contain this point."""
        return [z for z in self.zones if z.contains_point(x, y)]

    def check_bbox(self, x1: int, y1: int, x2: int, y2: int) -> List[Zone]:
        """Return all zones that contain the centroid of this bbox."""
        return [z for z in self.zones if z.contains_bbox(x1, y1, x2, y2)]

    def get_prohibited_zones_for_bbox(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> List[Zone]:
        return [z for z in self.check_bbox(x1, y1, x2, y2) if z.prohibited]

    def draw_all_zones(self, frame: np.ndarray) -> np.ndarray:
        for zone in self.zones:
            frame = zone.draw(frame)
        return frame

    def get_zone_ids(self, x1: int, y1: int, x2: int, y2: int) -> List[str]:
        return [z.id for z in self.check_bbox(x1, y1, x2, y2)]

    def get_zone_by_id(self, zone_id: str) -> Optional[Zone]:
        for z in self.zones:
            if z.id == zone_id:
                return z
        return None

    def to_dict(self) -> Dict:
        return {
            "zones": [
                {
                    "id": z.id,
                    "name": z.name,
                    "prohibited": z.prohibited,
                    "pixel_coords": z.pixel_coords,
                    "color": list(z.color),
                }
                for z in self.zones
            ]
        }