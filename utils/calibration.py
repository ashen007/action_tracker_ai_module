"""
calibration.py
--------------
Interactive camera calibration helper.
Used to collect GPS <-> pixel reference point pairs from the UI,
then compute the homography matrix.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.homography import CalibrationHelper, HomographyMapper


class CalibrationSession:
    """
    Manages a calibration workflow:
    1. User clicks a point on the frame → pixel coord recorded
    2. User enters GPS lat/lon for that point
    3. After 4+ points, compute homography
    """

    def __init__(self):
        self.helper = CalibrationHelper()
        self.mapper: Optional[HomographyMapper] = None
        self.status_message: str = "Click 4+ reference points on the frame."
        self.pending_pixel: Optional[Tuple[int, int]] = None  # waiting for GPS input

    def on_frame_click(self, px: int, py: int):
        """Call when user clicks a pixel coordinate on the frame."""
        self.pending_pixel = (px, py)
        self.status_message = f"Clicked ({px},{py}). Enter GPS lat/lon for this point."

    def on_gps_input(self, lat: float, lon: float) -> bool:
        """Call when user submits GPS for the pending pixel point."""
        if self.pending_pixel is None:
            self.status_message = "Click a point on the frame first."
            return False
        px, py = self.pending_pixel
        self.helper.add_point(px, py, lat, lon)
        self.pending_pixel = None
        n = len(self.helper.pixel_points)
        self.status_message = f"Point {n} added. {'Ready to calibrate!' if n >= 4 else f'Need {4-n} more points.'}"
        return True

    def calibrate(self) -> bool:
        """Compute homography from collected points."""
        if not self.helper.is_ready():
            self.status_message = "Need at least 4 reference points."
            return False
        self.mapper = self.helper.build_mapper()
        if self.mapper:
            self.status_message = "✓ Calibration successful!"
            return True
        self.status_message = "✗ Calibration failed — try different reference points."
        return False

    def save(self, path: str):
        """Save calibration data to JSON + homography .npy."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Save reference points
        data = {
            "pixel_points": self.helper.pixel_points,
            "gps_points": self.helper.gps_points,
        }
        with open(p.with_suffix(".json"), "w") as f:
            json.dump(data, f, indent=2)
        # Save homography
        if self.mapper:
            self.mapper.save(str(p.with_suffix(".npy")))

    def load(self, path: str) -> bool:
        """Load a saved calibration."""
        json_path = Path(path).with_suffix(".json")
        npy_path = Path(path).with_suffix(".npy")
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            self.helper.pixel_points = [tuple(p) for p in data["pixel_points"]]
            self.helper.gps_points = [tuple(p) for p in data["gps_points"]]
        if npy_path.exists():
            self.mapper = HomographyMapper()
            return self.mapper.load(str(npy_path))
        return False

    def reset(self):
        self.helper.clear()
        self.mapper = None
        self.pending_pixel = None
        self.status_message = "Calibration reset. Click 4+ points."

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        return self.helper.draw_collected_points(frame)

    def get_summary(self) -> dict:
        return {
            "points_collected": len(self.helper.pixel_points),
            "calibrated": self.mapper is not None,
            "status": self.status_message,
        }