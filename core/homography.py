"""
homography.py
-------------
Computes and applies homography transforms between GPS coordinates
and pixel coordinates using 4+ reference point pairs.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class HomographyMapper:
    """
    Maps between GPS coordinates (lat, lon) and pixel (x, y) coordinates
    using a perspective homography computed from calibration point pairs.
    """

    def __init__(self):
        self._H: Optional[np.ndarray] = None       # GPS -> pixel
        self._H_inv: Optional[np.ndarray] = None   # pixel -> GPS
        self._calibration_pts: List[Tuple] = []

    def calibrate(
        self,
        gps_points: List[Tuple[float, float]],
        pixel_points: List[Tuple[int, int]],
    ) -> bool:
        """
        Compute homography from at least 4 point correspondences.

        gps_points:   list of (lat, lon) pairs
        pixel_points: list of (x, y) pairs — same order as gps_points
        Returns True on success.
        """
        if len(gps_points) < 4 or len(gps_points) != len(pixel_points):
            print("[Homography] Need at least 4 matching point pairs.")
            return False

        src = np.array([[lon, lat] for lat, lon in gps_points], dtype=np.float64)
        dst = np.array(pixel_points, dtype=np.float64)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            print("[Homography] Failed to compute homography matrix.")
            return False

        self._H = H
        self._H_inv = np.linalg.inv(H)
        self._calibration_pts = list(zip(gps_points, pixel_points))
        print(f"[Homography] Calibrated with {len(gps_points)} points. Inliers: {mask.sum()}")
        return True

    def gps_to_pixel(
        self, lat: float, lon: float
    ) -> Optional[Tuple[int, int]]:
        """Convert GPS coordinate to pixel coordinate."""
        if self._H is None:
            return None
        pt = np.array([[[lon, lat]]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, self._H)
        x, y = result[0][0]
        return int(round(x)), int(round(y))

    def pixel_to_gps(
        self, x: int, y: int
    ) -> Optional[Tuple[float, float]]:
        """Convert pixel coordinate to GPS coordinate."""
        if self._H_inv is None:
            return None
        pt = np.array([[[float(x), float(y)]]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, self._H_inv)
        lon, lat = result[0][0]
        return float(lat), float(lon)

    def is_calibrated(self) -> bool:
        return self._H is not None

    def convert_zone_gps_to_pixels(
        self, gps_coords: List[List[float]]
    ) -> List[List[int]]:
        """Convert a list of [lat, lon] zone corners to pixel coords."""
        pixels = []
        for lat, lon in gps_coords:
            px = self.gps_to_pixel(lat, lon)
            if px:
                pixels.append(list(px))
        return pixels

    def save(self, path: str):
        """Save homography matrix to file."""
        if self._H is not None:
            np.save(path, self._H)

    def load(self, path: str) -> bool:
        """Load homography matrix from file."""
        try:
            self._H = np.load(path)
            self._H_inv = np.linalg.inv(self._H)
            return True
        except Exception as e:
            print(f"[Homography] Could not load: {e}")
            return False


class CalibrationHelper:
    """
    Interactive helper for collecting calibration point pairs.
    Call add_point() each time the user clicks a frame location
    and enters a GPS coordinate.
    """

    def __init__(self):
        self.pixel_points: List[Tuple[int, int]] = []
        self.gps_points: List[Tuple[float, float]] = []

    def add_point(self, px: int, py: int, lat: float, lon: float):
        self.pixel_points.append((px, py))
        self.gps_points.append((lat, lon))
        print(f"[Calibration] Added point #{len(self.pixel_points)}: "
              f"pixel=({px},{py}) <-> GPS=({lat:.6f},{lon:.6f})")

    def clear(self):
        self.pixel_points.clear()
        self.gps_points.clear()

    def is_ready(self) -> bool:
        return len(self.pixel_points) >= 4

    def build_mapper(self) -> Optional[HomographyMapper]:
        if not self.is_ready():
            return None
        mapper = HomographyMapper()
        if mapper.calibrate(self.gps_points, self.pixel_points):
            return mapper
        return None

    def draw_collected_points(self, frame: np.ndarray) -> np.ndarray:
        """Visualize collected calibration points on a frame."""
        for i, (px, py) in enumerate(self.pixel_points):
            cv2.circle(frame, (px, py), 8, (0, 255, 255), -1)
            lat, lon = self.gps_points[i]
            cv2.putText(
                frame,
                f"P{i+1} ({lat:.4f},{lon:.4f})",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
        return frame