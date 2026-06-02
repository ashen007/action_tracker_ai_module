"""
overlay.py
----------
Drawing helpers for annotating video frames with bounding boxes,
labels, zone violations, dwell time counters, and heatmaps.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Palette of 20 distinct colors for track IDs
_PALETTE = [
    (255, 56,  56),  (255, 157,  151), (255, 112, 31),
    (255, 178,  29), (207, 210,  49),  (72,  249, 10),
    (146, 204, 23),  (61,  219, 134),  (26,  147, 52),
    (0,   212, 187), (44,  153, 168),  (0,   194, 255),
    (52,  69,  147), (100, 115, 255),  (0,   24,  236),
    (132, 56,  255), (82,  0,   133),  (203, 56,  255),
    (255, 149, 200), (255, 55,  199),
]

def track_color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_person(
    frame: np.ndarray,
    track_id: int,
    x1: int, y1: int, x2: int, y2: int,
    action: str = "",
    dwell_str: str = "",
    is_violation: bool = False,
    is_loitering: bool = False,
    conf: float = 0.0,
) -> np.ndarray:
    """Draw bounding box, track label, action, and dwell time."""
    color = (0, 0, 255) if is_violation else (
            (255, 165, 0) if is_loitering else track_color(track_id))

    thickness = 3 if is_violation else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Flashing red outline for violation
    if is_violation:
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 2)

    # Label background
    label_parts = [f"ID:{track_id}"]
    if action:
        label_parts.append(action)
    if dwell_str:
        label_parts.append(dwell_str)
    label = "  ".join(label_parts)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_y = max(y1 - 5, th + 5)
    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
    cv2.putText(frame, label, (x1 + 2, label_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Violation badge
    if is_violation:
        badge = "⚠ VIOLATION"
        cv2.putText(frame, badge, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return frame


def draw_fps_counter(frame: np.ndarray, fps: float, frame_idx: int) -> np.ndarray:
    txt = f"FPS: {fps:.1f}  Frame: {frame_idx}"
    cv2.putText(frame, txt, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_alert_banner(frame: np.ndarray, message: str) -> np.ndarray:
    """Draw a red alert banner across the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, f"⚠  {message}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_stats_overlay(
    frame: np.ndarray,
    total_persons: int,
    current_persons: int,
    violations: int,
    avg_dwell: float,
) -> np.ndarray:
    """Draw a compact stats box in the bottom-left corner."""
    h, w = frame.shape[:2]
    lines = [
        f"Total detected: {total_persons}",
        f"In frame: {current_persons}",
        f"Violations: {violations}",
        f"Avg dwell: {avg_dwell:.0f}s",
    ]
    box_h = len(lines) * 22 + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - box_h - 5), (220, h - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, h - box_h + i * 22 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
    return frame


def build_heatmap_image(
    heatmap: np.ndarray,
    frame_shape: Tuple[int, int],
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Convert position accumulator to colored heatmap image."""
    if heatmap is None or heatmap.sum() == 0:
        blank = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        cv2.putText(blank, "No data yet", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        return blank

    # Resize to match frame shape
    resized = cv2.resize(heatmap.astype(np.float32),
                         (frame_shape[1], frame_shape[0]))
    # Normalize
    norm = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(norm.astype(np.uint8), colormap)
    return colored


def update_heatmap(
    heatmap: np.ndarray,
    track_list: List[Tuple[int, int, int, int, int, float]],
    sigma: int = 20,
) -> np.ndarray:
    """Accumulate gaussian blobs at person centroids into the heatmap."""
    for tid, x1, y1, x2, y2, _ in track_list:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if 0 <= cy < heatmap.shape[0] and 0 <= cx < heatmap.shape[1]:
            heatmap[cy, cx] += 1
    # Light Gaussian blur to spread heat
    return cv2.GaussianBlur(heatmap.astype(np.float32), (sigma * 2 + 1, sigma * 2 + 1), sigma)


def frame_to_rgb_bytes(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB for Streamlit display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)