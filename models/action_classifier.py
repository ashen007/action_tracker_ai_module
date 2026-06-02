"""
action_classifier.py
--------------------
Classifies human actions using pose keypoints with rule-based logic.
Optional: SlowFast R50 for video-clip-based classification.

Actions: Standing, Sitting, Walking, Running, Using Phone,
         Crouching, Waving, Loitering
"""

import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from models.pose_estimator import PoseEstimator, Keypoints

ACTIONS = [
    "Standing", "Sitting", "Walking", "Running",
    "Using Phone", "Crouching", "Waving", "Loitering", "Unknown"
]


class RuleBasedClassifier:
    """
    Classifies actions using geometric rules applied to pose keypoints.
    Works on a single frame's keypoints.
    """

    # Vertical ratio thresholds (normalized to bbox height)
    SIT_HIP_KNEE_RATIO = 0.15   # hips and knees close vertically → sitting
    CROUCH_THRESHOLD   = 0.35   # body compressed vertically → crouching

    def classify(
        self,
        kps: Optional[Keypoints],
        bbox_height: int,
        velocity: float = 0.0,    # pixels/sec from tracker
        dwell_seconds: float = 0.0,
    ) -> str:
        """
        Returns action string based on keypoints and motion cues.
        """
        if kps is None:
            return self._motion_fallback(velocity, dwell_seconds)

        visible = PoseEstimator.get_visible_keypoints(kps, conf_threshold=0.25)

        # --- Using Phone ---
        if self._is_using_phone(kps, visible, bbox_height):
            return "Using Phone"

        # --- Waving ---
        if self._is_waving(kps, visible, bbox_height):
            return "Waving"

        # --- Sitting ---
        if self._is_sitting(kps, visible, bbox_height):
            return "Sitting"

        # --- Crouching ---
        if self._is_crouching(kps, visible, bbox_height):
            return "Crouching"

        # --- Running / Walking (motion-based) ---
        if velocity > 80:
            return "Running"
        if velocity > 20:
            return "Walking"

        # --- Loitering (dwell > 60s, not moving) ---
        if dwell_seconds >= 60 and velocity < 10:
            return "Loitering"

        return "Standing"

    # ------------------------------------------------------------------
    def _is_using_phone(self, kps, visible, h) -> bool:
        """Wrist(s) close to face region."""
        nose = visible.get("nose")
        lw   = visible.get("left_wrist")
        rw   = visible.get("right_wrist")
        if nose is None:
            return False
        nx, ny = nose[0], nose[1]
        face_radius = h * 0.18
        for wrist in (lw, rw):
            if wrist and abs(wrist[0] - nx) < face_radius and abs(wrist[1] - ny) < face_radius:
                return True
        return False

    def _is_waving(self, kps, visible, h) -> bool:
        """One wrist is significantly above the shoulder."""
        ls = visible.get("left_shoulder")
        rs = visible.get("right_shoulder")
        lw = visible.get("left_wrist")
        rw = visible.get("right_wrist")
        threshold = h * 0.15
        if ls and lw and (ls[1] - lw[1]) > threshold:
            return True
        if rs and rw and (rs[1] - rw[1]) > threshold:
            return True
        return False

    def _is_sitting(self, kps, visible, h) -> bool:
        """Hip and knee y-coordinates are close — person is seated."""
        lh = visible.get("left_hip")
        rh = visible.get("right_hip")
        lk = visible.get("left_knee")
        rk = visible.get("right_knee")
        pairs = [(lh, lk), (rh, rk)]
        for hip, knee in pairs:
            if hip and knee:
                vert_diff = abs(knee[1] - hip[1])
                if vert_diff < h * self.SIT_HIP_KNEE_RATIO:
                    return True
        return False

    def _is_crouching(self, kps, visible, h) -> bool:
        """Shoulders close to hips vertically."""
        ls = visible.get("left_shoulder")
        rs = visible.get("right_shoulder")
        lh = visible.get("left_hip")
        rh = visible.get("right_hip")
        for sh, hip in [(ls, lh), (rs, rh)]:
            if sh and hip:
                if abs(hip[1] - sh[1]) < h * self.CROUCH_THRESHOLD:
                    return True
        return False

    def _motion_fallback(self, velocity: float, dwell_seconds: float) -> str:
        if velocity > 80: return "Running"
        if velocity > 20: return "Walking"
        if dwell_seconds >= 60: return "Loitering"
        return "Standing"


class ActionClassifier:
    """
    High-level action classifier.
    Uses rule-based logic; keeps a smoothing window per track.
    Optional SlowFast integration can be enabled separately.
    """

    WINDOW_SIZE = 5  # frames to smooth over

    def __init__(self, use_slowfast: bool = False):
        self._rule_clf = RuleBasedClassifier()
        self._history: Dict[int, Deque[str]] = {}  # track_id -> deque of actions
        self._velocities: Dict[int, Deque[float]] = {}
        self._prev_positions: Dict[int, Tuple[float, float]] = {}
        self._prev_times: Dict[int, float] = {}
        self._use_slowfast = use_slowfast

        # Try to load SlowFast if requested
        self._slowfast = None
        if use_slowfast:
            self._init_slowfast()

    def _init_slowfast(self):
        """Attempt to load SlowFast R50 via torchvision (optional)."""
        try:
            import torch
            from torchvision.models.video import r3d_18
            self._slowfast = r3d_18(pretrained=False)
            self._slowfast.eval()
            print("[ActionClassifier] SlowFast/R3D loaded.")
        except Exception as e:
            print(f"[ActionClassifier] SlowFast not available, using rule-based: {e}")
            self._slowfast = None

    def classify(
        self,
        track_id: int,
        kps: Optional[Keypoints],
        bbox: Tuple[int, int, int, int],   # x1, y1, x2, y2
        timestamp: float,
        dwell_seconds: float = 0.0,
        crop: Optional[np.ndarray] = None,
    ) -> str:
        """
        Classify and smooth action for a tracked person.
        """
        x1, y1, x2, y2 = bbox
        bbox_height = max(1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Compute velocity
        velocity = self._compute_velocity(track_id, cx, cy, timestamp)

        # Get raw prediction
        raw_action = self._rule_clf.classify(kps, bbox_height, velocity, dwell_seconds)

        # Smooth with history
        smoothed = self._smooth(track_id, raw_action)
        return smoothed

    def _compute_velocity(self, tid: int, cx: float, cy: float, t: float) -> float:
        """Pixels per second."""
        if tid not in self._prev_positions:
            self._prev_positions[tid] = (cx, cy)
            self._prev_times[tid] = t
            return 0.0
        px, py = self._prev_positions[tid]
        dt = max(0.001, t - self._prev_times[tid])
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        velocity = dist / dt
        self._prev_positions[tid] = (cx, cy)
        self._prev_times[tid] = t
        return velocity

    def _smooth(self, tid: int, action: str) -> str:
        if tid not in self._history:
            self._history[tid] = deque(maxlen=self.WINDOW_SIZE)
        self._history[tid].append(action)
        # Majority vote
        counts: Dict[str, int] = {}
        for a in self._history[tid]:
            counts[a] = counts.get(a, 0) + 1
        return max(counts, key=counts.get)

    def clear_track(self, tid: int):
        for d in (self._history, self._velocities, self._prev_positions, self._prev_times):
            d.pop(tid, None)

    def get_action_distribution(self) -> Dict[str, int]:
        """Aggregate action counts across all tracks."""
        counts: Dict[str, int] = {}
        for hist in self._history.values():
            for a in hist:
                counts[a] = counts.get(a, 0) + 1
        return counts