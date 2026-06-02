"""
pose_estimator.py
-----------------
YOLO11n-pose wrapper for body keypoint extraction on person crops.
Returns 17 COCO keypoints per person.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# COCO keypoint indices
KP = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# Keypoint result: array of shape (17, 3) — [x, y, confidence]
Keypoints = np.ndarray


class PoseEstimator:
    """
    Runs YOLOv8n-pose on cropped person images.
    Returns normalized keypoints relative to the crop.
    """

    def __init__(self, model_path: str = "yolo11n-pose.pt", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            print(f"[Pose] Loading pose model '{self.model_path}'...")
            self._model = YOLO(self.model_path)
            print("[Pose] Pose model loaded.")
        except Exception as e:
            print(f"[Pose] ERROR loading model: {e}")
            self._model = None

    def estimate(self, crop: np.ndarray) -> Optional[Keypoints]:
        """
        Run pose on a single BGR crop.
        Returns (17, 3) array or None.
        """
        if self._model is None or crop is None or crop.size == 0:
            return None
        if crop.shape[0] < 20 or crop.shape[1] < 10:
            return None

        try:
            results = self._model(crop, verbose=False, device=self.device)
            for r in results:
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    kps = r.keypoints.data[0].cpu().numpy()  # (17, 3)
                    if kps.shape == (17, 3):
                        return kps
            return None
        except Exception as e:
            print(f"[Pose] Inference error: {e}")
            return None

    def is_ready(self) -> bool:
        return self._model is not None

    @staticmethod
    def get_keypoint(kps: Keypoints, name: str) -> Optional[Tuple[float, float, float]]:
        """Get (x, y, conf) for a named keypoint."""
        idx = KP.get(name)
        if idx is None or kps is None:
            return None
        return tuple(kps[idx])

    @staticmethod
    def get_visible_keypoints(kps: Keypoints, conf_threshold: float = 0.3) -> Dict[str, Tuple]:
        """Return dict of name -> (x, y, conf) for visible keypoints."""
        if kps is None or kps.ndim < 2 or kps.shape[0] < 17:
            return {}
        visible = {}
        for name, idx in KP.items():
            x, y, c = kps[idx]
            if c >= conf_threshold:
                visible[name] = (float(x), float(y), float(c))
        return visible