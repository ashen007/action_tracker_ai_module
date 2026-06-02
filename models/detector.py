"""
detector.py
-----------
YOLOv8s person detection wrapper using ultralytics.
Filters results to 'person' class only (class_id=0).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Detection result type: (x1, y1, x2, y2, confidence, class_id)
Detection = Tuple[int, int, int, int, float, int]


class PersonDetector:
    """
    Wraps YOLOv8s for person-only detection.
    Auto-downloads model weights on first run.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            print(f"[Detector] Loading YOLOv8 from '{self.model_path}' on {self.device}...")
            self._model = YOLO(self.model_path)
            print("[Detector] Model loaded successfully.")
        except Exception as e:
            print(f"[Detector] ERROR loading model: {e}")
            self._model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.
        Returns list of (x1, y1, x2, y2, confidence, class_id).
        """
        if self._model is None or frame is None or frame.size == 0:
            return []

        try:
            results = self._model(
                frame,
                conf=self.conf_threshold,
                classes=[self.PERSON_CLASS_ID],
                device=self.device,
                verbose=False,
            )
            detections: List[Detection] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == self.PERSON_CLASS_ID and conf >= self.conf_threshold:
                        detections.append((x1, y1, x2, y2, conf, cls))
            return detections
        except Exception as e:
            print(f"[Detector] Inference error: {e}")
            return []

    def set_confidence(self, threshold: float):
        self.conf_threshold = max(0.1, min(0.99, threshold))

    def is_ready(self) -> bool:
        return self._model is not None

    # Format detections for DeepSORT: [[x1,y1,x2,y2,conf], ...]
    @staticmethod
    def to_deepsort_format(
        detections: List[Detection],
    ) -> List[List]:
        """Convert to format expected by DeepSORT tracker."""
        result = []
        for x1, y1, x2, y2, conf, _ in detections:
            # DeepSORT expects [x1, y1, w, h] for bounding box
            w = x2 - x1
            h = y2 - y1
            result.append(([x1, y1, w, h], conf, "person"))
        return result