"""
tracker.py
----------
DeepSORT multi-object tracking wrapper.
Assigns persistent Track IDs across frames.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Track result: (track_id, x1, y1, x2, y2, confidence)
TrackResult = Tuple[int, int, int, int, int, float]


class PersonTracker:
    """
    Wraps deep-sort-realtime for multi-person tracking.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        embedder: str = "mobilenet",
    ):
        """
        max_age: frames to keep a lost track before deletion.
        n_init: frames before a tentative track is confirmed.
        """
        self.max_age = max_age
        self.n_init = n_init
        self._tracker = None
        self._embedder = embedder
        self._load_tracker()

    def _load_tracker(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_cosine_distance=0.4,
                nn_budget=None,
                embedder=self._embedder,
                half=False,
                bgr=True,
                embedder_gpu=False,
            )
            print("[Tracker] DeepSORT initialized.")
        except Exception as e:
            print(f"[Tracker] ERROR initializing DeepSORT: {e}")
            self._tracker = None

    def update(
        self,
        detections: List[List],  # [[x1,y1,w,h], conf, class_name]
        frame: np.ndarray,
    ) -> List[TrackResult]:
        """
        Update tracker with new detections.
        Returns list of (track_id, x1, y1, x2, y2, confidence).
        """
        if self._tracker is None or frame is None or frame.size == 0:
            return []

        if not detections:
            # Update with empty list to age out existing tracks
            try:
                tracks = self._tracker.update_tracks([], frame=frame)
            except Exception:
                return []
            return self._extract_confirmed(tracks)

        try:
            tracks = self._tracker.update_tracks(detections, frame=frame)
            return self._extract_confirmed(tracks)
        except Exception as e:
            print(f"[Tracker] Update error: {e}")
            return []

    @staticmethod
    def _extract_confirmed(tracks) -> List[TrackResult]:
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            conf = float(track.det_conf) if track.det_conf is not None else 0.0
            results.append((track_id, x1, y1, x2, y2, conf))
        return results

    def is_ready(self) -> bool:
        return self._tracker is not None


class FallbackTracker:
    """
    Simple IoU-based tracker used when DeepSORT is unavailable.
    Assigns incremental IDs based on bounding box overlap.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 20):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: Dict[int, Dict] = {}
        self._next_id = 1

    def update(
        self,
        detections: List,
        frame: np.ndarray,
    ) -> List[TrackResult]:
        if not detections:
            self._age_tracks()
            return []

        # Convert detections from [x1,y1,w,h] to [x1,y1,x2,y2]
        det_boxes = []
        det_confs = []
        for det in detections:
            bbox, conf, _ = det
            x1, y1, w, h = bbox
            det_boxes.append((x1, y1, x1 + w, y1 + h))
            det_confs.append(conf)

        # Match to existing tracks by IoU
        matched = {}
        unmatched = list(range(len(det_boxes)))

        for tid, track in self._tracks.items():
            best_iou = self.iou_threshold
            best_idx = -1
            for i in unmatched:
                iou = self._iou(track["bbox"], det_boxes[i])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                matched[tid] = best_idx
                unmatched.remove(best_idx)

        # Update matched tracks
        for tid, idx in matched.items():
            self._tracks[tid]["bbox"] = det_boxes[idx]
            self._tracks[tid]["conf"] = det_confs[idx]
            self._tracks[tid]["age"] = 0

        # Create new tracks for unmatched detections
        for idx in unmatched:
            self._tracks[self._next_id] = {
                "bbox": det_boxes[idx],
                "conf": det_confs[idx],
                "age": 0,
            }
            self._next_id += 1

        results = []
        for tid, track in self._tracks.items():
            if track["age"] == 0:  # only active this frame
                x1, y1, x2, y2 = track["bbox"]
                results.append((tid, x1, y1, x2, y2, track["conf"]))

        self._age_tracks()
        return results

    def _age_tracks(self):
        dead = []
        for tid in self._tracks:
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > self.max_age:
                dead.append(tid)
        for tid in dead:
            del self._tracks[tid]

    @staticmethod
    def _iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter)

    def is_ready(self) -> bool:
        return True