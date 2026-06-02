"""
video_source.py
---------------
Unified video source abstraction for file playback and RTSP streams.
Supports pause/resume for file playback.
"""

import time
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoSource:
    """
    Wraps OpenCV VideoCapture for both file and RTSP sources.
    Thread-safe frame reading with optional buffering for RTSP.
    """

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._source: Optional[str] = None
        self._is_file: bool = False
        self._paused: bool = False
        self._stopped: bool = False

        # Cached source info
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0.0
        self.total_frames: int = 0
        self.current_frame: int = 0

        # RTSP buffer
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._rtsp_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------

    def open_file(self, path: str) -> bool:
        """Open a video file (MP4, AVI, MOV, etc.)."""
        if not Path(path).exists():
            print(f"[VideoSource] File not found: {path}")
            return False
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[VideoSource] Cannot open file: {path}")
            return False
        self._cap = cap
        self._source = path
        self._is_file = True
        self._stopped = False
        self._paused = False
        self._read_info()
        print(f"[VideoSource] Opened file: {path} "
              f"({self.width}x{self.height} @ {self.fps:.1f} fps, "
              f"{self.total_frames} frames)")
        return True

    def open_rtsp(self, url: str, timeout: float = 10.0) -> bool:
        """Open an RTSP stream URL."""
        # Validate URL format
        if not (url.startswith("rtsp://") or url.startswith("rtmp://")
                or url.startswith("http://") or url.startswith("https://")):
            print(f"[VideoSource] Invalid stream URL: {url}")
            return False

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency

        # Test connection
        deadline = time.time() + timeout
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                break
        else:
            print(f"[VideoSource] Could not connect to RTSP: {url}")
            cap.release()
            return False

        self._cap = cap
        self._source = url
        self._is_file = False
        self._stopped = False
        self._paused = False
        self._read_info()

        # Start background reader thread for RTSP
        self._latest_frame = frame
        self._rtsp_thread = threading.Thread(
            target=self._rtsp_reader, daemon=True
        )
        self._rtsp_thread.start()

        print(f"[VideoSource] Connected to RTSP: {url} "
              f"({self.width}x{self.height} @ {self.fps:.1f} fps)")
        return True

    def _read_info(self):
        if self._cap:
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _rtsp_reader(self):
        """Background thread to keep latest RTSP frame fresh."""
        while not self._stopped and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.01)

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame. Respects pause state."""
        if self._stopped or self._cap is None:
            return False, None
        if self._paused and self._is_file:
            return True, self._latest_frame  # return last frame while paused

        if self._is_file:
            ret, frame = self._cap.read()
            if ret:
                self.current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
                self._latest_frame = frame
            return ret, frame if ret else None
        else:
            # RTSP: return latest buffered frame
            with self._frame_lock:
                frame = self._latest_frame
            return frame is not None, frame

    def get_timestamp(self) -> float:
        """Current video timestamp in seconds."""
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return time.time()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def toggle_pause(self):
        self._paused = not self._paused

    @property
    def is_paused(self) -> bool:
        return self._paused

    def seek(self, frame_num: int):
        """Seek to frame number (file only)."""
        if self._is_file and self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    def stop(self):
        self._stopped = True
        if self._cap:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened() and not self._stopped

    def is_file(self) -> bool:
        return self._is_file

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        return {
            "source": self._source,
            "type": "file" if self._is_file else "rtsp",
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames if self._is_file else -1,
            "current_frame": self.current_frame,
            "duration_seconds": round(self.total_frames / self.fps, 1)
            if self._is_file and self.fps > 0 else -1,
        }

    def progress(self) -> float:
        """0.0 to 1.0 playback progress (file only)."""
        if self._is_file and self.total_frames > 0:
            return self.current_frame / self.total_frames
        return 0.0