import cv2
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from typing import Callable, Optional


class RatTracker:
    """
    Trackea la posición (x, y) de una rata frame a frame.
    Soporta dos modos:
      - 'mog2'      : Background Subtraction adaptativo (cámara estática, iluminación variable)
      - 'threshold' : Umbralización de Otsu (fondo completamente estático y uniforme)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.method = cfg.get("method", "mog2")
        self.invert = cfg.get("invert", False)
        self.min_area = cfg.get("min_area", 200)
        self.max_area = cfg.get("max_area", 10_000)
        self.blur_k = cfg.get("blur_size", 3) * 2 + 1  # siempre impar

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=cfg.get("history", 500),
            varThreshold=cfg.get("var_threshold", 40),
            detectShadows=False,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kalman = self._build_kalman()
        self._kf_ready = False

    # ── Kalman Filter (estado: [x, y, vx, vy]) ────────────────────────
    @staticmethod
    def _build_kalman() -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        kf.R = np.eye(2) * 5
        kf.P = np.eye(4) * 500
        kf.Q = np.eye(4) * 0.05
        return kf

    # ── Utilidades ────────────────────────────────────────────────────
    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    # ── Máscaras ──────────────────────────────────────────────────────
    def _mask_mog2(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (self.blur_k, self.blur_k), 0)
        raw = self.bg_sub.apply(blurred)
        _, binary = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
        return self._clean_mask(binary)

    def _mask_threshold(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (self.blur_k, self.blur_k), 0)
        flag = cv2.THRESH_BINARY if self.invert else cv2.THRESH_BINARY_INV
        _, binary = cv2.threshold(blurred, 0, 255, flag | cv2.THRESH_OTSU)
        return self._clean_mask(binary)

    # ── Detección por frame ───────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> tuple[Optional[tuple[float, float]], np.ndarray]:
        gray = self._to_gray(frame)
        mask = self._mask_mog2(gray) if self.method == "mog2" else self._mask_threshold(gray)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if self.min_area < cv2.contourArea(c) < self.max_area]

        if not valid:
            return None, mask

        largest = max(valid, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None, mask

        return (M["m10"] / M["m00"], M["m01"] / M["m00"]), mask

    # ── Loop principal ────────────────────────────────────────────────
    def process_video(
        self,
        video_path: str,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> tuple[pd.DataFrame, float]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        records: list[dict] = []
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t = idx / fps
            detection, _ = self.detect(frame)

            if detection:
                cx, cy = detection
                if not self._kf_ready:
                    self.kalman.x = np.array([[cx], [cy], [0.0], [0.0]])
                    self._kf_ready = True
                self.kalman.predict()
                self.kalman.update(np.array([[cx], [cy]]))
                sx, sy = float(self.kalman.x[0]), float(self.kalman.x[1])
                records.append({"frame": idx, "timestamp": t,
                                 "x_raw": cx, "y_raw": cy,
                                 "x": sx, "y": sy, "detected": True})
            else:
                if self._kf_ready:
                    self.kalman.predict()
                    sx, sy = float(self.kalman.x[0]), float(self.kalman.x[1])
                else:
                    sx = sy = None
                records.append({"frame": idx, "timestamp": t,
                                 "x_raw": None, "y_raw": None,
                                 "x": sx, "y": sy, "detected": False})

            if progress_cb and idx % 15 == 0:
                progress_cb(idx / total)
            idx += 1

        cap.release()
        return pd.DataFrame(records), fps

    # ── Frame de preview con anotación ───────────────────────────────
    @staticmethod
    def annotate_frame(frame: np.ndarray, x: float, y: float) -> np.ndarray:
        out = frame.copy()
        if frame.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(out, (ix, iy), 8, (0, 255, 0), 2)
        cv2.drawMarker(out, (ix, iy), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
        return out
