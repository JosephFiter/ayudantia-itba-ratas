import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from filterpy.kalman import KalmanFilter
from typing import Callable, Optional


class RatTracker:
    """
    Trackea la posición (x, y) de una rata frame a frame.
    Modos: 'mog2' | 'threshold' | 'color'
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.method        = cfg.get("method", "mog2")
        self.invert        = cfg.get("invert", False)
        self.min_area      = cfg.get("min_area", 200)
        self.max_area      = cfg.get("max_area", 10_000)
        self.blur_k        = cfg.get("blur_size", 3) * 2 + 1  # siempre impar

        # Filtros de calidad de detección
        self.min_circularity = cfg.get("min_circularity", 0.35)
        self.max_jump_px     = cfg.get("max_jump_px", 120)
        self.still_speed_px  = cfg.get("still_speed_px", 4.0)

        # Parámetros de detección por color (rata blanca)
        self.hue_min = cfg.get("hue_min", 0)
        self.hue_max = cfg.get("hue_max", 179)
        self.sat_max = cfg.get("sat_max", 30)
        self.val_min = cfg.get("val_min", 200)
        self.val_max = cfg.get("val_max", 255)

        # ROI / límites del laberinto
        self.boundary_points = cfg.get("boundary_points", [])

        # Modelo YOLO (opcional)
        self.yolo_model = None
        if self.method == "yolo":
            model_path = cfg.get("yolo_model_path", "")
            if model_path and Path(model_path).exists():
                from ultralytics import YOLO
                self.yolo_model = YOLO(model_path)

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=cfg.get("history", 500),
            varThreshold=cfg.get("var_threshold", 40),
            detectShadows=False,
        )
        self.kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kalman  = self._build_kalman()
        self._kf_ready = False

    # ── Kalman Filter (estado: [x, y, vx, vy]) ───────────────────────────────
    @staticmethod
    def _build_kalman() -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        kf.R = np.eye(2) * 15
        kf.P = np.eye(4) * 1000
        kf.Q = np.eye(4) * 0.01
        return kf

    # ── Utilidades ────────────────────────────────────────────────────────────
    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _kalman_speed(self) -> float:
        """Velocidad actual estimada por Kalman en px/frame."""
        if not self._kf_ready:
            return float("inf")
        vx = float(self.kalman.x[2, 0])
        vy = float(self.kalman.x[3, 0])
        return float(np.sqrt(vx**2 + vy**2))

    def _kalman_pos(self) -> Optional[tuple[float, float]]:
        if not self._kf_ready:
            return None
        return float(self.kalman.x[0, 0]), float(self.kalman.x[1, 0])

    # ── Máscaras ──────────────────────────────────────────────────────────────
    def _mask_mog2(self, gray: np.ndarray, learning_rate: float = -1) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (self.blur_k, self.blur_k), 0)
        raw = self.bg_sub.apply(blurred, learningRate=learning_rate)
        _, binary = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
        return self._clean_mask(binary)

    def _mask_threshold(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (self.blur_k, self.blur_k), 0)
        flag = cv2.THRESH_BINARY if self.invert else cv2.THRESH_BINARY_INV
        _, binary = cv2.threshold(blurred, 0, 255, flag | cv2.THRESH_OTSU)
        return self._clean_mask(binary)

    def _mask_color(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.hue_min, 0,            self.val_min])
        upper = np.array([self.hue_max, self.sat_max,  self.val_max])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (self.blur_k, self.blur_k), 0)
        return self._clean_mask(mask)

    def _boundary_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.boundary_points or len(self.boundary_points) < 3:
            return None
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(self.boundary_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    # ── Selección inteligente de contorno ────────────────────────────────────
    def _pick_contour(
        self,
        contours: list,
        predicted: Optional[tuple[float, float]],
    ) -> Optional[tuple[float, float]]:
        """
        Entre los contornos válidos por área, elige el que más se parece al cuerpo
        de la rata: descarta formas elongadas (cola) por circularidad y rechaza
        detecciones demasiado lejos de la predicción de Kalman.
        Hace dos pasadas: primero con todos los filtros, luego relajando circularidad
        si la primera pasada no encuentra nada.
        """
        for relaxed in (False, True):
            best_cx = best_cy = None
            best_area = -1

            for c in contours:
                area = cv2.contourArea(c)

                # Filtro de circularidad — descarta la cola
                if not relaxed:
                    perimeter = cv2.arcLength(c, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / perimeter ** 2
                        if circularity < self.min_circularity:
                            continue

                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                # Filtro de salto — descarta teleportaciones
                if predicted is not None:
                    dist = np.sqrt((cx - predicted[0])**2 + (cy - predicted[1])**2)
                    if dist > self.max_jump_px:
                        continue

                if area > best_area:
                    best_area = area
                    best_cx, best_cy = cx, cy

            if best_cx is not None:
                return best_cx, best_cy

        return None

    # ── Detección YOLO ────────────────────────────────────────────────────────
    def _detect_yolo(self, frame: np.ndarray) -> Optional[tuple[float, float]]:
        if self.yolo_model is None:
            return None
        results = self.yolo_model(frame, verbose=False)
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        best  = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Aplicar filtro de salto también con YOLO
        pred = self._kalman_pos()
        if pred is not None:
            dist = np.sqrt((cx - pred[0])**2 + (cy - pred[1])**2)
            if dist > self.max_jump_px:
                return None
        return cx, cy

    # ── Detección por frame ───────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> tuple[Optional[tuple[float, float]], np.ndarray]:
        # Método YOLO: no produce máscara, devuelve imagen vacía como placeholder
        if self.method == "yolo":
            blank = np.zeros(frame.shape[:2], dtype=np.uint8)
            return self._detect_yolo(frame), blank

        if self.method == "color":
            mask = self._mask_color(frame)
        elif self.method == "threshold":
            gray = self._to_gray(frame)
            mask = self._mask_threshold(gray)
        else:  # mog2
            gray = self._to_gray(frame)
            lr = 0.0 if self._kalman_speed() < self.still_speed_px else -1
            mask = self._mask_mog2(gray, lr)

        # Aplicar ROI si hay límites definidos
        boundary = self._boundary_mask(frame)
        if boundary is not None:
            mask = cv2.bitwise_and(mask, mask, mask=boundary)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if self.min_area < cv2.contourArea(c) < self.max_area]

        if not valid:
            return None, mask

        detection = self._pick_contour(valid, self._kalman_pos())
        return detection, mask

    # ── Loop principal ────────────────────────────────────────────────────────
    def process_video(
        self,
        video_path: str,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> tuple[pd.DataFrame, float]:
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
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
                sx, sy = float(self.kalman.x[0, 0]), float(self.kalman.x[1, 0])
                records.append({"frame": idx, "timestamp": t,
                                 "x_raw": cx, "y_raw": cy,
                                 "x": sx, "y": sy, "detected": True})
            else:
                if self._kf_ready:
                    self.kalman.predict()
                    sx, sy = float(self.kalman.x[0, 0]), float(self.kalman.x[1, 0])
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

    # ── Frame de preview con anotación ───────────────────────────────────────
    @staticmethod
    def annotate_frame(frame: np.ndarray, x: float, y: float) -> np.ndarray:
        out = frame.copy()
        if frame.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(out, (ix, iy), 8, (0, 255, 0), 2)
        cv2.drawMarker(out, (ix, iy), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
        return out
