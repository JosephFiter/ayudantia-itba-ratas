"""
Entrena YOLOv8 nano sobre frames del propio video.

Flujo:
  1. Recorre el video completo alimentando MOG2 frame a frame (así aprende el fondo).
  2. En los frames muestreados, guarda imagen + bbox detectado como label YOLO.
  3. Fine-tunea yolov8n.pt con esas etiquetas.
  4. Devuelve la ruta al modelo entrenado.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
import yaml


def extract_and_label(
    video_path: str,
    output_dir: Path,
    tracker_config: dict,
    n_frames: int = 300,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> int:
    """
    Recorre el video completo. MOG2 se actualiza en cada frame para que el
    modelo de fondo sea correcto. En los frames muestreados guarda la detección.
    Devuelve la cantidad de frames etiquetados.
    """
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Limpiar dataset anterior
    for f in img_dir.glob("*.jpg"):
        f.unlink()
    for f in lbl_dir.glob("*.txt"):
        f.unlink()

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Warmup: usar ~5 segundos para que MOG2 aprenda el fondo antes de etiquetar
    warmup_frames = min(int(fps * 5), total // 4)
    # Muestrear fuera del warmup
    usable = total - warmup_frames
    step   = max(1, usable // n_frames)

    blur_k   = tracker_config.get("blur_size", 3) * 2 + 1
    min_area = tracker_config.get("min_area", 200)
    max_area = tracker_config.get("max_area", 10_000)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    boundary_pts  = tracker_config.get("boundary_points", [])
    boundary_mask = None

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=tracker_config.get("history", 500),
        varThreshold=tracker_config.get("var_threshold", 40),
        detectShadows=False,
    )

    labeled = 0
    idx     = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]

        # Construir máscara de límites la primera vez
        if boundary_mask is None and len(boundary_pts) >= 3:
            mask = np.zeros((fh, fw), dtype=np.uint8)
            pts  = np.array(boundary_pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            boundary_mask = mask

        # Preprocesar y alimentar MOG2 en CADA frame (imprescindible para fondo estable)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        raw     = bg_sub.apply(blurred)

        # Durante el warmup solo aprendemos el fondo, sin etiquetar
        if idx < warmup_frames:
            idx += 1
            if progress_cb and idx % 30 == 0:
                progress_cb(idx / total * 0.3, f"Aprendiendo fondo… {idx}/{warmup_frames}")
            continue

        # Muestrear frames para el dataset
        if (idx - warmup_frames) % step == 0:
            _, binary = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            if boundary_mask is not None:
                binary = cv2.bitwise_and(binary, binary, mask=boundary_mask)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

            if valid:
                best = max(valid, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(best)

                # Expandir bbox un 25% para incluir pelo/cola cercana al cuerpo
                pad_x = max(5, int(bw * 0.25))
                pad_y = max(5, int(bh * 0.25))
                x1 = max(0,  x - pad_x)
                y1 = max(0,  y - pad_y)
                x2 = min(fw, x + bw + pad_x)
                y2 = min(fh, y + bh + pad_y)

                name = f"frame_{idx:07d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)

                cx_n = (x1 + x2) / 2 / fw
                cy_n = (y1 + y2) / 2 / fh
                bw_n = (x2 - x1) / fw
                bh_n = (y2 - y1) / fh
                with open(lbl_dir / f"{name}.txt", "w") as f:
                    f.write(f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")
                labeled += 1

            if progress_cb and labeled % 10 == 0:
                pct = 0.3 + (idx - warmup_frames) / usable * 0.6
                progress_cb(min(pct, 0.9), f"Etiquetando… {labeled} frames guardados")

        idx += 1

    cap.release()
    return labeled


def train_rat_detector(
    dataset_dir: Path,
    model_out_dir: Path,
    epochs: int = 50,
    imgsz: int = 640,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """Fine-tunea YOLOv8 nano. Devuelve la ruta al modelo .pt resultante."""
    from ultralytics import YOLO

    yaml_path = dataset_dir.parent / "dataset.yaml"
    yaml_content = {
        "path": str(dataset_dir),
        "train": "images",
        "val":   "images",
        "nc":    1,
        "names": ["rata"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    if progress_cb:
        progress_cb(0.0, "Descargando pesos base YOLOv8n…")

    model = YOLO("yolov8n.pt")

    if progress_cb:
        progress_cb(0.05, f"Entrenando {epochs} épocas… (puede tardar varios minutos en CPU)")

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        project=str(model_out_dir),
        name="rat_detector",
        exist_ok=True,
        verbose=False,
        plots=False,
        device="cpu",
    )

    return model_out_dir / "rat_detector" / "weights" / "best.pt"
