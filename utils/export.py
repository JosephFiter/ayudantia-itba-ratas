import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def export_csv(tracking_df: pd.DataFrame, path: str) -> None:
    tracking_df.to_csv(path, index=False)


def export_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def export_annotated_video(
    video_path: str,
    tracking_df: pd.DataFrame,
    output_path: str,
    px_per_cm: float = 1.0,
    show_trail: int = 60,
) -> None:
    """Genera video con trayectoria anotada y velocidad."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    df = tracking_df.set_index("frame")
    trail: list[tuple[int, int]] = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in df.index:
            row = df.loc[frame_idx]
            if pd.notna(row["x"]) and pd.notna(row["y"]):
                cx, cy = int(row["x"]), int(row["y"])
                trail.append((cx, cy))
                if len(trail) > show_trail:
                    trail.pop(0)

                # Dibujar trail
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    color = (int(255 * alpha), int(100 * alpha), 0)
                    cv2.line(frame, trail[i - 1], trail[i], color, 2)

                # Círculo y centroide
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2)
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)

                # Velocidad
                if "speed_smooth" in df.columns and pd.notna(row["speed_smooth"]):
                    spd = row["speed_smooth"]
                    cv2.putText(frame, f"{spd:.1f} cm/s",
                                (cx + 12, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)

        # Timestamp
        t = frame_idx / fps
        cv2.putText(frame, f"t={t:.2f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
