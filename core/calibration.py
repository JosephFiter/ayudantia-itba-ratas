import numpy as np


def compute_px_per_cm(
    p1: tuple[float, float],
    p2: tuple[float, float],
    real_cm: float,
) -> float:
    """Calcula píxeles por centímetro a partir de dos puntos y la distancia real."""
    dist_px = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    if real_cm <= 0 or dist_px <= 0:
        raise ValueError("Distancia real y distancia en píxeles deben ser > 0")
    return dist_px / real_cm


def px_to_cm(px: float, px_per_cm: float) -> float:
    return px / px_per_cm


def cm_to_px(cm: float, px_per_cm: float) -> float:
    return cm * px_per_cm
