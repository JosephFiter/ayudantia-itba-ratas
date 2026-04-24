import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame, px_per_cm: float) -> pd.DataFrame:
    """
    Agrega columnas de velocidad y distancia al DataFrame de tracking.
    Requiere columnas: timestamp, x, y
    """
    df = df.copy()

    dx = df["x"].diff()
    dy = df["y"].diff()
    dt = df["timestamp"].diff().replace(0, np.nan)

    df["dist_px"] = np.sqrt(dx**2 + dy**2)
    df["dist_cm"] = df["dist_px"] / px_per_cm
    df["speed_cms"] = df["dist_cm"] / dt

    # Suavizado: ventana de ~0.5 s
    fps_approx = 1.0 / dt.median() if dt.median() > 0 else 25.0
    window = max(3, int(fps_approx * 0.5))
    df["speed_smooth"] = (
        df["speed_cms"]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )
    df["total_dist_cm"] = df["dist_cm"].fillna(0).cumsum()

    return df


def summary_stats(df: pd.DataFrame) -> dict:
    active = df[df["detected"]]
    return {
        "total_dist_cm": df["total_dist_cm"].iloc[-1] if len(df) else 0,
        "mean_speed_cms": active["speed_smooth"].mean() if len(active) else 0,
        "max_speed_cms": active["speed_smooth"].max() if len(active) else 0,
        "detection_pct": 100 * df["detected"].sum() / max(len(df), 1),
        "duration_s": df["timestamp"].iloc[-1] if len(df) else 0,
    }
