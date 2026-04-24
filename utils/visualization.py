import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2


# ── Heatmap de ocupación ──────────────────────────────────────────────────────

def build_heatmap(
    tracking_df: pd.DataFrame,
    frame_shape: tuple[int, int],   # (height, width)
    bg_frame: np.ndarray | None = None,
    bins: int = 60,
) -> matplotlib.figure.Figure:
    df = tracking_df.dropna(subset=["x", "y"])
    h, w = frame_shape

    fig, ax = plt.subplots(figsize=(8, 8 * h / w))

    if bg_frame is not None:
        bg = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB) if bg_frame.ndim == 3 else bg_frame
        ax.imshow(bg, extent=[0, w, h, 0], cmap="gray", alpha=0.4)

    hb = ax.hexbin(
        df["x"], df["y"],
        gridsize=bins,
        cmap="hot",
        mincnt=1,
        extent=[0, w, 0, h],
        alpha=0.75,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Heatmap de ocupación")
    fig.colorbar(hb, ax=ax, label="Frecuencia de visitas")
    fig.tight_layout()
    return fig


def build_trajectory(
    tracking_df: pd.DataFrame,
    frame_shape: tuple[int, int],
    bg_frame: np.ndarray | None = None,
) -> matplotlib.figure.Figure:
    df = tracking_df.dropna(subset=["x", "y"])
    h, w = frame_shape

    fig, ax = plt.subplots(figsize=(8, 8 * h / w))
    if bg_frame is not None:
        bg = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB) if bg_frame.ndim == 3 else bg_frame
        ax.imshow(bg, extent=[0, w, h, 0], cmap="gray", alpha=0.5)

    sc = ax.scatter(df["x"], df["y"], c=df["timestamp"],
                    cmap="plasma", s=1, alpha=0.6)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title("Trayectoria (color = tiempo)")
    fig.colorbar(sc, ax=ax, label="Tiempo (s)")
    fig.tight_layout()
    return fig


# ── Timeline velocidad + USV ──────────────────────────────────────────────────

def build_timeline(
    tracking_df: pd.DataFrame,
    usv_df: pd.DataFrame,
    audio_offset: float = 0.0,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("Velocidad instantánea (cm/s)", "Vocalizaciones ultrasónicas (USV)"),
    )

    # Fila 1 — velocidad
    df = tracking_df.dropna(subset=["speed_smooth"])
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["speed_smooth"],
            mode="lines", name="Velocidad",
            line=dict(color="#00b4d8", width=1.5),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.15)",
        ),
        row=1, col=1,
    )

    # Puntos donde no hubo detección
    no_det = tracking_df[~tracking_df["detected"]]
    if len(no_det):
        fig.add_trace(
            go.Scatter(
                x=no_det["timestamp"],
                y=[0] * len(no_det),
                mode="markers", name="Sin detección",
                marker=dict(color="red", size=3, symbol="x"),
                showlegend=True,
            ),
            row=1, col=1,
        )

    # Fila 2 — eventos USV como barras verticales
    if len(usv_df):
        usv = usv_df.copy()
        usv["time_start"] += audio_offset
        usv["time_end"] += audio_offset

        for _, ev in usv.iterrows():
            fig.add_vrect(
                x0=ev["time_start"], x1=ev["time_end"],
                fillcolor="rgba(255,165,0,0.6)",
                layer="below", line_width=0,
                row=2, col=1,
            )
        # Línea fantasma para leyenda
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                name="USV", line=dict(color="orange", width=6),
            ),
            row=2, col=1,
        )
        # Puntos de frecuencia pico
        fig.add_trace(
            go.Scatter(
                x=(usv["time_start"] + usv["time_end"]) / 2,
                y=usv["freq_peak_hz"] / 1000,
                mode="markers", name="Frec. pico (kHz)",
                marker=dict(color="darkorange", size=6),
                yaxis="y3",
            ),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="cm/s", row=1, col=1)
    fig.update_yaxes(title_text="kHz", row=2, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
    fig.update_layout(
        height=520,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


# ── Espectrograma USV ─────────────────────────────────────────────────────────

def build_spectrogram(
    times: np.ndarray,
    freqs: np.ndarray,
    spec_db: np.ndarray,
    freq_min: int = 30_000,
    freq_max: int = 90_000,
    audio_offset: float = 0.0,
) -> go.Figure:
    band_idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    if len(band_idx) == 0:
        return go.Figure()

    band_spec = spec_db[band_idx, :]
    band_freqs = freqs[band_idx] / 1000  # kHz
    shifted_times = times + audio_offset

    fig = go.Figure(
        go.Heatmap(
            x=shifted_times,
            y=band_freqs,
            z=band_spec,
            colorscale="Inferno",
            colorbar=dict(title="dB"),
            zmin=-60, zmax=0,
        )
    )
    fig.update_layout(
        title="Espectrograma USV",
        xaxis_title="Tiempo (s)",
        yaxis_title="Frecuencia (kHz)",
        height=320,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig
