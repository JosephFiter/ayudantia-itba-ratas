import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from config import APP_TITLE
from utils.visualization import build_timeline, build_spectrogram

st.set_page_config(page_title=f"Sincronización — {APP_TITLE}", layout="wide")
st.title("4 · Sincronización Audio ↔ Video")

if st.session_state.get("tracking_df") is None:
    st.warning("Primero completá el tracking en la página **3 Tracking**.")
    st.stop()

if st.session_state.get("usv_events") is None:
    st.info("No hay datos de audio. Podés continuar sin sincronización o cargar audio en **1 Upload**.")
    usv_df = pd.DataFrame()
else:
    usv_df = st.session_state.usv_events

tracking_df = st.session_state.tracking_df

# ── Control de offset ─────────────────────────────────────────────────────────
st.subheader("Ajuste de offset")
st.markdown("""
El **offset** compensa que el audio y el video no arrancaron al mismo tiempo.
- **Positivo**: el audio arrancó *antes* que el video (se corre el audio hacia adelante).
- **Negativo**: el audio arrancó *después* del video.

Ajustá el slider hasta que los picos de velocidad coincidan visualmente con los eventos USV.
""")

col_slider, col_fine = st.columns([3, 1])
with col_slider:
    offset = st.slider(
        "Offset de audio (segundos)",
        min_value=-60.0, max_value=60.0,
        value=float(st.session_state.get("audio_offset", 0.0)),
        step=0.05,
        help="Desplazamiento del eje temporal del audio respecto al video.",
    )
with col_fine:
    offset_fine = st.number_input(
        "Ajuste fino (s)", value=offset, step=0.01, format="%.3f"
    )
    if offset_fine != offset:
        offset = offset_fine

st.session_state.audio_offset = offset
st.caption(f"Offset actual: **{offset:+.3f} s**")

st.divider()

# ── Timeline interactivo ──────────────────────────────────────────────────────
st.subheader("Timeline: Velocidad + USV")

fig_timeline = build_timeline(tracking_df, usv_df, audio_offset=offset)
st.plotly_chart(fig_timeline, use_container_width=True)

# ── Espectrograma ─────────────────────────────────────────────────────────────
if st.session_state.get("spec_data") is not None:
    st.subheader("Espectrograma USV")
    times, freqs, spec_db = st.session_state.spec_data
    cfg = st.session_state.get("usv_config", {"freq_min": 30_000, "freq_max": 90_000})
    fig_spec = build_spectrogram(
        times, freqs, spec_db,
        freq_min=cfg["freq_min"],
        freq_max=cfg["freq_max"],
        audio_offset=offset,
    )
    st.plotly_chart(fig_spec, use_container_width=True)

st.divider()

# ── Resumen de eventos con offset aplicado ────────────────────────────────────
if len(usv_df):
    st.subheader("Eventos USV sincronizados")
    usv_synced = usv_df.copy()
    usv_synced["time_start"] = usv_synced["time_start"] + offset
    usv_synced["time_end"] = usv_synced["time_end"] + offset

    # Buscar velocidad media durante cada evento USV
    def speed_at_event(row):
        mask = (
            (tracking_df["timestamp"] >= row["time_start"]) &
            (tracking_df["timestamp"] <= row["time_end"])
        )
        seg = tracking_df[mask]
        return seg["speed_smooth"].mean() if len(seg) else float("nan")

    usv_synced["speed_during_cms"] = usv_synced.apply(speed_at_event, axis=1)

    st.dataframe(
        usv_synced[["time_start", "time_end", "duration_ms",
                    "freq_peak_hz", "power_db", "speed_during_cms"]]
        .round(3),
        use_container_width=True,
    )
    st.caption(f"Total de eventos: **{len(usv_synced)}**")

st.divider()
st.markdown("➡️ Continuá en **5 Análisis**")
