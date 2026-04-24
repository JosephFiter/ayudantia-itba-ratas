import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import cv2
import shutil
from config import UPLOADS_DIR, APP_TITLE

st.set_page_config(page_title=f"Upload — {APP_TITLE}", layout="wide")
st.title("1 · Cargar archivos")

# ── Inicializar session state ────────────────────────────────────────────────
defaults = {
    "video_path": None, "audio_path": None, "fps": 25.0,
    "frame_shape": None, "first_frame": None, "tracking_df": None,
    "usv_events": None, "spec_data": None, "px_per_cm": 1.0,
    "calib_done": False, "audio_offset": 0.0, "tracker_config": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Upload video ──────────────────────────────────────────────────────────────
st.subheader("Video cenital (B&N)")
video_file = st.file_uploader(
    "Formatos soportados: mp4, avi, mov, mkv",
    type=["mp4", "avi", "mov", "mkv"],
    key="video_upload",
)

if video_file:
    dest = UPLOADS_DIR / video_file.name
    with open(dest, "wb") as f:
        shutil.copyfileobj(video_file, f)

    cap = cv2.VideoCapture(str(dest))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, first_frame = cap.read()
    cap.release()

    if ret:
        st.session_state.video_path = str(dest)
        st.session_state.fps = fps
        st.session_state.frame_shape = (h, w)
        st.session_state.first_frame = first_frame
        # Invalidar resultados previos si cambia el video
        st.session_state.tracking_df = None
        st.session_state.calib_done = False

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("FPS", f"{fps:.1f}")
        col2.metric("Frames totales", f"{total_frames:,}")
        col3.metric("Resolución", f"{w}×{h}")
        col4.metric("Duración", f"{total_frames/fps:.1f} s")

        st.image(
            cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
            caption="Primer frame del video",
            use_container_width=True,
        )
        st.success(f"Video cargado: {video_file.name}")
    else:
        st.error("No se pudo leer el video. Verificá el formato.")

elif st.session_state.video_path:
    st.info(f"Video ya cargado: `{Path(st.session_state.video_path).name}`")

st.divider()

# ── Upload audio ──────────────────────────────────────────────────────────────
st.subheader("Audio de ultrasonido")
audio_file = st.file_uploader(
    "Formatos soportados: wav, flac, aiff  |  Tasa de muestreo recomendada: ≥ 192 kHz",
    type=["wav", "flac", "aif", "aiff"],
    key="audio_upload",
)

if audio_file:
    dest_audio = UPLOADS_DIR / audio_file.name
    with open(dest_audio, "wb") as f:
        shutil.copyfileobj(audio_file, f)
    st.session_state.audio_path = str(dest_audio)
    st.session_state.usv_events = None  # resetear detección previa
    st.success(f"Audio cargado: {audio_file.name}")

elif st.session_state.audio_path:
    st.info(f"Audio ya cargado: `{Path(st.session_state.audio_path).name}`")
else:
    st.warning("El audio es opcional. Sin él, no se detectarán USV.")

st.divider()

# ── Parámetros de detección USV ───────────────────────────────────────────────
if st.session_state.audio_path:
    st.subheader("Parámetros de detección USV")
    c1, c2, c3 = st.columns(3)
    freq_min = c1.number_input("Frecuencia mínima (Hz)", value=30_000, step=1_000)
    freq_max = c2.number_input("Frecuencia máxima (Hz)", value=90_000, step=1_000)
    threshold_db = c3.number_input("Umbral de potencia (dB)", value=-35, step=1)
    min_dur = st.number_input("Duración mínima de evento (ms)", value=5, step=1)

    st.session_state.usv_config = {
        "freq_min": int(freq_min),
        "freq_max": int(freq_max),
        "threshold_db": float(threshold_db),
        "min_duration_s": min_dur / 1000,
    }

    if st.button("Analizar audio (detectar USV)", type="primary"):
        with st.spinner("Procesando espectrograma..."):
            from core.audio_processor import detect_usv
            try:
                cfg = st.session_state.usv_config
                events_df, times, freqs, spec_db = detect_usv(
                    st.session_state.audio_path,
                    freq_min=cfg["freq_min"],
                    freq_max=cfg["freq_max"],
                    threshold_db=cfg["threshold_db"],
                    min_duration_s=cfg["min_duration_s"],
                )
                st.session_state.usv_events = events_df
                st.session_state.spec_data = (times, freqs, spec_db)
                st.success(f"Detectados **{len(events_df)}** eventos USV.")
                if len(events_df):
                    st.dataframe(events_df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Error al procesar audio: {e}")

st.divider()
st.markdown("➡️ Continuá en **2 Calibración**")
