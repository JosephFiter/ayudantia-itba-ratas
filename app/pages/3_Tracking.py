import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from config import APP_TITLE, CACHE_DIR, TRACKER_DEFAULTS
from core.tracker import RatTracker
from core.metrics import compute_metrics, summary_stats

st.set_page_config(page_title=f"Tracking — {APP_TITLE}", layout="wide")
st.title("3 · Tracking")

if not st.session_state.get("video_path"):
    st.warning("Primero cargá un video en la página **1 Upload**.")
    st.stop()

# ── Configuración del tracker ────────────────────────────────────────────────
st.subheader("Parámetros del algoritmo")

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox(
        "Método de detección",
        ["mog2", "threshold"],
        help="mog2: Background Subtraction adaptativo (recomendado). threshold: Umbralización de Otsu (fondo 100% estático).",
    )
    invert = st.checkbox(
        "Invertir (rata clara sobre fondo oscuro)",
        value=TRACKER_DEFAULTS["invert"],
    )
    min_area = st.number_input("Área mínima de contorno (px²)", value=TRACKER_DEFAULTS["min_area"], step=50)

with col2:
    max_area = st.number_input("Área máxima de contorno (px²)", value=TRACKER_DEFAULTS["max_area"], step=100)
    var_threshold = st.slider(
        "Sensibilidad MOG2 (var_threshold)",
        min_value=10, max_value=200, value=TRACKER_DEFAULTS["var_threshold"],
        help="Valores bajos = más sensible al movimiento, más falsos positivos.",
    )
    blur_size = st.slider("Radio de blur (px)", 1, 10, TRACKER_DEFAULTS["blur_size"])

tracker_config = {
    "method": method,
    "invert": invert,
    "min_area": int(min_area),
    "max_area": int(max_area),
    "var_threshold": int(var_threshold),
    "blur_size": int(blur_size),
}

st.divider()

# ── Test en un frame ──────────────────────────────────────────────────────────
st.subheader("Preview en un frame")
cap_test = cv2.VideoCapture(st.session_state.video_path)
total_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
cap_test.release()

test_frame_idx = st.slider("Frame de prueba", 0, max(0, total_frames - 1), total_frames // 2)

if st.button("Probar detección en este frame"):
    cap = cv2.VideoCapture(st.session_state.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        tracker_test = RatTracker(tracker_config)
        # Calentar el sustractor con frames previos
        cap2 = cv2.VideoCapture(st.session_state.video_path)
        warm_start = max(0, test_frame_idx - 50)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, warm_start)
        for _ in range(min(50, test_frame_idx)):
            r, f = cap2.read()
            if r:
                tracker_test.detect(f)
        cap2.release()

        detection, mask = tracker_test.detect(frame)
        col_a, col_b = st.columns(2)
        col_a.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame,
            caption="Frame original",
            use_container_width=True,
        )
        col_b.image(mask, caption="Máscara (foreground)", use_container_width=True)

        if detection:
            cx, cy = detection
            annotated = RatTracker.annotate_frame(frame, cx, cy)
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"Detección: ({cx:.0f}, {cy:.0f})",
                use_container_width=True,
            )
            st.success(f"Rata detectada en ({cx:.1f}, {cy:.1f})")
        else:
            st.error("No se detectó la rata en este frame. Ajustá los parámetros.")

st.divider()

# ── Ejecutar tracking completo ────────────────────────────────────────────────
st.subheader("Ejecutar tracking completo")

cache_file = CACHE_DIR / f"{Path(st.session_state.video_path).stem}_tracking.parquet"

if cache_file.exists() and st.session_state.tracking_df is not None:
    st.info("Ya existe un resultado de tracking en memoria. Podés re-ejecutarlo si cambiaste parámetros.")

col_run, col_load = st.columns(2)

with col_run:
    if st.button("▶ Ejecutar tracking", type="primary"):
        st.session_state.tracker_config = tracker_config
        tracker = RatTracker(tracker_config)
        fps = st.session_state.fps
        px_per_cm = st.session_state.px_per_cm

        progress_bar = st.progress(0.0, text="Procesando video...")
        status_text = st.empty()

        def update_progress(p: float):
            progress_bar.progress(min(p, 1.0), text=f"Procesando... {p*100:.0f}%")

        try:
            df, detected_fps = tracker.process_video(
                st.session_state.video_path,
                progress_cb=update_progress,
            )
            progress_bar.progress(1.0, text="Calculando métricas...")
            df = compute_metrics(df, px_per_cm)
            df.to_parquet(cache_file, index=False)

            st.session_state.tracking_df = df
            progress_bar.empty()
            st.success(f"Tracking completado: {len(df)} frames procesados.")
        except Exception as e:
            st.error(f"Error durante el tracking: {e}")

with col_load:
    if cache_file.exists():
        if st.button("Cargar resultado previo desde caché"):
            df = pd.read_parquet(cache_file)
            if "speed_smooth" not in df.columns:
                df = compute_metrics(df, st.session_state.px_per_cm)
            st.session_state.tracking_df = df
            st.success("Cargado desde caché.")

# ── Resumen ───────────────────────────────────────────────────────────────────
if st.session_state.tracking_df is not None:
    df = st.session_state.tracking_df
    stats = summary_stats(df)

    st.divider()
    st.subheader("Resumen")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dist. total", f"{stats['total_dist_cm']:.1f} cm")
    c2.metric("Vel. media", f"{stats['mean_speed_cms']:.1f} cm/s")
    c3.metric("Vel. máx.", f"{stats['max_speed_cms']:.1f} cm/s")
    c4.metric("Detección", f"{stats['detection_pct']:.1f} %")
    c5.metric("Duración", f"{stats['duration_s']:.1f} s")

    with st.expander("Ver primeras filas del DataFrame"):
        st.dataframe(df.head(50), use_container_width=True)

    st.markdown("➡️ Continuá en **4 Sincronización**")
