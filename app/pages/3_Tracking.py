import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from config import APP_TITLE, CACHE_DIR, YOLO_DIR, TRACKER_DEFAULTS
from core.tracker import RatTracker
from core.metrics import compute_metrics, summary_stats

st.set_page_config(page_title=f"Tracking — {APP_TITLE}", layout="wide")
st.title("3 · Tracking")

if not st.session_state.get("video_path"):
    st.warning("Primero cargá un video en la página **1 Upload**.")
    st.stop()

for key, default in [
    ("boundary_points", []),
    ("last_boundary_click", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Definir límites del laberinto ───────────────────────────────────────────
st.subheader("Límites del laberinto")
st.info("Hacé click en varios puntos alrededor del laberinto. El tracker solo usará esa región para evitar detecciones en la luz o en el borde.")

frame = st.session_state.first_frame
h, w = st.session_state.frame_shape
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame

CANVAS_W = min(w, 900)
scale = CANVAS_W / w
CANVAS_H = int(h * scale)

display = Image.fromarray(rgb).resize((CANVAS_W, CANVAS_H))
draw = ImageDraw.Draw(display)

points = st.session_state.boundary_points
for i, pt in enumerate(points):
    x, y = int(pt[0] * scale), int(pt[1] * scale)
    draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill="lime", outline="white")
    draw.text((x + 8, y - 8), str(i + 1), fill="lime")

if len(points) > 1:
    line = [(int(x * scale), int(y * scale)) for x, y in points]
    if len(points) >= 3:
        draw.polygon(line, outline="cyan")
    draw.line(line + [line[0]], fill="cyan", width=2)

coords = streamlit_image_coordinates(display, key="boundary_img")
if coords and coords != st.session_state.last_boundary_click:
    st.session_state.last_boundary_click = coords
    new_point = (int(coords["x"] / scale), int(coords["y"] / scale))
    st.session_state.boundary_points.append(new_point)

col_left, col_right = st.columns([3, 1])
with col_left:
    if st.button("Borrar último punto"):
        if st.session_state.boundary_points:
            st.session_state.boundary_points.pop()
            st.session_state.last_boundary_click = None
with col_right:
    if st.button("Reiniciar límites"):
        st.session_state.boundary_points = []
        st.session_state.last_boundary_click = None

if points:
    st.caption(f"Puntos definidos: {len(points)}. Deben ser al menos 3 para crear una región válida.")

st.image(display, caption="Haz click para definir los límites", use_column_width=True)

st.divider()

# ── Configuración del tracker ────────────────────────────────────────────────
st.subheader("Parámetros del algoritmo")

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox(
        "Método de detección",
        ["mog2", "threshold", "color", "yolo"],
        help="mog2: Background Subtraction adaptativo. threshold: Umbralización de Otsu. color: Detección de rata blanca. yolo: Modelo IA entrenado (recomendado para visión nocturna).",
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

st.markdown("**Filtros de calidad de detección**")
col3, col4, col5 = st.columns(3)
with col3:
    min_circularity = st.slider(
        "Circularidad mínima",
        min_value=0.0, max_value=0.8,
        value=float(TRACKER_DEFAULTS["min_circularity"]),
        step=0.05,
        help="Filtra la cola (elongada). 0 = sin filtro. Subí si trackea la cola, bajá si pierde el cuerpo.",
    )
with col4:
    max_jump_px = st.slider(
        "Salto máximo (px)",
        min_value=20, max_value=400,
        value=int(TRACKER_DEFAULTS["max_jump_px"]),
        step=10,
        help="Distancia máxima desde la última posición. Evita teleportaciones a detecciones falsas.",
    )
with col5:
    still_speed_px = st.slider(
        "Umbral de quietud (px/frame)",
        min_value=1.0, max_value=20.0,
        value=float(TRACKER_DEFAULTS["still_speed_px"]),
        step=0.5,
        help="Por debajo de esta velocidad se frena el aprendizaje del fondo en MOG2. Evita que la rata 'desaparezca' cuando está quieta.",
    )

if method == "color":
    st.subheader("Ajuste de color blanco")
    col3, col4 = st.columns(2)
    with col3:
        sat_max = st.slider(
            "Saturación máxima",
            min_value=0, max_value=255, value=TRACKER_DEFAULTS["sat_max"],
            help="Color blanco tiene baja saturación.",
        )
        val_min = st.slider(
            "Brillo mínimo",
            min_value=0, max_value=255, value=TRACKER_DEFAULTS["val_min"],
            help="La rata blanca es brillante en el video.",
        )
    with col4:
        hue_min = st.slider("Matiz mínimo", 0, 179, TRACKER_DEFAULTS["hue_min"])
        hue_max = st.slider("Matiz máximo", 0, 179, TRACKER_DEFAULTS["hue_max"])

tracker_config = {
    "method": method,
    "invert": invert,
    "min_area": int(min_area),
    "max_area": int(max_area),
    "var_threshold": int(var_threshold),
    "blur_size": int(blur_size),
    "min_circularity": float(min_circularity),
    "max_jump_px": int(max_jump_px),
    "still_speed_px": float(still_speed_px),
}

if method == "color":
    tracker_config.update({
        "hue_min": int(hue_min),
        "hue_max": int(hue_max),
        "sat_max": int(sat_max),
        "val_min": int(val_min),
        "val_max": TRACKER_DEFAULTS["val_max"],
    })

if len(st.session_state.boundary_points) >= 3:
    tracker_config["boundary_points"] = st.session_state.boundary_points

st.divider()

# ── Entrenar modelo IA (YOLOv8) ───────────────────────────────────────────────
st.subheader("Modelo IA (YOLOv8)")

YOLO_MODEL_PATH = YOLO_DIR / "rat_detector" / "weights" / "best.pt"

if YOLO_MODEL_PATH.exists():
    st.success(f"Modelo entrenado disponible: `{YOLO_MODEL_PATH.name}`")
    if st.button("Reentrenar modelo", help="Volvé a entrenar si cambiaron los videos o los parámetros."):
        st.session_state["run_training"] = True
else:
    st.info("No hay modelo IA entrenado todavía. Entrenalo una vez y después usá el método **yolo** en el tracking.")
    if st.button("Entrenar modelo IA", type="primary"):
        st.session_state["run_training"] = True

if st.session_state.get("run_training"):
    st.session_state["run_training"] = False
    from core.trainer import extract_and_label, train_rat_detector

    col_train1, col_train2 = st.columns(2)
    with col_train1:
        n_frames = st.number_input("Frames a etiquetar", min_value=50, max_value=1000, value=300, step=50)
    with col_train2:
        epochs = st.number_input("Épocas de entrenamiento", min_value=10, max_value=200, value=50, step=10)

    prog = st.progress(0.0, text="Iniciando…")
    log  = st.empty()

    def cb(pct, msg):
        prog.progress(float(pct), text=msg)
        log.caption(msg)

    try:
        dataset_dir = YOLO_DIR / "dataset"
        cb(0.0, "Extrayendo y etiquetando frames con tracker CV…")
        labeled = extract_and_label(
            st.session_state.video_path,
            dataset_dir,
            tracker_config,
            n_frames=int(n_frames),
            progress_cb=cb,
        )
        if labeled < 10:
            st.error(f"Solo se encontraron {labeled} detecciones. Ajustá los parámetros del tracker CV primero y volvé a intentar.")
        else:
            cb(0.6, f"{labeled} frames etiquetados. Entrenando YOLOv8 nano… (puede tardar varios minutos)")
            trained_path = train_rat_detector(
                dataset_dir,
                YOLO_DIR,
                epochs=int(epochs),
                progress_cb=cb,
            )
            prog.progress(1.0, text="Entrenamiento completado.")
            st.success(f"Modelo guardado en `{trained_path}`. Ahora seleccioná el método **yolo** y ejecutá el tracking.")
    except Exception as e:
        st.error(f"Error durante el entrenamiento: {e}")

if YOLO_MODEL_PATH.exists() and method == "yolo":
    tracker_config["yolo_model_path"] = str(YOLO_MODEL_PATH)
elif method == "yolo" and not YOLO_MODEL_PATH.exists():
    st.warning("Método **yolo** seleccionado pero no hay modelo entrenado. Entrenalo primero.")

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

    st.markdown("➡️ Continuá en **5 Análisis**")
