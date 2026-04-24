import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from core.calibration import compute_px_per_cm
from config import APP_TITLE

st.set_page_config(page_title=f"Calibración — {APP_TITLE}", layout="wide")
st.title("2 · Calibración píxel / cm")

if not st.session_state.get("video_path"):
    st.warning("Primero cargá un video en la página **1 Upload**.")
    st.stop()

frame = st.session_state.first_frame
h, w = st.session_state.frame_shape
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame

for key, default in [("calib_p1", None), ("calib_p2", None), ("last_calib_click", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

p1 = st.session_state.calib_p1
p2 = st.session_state.calib_p2

# ── Instrucciones dinámicas ───────────────────────────────────────────────────
if p1 is None:
    st.info("Hacé click en el **punto 1** de la línea de calibración.")
elif p2 is None:
    st.info("Ahora hacé click en el **punto 2**.")
else:
    st.success("Línea trazada. Ingresá la distancia real y guardá.")

# ── Imagen con overlay ────────────────────────────────────────────────────────
CANVAS_W = min(w, 800)
scale = CANVAS_W / w
CANVAS_H = int(h * scale)

display = Image.fromarray(rgb).resize((CANVAS_W, CANVAS_H))
draw = ImageDraw.Draw(display)

def to_display(pt):
    return (int(pt[0] * scale), int(pt[1] * scale))

if p1:
    dp1 = to_display(p1)
    draw.ellipse([dp1[0]-6, dp1[1]-6, dp1[0]+6, dp1[1]+6], fill="lime", outline="white", width=1)
    draw.text((dp1[0]+8, dp1[1]-8), "P1", fill="lime")
if p2:
    dp2 = to_display(p2)
    draw.ellipse([dp2[0]-6, dp2[1]-6, dp2[0]+6, dp2[1]+6], fill="red", outline="white", width=1)
    draw.text((dp2[0]+8, dp2[1]-8), "P2", fill="red")
if p1 and p2:
    draw.line([to_display(p1), to_display(p2)], fill="cyan", width=2)

coords = streamlit_image_coordinates(display, key="calib_img")

# Solo procesar click nuevo (distinto al anterior)
if coords and coords != st.session_state.last_calib_click:
    st.session_state.last_calib_click = coords
    actual = (int(coords["x"] / scale), int(coords["y"] / scale))
    if p1 is None:
        st.session_state.calib_p1 = actual
    elif p2 is None:
        st.session_state.calib_p2 = actual
    st.rerun()

st.divider()

# ── Controles ─────────────────────────────────────────────────────────────────
real_cm = st.number_input("Distancia real de la línea (cm)", min_value=0.1, value=30.0, step=0.5)

col_reset, col_save = st.columns([1, 2])

with col_reset:
    if st.button("Reiniciar puntos"):
        st.session_state.calib_p1 = None
        st.session_state.calib_p2 = None
        st.session_state.last_calib_click = None
        st.rerun()

if p1 and p2:
    dist_px = float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))
    st.caption(f"Distancia en píxeles: **{dist_px:.1f} px** → **{dist_px / real_cm:.2f} px/cm**")

    with col_save:
        if st.button("Guardar calibración", type="primary"):
            try:
                px_per_cm = compute_px_per_cm(p1, p2, real_cm)
                st.session_state.px_per_cm = px_per_cm
                st.session_state.calib_done = True
                st.success(f"Calibración guardada: **{px_per_cm:.3f} px/cm** ({1/px_per_cm*10:.2f} mm/px)")
            except ValueError as e:
                st.error(str(e))

if st.session_state.get("calib_done"):
    st.metric("px/cm actual", f"{st.session_state.px_per_cm:.3f}")
    st.markdown("➡️ Continuá en **3 Tracking**")
