import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from core.calibration import compute_px_per_cm
from config import APP_TITLE

st.set_page_config(page_title=f"Calibración — {APP_TITLE}", layout="wide")
st.title("2 · Calibración píxel / cm")

if not st.session_state.get("video_path"):
    st.warning("Primero cargá un video en la página **1 Upload**.")
    st.stop()

frame = st.session_state.first_frame
h, w = st.session_state.frame_shape

st.markdown("""
**Cómo calibrar:**
1. Mirá la imagen de abajo. Pasá el cursor por dos puntos cuya distancia real conocés
   (ej: dos extremos del laberinto, una regla en el piso, marcas en la pared).
2. Los coordenadas aparecen en la esquina al hacer hover.
3. Ingresá esas coordenadas y la distancia real en cm.
""")

# ── Mostrar frame con plotly (coordenadas en hover) ───────────────────────────
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame

fig = go.Figure()
fig.add_trace(go.Image(z=rgb))
fig.update_layout(
    width=min(w, 900),
    height=int(min(w, 900) * h / w),
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(showticklabels=True, title="x (px)"),
    yaxis=dict(showticklabels=True, title="y (px)", autorange="reversed"),
)
st.plotly_chart(fig, use_container_width=True)

st.caption("Pasá el mouse sobre la imagen para ver las coordenadas (x, y) en la barra de Plotly.")

st.divider()

# ── Inputs de calibración ────────────────────────────────────────────────────
st.subheader("Ingresar puntos de referencia")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Punto 1**")
    x1 = st.number_input("x₁ (px)", min_value=0, max_value=w, value=50, key="cx1")
    y1 = st.number_input("y₁ (px)", min_value=0, max_value=h, value=50, key="cy1")

with col2:
    st.markdown("**Punto 2**")
    x2 = st.number_input("x₂ (px)", min_value=0, max_value=w, value=w - 50, key="cx2")
    y2 = st.number_input("y₂ (px)", min_value=0, max_value=h, value=50, key="cy2")

real_cm = st.number_input(
    "Distancia real entre los dos puntos (cm)",
    min_value=0.1, value=30.0, step=0.5,
)

# Preview de la línea de calibración
dist_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
st.info(f"Distancia en píxeles: **{dist_px:.1f} px** → **{dist_px / real_cm:.2f} px/cm**")

fig2 = go.Figure()
fig2.add_trace(go.Image(z=rgb))
fig2.add_trace(go.Scatter(
    x=[x1, x2], y=[y1, y2],
    mode="lines+markers+text",
    line=dict(color="cyan", width=2),
    marker=dict(size=10, color=["lime", "red"]),
    text=["P1", "P2"], textposition="top center",
    textfont=dict(color="white", size=13),
    name="Línea de calibración",
))
fig2.update_layout(
    width=min(w, 900),
    height=int(min(w, 900) * h / w),
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(showticklabels=True, title="x (px)"),
    yaxis=dict(showticklabels=True, title="y (px)", autorange="reversed"),
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

if st.button("Guardar calibración", type="primary"):
    try:
        px_per_cm = compute_px_per_cm((x1, y1), (x2, y2), real_cm)
        st.session_state.px_per_cm = px_per_cm
        st.session_state.calib_done = True
        st.success(f"Calibración guardada: **{px_per_cm:.3f} px/cm** ({1/px_per_cm*10:.2f} mm/px)")
    except ValueError as e:
        st.error(str(e))

if st.session_state.get("calib_done"):
    st.metric("px/cm actual", f"{st.session_state.px_per_cm:.3f}")
    st.markdown("➡️ Continuá en **3 Tracking**")
