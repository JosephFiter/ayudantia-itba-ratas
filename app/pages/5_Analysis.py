import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import io
from config import APP_TITLE, RESULTS_DIR
from core.metrics import summary_stats
from utils.visualization import build_heatmap, build_trajectory, build_timeline
from utils.export import export_csv, export_annotated_video

st.set_page_config(page_title=f"Análisis — {APP_TITLE}", layout="wide")
st.title("5 · Análisis y Exportación")

if st.session_state.get("tracking_df") is None:
    st.warning("Primero completá el tracking en la página **3 Tracking**.")
    st.stop()

df = st.session_state.tracking_df
usv_df = st.session_state.get("usv_events") or pd.DataFrame()
offset = st.session_state.get("audio_offset", 0.0)
frame_shape = st.session_state.frame_shape
first_frame = st.session_state.first_frame
px_per_cm = st.session_state.px_per_cm

# ── Métricas globales ─────────────────────────────────────────────────────────
st.subheader("Métricas globales")
stats = summary_stats(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dist. total", f"{stats['total_dist_cm']:.1f} cm")
c2.metric("Vel. media", f"{stats['mean_speed_cms']:.1f} cm/s")
c3.metric("Vel. máx.", f"{stats['max_speed_cms']:.1f} cm/s")
c4.metric("Detección", f"{stats['detection_pct']:.1f} %")
c5.metric("Duración", f"{stats['duration_s']:.1f} s")

if len(usv_df):
    st.metric("Eventos USV detectados", len(usv_df))

st.divider()

# ── Filtro temporal ───────────────────────────────────────────────────────────
st.subheader("Filtro temporal")
t_max = float(df["timestamp"].max())
t_range = st.slider(
    "Rango de análisis (s)",
    0.0, t_max, (0.0, t_max), step=0.5,
)
df_filtered = df[(df["timestamp"] >= t_range[0]) & (df["timestamp"] <= t_range[1])]
st.caption(f"Frames en rango: **{len(df_filtered):,}** de {len(df):,}")

st.divider()

# ── Visualizaciones ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔥 Heatmap", "🗺️ Trayectoria", "📈 Timeline"])

with tab1:
    st.markdown("### Heatmap de ocupación")
    bins = st.slider("Resolución del heatmap (bins)", 20, 100, 60, key="hm_bins")
    show_bg = st.checkbox("Mostrar fondo del video", value=True, key="hm_bg")
    bg = first_frame if show_bg else None
    fig_hm = build_heatmap(df_filtered, frame_shape, bg_frame=bg, bins=bins)
    st.pyplot(fig_hm, use_container_width=True)

with tab2:
    st.markdown("### Trayectoria (coloreada por tiempo)")
    fig_traj = build_trajectory(df_filtered, frame_shape, bg_frame=first_frame)
    st.pyplot(fig_traj, use_container_width=True)

with tab3:
    st.markdown("### Velocidad + USV sincronizados")
    fig_tl = build_timeline(df_filtered, usv_df, audio_offset=offset)
    st.plotly_chart(fig_tl, use_container_width=True)

st.divider()

# ── Análisis por zonas (opcional) ─────────────────────────────────────────────
with st.expander("Análisis por zonas del laberinto (opcional)"):
    st.markdown("""
    Definí zonas rectangulares del laberinto para calcular el tiempo de permanencia en cada una.
    Coordinadas en **píxeles** (podés leerlas en el hover del heatmap o la trayectoria).
    """)
    h_frame, w_frame = frame_shape
    n_zones = st.number_input("Cantidad de zonas", 1, 8, 2, step=1)
    zone_results = []
    for i in range(int(n_zones)):
        with st.container():
            st.markdown(f"**Zona {i+1}**")
            zc = st.columns(4)
            zx1 = zc[0].number_input(f"x1_z{i}", 0, w_frame, 0, key=f"zx1_{i}", label_visibility="collapsed")
            zy1 = zc[1].number_input(f"y1_z{i}", 0, h_frame, 0, key=f"zy1_{i}", label_visibility="collapsed")
            zx2 = zc[2].number_input(f"x2_z{i}", 0, w_frame, w_frame, key=f"zx2_{i}", label_visibility="collapsed")
            zy2 = zc[3].number_input(f"y2_z{i}", 0, h_frame, h_frame, key=f"zy2_{i}", label_visibility="collapsed")
            zone_mask = (
                (df_filtered["x"] >= zx1) & (df_filtered["x"] <= zx2) &
                (df_filtered["y"] >= zy1) & (df_filtered["y"] <= zy2)
            )
            frames_in = zone_mask.sum()
            time_in = frames_in / st.session_state.fps
            pct = 100 * frames_in / max(len(df_filtered), 1)
            zone_results.append({"Zona": i+1, "Frames": frames_in,
                                  "Tiempo (s)": round(time_in, 2), "% tiempo": round(pct, 1)})
    st.dataframe(pd.DataFrame(zone_results), use_container_width=True)

st.divider()

# ── Exportar ───────────────────────────────────────────────────────────────────
st.subheader("Exportar resultados")

stem = Path(st.session_state.video_path).stem

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**CSV — Tracking completo**")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Descargar CSV",
        data=csv_buf.getvalue().encode(),
        file_name=f"{stem}_tracking.csv",
        mime="text/csv",
    )

with col_b:
    if len(usv_df):
        st.markdown("**CSV — Eventos USV sincronizados**")
        usv_synced = usv_df.copy()
        usv_synced["time_start"] += offset
        usv_synced["time_end"] += offset
        csv_usv = io.StringIO()
        usv_synced.to_csv(csv_usv, index=False)
        st.download_button(
            "Descargar CSV USV",
            data=csv_usv.getvalue().encode(),
            file_name=f"{stem}_usv_synced.csv",
            mime="text/csv",
        )

with col_c:
    st.markdown("**Video anotado (mp4)**")
    out_video = str(RESULTS_DIR / f"{stem}_annotated.mp4")
    if st.button("Generar video anotado"):
        with st.spinner("Renderizando video anotado..."):
            try:
                export_annotated_video(
                    st.session_state.video_path,
                    df,
                    out_video,
                    px_per_cm=px_per_cm,
                )
                st.success(f"Video guardado en:\n`{out_video}`")
            except Exception as e:
                st.error(f"Error: {e}")
