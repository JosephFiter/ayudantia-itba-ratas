import sys
from pathlib import Path

# Asegurar que el root del proyecto esté en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from config import APP_TITLE

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🐀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🐀 Rat Tracker")
st.caption("Análisis de comportamiento animal — video cenital + ultrasonido")

st.markdown("""
### Flujo de trabajo

| Paso | Página | Descripción |
|------|--------|-------------|
| 1 | **Upload** | Cargar video y archivo de audio |
| 2 | **Calibración** | Definir la relación píxel/cm |
| 3 | **Tracking** | Ejecutar el algoritmo de seguimiento |
| 4 | **Sincronización** | Ajustar el offset temporal audio-video |
| 5 | **Análisis** | Heatmap, velocidad, exportar resultados |

Usá el menú de la izquierda para navegar entre pasos.
""")

st.info("Empezá por la página **1 Upload** en el menú lateral.")

# Inicializar session_state con valores por defecto
defaults = {
    "video_path": None,
    "audio_path": None,
    "fps": 25.0,
    "frame_shape": None,
    "first_frame": None,
    "tracking_df": None,
    "usv_events": None,
    "spec_data": None,
    "px_per_cm": 1.0,
    "calib_done": False,
    "audio_offset": 0.0,
    "tracker_config": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
