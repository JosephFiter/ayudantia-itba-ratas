from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"

YOLO_DIR = DATA_DIR / "yolo"

for _d in (UPLOADS_DIR, CACHE_DIR, RESULTS_DIR, YOLO_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRACKER_DEFAULTS = {
    "method": "mog2",        # "mog2" | "threshold" | "color"
    "invert": False,
    "min_area": 200,
    "max_area": 10_000,
    "blur_size": 3,
    "history": 500,
    "var_threshold": 40,
    # Filtrado de forma (evita trackear la cola)
    "min_circularity": 0.35,  # 0 = sin filtro, 1 = solo círculos perfectos
    # Filtro de salto (evita teleportaciones)
    "max_jump_px": 120,        # px máx desde la predicción de Kalman
    # Congelamiento de MOG2 cuando la rata está quieta
    "still_speed_px": 4.0,    # px/frame — por debajo de esto se frena el aprendizaje
    # Color detection (rata blanca)
    "hue_min": 0,
    "hue_max": 179,
    "sat_max": 30,
    "val_min": 200,
    "val_max": 255,
}

USV_DEFAULTS = {
    "freq_min": 30_000,
    "freq_max": 90_000,
    "threshold_db": -35,
    "min_duration_s": 0.005,
}

APP_TITLE = "Rat Tracker — Análisis de Comportamiento Animal"
