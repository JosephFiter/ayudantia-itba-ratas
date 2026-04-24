from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"

for _d in (UPLOADS_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRACKER_DEFAULTS = {
    "method": "mog2",        # "mog2" | "threshold"
    "invert": False,          # True si la rata es clara sobre fondo oscuro
    "min_area": 200,
    "max_area": 10_000,
    "blur_size": 3,
    "history": 500,
    "var_threshold": 40,
}

USV_DEFAULTS = {
    "freq_min": 30_000,
    "freq_max": 90_000,
    "threshold_db": -35,
    "min_duration_s": 0.005,
}

APP_TITLE = "Rat Tracker — Análisis de Comportamiento Animal"
