import numpy as np
import pandas as pd
import librosa
import soundfile as sf


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Carga audio manteniendo la tasa de muestreo original (crítico para ultrasonido)."""
    y, sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def detect_usv(
    audio_path: str,
    freq_min: int = 30_000,
    freq_max: int = 90_000,
    threshold_db: float = -35,
    min_duration_s: float = 0.005,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detecta Vocalizaciones Ultrasónicas (USV).

    Returns:
        events_df    : DataFrame con [time_start, time_end, duration_ms, freq_peak_hz, power_db]
        times        : eje temporal del espectrograma
        freqs        : eje de frecuencias
        spec_db      : espectrograma completo en dB (para visualización)
    """
    y, sr = load_audio(audio_path)

    if sr < freq_max * 2:
        raise ValueError(
            f"Tasa de muestreo {sr} Hz insuficiente para detectar USV hasta {freq_max} Hz. "
            f"Se necesita al menos {freq_max * 2} Hz."
        )

    n_fft = 2048
    hop_length = 512

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)

    spec_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Filtrar banda USV
    band_idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    if len(band_idx) == 0:
        return pd.DataFrame(), times, freqs, spec_db

    band_power = spec_db[band_idx, :]
    energy = band_power.max(axis=0)

    # Detectar eventos (franjas sobre umbral)
    active = energy > threshold_db
    events: list[dict] = []
    in_event = False
    t_start = 0.0
    peak_pow = -np.inf
    peak_freq_idx = band_idx[0]

    for i, (t, a) in enumerate(zip(times, active)):
        if a and not in_event:
            in_event = True
            t_start = t
            peak_pow = energy[i]
            peak_freq_idx = band_idx[band_power[:, i].argmax()]
        elif a and in_event:
            if energy[i] > peak_pow:
                peak_pow = energy[i]
                peak_freq_idx = band_idx[band_power[:, i].argmax()]
        elif not a and in_event:
            duration = t - t_start
            if duration >= min_duration_s:
                events.append({
                    "time_start":    t_start,
                    "time_end":      t,
                    "duration_ms":   round(duration * 1000, 1),
                    "freq_peak_hz":  float(freqs[peak_freq_idx]),
                    "power_db":      round(float(peak_pow), 1),
                })
            in_event = False

    if in_event:
        duration = times[-1] - t_start
        if duration >= min_duration_s:
            events.append({
                "time_start":   t_start,
                "time_end":     times[-1],
                "duration_ms":  round(duration * 1000, 1),
                "freq_peak_hz": float(freqs[peak_freq_idx]),
                "power_db":     round(float(peak_pow), 1),
            })

    return pd.DataFrame(events), times, freqs, spec_db
