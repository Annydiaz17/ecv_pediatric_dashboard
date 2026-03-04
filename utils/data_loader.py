"""
Cargador de datos del dashboard pediátrico.
Carga SOLO el dataset Timbiquí (5,000 muestras) del notebook SEMMA.
Aplica limpieza replicando el notebook SEMMA.
"""
import os
import numpy as np
import pandas as pd

_data_cache = {}
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Ruta ─────────────────────────────────────────────────────────────────────
_PATH_TIMBIQUI = os.path.join(_BASE_DIR, "data", "dataset_timbiqui.csv")

# ── Columnas unificadas ─────────────────────────────────────────────────────
FEATURES_MODELO = [
    "edad", "genero", "peso_kg", "pa_sistolica",
    "frecuencia_cardiaca", "colesterol_mgdl"
]
TARGET = "riesgo_cv"
ALL_COLS = FEATURES_MODELO + [TARGET]


# ── Limpieza del Timbiquí (replica SEMMA notebook) ──────────────────────────

def _limpiar_genero(val):
    """Estandariza la columna genero a 0 (femenino) y 1 (masculino)."""
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in ("masculino", "m", "masc", "male", "h", "hombre"):
        return 1
    if v in ("femenino", "f", "fem", "female", "mujer"):
        return 0
    return np.nan


def _limpiar_pa_sistolica(val):
    """Convierte PA_Sistolica de string con coma a float."""
    if pd.isna(val):
        return np.nan
    v = str(val).replace(",", ".").strip()
    try:
        return float(v)
    except ValueError:
        return np.nan


def _cargar_timbiqui():
    """Carga y limpia el dataset Timbiquí original (5,000 muestras)."""
    if not os.path.exists(_PATH_TIMBIQUI):
        raise FileNotFoundError(
            f"No se encuentra: dataset_timbiqui.csv en {_PATH_TIMBIQUI}"
        )

    df = pd.read_csv(_PATH_TIMBIQUI)

    # Renombrar columnas (nombres originales → nombres estándar)
    rename_map = {
        "EDAD": "edad",
        " genero": "genero",       # tiene espacio al inicio
        "Peso_kg": "peso_kg",
        "PA_Sistolica": "pa_sistolica",
        "frecuencia_Cardiaca": "frecuencia_cardiaca",
        "Colesterol_mgdl": "colesterol_mgdl",
        "RIESGO": "riesgo_cv",
    }
    # Limpiar espacios en nombres de columnas primero
    df.columns = [c.strip() if c.strip() in [v for v in rename_map.keys() if v.strip() == v]
                  else c for c in df.columns]
    df = df.rename(columns=rename_map)

    # Limpiar genero
    df["genero"] = df["genero"].apply(_limpiar_genero)

    # Limpiar pa_sistolica (string con comas → float)
    df["pa_sistolica"] = df["pa_sistolica"].apply(_limpiar_pa_sistolica)

    # Imputar nulos con mediana (como en el notebook SEMMA)
    for col in ["peso_kg", "colesterol_mgdl", "pa_sistolica", "genero"]:
        if col in df.columns and df[col].isnull().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)

    return df[FEATURES_MODELO + [TARGET]].copy()


# ── API Pública ──────────────────────────────────────────────────────────────

def get_data(source="combinado"):
    """
    Carga y cachea el dataset Timbiquí.

    Args:
        source: cualquier valor retorna el dataset Timbiquí.
                Se mantienen "timbiqui" y "combinado" por compatibilidad.
    """
    # Todo apunta al mismo dataset Timbiquí
    cache_key = "timbiqui"
    if cache_key in _data_cache:
        return _data_cache[cache_key].copy()

    df = _cargar_timbiqui()
    _data_cache[cache_key] = df
    return df.copy()


def get_combined_stats():
    """Estadísticas del dataset Timbiquí."""
    df = get_data()

    return {
        "n_total": len(df),
        "prevalencia": float(df[TARGET].mean()),
    }


def filter_data(df, genero=None, edad_min=None, edad_max=None,
                col_min=None, col_max=None,
                pas_min=None, pas_max=None,
                peso_min=None, peso_max=None,
                fc_min=None, fc_max=None,
                **kwargs):
    """Aplica filtros al dataset. No modifica datos, solo filtra filas."""
    filt = df.copy()

    if genero is not None and genero != "todos":
        if str(genero) in ("1", "Masculino"):
            filt = filt[filt["genero"] == 1]
        elif str(genero) in ("0", "Femenino"):
            filt = filt[filt["genero"] == 0]

    if edad_min is not None:
        filt = filt[filt["edad"] >= edad_min]
    if edad_max is not None:
        filt = filt[filt["edad"] <= edad_max]

    if col_min is not None:
        filt = filt[filt["colesterol_mgdl"] >= col_min]
    if col_max is not None:
        filt = filt[filt["colesterol_mgdl"] <= col_max]

    if pas_min is not None:
        filt = filt[filt["pa_sistolica"] >= pas_min]
    if pas_max is not None:
        filt = filt[filt["pa_sistolica"] <= pas_max]

    if peso_min is not None:
        filt = filt[filt["peso_kg"] >= peso_min]
    if peso_max is not None:
        filt = filt[filt["peso_kg"] <= peso_max]

    if fc_min is not None:
        filt = filt[filt["frecuencia_cardiaca"] >= fc_min]
    if fc_max is not None:
        filt = filt[filt["frecuencia_cardiaca"] <= fc_max]

    return filt
