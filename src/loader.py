"""
loader.py — Descarga y cacheo de modelos desde Hugging Face Hub
================================================================
Los modelos nunca se suben al repo de GitHub (demasiado pesados).
Se alojan en Hugging Face Hub y se descargan una sola vez al arrancar.

Modelos usados en el dashboard:
  - HistGB_R       (1.05 MB) → Regresión: TASA_MORTALIDAD_EVITABLE_100K
  - LightGBM_C     (1.65 MB) → Clasificación: FLG_ZONA_ALTO_RIESGO
  - meta_modelos_v3 (0.01 MB) → Features, umbrales, métricas

ExtraTrees_R (487 MB) se excluye del dashboard por tamaño.
HistGB_R ofrece R²=0.97, RMSE=18.1 — prácticamente idéntico.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from huggingface_hub import hf_hub_download

# ── Repositorio Hugging Face ──────────────────────────────────────────────────
# Cambiar HF_REPO_ID por tu repo una vez que subas los archivos.
# Formato: "tu-usuario-hf/salud-colombia-modelos"
HF_REPO_ID = os.getenv("HF_REPO_ID", "nicoacos/salud-colombia-modelos")

CACHE_DIR = Path(".cache/modelos")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── DropAllNanCols ────────────────────────────────────────────────────────────
# Transformer custom del notebook — se necesita para desempaquetar los .pkl
class DropAllNanCols(BaseEstimator, TransformerMixin):
    """
    Elimina columnas all-NaN ANTES de que SimpleImputer las vea.
    Fue parte del pipeline de entrenamiento (v3); se replica aquí
    para que joblib.load() pueda desempaquetar los modelos correctamente.
    """
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.cols_validas_ = [c for c in X.columns if not X[c].isna().all()]
            self.feature_names_in_ = list(X.columns)
        else:
            mask = ~np.all(np.isnan(X.astype(float)), axis=0)
            self.cols_validas_ = list(np.where(mask)[0])
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            cols_ok = [c for c in self.cols_validas_ if c in X.columns]
            return X[cols_ok]
        return X[:, self.cols_validas_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_validas_)


def _hf_download(filename: str) -> Path:
    """Descarga un archivo de HF Hub al cache local."""
    cached = CACHE_DIR / filename
    if cached.exists():
        return cached
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=str(CACHE_DIR),
    )
    return Path(path)


@st.cache_resource(show_spinner="Cargando modelo de regresión…")
def cargar_modelo_regresion():
    """Descarga y cachea el pipeline de regresión (ExtraTrees_R)."""
    path = _hf_download("modelo_ExtraTrees_R.pkl")
    return joblib.load(path)


@st.cache_resource(show_spinner="Cargando modelo de clasificación…")
def cargar_modelo_clasificacion():
    """Descarga y cachea el pipeline de clasificación (LightGBM_C)."""
    path = _hf_download("modelo_LightGBM_C.pkl")
    return joblib.load(path)


@st.cache_resource(show_spinner="Cargando metadatos…")
def cargar_meta() -> dict:
    """Descarga y cachea el diccionario de metadatos v3."""
    path = _hf_download("meta_modelos_v3.pkl")
    return joblib.load(path)


@st.cache_data(show_spinner="Cargando datos históricos…", ttl=3600)
def cargar_datos_historicos() -> pd.DataFrame:
    """
    Descarga el parquet de la tabla de hechos y lo filtra para el dashboard.
    Solo se leen las columnas necesarias para el mapa y las tablas.
    """
    path = _hf_download("tabla_hechos_salud_colombia.parquet")
    cols_necesarias = [
        "COD_MPIO", "NOM_MPIO", "COD_DPTO", "NOM_DPTO",
        "ANIO", "LATITUD", "LONGITUD",
        "TASA_MORTALIDAD_EVITABLE_100K", "FLG_ZONA_ALTO_RIESGO",
        "POBLACION_TOTAL", "MUERTES_ATRIBUIBLES", "TOTAL_FALLECIDOS",
        "TASA_MORTALIDAD_TOTAL_100K", "PCT_AFILIACION",
    ]
    df = pd.read_parquet(path, columns=cols_necesarias)
    df["ANIO"] = df["ANIO"].astype(int)
    return df


def cargar_todo() -> tuple:
    """
    Carga todos los artefactos en paralelo.
    Retorna (pipe_reg, pipe_clf, meta, df_historico).
    """
    pipe_r = cargar_modelo_regresion()
    pipe_c = cargar_modelo_clasificacion()
    meta   = cargar_meta()
    df     = cargar_datos_historicos()
    return pipe_r, pipe_c, meta, df
