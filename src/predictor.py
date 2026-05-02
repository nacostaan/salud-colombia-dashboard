"""
predictor.py — Lógica de predicción y construcción de features
==============================================================
Versión sincronizada con FEATURES_FINALES v3 del notebook ml_salud.ipynb.
34 features exactas en el orden que espera el pipeline entrenado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ── 34 features exactas del entrenamiento (orden crítico) ────────────────────
FEATURES_FINALES = [
    "TENDENCIA_MPIO", "ROLL3_MORT", "LAG1_MORT", "LAG2_MORT", "DELTA_YOY",
    "TASA_ATENCIONES_RIPS_100K", "COBERTURA_RIPS_POR_AFIL", "EFICIENCIA_GIRO",
    "VALOR_GIRADO_NORMAL", "PCT_AFILIACION", "RIPS_HOSPITALIZACIONES",
    "RATIO_VS_TEND", "RIPS_CONSULTAS", "TOTAL_ATENCIONES_RIPS",
    "RIPS_PROCEDIMIENTOS_DE_SALUD", "GIROS_ATIPICOS_PCT", "N_GIROS_ATIPICOS",
    "LATITUD", "LATITUD_ANIO", "RIPS_URGENCIAS", "VALOR_GIRADO_TOTAL",
    "GIRO_PROMEDIO_MES", "PCT_URGENCIAS_100K", "RATIO_URGENCIAS", "N_GIROS",
    "BDUA_MASCULINO", "PCT_SUBSIDIADO_BDUA", "BDUA_REG_SUBSIDIADO",
    "BDUA_TOTAL_AFILIADOS", "LONGITUD_ANIO", "BDUA_FEMENINO",
    "LONGITUD", "ANIO", "POBLACION_TOTAL",
]

# Etiquetas amigables y metadatos para los inputs de la UI
FEATURE_LABELS: dict[str, dict] = {
    "LAG1_MORT": {
        "label":   "Tasa mortalidad año anterior (por 100k hab)",
        "help":    "Tasa de mortalidad evitable del municipio en el año inmediatamente anterior",
        "min": 0.0, "max": 800.0, "default": 130.0, "step": 1.0,
    },
    "LAG2_MORT": {
        "label":   "Tasa mortalidad hace 2 años (por 100k hab)",
        "help":    "Tasa de mortalidad evitable del municipio hace dos años",
        "min": 0.0, "max": 800.0, "default": 135.0, "step": 1.0,
    },
    "ROLL3_MORT": {
        "label":   "Promedio móvil 3 años — mortalidad",
        "help":    "Media de la tasa de mortalidad de los últimos 3 años",
        "min": 0.0, "max": 800.0, "default": 132.0, "step": 1.0,
    },
    "TASA_ATENCIONES_RIPS_100K": {
        "label":   "Atenciones RIPS (por 100k hab)",
        "help":    "Total de atenciones en salud registradas por cada 100k habitantes",
        "min": 0.0, "max": 8_000_000.0, "default": 228_000.0, "step": 1_000.0,
    },
    "PCT_AFILIACION": {
        "label":   "% Afiliación al sistema de salud",
        "help":    "Porcentaje de la población afiliada a cualquier régimen",
        "min": 0.0, "max": 100.0, "default": 66.0, "step": 0.5,
    },
    "EFICIENCIA_GIRO": {
        "label":   "Eficiencia de giro SGP (índice)",
        "help":    "1.0 = media nacional. >1.0 = más eficiente que el promedio",
        "min": 0.0, "max": 5.0, "default": 1.0, "step": 0.05,
    },
    "POBLACION_TOTAL": {
        "label":   "Población total",
        "help":    "Proyección DANE de población total del municipio",
        "min": 500.0, "max": 10_000_000.0, "default": 12_915.0, "step": 500.0,
    },
    "BDUA_TOTAL_AFILIADOS": {
        "label":   "Afiliados BDUA",
        "help":    "Total de personas afiliadas según Base de Datos Única de Afiliados",
        "min": 0.0, "max": 5_000_000.0, "default": 8_382.0, "step": 100.0,
    },
    "VALOR_GIRADO_NORMAL": {
        "label":   "Valor girado SGP sin atípicos (COP)",
        "help":    "Recursos SGP girados excluyendo pagos atípicos",
        "min": 0.0, "max": 1_000_000_000_000.0, "default": 121_436_350.0, "step": 1_000_000.0,
    },
    "ANIO": {
        "label":   "Año de predicción",
        "help":    "Año para el que se desea predecir",
        "min": 2020, "max": 2027, "default": 2025, "step": 1,
    },
}


def _derivar_features(inputs: dict) -> dict:
    """
    Calcula las 24 features derivadas que el modelo espera
    pero el usuario no ingresa directamente en el formulario.
    Se calculan con heurísticas basadas en los valores ingresados.
    """
    anio       = float(inputs.get("ANIO", 2025))
    lat        = float(inputs.get("LATITUD", 4.5))
    lon        = float(inputs.get("LONGITUD", -74.0))
    lag1       = float(inputs.get("LAG1_MORT", 130.0))
    lag2       = float(inputs.get("LAG2_MORT", 135.0))
    roll3      = float(inputs.get("ROLL3_MORT", (lag1 + lag2) / 2))
    val_normal = float(inputs.get("VALOR_GIRADO_NORMAL", 121_436_350.0))
    pob        = max(float(inputs.get("POBLACION_TOTAL", 12_915.0)), 1.0)
    bdua       = float(inputs.get("BDUA_TOTAL_AFILIADOS", 8_382.0))
    tasa_rips  = float(inputs.get("TASA_ATENCIONES_RIPS_100K", 228_000.0))
    pct_afil   = float(inputs.get("PCT_AFILIACION", 66.0))
    n_giros    = float(inputs.get("N_GIROS", 12.0))
    efic_giro  = float(inputs.get("EFICIENCIA_GIRO", 1.0))

    # Temporal
    anio_norm      = (anio - 1979) / (2024 - 1979)

    # Lag features derivadas
    tendencia      = lag1 - lag2
    delta_yoy      = lag1 - roll3
    ratio_vs_tend  = lag1 / max(roll3, 0.001)

    # Giros
    val_total      = val_normal * 1.02
    giro_mes       = val_total / 12.0
    giros_atip_pct = 0.02
    n_giros_atip   = max(int(n_giros * giros_atip_pct), 0)

    # RIPS desagregado (proporciones promedio nacionales)
    total_rips  = tasa_rips * pob / 100_000.0
    rips_cons   = total_rips * 0.51
    rips_hosp   = total_rips * 0.006
    rips_proc   = total_rips * 0.47
    rips_urg    = total_rips * 0.016
    pct_urg_100k= (rips_urg / max(pob / 100_000.0, 0.001))
    ratio_urg   = tasa_rips * 0.016

    # BDUA desagregado
    bdua_subs   = bdua * (pct_afil / 100.0) * 0.85
    bdua_masc   = bdua * 0.48
    bdua_fem    = bdua * 0.52
    pct_subs    = (bdua_subs / max(pob, 1.0)) * 100.0

    # Cobertura RIPS vs BDUA
    cob_rips    = total_rips / max(bdua, 1.0)

    return {
        "TENDENCIA_MPIO"              : tendencia,
        "DELTA_YOY"                   : delta_yoy,
        "RATIO_VS_TEND"               : ratio_vs_tend,
        "COBERTURA_RIPS_POR_AFIL"     : cob_rips,
        "LATITUD_ANIO"                : lat * anio_norm,
        "LONGITUD_ANIO"               : lon * anio_norm,
        "VALOR_GIRADO_TOTAL"          : val_total,
        "GIRO_PROMEDIO_MES"           : giro_mes,
        "GIROS_ATIPICOS_PCT"          : giros_atip_pct,
        "N_GIROS_ATIPICOS"            : float(n_giros_atip),
        "RATIO_URGENCIAS"             : ratio_urg,
        "PCT_SUBSIDIADO_BDUA"         : pct_subs,
        "PCT_URGENCIAS_100K"          : pct_urg_100k,
        "RIPS_CONSULTAS"              : rips_cons,
        "RIPS_HOSPITALIZACIONES"      : rips_hosp,
        "RIPS_PROCEDIMIENTOS_DE_SALUD": rips_proc,
        "RIPS_URGENCIAS"              : rips_urg,
        "TOTAL_ATENCIONES_RIPS"       : total_rips,
        "BDUA_FEMENINO"               : bdua_fem,
        "BDUA_MASCULINO"              : bdua_masc,
        "BDUA_REG_SUBSIDIADO"         : bdua_subs,
    }


def construir_fila_prediccion(inputs: dict) -> pd.DataFrame:
    """
    Combina inputs del usuario + features derivadas automáticamente,
    en el orden exacto que el pipeline de sklearn espera (FEATURES_FINALES).
    """
    fila = dict(inputs)
    fila.update(_derivar_features(inputs))
    row = {f: fila.get(f, np.nan) for f in FEATURES_FINALES}
    return pd.DataFrame([row])


def predecir_regresion(pipe, inputs: dict) -> dict:
    """
    Predice TASA_MORTALIDAD_EVITABLE_100K con ExtraTrees_R.
    Retorna tasa, nivel de riesgo, color y emoji para la UI.
    """
    X    = construir_fila_prediccion(inputs)
    tasa = float(pipe.predict(X)[0])
    tasa = max(tasa, 0.0)

    if tasa < 75:
        nivel, color, emoji = "Bajo",     "#2ECC71", "🟢"
    elif tasa < 150:
        nivel, color, emoji = "Moderado", "#F39C12", "🟡"
    elif tasa < 250:
        nivel, color, emoji = "Alto",     "#E67E22", "🟠"
    else:
        nivel, color, emoji = "Crítico",  "#E63946", "🔴"

    return {"tasa": round(tasa, 1), "nivel": nivel,
            "color": color, "emoji": emoji}


def predecir_clasificacion(pipe, inputs: dict,
                            umbral: float = 0.65) -> dict:
    """
    Predice FLG_ZONA_ALTO_RIESGO con LightGBM_C.
    Retorna probabilidad, clasificación binaria y semáforo.
    Umbral óptimo pre-calculado en entrenamiento: 0.65.
    """
    X     = construir_fila_prediccion(inputs)
    proba = float(pipe.predict_proba(X)[0][1])
    alto  = proba >= umbral

    if proba < 0.30:
        nivel, color, emoji = "Sin riesgo",     "#2ECC71", "🟢"
    elif proba < 0.50:
        nivel, color, emoji = "Riesgo bajo",    "#27AE60", "🟢"
    elif proba < umbral:
        nivel, color, emoji = "Riesgo moderado","#F39C12", "🟡"
    elif proba < 0.85:
        nivel, color, emoji = "Alto riesgo",    "#E67E22", "🟠"
    else:
        nivel, color, emoji = "Riesgo crítico", "#E63946", "🔴"

    return {
        "proba":  round(proba * 100, 1),
        "alto":   alto,
        "nivel":  nivel,
        "color":  color,
        "emoji":  emoji,
        "umbral": umbral,
    }


def agregar_por_departamento(df: pd.DataFrame,
                              anio: Optional[int] = None) -> pd.DataFrame:
    """Agrega la tabla de hechos por departamento para el mapa nacional."""
    df_f = df.dropna(subset=["TASA_MORTALIDAD_EVITABLE_100K"])
    if anio is not None:
        df_f = df_f[df_f["ANIO"] == anio]
    else:
        df_f = df_f[df_f["ANIO"] == int(df_f["ANIO"].max())]

    agg = (
        df_f.groupby(["COD_DPTO", "NOM_DPTO"], as_index=False)
        .agg(
            TASA_MEDIA             = ("TASA_MORTALIDAD_EVITABLE_100K", "mean"),
            MUERTES_TOT            = ("MUERTES_ATRIBUIBLES", "sum"),
            N_MUNICIPIOS           = ("COD_MPIO", "nunique"),
            MUNICIPIOS_ALTO_RIESGO = ("FLG_ZONA_ALTO_RIESGO", "sum"),
            LATITUD                = ("LATITUD",  "mean"),
            LONGITUD               = ("LONGITUD", "mean"),
        )
        .round(2)
    )
    p25 = agg["TASA_MEDIA"].quantile(0.25)
    p50 = agg["TASA_MEDIA"].quantile(0.50)
    p75 = agg["TASA_MEDIA"].quantile(0.75)

    def semaforo(t):
        if t < p25: return "🟢 Bajo"
        if t < p50: return "🟡 Moderado"
        if t < p75: return "🟠 Alto"
        return "🔴 Crítico"

    agg["SEMAFORO"] = agg["TASA_MEDIA"].apply(semaforo)
    return agg.sort_values("TASA_MEDIA", ascending=False)


def top_municipios_criticos(df: pd.DataFrame, n: int = 20,
                             anio: Optional[int] = None) -> pd.DataFrame:
    """Top n municipios por tasa de mortalidad evitable."""
    df_f = df.dropna(subset=["TASA_MORTALIDAD_EVITABLE_100K"])
    if anio:
        df_f = df_f[df_f["ANIO"] == anio]
    else:
        df_f = df_f[df_f["ANIO"] == int(df_f["ANIO"].max())]

    return (
        df_f[["COD_MPIO", "NOM_MPIO", "NOM_DPTO",
              "TASA_MORTALIDAD_EVITABLE_100K", "MUERTES_ATRIBUIBLES",
              "POBLACION_TOTAL", "FLG_ZONA_ALTO_RIESGO"]]
        .sort_values("TASA_MORTALIDAD_EVITABLE_100K", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
