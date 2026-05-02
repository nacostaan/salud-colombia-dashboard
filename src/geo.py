"""
geo.py — Utilidades geográficas para el mapa coroplético de Colombia
=====================================================================
Usa Plotly scatter_geo con coordenadas reales del dataset (LATITUD/LONGITUD).
No requiere GeoJSON externo, lo que elimina una dependencia de red y
garantiza que el mapa funcione sin archivos adicionales en el repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Paleta de riesgo (alineada con config.toml y charts.py) ─────────────────
BG       = "#0D1117"
BG2      = "#161B22"
GRID     = "#21262D"
TEXT     = "#F0F6FC"
TEXT_DIM = "#8B949E"
RED      = "#E63946"
ORANGE   = "#E67E22"
YELLOW   = "#F39C12"
GREEN    = "#2ECC71"


def _color_semaforo(tasa: float, p25: float, p50: float, p75: float) -> str:
    if tasa < p25:   return GREEN
    if tasa < p50:   return YELLOW
    if tasa < p75:   return ORANGE
    return RED


def mapa_nacional_municipios(df: pd.DataFrame, anio: int) -> go.Figure:
    """
    Mapa scatter_geo a nivel municipal centrado en Colombia.

    Cada punto representa un municipio.
    - Color: semáforo de riesgo (verde → rojo)
    - Tamaño: proporcional a la tasa de mortalidad evitable
    - Tooltip: municipio, departamento, tasa, semáforo

    Args:
        df   : DataFrame con columnas COD_MPIO, NOM_MPIO, NOM_DPTO,
               LATITUD, LONGITUD, TASA_MORTALIDAD_EVITABLE_100K, ANIO
        anio : año a visualizar
    """
    df_a = (
        df[df["ANIO"] == anio]
        .dropna(subset=["TASA_MORTALIDAD_EVITABLE_100K", "LATITUD", "LONGITUD"])
        .copy()
    )

    if df_a.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=BG,
            title=dict(text=f"Sin datos disponibles para {anio}", font_color=TEXT),
        )
        return fig

    p25 = df_a["TASA_MORTALIDAD_EVITABLE_100K"].quantile(0.25)
    p50 = df_a["TASA_MORTALIDAD_EVITABLE_100K"].quantile(0.50)
    p75 = df_a["TASA_MORTALIDAD_EVITABLE_100K"].quantile(0.75)

    df_a["COLOR"] = df_a["TASA_MORTALIDAD_EVITABLE_100K"].apply(
        lambda t: _color_semaforo(t, p25, p50, p75)
    )
    df_a["SEMAFORO"] = df_a["TASA_MORTALIDAD_EVITABLE_100K"].apply(
        lambda t: "🟢 Bajo" if t < p25 else
                  ("🟡 Moderado" if t < p50 else
                   ("🟠 Alto" if t < p75 else "🔴 Crítico"))
    )
    # Tamaño: normalizar a [6, 22] para legibilidad
    tasa_norm = df_a["TASA_MORTALIDAD_EVITABLE_100K"]
    t_min, t_max = tasa_norm.min(), tasa_norm.max()
    if t_max > t_min:
        df_a["SIZE"] = 6 + 16 * (tasa_norm - t_min) / (t_max - t_min)
    else:
        df_a["SIZE"] = 10.0

    df_a["HOVER"] = (
        "<b>" + df_a["NOM_MPIO"] + "</b> · " + df_a["NOM_DPTO"] + "<br>"
        + "Tasa: " + df_a["TASA_MORTALIDAD_EVITABLE_100K"].round(1).astype(str)
        + " / 100k hab<br>"
        + "Zona: " + df_a["SEMAFORO"]
    )

    # Traza por nivel de riesgo (4 traces → leyenda correcta)
    niveles = [
        ("🔴 Crítico",  RED,    df_a[df_a["SEMAFORO"] == "🔴 Crítico"]),
        ("🟠 Alto",     ORANGE, df_a[df_a["SEMAFORO"] == "🟠 Alto"]),
        ("🟡 Moderado", YELLOW, df_a[df_a["SEMAFORO"] == "🟡 Moderado"]),
        ("🟢 Bajo",     GREEN,  df_a[df_a["SEMAFORO"] == "🟢 Bajo"]),
    ]

    traces = []
    for nombre, color, sub in niveles:
        if sub.empty:
            continue
        traces.append(go.Scattergeo(
            lat=sub["LATITUD"],
            lon=sub["LONGITUD"],
            text=sub["HOVER"],
            hoverinfo="text",
            mode="markers",
            name=nombre,
            marker=dict(
                size=sub["SIZE"],
                color=color,
                opacity=0.82,
                line=dict(width=0.4, color=BG),
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_geos(
        center=dict(lat=4.5, lon=-74.0),
        projection_type="mercator",
        projection_scale=8,
        showland=True, landcolor="#1C2128",
        showocean=True, oceancolor=BG,
        showlakes=True, lakecolor=BG,
        showrivers=True, rivercolor="#21262D",
        showcoastlines=True, coastlinecolor=GRID,
        showcountries=True, countrycolor=GRID,
        showframe=False,
        bgcolor=BG,
    )
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        geo_bgcolor=BG,
        font=dict(color=TEXT, family="sans-serif"),
        title=dict(
            text=f"Colombia — Mortalidad Evitable por Municipio · {anio}",
            font=dict(color=TEXT, size=15),
            x=0.02,
        ),
        legend=dict(
            title=dict(text="Zona de riesgo", font_color=TEXT_DIM),
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor=GRID,
            font_color=TEXT,
            x=0.01, y=0.01,
            orientation="v",
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        height=560,
    )
    return fig


def mapa_departamento(df: pd.DataFrame, cod_dpto: str,
                       anio: int) -> go.Figure:
    """
    Mapa scatter_geo con zoom en un departamento específico.

    Muestra todos los municipios de ese departamento para el año dado.
    Útil en la vista de predicción para mostrar contexto geográfico.
    """
    df_d = (
        df[(df["COD_DPTO"] == cod_dpto) & (df["ANIO"] == anio)]
        .dropna(subset=["LATITUD", "LONGITUD"])
        .copy()
    )

    if df_d.empty:
        # Intentar sin filtro de año
        df_d = df[df["COD_DPTO"] == cod_dpto].dropna(subset=["LATITUD", "LONGITUD"])
        if df_d.empty:
            fig = go.Figure()
            fig.update_layout(paper_bgcolor=BG,
                              title=dict(text="Sin datos geográficos", font_color=TEXT))
            return fig

    lat_c = df_d["LATITUD"].mean()
    lon_c = df_d["LONGITUD"].mean()

    tiene_tasa = df_d["TASA_MORTALIDAD_EVITABLE_100K"].notna().any()

    if tiene_tasa:
        p25 = df_d["TASA_MORTALIDAD_EVITABLE_100K"].quantile(0.25)
        p50 = df_d["TASA_MORTALIDAD_EVITABLE_100K"].quantile(0.50)
        p75 = df_d["TASA_MORTALIDAD_EVITABLE_100K"].quantile(0.75)
        df_d["COLOR"] = df_d["TASA_MORTALIDAD_EVITABLE_100K"].apply(
            lambda t: _color_semaforo(t, p25, p50, p75) if pd.notna(t) else TEXT_DIM
        )
        df_d["HOVER"] = (
            "<b>" + df_d["NOM_MPIO"] + "</b><br>"
            + "Tasa: " + df_d["TASA_MORTALIDAD_EVITABLE_100K"]
                              .fillna(-1).round(1).astype(str).replace("-1.0", "S/D")
            + " / 100k<br>"
        )
        size_col = df_d["TASA_MORTALIDAD_EVITABLE_100K"].fillna(
            df_d["TASA_MORTALIDAD_EVITABLE_100K"].median()
        )
        t_min, t_max = size_col.min(), size_col.max()
        df_d["SIZE"] = 8 + 14 * (size_col - t_min) / max(t_max - t_min, 1)
    else:
        df_d["COLOR"] = YELLOW
        df_d["HOVER"] = "<b>" + df_d["NOM_MPIO"] + "</b><br>Sin datos de tasa"
        df_d["SIZE"]  = 10.0

    fig = go.Figure(go.Scattergeo(
        lat=df_d["LATITUD"],
        lon=df_d["LONGITUD"],
        text=df_d["NOM_MPIO"],            # ← label visible en el mapa
        hovertext=df_d["HOVER"],          # ← texto del tooltip
        hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=8, color=TEXT_DIM),
        marker=dict(
            size=df_d["SIZE"],
            color=df_d["COLOR"],
            opacity=0.88,
            line=dict(width=0.6, color=BG),
        ),
        showlegend=False,
    ))
    fig.update_geos(
        center=dict(lat=lat_c, lon=lon_c),
        projection_type="mercator",
        projection_scale=9,
        showland=True, landcolor="#1C2128",
        showocean=True, oceancolor=BG,
        showcoastlines=True, coastlinecolor=GRID,
        showcountries=True, countrycolor=GRID,
        showframe=False,
        bgcolor=BG,
    )
    nom_dpto = df_d["NOM_DPTO"].iloc[0] if "NOM_DPTO" in df_d.columns else cod_dpto
    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(color=TEXT),
        title=dict(
            text=f"{nom_dpto} — Municipios · {anio}",
            font=dict(color=TEXT, size=13),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=340,
    )
    return fig


def kpis_nacionales(df: pd.DataFrame, anio: int) -> dict:
    """
    Calcula los KPIs del encabezado del dashboard para un año dado.

    Retorna:
        dict con claves: tasa_nacional, municipios_criticos, pct_criticos,
        muertes_atribuibles, delta_vs_anterior
    """
    df_a = df[df["ANIO"] == anio].dropna(subset=["TASA_MORTALIDAD_EVITABLE_100K"])
    df_prev = df[df["ANIO"] == anio - 1].dropna(subset=["TASA_MORTALIDAD_EVITABLE_100K"])

    tasa_actual = df_a["TASA_MORTALIDAD_EVITABLE_100K"].mean()
    tasa_prev   = df_prev["TASA_MORTALIDAD_EVITABLE_100K"].mean() if not df_prev.empty else None

    n_criticos  = int((df_a["FLG_ZONA_ALTO_RIESGO"] == 1).sum())
    n_total     = len(df_a)
    pct_crit    = n_criticos / max(n_total, 1) * 100

    muertes     = df_a["MUERTES_ATRIBUIBLES"].sum() if "MUERTES_ATRIBUIBLES" in df_a.columns else None

    delta = None
    if tasa_prev is not None and not np.isnan(tasa_prev):
        delta = tasa_actual - tasa_prev

    return {
        "anio"                : anio,
        "tasa_nacional"       : round(tasa_actual, 1),
        "municipios_criticos" : n_criticos,
        "total_municipios"    : n_total,
        "pct_criticos"        : round(pct_crit, 1),
        "muertes_atribuibles" : int(muertes) if muertes is not None else None,
        "delta_vs_anterior"   : round(delta, 1) if delta is not None else None,
    }


def lista_municipios_por_dpto(df: pd.DataFrame, cod_dpto: str) -> list[dict]:
    """
    Retorna lista de municipios de un departamento con su tasa más reciente.
    Usado para poblar el selector de municipio en la vista de predicción.
    """
    sub = (
        df[df["COD_DPTO"] == cod_dpto]
        .dropna(subset=["NOM_MPIO"])
        [["COD_MPIO", "NOM_MPIO", "LATITUD", "LONGITUD",
          "TASA_MORTALIDAD_EVITABLE_100K", "ANIO", "POBLACION_TOTAL"]]
        .sort_values("ANIO", ascending=False)
        .drop_duplicates(subset=["COD_MPIO"])
        .sort_values("NOM_MPIO")
    )
    return sub.to_dict("records")
