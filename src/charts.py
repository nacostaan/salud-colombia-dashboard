"""
charts.py — Todas las gráficas Plotly del dashboard
====================================================
Cada función retorna un go.Figure listo para st.plotly_chart().
Paleta y estilos alineados con config.toml (tema oscuro, rojo #E63946).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Constantes de estilo ──────────────────────────────────────────────────────
BG       = "#0D1117"
BG2      = "#161B22"
GRID     = "#21262D"
TEXT     = "#F0F6FC"
TEXT_DIM = "#8B949E"
RED      = "#E63946"
GREEN    = "#2ECC71"
YELLOW   = "#F39C12"
ORANGE   = "#E67E22"
BLUE     = "#58A6FF"

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG2,
    font=dict(family="'DM Sans', 'IBM Plex Sans', sans-serif", color=TEXT, size=12),
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID, font_color=TEXT_DIM),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont_color=TEXT_DIM),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont_color=TEXT_DIM),
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font_color=TEXT, font_size=15))
    return fig


# ── 1. Mapa de burbujas Colombia ──────────────────────────────────────────────
def mapa_burbujas_colombia(df_dpto: pd.DataFrame) -> go.Figure:
    """
    Mapa scatter_geo centrado en Colombia mostrando una burbuja por
    departamento. El tamaño codifica la tasa media de mortalidad;
    el color va de verde (bajo) a rojo (crítico).
    """
    df = df_dpto.copy()
    df["TASA_NORM"] = df["TASA_MEDIA"].clip(upper=df["TASA_MEDIA"].quantile(0.97))
    df["hover"] = (
        "<b>" + df["NOM_DPTO"] + "</b><br>"
        "Tasa media: " + df["TASA_MEDIA"].round(1).astype(str) + "/100k<br>"
        "Municipios: " + df["N_MUNICIPIOS"].astype(str) + "<br>"
        "Semáforo: " + df["SEMAFORO"]
    )

    fig = go.Figure(go.Scattergeo(
        lat=df["LATITUD"],
        lon=df["LONGITUD"],
        text=df["hover"],
        hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=9, color=TEXT_DIM),
        marker=dict(
            size=df["TASA_NORM"].clip(lower=5) / 8,
            color=df["TASA_MEDIA"],
            colorscale=[[0, GREEN], [0.33, YELLOW], [0.66, ORANGE], [1, RED]],
            showscale=True,
            colorbar=dict(
                title="Tasa<br>/100k",
                tickfont_color=TEXT_DIM,
                bgcolor=BG2,
                bordercolor=GRID,
            ),
            line=dict(width=1, color=BG),
            opacity=0.85,
        ),
    ))

    fig.update_geos(
        center=dict(lat=4.5, lon=-74.0),
        projection_scale=12,
        showland=True, landcolor="#1C2128",
        showocean=True, oceancolor=BG,
        showcoastlines=True, coastlinecolor=GRID,
        showcountries=True, countrycolor=GRID,
        showframe=False,
        bgcolor=BG,
    )
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        title=dict(text="Colombia — Tasa de Mortalidad Evitable por Departamento",
                   font_color=TEXT, font_size=15),
        geo=dict(bgcolor=BG),
        height=520,
    )
    return fig


# ── 2. Semáforo por departamento (barras horizontales) ───────────────────────
def barras_departamentos(df_dpto: pd.DataFrame, top_n: int = 20) -> go.Figure:
    df = df_dpto.head(top_n).sort_values("TASA_MEDIA")

    colors = []
    for s in df["SEMAFORO"]:
        if "Crítico" in s:  colors.append(RED)
        elif "Alto" in s:   colors.append(ORANGE)
        elif "Mod" in s:    colors.append(YELLOW)
        else:               colors.append(GREEN)

    fig = go.Figure(go.Bar(
        y=df["NOM_DPTO"],
        x=df["TASA_MEDIA"],
        orientation="h",
        marker_color=colors,
        text=df["TASA_MEDIA"].round(1).astype(str) + "/100k",
        textposition="outside",
        textfont_color=TEXT_DIM,
        hovertemplate="<b>%{y}</b><br>Tasa: %{x:.1f}/100k<extra></extra>",
    ))
    _apply_base(fig, f"Top {top_n} Departamentos — Tasa Mortalidad Evitable")
    fig.update_layout(height=max(350, top_n * 22), yaxis_tickfont_size=11)
    fig.update_xaxes(title="Muertes / 100k hab")
    return fig


# ── 3. Gauge de predicción de tasa ───────────────────────────────────────────
def gauge_tasa(tasa: float, nivel: str, color: str) -> go.Figure:
    """Velocímetro circular para la tasa de mortalidad predicha."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=tasa,
        number=dict(suffix=" / 100k", font_size=32, font_color=TEXT),
        delta=dict(reference=130, suffix=" vs media", increasing_color=RED,
                   decreasing_color=GREEN),
        gauge=dict(
            axis=dict(range=[0, 500], tickfont_color=TEXT_DIM,
                      tickcolor=GRID, tickwidth=1),
            bar=dict(color=color, thickness=0.22),
            bgcolor=BG2,
            borderwidth=1, bordercolor=GRID,
            steps=[
                dict(range=[0, 75],    color="#1C2128"),
                dict(range=[75, 150],  color="#1C2128"),
                dict(range=[150, 250], color="#1C2128"),
                dict(range=[250, 500], color="#1C2128"),
            ],
            threshold=dict(line=dict(color=RED, width=2), thickness=0.75, value=130),
        ),
        title=dict(text=f"Tasa predicha · <b>{nivel}</b>",
                   font_size=14, font_color=TEXT_DIM),
    ))
    fig.update_layout(**{k: v for k, v in LAYOUT_BASE.items() if "axis" not in k},
                      height=280, margin=dict(l=24, r=24, t=60, b=8))
    return fig


# ── 4. Gauge de probabilidad de alto riesgo ──────────────────────────────────
def gauge_riesgo(proba_pct: float, nivel: str, color: str,
                 umbral_pct: float = 65.0) -> go.Figure:
    """Velocímetro para la probabilidad de zona de alto riesgo."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba_pct,
        number=dict(suffix="%", font_size=32, font_color=TEXT),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont_color=TEXT_DIM,
                      tickcolor=GRID, tickwidth=1),
            bar=dict(color=color, thickness=0.22),
            bgcolor=BG2,
            borderwidth=1, bordercolor=GRID,
            steps=[
                dict(range=[0, 30],          color="#1C2128"),
                dict(range=[30, umbral_pct], color="#1C2128"),
                dict(range=[umbral_pct, 100],color="#29111355"),
            ],
            threshold=dict(line=dict(color=YELLOW, width=2),
                           thickness=0.75, value=umbral_pct),
        ),
        title=dict(text=f"Prob. zona alto riesgo · <b>{nivel}</b>",
                   font_size=14, font_color=TEXT_DIM),
    ))
    fig.update_layout(**{k: v for k, v in LAYOUT_BASE.items() if "axis" not in k},
                      height=280, margin=dict(l=24, r=24, t=60, b=8))
    return fig


# ── 5. Tabla top municipios críticos ─────────────────────────────────────────
def tabla_top_municipios(df_top: pd.DataFrame) -> go.Figure:
    df = df_top.copy()
    df["RIESGO"] = df["FLG_ZONA_ALTO_RIESGO"].apply(
        lambda x: "🔴 Alto" if x == 1 else ("🟢 Normal" if x == 0 else "—"))
    df["TASA"] = df["TASA_MORTALIDAD_EVITABLE_100K"].round(1)
    df["POB"]  = df["POBLACION_TOTAL"].fillna(0).astype(int).apply(lambda x: f"{x:,}")
    df["MUERTES"] = df["MUERTES_ATRIBUIBLES"].fillna(0).astype(int)
    df.index = range(1, len(df)+1)

    cols_show  = ["NOM_MPIO", "NOM_DPTO", "TASA", "MUERTES", "POB", "RIESGO"]
    col_labels = ["Municipio", "Departamento", "Tasa/100k", "Muertes Attr.", "Población", "Zona"]

    fill_colors = [[BG2 if i % 2 == 0 else "#111318" for i in range(len(df))]] * len(cols_show)

    fig = go.Figure(go.Table(
        columnwidth=[2, 2, 1, 1, 1.2, 1],
        header=dict(
            values=[f"<b>{c}</b>" for c in col_labels],
            fill_color=GRID,
            font_color=TEXT, font_size=12,
            align="left", height=32,
            line_color=BG,
        ),
        cells=dict(
            values=[df[c] for c in cols_show],
            fill_color=fill_colors,
            font_color=TEXT, font_size=11,
            align=["left","left","right","right","right","center"],
            height=28,
            line_color=BG,
        ),
    ))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if "axis" not in k},
        height=min(60 + len(df) * 30, 700),
        title=dict(text="Top Municipios — Mayor Tasa de Mortalidad Evitable",
                   font_color=TEXT, font_size=15),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    return fig


# ── 6. Comparativa de modelos ─────────────────────────────────────────────────
def comparativa_modelos_reg(metricas_reg: dict) -> go.Figure:
    """Barras de RMSE y R² para todos los modelos de regresión."""
    rmse  = metricas_reg.get("RMSE", {})
    r2    = metricas_reg.get("R2",   {})
    modelos = [m for m in rmse if not any(x in m for x in ["Ridge","ElasticNet","Huber","Bayesian","SVR"])]

    df_m = pd.DataFrame({
        "modelo": modelos,
        "RMSE":   [rmse[m] for m in modelos],
        "R2":     [r2[m]   for m in modelos],
    }).sort_values("RMSE")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["RMSE (menor = mejor)", "R² (mayor = mejor)"])

    color_rmse = [GREEN if v <= 15 else (YELLOW if v <= 25 else ORANGE) for v in df_m["RMSE"]]
    color_r2   = [GREEN if v >= 0.97 else (YELLOW if v >= 0.90 else ORANGE) for v in df_m["R2"]]

    fig.add_trace(go.Bar(
        y=df_m["modelo"], x=df_m["RMSE"],
        orientation="h", marker_color=color_rmse,
        text=df_m["RMSE"].round(1), textposition="outside",
        textfont_color=TEXT_DIM,
        hovertemplate="<b>%{y}</b><br>RMSE: %{x:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=df_m["modelo"], x=df_m["R2"],
        orientation="h", marker_color=color_r2,
        text=df_m["R2"].round(3), textposition="outside",
        textfont_color=TEXT_DIM,
        hovertemplate="<b>%{y}</b><br>R²: %{x:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if "axis" not in k},
        height=max(300, len(df_m) * 30 + 100),
        title=dict(text="Comparativa modelos de regresión — Test out-of-time",
                   font_color=TEXT, font_size=15),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont_color=TEXT_DIM)
    fig.update_yaxes(gridcolor=GRID, tickfont_color=TEXT_DIM)
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_DIM
    return fig


def comparativa_modelos_clf(metricas_clf: dict) -> go.Figure:
    """Barras de AUC, Recall y F1 para todos los modelos de clasificación."""
    auc    = metricas_clf.get("AUC",    {})
    recall = metricas_clf.get("Recall", {})
    f1     = metricas_clf.get("F1",     {})
    modelos = [m for m in auc if m != "LogisticReg"]

    df_m = pd.DataFrame({
        "modelo": modelos,
        "AUC":    [auc[m]    for m in modelos],
        "Recall": [recall[m] for m in modelos],
        "F1":     [f1[m]     for m in modelos],
    }).sort_values("AUC", ascending=False)

    fig = go.Figure()
    for col, col_name, clr in [("AUC", "AUC", BLUE), ("Recall", "Recall", GREEN), ("F1", "F1", YELLOW)]:
        fig.add_trace(go.Bar(
            name=col_name,
            y=df_m["modelo"],
            x=df_m[col],
            orientation="h",
            marker_color=clr,
            opacity=0.85,
            text=df_m[col].round(3),
            textposition="outside",
            textfont_color=TEXT_DIM,
            hovertemplate=f"<b>%{{y}}</b><br>{col_name}: %{{x:.3f}}<extra></extra>",
        ))

    # Líneas de umbral
    for thr, clr, label in [(0.90, BLUE, "AUC≥0.90"), (0.88, GREEN, "Recall≥0.88"), (0.85, YELLOW, "F1≥0.85")]:
        fig.add_vline(x=thr, line_dash="dot", line_color=clr, opacity=0.5,
                      annotation_text=label, annotation_font_color=clr,
                      annotation_position="top right", annotation_font_size=9)

    _apply_base(fig, "Comparativa modelos de clasificación — Test out-of-time")
    fig.update_layout(
        barmode="group",
        height=max(320, len(df_m) * 40 + 100),
        xaxis_range=[0, 1.05],
    )
    return fig


# ── 7. Evolución temporal de un municipio ────────────────────────────────────
def evolucion_municipio(df_hist: pd.DataFrame, cod_mpio: str,
                         nom_mpio: str) -> go.Figure:
    """Línea temporal de mortalidad para un municipio específico."""
    df_m = (
        df_hist[df_hist["COD_MPIO"] == cod_mpio]
        .dropna(subset=["TASA_MORTALIDAD_EVITABLE_100K"])
        .sort_values("ANIO")
    )
    if df_m.empty:
        fig = go.Figure()
        _apply_base(fig, f"{nom_mpio} — sin datos históricos disponibles")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_m["ANIO"], y=df_m["TASA_MORTALIDAD_EVITABLE_100K"],
        mode="lines+markers",
        line=dict(color=RED, width=2.5),
        marker=dict(size=6, color=RED, line=dict(color=BG, width=1)),
        name="Tasa real",
        hovertemplate="<b>%{x}</b>: %{y:.1f}/100k<extra></extra>",
    ))

    # Promedio móvil 3 años
    if len(df_m) >= 3:
        roll = df_m["TASA_MORTALIDAD_EVITABLE_100K"].rolling(3, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df_m["ANIO"], y=roll,
            mode="lines",
            line=dict(color=YELLOW, width=1.5, dash="dot"),
            name="Promedio 3 años",
            hovertemplate="Prom. 3a: %{y:.1f}/100k<extra></extra>",
        ))

    _apply_base(fig, f"{nom_mpio} — Evolución histórica mortalidad evitable")
    fig.update_xaxes(title="Año")
    fig.update_yaxes(title="Muertes / 100k hab")
    fig.update_layout(height=320, legend_orientation="h",
                      legend_y=1.08, legend_x=0)
    return fig
