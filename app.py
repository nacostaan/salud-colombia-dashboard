"""
app.py — Dashboard principal: Brechas en Salud Preventiva Colombia
==================================================================
Entrypoint de Streamlit Community Cloud.
Ejecución local: streamlit run app.py

Estructura de vistas:
  🏠 Inicio          → mapa nacional + KPIs + top 20 municipios críticos
  🔮 Predicción      → predictor interactivo por municipio
  📊 Modelos         → comparativa de métricas de todos los modelos
  ℹ️  Sobre el modelo → descripción técnica, variables, metodología
"""

from __future__ import annotations

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# ── Añadir src/ al path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from loader    import cargar_todo
from predictor import (
    predecir_regresion, predecir_clasificacion,
    agregar_por_departamento, top_municipios_criticos,
    FEATURE_LABELS,
)
from charts import (
    mapa_burbujas_colombia, barras_departamentos,
    gauge_tasa, gauge_riesgo,
    tabla_top_municipios,
    comparativa_modelos_reg, comparativa_modelos_clf,
    evolucion_municipio,
)
from geo import (
    mapa_nacional_municipios, mapa_departamento,
    kpis_nacionales, lista_municipios_por_dpto,
)

# ── Configuración global de la página ────────────────────────────────────────
st.set_page_config(
    page_title="Brechas en Salud Colombia",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": (
            "**Brechas en Salud Preventiva Colombia**\n\n"
            "Proyecto MIAD — Universidad de los Andes\n\n"
            "Modelos: ExtraTrees (R²=0.978) · LightGBM (AUC=0.999)"
        ),
    },
)

# ── CSS global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tarjetas KPI */
.kpi-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 18px 20px 14px;
    text-align: center;
}
.kpi-value  { font-size: 2.0rem; font-weight: 700; color: #F0F6FC; line-height: 1.1; }
.kpi-label  { font-size: 0.78rem; color: #8B949E; margin-top: 4px; }
.kpi-delta  { font-size: 0.82rem; margin-top: 6px; }
.delta-pos  { color: #E63946; }
.delta-neg  { color: #2ECC71; }
/* Badge de semáforo */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
}
/* Separador sección */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #8B949E;
    border-bottom: 1px solid #21262D;
    padding-bottom: 6px;
    margin: 18px 0 12px;
}
/* Ocultar menú hamburguesa y pie de Streamlit */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Carga de artefactos (cacheados) ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _cargar():
    with st.spinner("⏳ Cargando modelos y datos…"):
        return cargar_todo()

pipe_r, pipe_c, meta, df_hist = _cargar()

# Constantes del meta
UMBRAL_CLF   = meta["umbrales_prob_clf"].get("LightGBM_C", 0.65)
MEJOR_REG    = meta.get("mejor_reg", "HistGB_R")
MEJOR_CLF    = meta.get("mejor_clf", "LightGBM_C")
MET_REG      = meta.get("metricas_reg", {})
MET_CLF      = meta.get("metricas_clf", {})

# Años disponibles con datos de target
ANIOS_CON_DATOS = sorted(
    df_hist.dropna(subset=["TASA_MORTALIDAD_EVITABLE_100K"])["ANIO"]
    .astype(int).unique().tolist(),
    reverse=True,
)
ANIO_DEFAULT = ANIOS_CON_DATOS[0] if ANIOS_CON_DATOS else 2024

# Catálogo geográfico
DPTOS = (
    df_hist[["COD_DPTO", "NOM_DPTO"]]
    .drop_duplicates()
    .sort_values("NOM_DPTO")
    .values.tolist()
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://uniandes.edu.co/sites/default/files/logo-uniandes.png",
        width=160,
    )
    st.markdown("## 🏥 Brechas en Salud")
    st.markdown(
        "<small style='color:#8B949E'>Proyecto MIAD · PAAD<br>"
        "Universidad de los Andes</small>",
        unsafe_allow_html=True,
    )
    st.divider()

    vista = st.radio(
        "Navegación",
        ["🏠 Inicio", "🔮 Predicción", "📊 Modelos", "ℹ️ Sobre el modelo"],
        label_visibility="collapsed",
    )
    st.divider()

    # Selector de año global (usado en Inicio)
    anio_sel = st.select_slider(
        "Año visualizado",
        options=ANIOS_CON_DATOS,
        value=ANIO_DEFAULT,
    )

    st.markdown(
        f"<small style='color:#8B949E'>"
        f"**Regresión:** {MEJOR_REG}<br>"
        f"R² = {MET_REG.get('R2', {}).get(MEJOR_REG, 0):.3f} · "
        f"RMSE = {MET_REG.get('RMSE', {}).get(MEJOR_REG, 0):.1f}<br><br>"
        f"**Clasificación:** {MEJOR_CLF}<br>"
        f"AUC = {MET_CLF.get('AUC', {}).get(MEJOR_CLF, 0):.4f} · "
        f"Recall = {MET_CLF.get('Recall', {}).get(MEJOR_CLF, 0):.3f}"
        f"</small>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 1: INICIO
# ═══════════════════════════════════════════════════════════════════════════════
if vista == "🏠 Inicio":

    st.markdown("# 🏥 Brechas en Salud Preventiva · Colombia")
    st.markdown(
        f"<span style='color:#8B949E'>Mortalidad evitable por municipio — año **{anio_sel}**</span>",
        unsafe_allow_html=True,
    )

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis = kpis_nacionales(df_hist, anio_sel)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_html = ""
        if kpis["delta_vs_anterior"] is not None:
            d = kpis["delta_vs_anterior"]
            cls   = "delta-pos" if d > 0 else "delta-neg"
            signo = "▲" if d > 0 else "▼"
            delta_html = f"<div class='kpi-delta {cls}'>{signo} {abs(d):.1f} vs {anio_sel-1}</div>"
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{kpis['tasa_nacional']}</div>"
            f"<div class='kpi-label'>Tasa nacional /100k hab</div>"
            f"{delta_html}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value' style='color:#E63946'>{kpis['municipios_criticos']:,}</div>"
            f"<div class='kpi-label'>Municipios zona crítica</div>"
            f"<div class='kpi-delta' style='color:#8B949E'>"
            f"{kpis['pct_criticos']:.1f}% del total</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{kpis['total_municipios']:,}</div>"
            f"<div class='kpi-label'>Municipios con datos</div>"
            f"<div class='kpi-delta' style='color:#8B949E'>de 1,122 totales</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        muertes_str = f"{kpis['muertes_atribuibles']:,}" if kpis['muertes_atribuibles'] else "N/D"
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value' style='color:#E67E22'>{muertes_str}</div>"
            f"<div class='kpi-label'>Muertes atribuibles</div>"
            f"<div class='kpi-delta' style='color:#8B949E'>al sistema de salud</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Mapa + Top departamentos ───────────────────────────────────────────────
    col_mapa, col_barras = st.columns([1.6, 1], gap="medium")

    with col_mapa:
        st.markdown("<div class='section-header'>Mapa de municipios</div>",
                    unsafe_allow_html=True)
        fig_mapa = mapa_nacional_municipios(df_hist, anio_sel)
        st.plotly_chart(fig_mapa, use_container_width=True, config={"displayModeBar": False})

    with col_barras:
        st.markdown("<div class='section-header'>Ranking por departamento</div>",
                    unsafe_allow_html=True)
        df_dpto = agregar_por_departamento(df_hist, anio_sel)
        fig_bar  = barras_departamentos(df_dpto, top_n=15)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Top 20 municipios críticos ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>Top 20 municipios — Mayor mortalidad evitable</div>",
                unsafe_allow_html=True)
    df_top = top_municipios_criticos(df_hist, n=20, anio=anio_sel)
    fig_tabla = tabla_top_municipios(df_top)
    st.plotly_chart(fig_tabla, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 2: PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
elif vista == "🔮 Predicción":

    st.markdown("# 🔮 Predicción por Municipio")
    st.markdown(
        "<span style='color:#8B949E'>Configura los parámetros del municipio "
        "para obtener la predicción de mortalidad y clasificación de riesgo.</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Selector geográfico ────────────────────────────────────────────────────
    col_geo1, col_geo2, col_geo3 = st.columns([1, 1, 1])

    with col_geo1:
        dpto_opts = {nom: cod for cod, nom in DPTOS}
        nom_dpto  = st.selectbox("Departamento", list(dpto_opts.keys()))
        cod_dpto  = dpto_opts[nom_dpto]

    with col_geo2:
        munis = lista_municipios_por_dpto(df_hist, cod_dpto)
        if munis:
            muni_opts = {m["NOM_MPIO"]: m for m in munis}
            nom_mpio  = st.selectbox("Municipio", list(muni_opts.keys()))
            muni_sel  = muni_opts[nom_mpio]
        else:
            st.warning("Sin municipios para este departamento.")
            st.stop()

    with col_geo3:
        anio_pred = st.selectbox(
            "Año a predecir",
            list(range(2025, 2028)) + ANIOS_CON_DATOS[:3],
            index=0,
        )

    # Pre-poblar con datos históricos del municipio (último año disponible)
    hist_mpio = (
        df_hist[df_hist["COD_MPIO"] == muni_sel["COD_MPIO"]]
        .dropna(subset=["TASA_MORTALIDAD_EVITABLE_100K"])
        .sort_values("ANIO", ascending=False)
    )
    tasa_hist_lag1 = float(hist_mpio["TASA_MORTALIDAD_EVITABLE_100K"].iloc[0]) if not hist_mpio.empty else 130.0
    tasa_hist_lag2 = float(hist_mpio["TASA_MORTALIDAD_EVITABLE_100K"].iloc[1]) if len(hist_mpio) > 1 else 135.0

    # ── Formulario de features ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Parámetros del municipio</div>",
                unsafe_allow_html=True)

    with st.form("prediccion_form"):
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            lag1 = st.number_input(
                "Tasa mortalidad año anterior (/ 100k)",
                min_value=0.0, max_value=800.0,
                value=round(tasa_hist_lag1, 1), step=1.0,
                help="Tasa del año inmediatamente anterior (se pre-llena con datos históricos reales si existen)",
            )
            lag2 = st.number_input(
                "Tasa mortalidad hace 2 años (/ 100k)",
                min_value=0.0, max_value=800.0,
                value=round(tasa_hist_lag2, 1), step=1.0,
            )
            pob = st.number_input(
                "Población total",
                min_value=500, max_value=10_000_000,
                value=int(muni_sel.get("POBLACION_TOTAL") or 50_000),
                step=500,
            )

        with fc2:
            pct_afil = st.slider(
                "% Afiliación al sistema de salud",
                min_value=0.0, max_value=100.0,
                value=float(
                    df_hist[df_hist["COD_MPIO"] == muni_sel["COD_MPIO"]]
                    ["PCT_AFILIACION"].dropna().median() or 65.0
                ),
                step=0.5,
                help="Porcentaje de la población afiliada a cualquier régimen de salud",
            )
            tasa_rips = st.number_input(
                "Atenciones RIPS (/ 100k hab)",
                min_value=0.0, max_value=8_000_000.0,
                value=float(
                    df_hist[df_hist["COD_MPIO"] == muni_sel["COD_MPIO"]]
                    ["TASA_ATENCIONES_RIPS_100K"].dropna().median() or 228_000.0
                ),
                step=1_000.0,
            )
            bdua_afil = st.number_input(
                "Afiliados BDUA",
                min_value=0, max_value=5_000_000,
                value=int(
                    df_hist[df_hist["COD_MPIO"] == muni_sel["COD_MPIO"]]
                    ["BDUA_TOTAL_AFILIADOS"].dropna().median() or 8_000
                ),
                step=100,
            )

        with fc3:
            val_girado = st.number_input(
                "Valor girado SGP (COP)",
                min_value=0, max_value=1_000_000_000_000,
                value=int(
                    df_hist[df_hist["COD_MPIO"] == muni_sel["COD_MPIO"]]
                    ["VALOR_GIRADO_NORMAL"].dropna().median() or 121_000_000
                ),
                step=1_000_000,
                format="%d",
                help="Recursos del Sistema General de Participaciones para salud",
            )
            n_giros = st.slider("Número de giros SGP al año", 1, 24,
                                value=int(
                                    df_hist[df_hist["COD_MPIO"] == muni_sel["COD_MPIO"]]
                                    ["N_GIROS"].dropna().median() or 12
                                ))
            efic_giro = st.slider(
                "Eficiencia de giro (índice)",
                min_value=0.0, max_value=5.0, value=1.0, step=0.05,
                help="1.0 = eficiencia media nacional. >1.0 = por encima del promedio.",
            )

        submitted = st.form_submit_button(
            "🔮 Calcular predicción",
            use_container_width=True,
            type="primary",
        )

    # ── Resultados de predicción ───────────────────────────────────────────────
    if submitted:
        inputs = {
            "LAG1_MORT"                : lag1,
            "LAG2_MORT"                : lag2,
            "ROLL3_MORT"               : (lag1 + lag2) / 2,
            "POBLACION_TOTAL"          : float(pob),
            "PCT_AFILIACION"           : pct_afil,
            "TASA_ATENCIONES_RIPS_100K": tasa_rips,
            "BDUA_TOTAL_AFILIADOS"     : float(bdua_afil),
            "VALOR_GIRADO_NORMAL"      : float(val_girado),
            "N_GIROS"                  : float(n_giros),
            "EFICIENCIA_GIRO"          : efic_giro,
            "LATITUD"                  : float(muni_sel.get("LATITUD") or 4.5),
            "LONGITUD"                 : float(muni_sel.get("LONGITUD") or -74.0),
            "ANIO"                     : float(anio_pred),
        }

        with st.spinner("Calculando predicciones…"):
            res_reg = predecir_regresion(pipe_r, inputs)
            res_clf = predecir_clasificacion(pipe_c, inputs, umbral=UMBRAL_CLF)

        st.markdown("<div class='section-header'>Resultados</div>",
                    unsafe_allow_html=True)

        c_gauge1, c_gauge2 = st.columns(2)
        with c_gauge1:
            st.plotly_chart(
                gauge_tasa(res_reg["tasa"], res_reg["nivel"], res_reg["color"]),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.markdown(
                f"<div style='text-align:center'>"
                f"<span class='badge' style='background:{res_reg['color']}22;"
                f"color:{res_reg['color']};border:1px solid {res_reg['color']}'>"
                f"{res_reg['emoji']} Zona {res_reg['nivel']}"
                f"</span></div>",
                unsafe_allow_html=True,
            )

        with c_gauge2:
            st.plotly_chart(
                gauge_riesgo(res_clf["proba"], res_clf["nivel"], res_clf["color"],
                             umbral_pct=UMBRAL_CLF * 100),
                use_container_width=True, config={"displayModeBar": False},
            )
            clasificacion_txt = "🔴 ALTO RIESGO" if res_clf["alto"] else "🟢 SIN ALTO RIESGO"
            st.markdown(
                f"<div style='text-align:center'>"
                f"<span class='badge' style='background:{res_clf['color']}22;"
                f"color:{res_clf['color']};border:1px solid {res_clf['color']}'>"
                f"{clasificacion_txt}</span>"
                f"<br><small style='color:#8B949E'>Umbral: {UMBRAL_CLF*100:.0f}% · "
                f"Recall={MET_CLF.get('Recall',{}).get(MEJOR_CLF,0):.3f}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Contexto geográfico + evolución histórica
        st.markdown("<div class='section-header'>Contexto histórico</div>",
                    unsafe_allow_html=True)
        c_mapa_d, c_evol = st.columns([1, 1.4], gap="medium")
        with c_mapa_d:
            fig_d = mapa_departamento(df_hist, cod_dpto, ANIO_DEFAULT)
            st.plotly_chart(fig_d, use_container_width=True,
                            config={"displayModeBar": False})
        with c_evol:
            fig_evol = evolucion_municipio(
                df_hist, muni_sel["COD_MPIO"], nom_mpio)
            st.plotly_chart(fig_evol, use_container_width=True,
                            config={"displayModeBar": False})

    else:
        st.info(
            "👆 Ajusta los parámetros del municipio y haz clic en "
            "**Calcular predicción** para ver los resultados.",
            icon="💡",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 3: COMPARATIVA DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
elif vista == "📊 Modelos":

    st.markdown("# 📊 Comparativa de Modelos")
    st.markdown(
        "<span style='color:#8B949E'>Evaluación out-of-time (test: 2020–2024) "
        "sobre 15+ modelos de regresión y clasificación.</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    tab_reg, tab_clf = st.tabs(["📈 Regresión", "🎯 Clasificación"])

    with tab_reg:
        st.markdown(
            f"**Modelo ganador:** `{MEJOR_REG}` · "
            f"RMSE = {MET_REG.get('RMSE',{}).get(MEJOR_REG,0):.2f} · "
            f"MAE = {MET_REG.get('MAE',{}).get(MEJOR_REG,0):.2f} · "
            f"R² = {MET_REG.get('R2',{}).get(MEJOR_REG,0):.3f}"
        )
        st.markdown(
            "<small style='color:#8B949E'>Variable objetivo: "
            "TASA_MORTALIDAD_EVITABLE_100K (muertes / 100k hab)</small>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            comparativa_modelos_reg(MET_REG),
            use_container_width=True, config={"displayModeBar": False},
        )

        # Tabla de métricas completa
        with st.expander("Ver tabla de métricas completa"):
            df_reg_met = pd.DataFrame({
                "RMSE"   : MET_REG.get("RMSE",   {}),
                "MAE"    : MET_REG.get("MAE",    {}),
                "R²"     : MET_REG.get("R2",     {}),
                "MAPE %" : MET_REG.get("MAPE_%", {}),
                "Pearson": MET_REG.get("Pearson", {}),
            }).round(3).sort_values("RMSE")

            def highlight_best_reg(s):
                if s.name in ("RMSE", "MAE", "MAPE %"):
                    return ["background-color:#1a3a1a" if v == s.min() else "" for v in s]
                if s.name in ("R²", "Pearson"):
                    return ["background-color:#1a3a1a" if v == s.max() else "" for v in s]
                return [""] * len(s)

            st.dataframe(df_reg_met.style.apply(highlight_best_reg), use_container_width=True)

    with tab_clf:
        st.markdown(
            f"**Modelo ganador:** `{MEJOR_CLF}` · "
            f"AUC = {MET_CLF.get('AUC',{}).get(MEJOR_CLF,0):.4f} · "
            f"Recall = {MET_CLF.get('Recall',{}).get(MEJOR_CLF,0):.3f} · "
            f"F1 = {MET_CLF.get('F1',{}).get(MEJOR_CLF,0):.3f}"
        )
        st.markdown(
            "<small style='color:#8B949E'>Variable objetivo: "
            "FLG_ZONA_ALTO_RIESGO (municipio en percentil 80+ de mortalidad)</small>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            comparativa_modelos_clf(MET_CLF),
            use_container_width=True, config={"displayModeBar": False},
        )

        with st.expander("Ver tabla de métricas completa"):
            df_clf_met = pd.DataFrame({
                "AUC"      : MET_CLF.get("AUC",      {}),
                "Recall"   : MET_CLF.get("Recall",   {}),
                "F1"       : MET_CLF.get("F1",       {}),
                "Precision": MET_CLF.get("Precision",{}),
                "Accuracy" : MET_CLF.get("Accuracy", {}),
            }).round(4).sort_values("AUC", ascending=False)

            def highlight_best_clf(s):
                return ["background-color:#1a3a1a" if v == s.max() else "" for v in s]

            st.dataframe(df_clf_met.style.apply(highlight_best_clf), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 4: SOBRE EL MODELO
# ═══════════════════════════════════════════════════════════════════════════════
elif vista == "ℹ️ Sobre el modelo":

    st.markdown("# ℹ️ Sobre el Proyecto y el Modelo")
    st.divider()

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### 🎯 Objetivo")
        st.markdown("""
Identificar y priorizar **brechas territoriales en salud preventiva** en Colombia
a escala municipal, con dos modelos supervisados:

| Modelo | Tarea | Variable objetivo |
|--------|-------|-------------------|
| **Regresión** | Estimar tasa de mortalidad evitable | `TASA_MORTALIDAD_EVITABLE_100K` |
| **Clasificación** | Identificar zonas de alto riesgo | `FLG_ZONA_ALTO_RIESGO` (≥ P80) |
        """)

        st.markdown("### 🏆 Mejores resultados")
        st.markdown(f"""
**Regresión — `{MEJOR_REG}`**
- R² = `{MET_REG.get("R2", {}).get(MEJOR_REG, 0):.4f}` (varianza explicada)
- RMSE = `{MET_REG.get("RMSE", {}).get(MEJOR_REG, 0):.2f}` muertes/100k
- MAE  = `{MET_REG.get("MAE",  {}).get(MEJOR_REG, 0):.2f}` muertes/100k
- MAPE = `{MET_REG.get("MAPE_%", {}).get(MEJOR_REG, 0):.1f}%`

**Clasificación — `{MEJOR_CLF}`**
- AUC    = `{MET_CLF.get("AUC",       {}).get(MEJOR_CLF, 0):.4f}`
- Recall = `{MET_CLF.get("Recall",    {}).get(MEJOR_CLF, 0):.4f}`
- F1     = `{MET_CLF.get("F1",        {}).get(MEJOR_CLF, 0):.4f}`
- Umbral de probabilidad: `{UMBRAL_CLF:.2f}` (optimizado para Recall ≥ 0.88)
        """)

        st.markdown("### 📦 Fuentes de datos integradas")
        st.markdown("""
| Fuente | Descripción | Cobertura |
|--------|-------------|-----------|
| DANE Defunciones | Muertes registradas por municipio y causa | 1979–2024 |
| RIPS | Registros de Prestación de Servicios de Salud | 2009–2021 |
| Giros SGP | Recursos del Sistema General de Participaciones | 2015–2023 |
| BDUA | Base de Datos Única de Afiliados al SGSSS | Corte transversal |
| SIVIGILA | Vigilancia epidemiológica | 2021 |
| DANE Población | Proyecciones poblacionales | 1985–2042 |
| DIVIPOLA | Codificación geográfica oficial | Vigente |
        """)

    with col_b:
        st.markdown("### 🔑 Variables más importantes (features)")
        st.markdown("""
Las 34 features del modelo incluyen tres grupos principales:

**📈 Features de tendencia temporal (lag)**
- `LAG1_MORT` — tasa del año anterior (principal predictor)
- `LAG2_MORT` — tasa de hace 2 años
- `ROLL3_MORT` — promedio móvil 3 años
- `TENDENCIA_MPIO` — pendiente de regresión local
- `DELTA_YOY` — cambio año a año

**🏥 Capacidad del sistema de salud**
- `TASA_ATENCIONES_RIPS_100K` — uso del sistema por habitante
- `EFICIENCIA_GIRO` — eficiencia en la asignación de recursos SGP
- `PCT_AFILIACION` — cobertura del sistema de salud

**🗺️ Contexto geográfico y temporal**
- `LATITUD` / `LONGITUD` — localización municipal
- `LATITUD_ANIO` — interacción espacio-temporal
- `ANIO_SIN` / `ANIO_COS` — codificación cíclica del año
        """)

        st.markdown("### ⚙️ Stack técnico")
        st.markdown("""
```
Lenguaje    Python 3.11
ML          scikit-learn 1.6 · LightGBM 4.6
Secuencial  PyTorch (LSTM) · entrenado en Kaggle GPU
Pipeline    sklearn Pipeline + DropAllNanCols (custom)
Validación  Cross-validation temporal + out-of-time test
SHAP        Interpretabilidad de variables por municipio
Dashboard   Streamlit 1.45 · Plotly 6.1
Modelos     Hugging Face Hub (hosting gratuito)
Deploy      Streamlit Community Cloud
```
        """)

        st.markdown("### 🔬 Metodología de split temporal")
        st.markdown("""
Para evitar **data leakage** al usar features de lag (que requieren datos del pasado):

| Conjunto | Años | Descripción |
|----------|------|-------------|
| **Train** | 1979–2019 | Ajuste de hiperparámetros + entrenamiento |
| **Test**  | 2020–2024 | Evaluación out-of-time (nunca visto por el modelo) |

Las features `LAG1_MORT` y `LAG2_MORT` se calculan con el historial disponible
hasta el año anterior al de predicción, garantizando que no haya filtración
de información futura al modelo.
        """)

    st.divider()
    st.markdown(
        "<center><small style='color:#8B949E'>"
        "Maestría en Inteligencia y Analítica de Datos (MIAD) · "
        "Universidad de los Andes · "
        "Proyecto Aplicado en Analítica de Datos (PAAD)"
        "</small></center>",
        unsafe_allow_html=True,
    )
