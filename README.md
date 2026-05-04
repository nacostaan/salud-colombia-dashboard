# 🏥 Brechas en Salud Preventiva Colombia

> Dashboard interactivo de identificación y priorización de brechas territoriales en salud preventiva a nivel municipal en Colombia.

**🔗 Live demo:** [salud-colombia.streamlit.app](https://salud-colombia.streamlit.app)  
**📊 Modelos:** [huggingface.co/nicoacos/salud-colombia-modelos](https://huggingface.co/nicoacos/salud-colombia-modelos)

---

## Descripción

Este proyecto forma parte del **Proyecto Aplicado en Analítica de Datos (PAAD)** de la Maestría en Inteligencia y Analítica de Datos (MIAD) de la Universidad de los Andes.

Integra datos de **7 fuentes oficiales** (DANE, RIPS, SGP, BDUA, SIVIGILA, DIVIPOLA) para construir una tabla de hechos a nivel **municipio × año** (1979–2024) y entrenar modelos que permiten:

1. **Predecir** la tasa de mortalidad evitable por 100.000 habitantes en cualquier municipio de Colombia.
2. **Clasificar** si un municipio se encuentra en zona de alto riesgo (percentil ≥ 80).

---

## Capturas de pantalla

| Vista Inicio | Vista Predicción |
|:---:|:---:|
| Mapa nacional + KPIs + Top 20 municipios | Predictor interactivo + gauges de resultado |

---

## Modelos

| Tarea | Modelo | Métrica principal |
|-------|--------|-------------------|
| **Regresión** | ExtraTrees Regressor | R² = 0.978 · RMSE = 16.04 |
| **Clasificación** | LightGBM Classifier | AUC = 0.999 · Recall = 0.941 |

- **Split temporal:** entrenamiento ≤ 2019 · test out-of-time 2020–2024
- **Features:** 34 variables incluyendo lags temporales, indicadores RIPS, giros SGP y cobertura BDUA
- Los modelos se alojan en [Hugging Face Hub](https://huggingface.co/nicoacos/salud-colombia-modelos) (descarga automática en el primer arranque)

---

## Estructura del repositorio

```
salud-colombia-dashboard/
│
├── 📁 .streamlit/
│   └── config.toml                         # Tema oscuro personalizado
│
├── 📁 src/
│   ├── __init__.py
│   ├── loader.py                           # Descarga modelos desde Hugging Face
│   ├── predictor.py                        # Lógica de predicción y features
│   ├── charts.py                           # Gráficas Plotly reutilizables
│   └── geo.py                              # Mapas de Colombia y KPIs geográficos
│
├── 📁 Datasets_Limpios/                    # Datasets procesados listos para modelado
│   ├── df_BDUA_limpio.parquet
│   ├── df_defunciones_atribuibles.parquet
│   ├── df_poblacion_colombia.parquet
│   ├── df_SIVIGILA_limpio.parquet
│   ├── divipola_cabeceras_centros_poblados.parquet
│   └── Giros_final.parquet
│
├── 📁 Notebooks/
│   │
│   ├── 📁 Modelos/                         # Entrenamiento y evaluación de modelos ML
│   │   ├── 9-modelos-ml-salud-v4_1.ipynb
│   │   └── modelo-ml-paad-nicolas.ipynb
│   │
│   └── 📁 Tratamiento_datasets_bruto/      # Pipelines de limpieza por fuente
│       ├── 📁 BDUA/
│       │   └── da-bdua-preprocesamiento.ipynb
│       ├── 📁 Defunciones/
│       │   ├── 1-defunciones-raw.ipynb
│       │   ├── 2-defunciones-interim.ipynb
│       │   ├── 3-transformacion-defunciones.ipynb
│       │   └── 4-defunciones-atribuibles-salud.ipynb
│       ├── 📁 Divipola/
│       │   ├── 5-procesamiento-divipola-geografica.ipynb
│       │   └── 6-procesamiento-divipola-actualizado.ipynb
│       ├── 📁 Giros/
│       │   ├── 1-analisis-exploratorio-giros.ipynb
│       │   ├── 2-limpieza-de-datos-y-exploracion-giros.ipynb
│       │   └── 3-giros-final.ipynb
│       ├── 📁 Poblacion/
│       │   └── 7-procesamiento-poblacion.ipynb
│       ├── 📁 RIPS/
│       │   ├── 1-rips-analisis-exploratorio.ipynb
│       │   ├── 2-limpieza-de-datos-rips.ipynb
│       │   ├── 3-rips-final-parte-1.ipynb
│       │   └── 3-rips-final-parte-2.ipynb
│       └── 📁 Sivigila/
│           └── da-sivigila-preprocesamiento.ipynb
│
├── 📁 Tabla_de_Hechos/                     # Tabla de hechos final municipio × año
│   ├── 8-tabla-hechos-final-salud-colombia.ipynb
│   ├── tabla_hechos_salud_colombia.parquet
│   ├── tabla_hechos_test.parquet
│   ├── tabla_hechos_train.parquet
│   └── df_SIVIGILA_limpio.parquet
│
├── app.py                                  # Entrypoint Streamlit — 4 vistas
├── requirements.txt                        # Dependencias fijadas
├── runtime.txt                             # Python 3.12
├── .python-version                         # 3.12
├── LICENSE
└── README.md
```

---

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Frontend | Streamlit 1.45 · Plotly 6.1 |
| ML | scikit-learn 1.6 · LightGBM 4.6 |
| Datos | pandas 2.3 · pyarrow · numpy 2.0 |
| Hosting modelos | Hugging Face Hub |
| Despliegue | Streamlit Community Cloud |
| Lenguaje | Python 3.12 |

---

## Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/nacostaan/salud-colombia-dashboard.git
cd salud-colombia-dashboard

# 2. Crear entorno virtual con Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el dashboard
streamlit run app.py
```

El primer arranque descarga los modelos desde Hugging Face (~5 MB) y los cachea localmente en `.cache/modelos/`.

---

## Variables de entorno

| Variable | Descripción | Por defecto |
|----------|-------------|-------------|
| `HF_REPO_ID` | Repositorio Hugging Face con los modelos | `nicoacos/salud-colombia-modelos` |

Configurar en Streamlit Community Cloud: **Settings → Secrets** (si cambias el repo de HF).

---

## Fuentes de datos

| Fuente | Descripción | Cobertura |
|--------|-------------|-----------|
| DANE Defunciones | Muertes por municipio y causa (CIE-10) | 1979–2024 |
| RIPS | Registros Individuales de Prestación de Servicios | 2009–2021 |
| Giros SGP | Sistema General de Participaciones — Salud | 2015–2023 |
| BDUA | Base de Datos Única de Afiliados al SGSSS | Corte transversal |
| SIVIGILA | Vigilancia epidemiológica en salud pública | 2021 |
| DANE Población | Proyecciones poblacionales municipales | 1985–2042 |
| DIVIPOLA | Codificación geográfica oficial DANE | Vigente 2025 |

> **Nota:** Los datos crudos originales no se incluyen en este repositorio por su tamaño. Los datasets limpios (`.parquet`) se encuentran en `Datasets_Limpios/` y `Tabla_de_Hechos/`. Para replicar el pipeline completo desde cero, descarga las fuentes desde sus portales oficiales y ejecuta los notebooks en `Notebooks/Tratamiento_datasets_bruto/` en orden numérico.

---

## Autores

Proyecto MIAD — Maestría en Inteligencia y Analítica de Datos  
Universidad de los Andes · Bogotá, Colombia

---

## Licencia

MIT License — ver [`LICENSE`](LICENSE) para detalles.
