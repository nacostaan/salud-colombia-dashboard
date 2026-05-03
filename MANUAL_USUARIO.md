# Manual de Usuario
## Dashboard — Brechas en Salud Preventiva Colombia

**URL:** https://salud-colombia.streamlit.app  
**Versión:** 1.0 · Mayo 2026  
**Proyecto:** MIAD PAAD — Universidad de los Andes

---

## Índice

1. [Acceso y carga inicial](#1-acceso-y-carga-inicial)
2. [Navegación general](#2-navegación-general)
3. [Vista 🏠 Inicio](#3-vista--inicio)
4. [Vista 🔮 Predicción](#4-vista--predicción)
5. [Vista 📊 Modelos](#5-vista--modelos)
6. [Vista ℹ️ Sobre el modelo](#6-vista-ℹ️-sobre-el-modelo)
7. [Preguntas frecuentes](#7-preguntas-frecuentes)
8. [Glosario](#8-glosario)

---

## 1. Acceso y carga inicial

### Abrir el dashboard

1. Abrir un navegador web (Chrome, Firefox o Edge recomendados).
2. Ir a: **https://salud-colombia.streamlit.app**
3. Al acceder por primera vez, verá un mensaje de carga:

```
⏳ Cargando modelos y datos…
```

Este proceso tarda entre **20 y 40 segundos** en el primer acceso porque el sistema descarga los modelos de inteligencia artificial desde Hugging Face Hub. En accesos posteriores (misma sesión) la carga es instantánea.

### ¿El dashboard tarda demasiado?

Si después de 60 segundos no carga, intente:
- Recargar la página (`F5` o `Ctrl + R`)
- Limpiar la caché del navegador
- Usar una conexión de Internet diferente

---

## 2. Navegación general

### Panel lateral (sidebar)

El panel lateral izquierdo contiene los controles de navegación principales:

| Elemento | Función |
|----------|---------|
| **Logo UNIANDES** | Identificación del proyecto |
| **Radio de vistas** | Cambia entre las 4 secciones del dashboard |
| **Selector de año** | Controla el año visualizado en el mapa y KPIs de la vista Inicio |
| **Métricas del modelo** | Muestra el R², RMSE y AUC de los modelos activos |

### Selector de año (sidebar)

- Usar el control deslizante para cambiar el año visualizado.
- El rango disponible corresponde a los años con datos de mortalidad real en el dataset.
- El cambio de año actualiza automáticamente el mapa, los KPIs y la tabla de municipios críticos.

---

## 3. Vista 🏠 Inicio

Esta es la vista principal del dashboard. Muestra el panorama nacional de mortalidad evitable.

### 3.1 Tarjetas KPI

En la parte superior aparecen 4 indicadores para el año seleccionado:

| Tarjeta | Descripción |
|---------|-------------|
| **Tasa nacional / 100k hab** | Promedio nacional de la tasa de mortalidad evitable. La flecha (▲/▼) compara con el año anterior — rojo es peor, verde es mejor. |
| **Municipios zona crítica** | Número de municipios clasificados en zona de alto riesgo (percentil ≥ 80). Incluye el porcentaje sobre el total. |
| **Municipios con datos** | Total de municipios con registros de mortalidad para ese año (de 1.122 totales en Colombia). |
| **Muertes atribuibles** | Suma total de muertes atribuibles al sistema de salud en todos los municipios. |

### 3.2 Mapa de municipios

El mapa central muestra todos los municipios de Colombia como burbujas coloreadas:

**Código de colores (semáforo de riesgo):**

| Color | Zona | Criterio |
|-------|------|----------|
| 🟢 Verde | Bajo | Tasa < percentil 25 |
| 🟡 Amarillo | Moderado | Percentil 25–50 |
| 🟠 Naranja | Alto | Percentil 50–75 |
| 🔴 Rojo | Crítico | Tasa > percentil 75 |

**Interacción con el mapa:**
- **Pasar el cursor** sobre una burbuja muestra el nombre del municipio, departamento y tasa exacta.
- El **tamaño** de la burbuja es proporcional a la tasa de mortalidad (burbujas más grandes = más crítico).
- La **leyenda** en la esquina inferior izquierda permite hacer clic para ocultar/mostrar niveles de riesgo.

### 3.3 Ranking por departamento

El panel derecho muestra una barra horizontal con los 15 departamentos con mayor tasa de mortalidad evitable. Las barras siguen el mismo código de colores del semáforo.

### 3.4 Tabla Top 20 municipios

Al final de la página aparece una tabla con los 20 municipios más críticos para el año seleccionado, ordenados de mayor a menor tasa. Contiene:

- **Municipio y departamento**
- **Tasa / 100k** — muertes evitables por cada 100.000 habitantes
- **Muertes attr.** — número absoluto de muertes atribuibles al sistema
- **Población** — proyección DANE
- **Zona** — semáforo de riesgo del clasificador

---

## 4. Vista 🔮 Predicción

Esta vista permite obtener una predicción personalizada para cualquier municipio del país.

### 4.1 Selección geográfica

En la fila superior aparecen tres selectores:

1. **Departamento** — elegir el departamento de interés (listado completo de 33 departamentos).
2. **Municipio** — se filtra automáticamente según el departamento. Muestra todos los municipios disponibles.
3. **Año a predecir** — elegir el año para el que se quiere la predicción (2025–2027 o años históricos recientes).

> 💡 Al seleccionar el municipio, los campos del formulario se **pre-llenan automáticamente** con los datos históricos reales del municipio (mediana de los últimos años disponibles). Puede aceptar estos valores o modificarlos.

### 4.2 Formulario de parámetros

El formulario se divide en tres columnas:

#### Columna izquierda — Historial de mortalidad

| Campo | Descripción | Rango |
|-------|-------------|-------|
| **Tasa mortalidad año anterior** | Tasa de mortalidad evitable del mismo municipio en el año t-1 | 0 – 800 por 100k |
| **Tasa mortalidad hace 2 años** | Tasa del año t-2 | 0 – 800 por 100k |
| **Población total** | Proyección DANE de habitantes | 500 – 10.000.000 |

#### Columna central — Sistema de salud

| Campo | Descripción | Rango |
|-------|-------------|-------|
| **% Afiliación al sistema** | Porcentaje de la población afiliada a cualquier régimen de salud | 0 – 100% |
| **Atenciones RIPS / 100k** | Total de atenciones en salud (consultas + urgencias + hospitalizaciones + procedimientos) por cada 100k habitantes | 0 – 8.000.000 |
| **Afiliados BDUA** | Total de personas afiliadas según la Base de Datos Única de Afiliados | 0 – 5.000.000 |

#### Columna derecha — Recursos financieros

| Campo | Descripción | Rango |
|-------|-------------|-------|
| **Valor girado SGP (COP)** | Recursos del Sistema General de Participaciones para salud, sin pagos atípicos | 0 – $1 billón |
| **Número de giros SGP** | Cantidad de pagos SGP realizados en el año | 1 – 24 |
| **Eficiencia de giro** | Índice: 1.0 = media nacional; >1.0 = más eficiente que el promedio | 0.0 – 5.0 |

### 4.3 Calcular predicción

Después de revisar o ajustar los parámetros, hacer clic en el botón rojo **🔮 Calcular predicción**.

### 4.4 Interpretación de resultados

Aparecen dos gauges (velocímetros) y una sección de contexto histórico:

#### Gauge izquierdo — Tasa de mortalidad predicha

- Muestra la tasa estimada de mortalidad evitable en muertes por cada 100.000 habitantes.
- La línea roja punteada indica la **media nacional** (referencia: ~130/100k).
- El color del gauge indica el nivel de riesgo.
- **El delta** muestra cuánto se desvía de la media nacional.

| Valor | Interpretación |
|-------|---------------|
| < 75 / 100k | 🟢 Zona de riesgo **Bajo** |
| 75 – 150 / 100k | 🟡 Zona de riesgo **Moderado** |
| 150 – 250 / 100k | 🟠 Zona de riesgo **Alto** |
| > 250 / 100k | 🔴 Zona de riesgo **Crítico** |

#### Gauge derecho — Probabilidad de zona de alto riesgo

- Muestra el porcentaje de probabilidad de que el municipio sea clasificado como **zona de alto riesgo**.
- El umbral de clasificación es **65%** (optimizado para maximizar Recall ≥ 0.88).
- La línea punteada amarilla indica el umbral de decisión.

| Probabilidad | Clasificación |
|-------------|--------------|
| < 30% | 🟢 Sin riesgo |
| 30 – 50% | 🟢 Riesgo bajo |
| 50 – 65% | 🟡 Riesgo moderado |
| 65 – 85% | 🟠 **Alto riesgo** ← supera el umbral |
| > 85% | 🔴 **Riesgo crítico** |

> ⚠️ **Nota metodológica:** el modelo de clasificación prioriza el **Recall** (sensibilidad) sobre la precisión. Esto significa que prefiere identificar todos los municipios verdaderamente en riesgo aunque algunos no lo sean (falsos positivos), en lugar de dejar pasar municipios críticos sin detectar (falsos negativos). Esta elección es deliberada en el contexto de salud pública.

#### Contexto histórico

Debajo de los gauges aparecen dos paneles:

- **Mapa del departamento:** muestra la ubicación del municipio en el departamento y la tasa de los municipios vecinos para el último año disponible.
- **Evolución histórica:** línea temporal con la tasa de mortalidad real del municipio desde el inicio del registro hasta el último año disponible, incluyendo el promedio móvil de 3 años (línea punteada amarilla).

---

## 5. Vista 📊 Modelos

Muestra la comparativa de todos los modelos evaluados durante el proceso de entrenamiento.

### 5.1 Pestaña Regresión

- **Barra de métricas:** RMSE (menor es mejor) y R² (mayor es mejor) para cada modelo.
  - Barras en **verde**: el modelo cumple el umbral mínimo aceptado (RMSE ≤ 15 | R² ≥ 0.75).
  - Barras en **naranja/rojo**: el modelo no cumple el umbral.
- **Tabla expandible:** al hacer clic en *"Ver tabla de métricas completa"* se muestra la tabla detallada con RMSE, MAE, R², MAPE% y correlación de Pearson para todos los modelos. La celda con mejor valor de cada columna aparece resaltada en verde.

**Modelos evaluados (regresión):** Ridge · ElasticNet · RandomForest · ExtraTrees · XGBoost · LightGBM · HistGradientBoosting · MLP · LSTM

### 5.2 Pestaña Clasificación

- **Gráfico de barras agrupadas:** AUC, Recall y F1 para cada modelo, con líneas punteadas que marcan los umbrales mínimos aceptados.
- **Tabla expandible:** métricas completas (AUC, Recall, F1, Precision, Accuracy) con resaltado del mejor valor.

**Modelos evaluados (clasificación):** LogisticRegression · RandomForest · ExtraTrees · XGBoost · LightGBM · MLP

---

## 6. Vista ℹ️ Sobre el modelo

Documentación técnica del proyecto, organizada en dos columnas:

### Columna izquierda
- **Objetivo:** descripción de los dos problemas de ML y sus variables objetivo.
- **Mejores resultados:** métricas exactas de los modelos ganadores (ExtraTrees y LightGBM).
- **Fuentes de datos:** tabla de las 7 fuentes integradas con su cobertura temporal.

### Columna derecha
- **Variables más importantes:** descripción de las 34 features agrupadas por categoría (lags temporales, capacidad del sistema, contexto geográfico).
- **Stack técnico:** versiones exactas de todas las librerías usadas.
- **Metodología de split temporal:** explicación del enfoque out-of-time para evitar data leakage.

---

## 7. Preguntas frecuentes

**¿Por qué tarda tanto el primer acceso?**  
Los modelos se descargan desde Hugging Face Hub (≈5 MB en total) en el primer arranque. En sesiones posteriores están en caché y la carga es casi inmediata.

**¿Por qué los campos del formulario ya tienen valores cuando selecciono un municipio?**  
El sistema pre-llena los campos con la mediana histórica real del municipio extraída del dataset. Estos valores son una buena estimación de partida; puede modificarlos para hacer análisis de escenarios.

**¿Qué significa que el modelo tiene R² = 0.978?**  
El modelo explica el 97.8% de la variabilidad de la tasa de mortalidad en el conjunto de prueba. Un valor de 1.0 sería predicción perfecta; valores por encima de 0.75 son considerados buenos en este contexto.

**¿Por qué el umbral de clasificación es 65% y no 50%?**  
El umbral se calibró en el entrenamiento para maximizar el Recall (sensibilidad). Con umbral 65% el modelo alcanza Recall = 0.941, lo que significa que detecta el 94.1% de los municipios verdaderamente en zona crítica. Bajar el umbral aumentaría las falsas alarmas; subirlo dejaría pasar municipios críticos sin detectar.

**¿Los datos son en tiempo real?**  
No. El dataset más reciente integrado es hasta 2024 (proyecciones DANE). Las predicciones para 2025–2027 son proyecciones del modelo basadas en los parámetros ingresados por el usuario.

**¿Con qué frecuencia se actualizan los modelos?**  
Los modelos se actualizarán cuando estén disponibles nuevas fuentes de datos oficiales del DANE, RIPS o SGP.

**¿Puedo usar el dashboard sin conexión a Internet?**  
No. La primera carga requiere conexión para descargar los modelos. Después de la carga inicial, la sesión activa funciona sin descargas adicionales.

---

## 8. Glosario

| Término | Definición |
|---------|-----------|
| **AUC** | Área bajo la curva ROC. Mide la capacidad de discriminación del clasificador. 1.0 = perfecto; 0.5 = aleatorio. |
| **BDUA** | Base de Datos Única de Afiliados al Sistema General de Seguridad Social en Salud. |
| **CIE-10** | Clasificación Internacional de Enfermedades, décima edición. Sistema de codificación de causas de muerte. |
| **DANE** | Departamento Administrativo Nacional de Estadística. |
| **DIVIPOLA** | División político-administrativa de Colombia. Codificación oficial de departamentos y municipios. |
| **F1** | Media armónica entre Precisión y Recall. Balance entre ambas métricas. |
| **Lag** | Valor de una variable en un período anterior (ej: LAG1 = valor del año anterior). |
| **MAE** | Error Absoluto Medio. Promedio del error de predicción en las mismas unidades que el target. |
| **MAPE** | Error Porcentual Absoluto Medio. Error relativo expresado como porcentaje. |
| **Mortalidad evitable** | Muertes que podrían haberse prevenido con acceso oportuno y adecuado al sistema de salud. |
| **Percentil 80 (P80)** | Valor por encima del cual se encuentran el 20% de los municipios con mayor mortalidad. Umbral para "zona de alto riesgo". |
| **R²** | Coeficiente de determinación. Proporción de varianza del target explicada por el modelo. |
| **Recall** | Tasa de verdaderos positivos. De todos los municipios realmente críticos, cuántos identifica el modelo. |
| **RIPS** | Registros Individuales de Prestación de Servicios de Salud. |
| **RMSE** | Raíz del Error Cuadrático Medio. Penaliza errores grandes; en las mismas unidades que el target. |
| **SGP** | Sistema General de Participaciones. Recursos transferidos del Gobierno nacional a municipios para salud, educación y saneamiento. |
| **SIVIGILA** | Sistema de Vigilancia en Salud Pública del Instituto Nacional de Salud. |
| **Split temporal** | División del dataset en entrenamiento (años históricos) y prueba (años recientes) sin mezclar períodos, para evaluar el modelo como si predijera el futuro real. |
| **Tasa / 100k** | Número de eventos (muertes) por cada 100.000 habitantes. Permite comparar municipios de distinto tamaño poblacional. |
