# Databricks notebook source
# MAGIC %md
# MAGIC # Avance 5 · Modelo final: ensambles, ajuste fino y evaluación en datos no vistos
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Entregable `Avance5.43`**
# MAGIC
# MAGIC **Autores:** Daniela Juárez · Kevin Valdez · Yael Márquez
# MAGIC
# MAGIC **Problema.** Pronosticar la venta diaria en pesos de cada **tienda × categoría** para los **14 días siguientes**
# MAGIC (horizonte h = 1…14) usando sólo información disponible el día en que se pronostica (el **origen** T). La métrica
# MAGIC principal es el **WMAPE** = Σ|real − pronóstico| / Σ real: pondera cada error por la venta, que es lo que importa para
# MAGIC surtir. Se reportan además el **sesgo** (Σ pronóstico / Σ real − 1), el MAE y el RMSE relativos a la referencia,
# MAGIC el error en mercancía (lo que se surte físicamente) y los tiempos de entrenamiento e inferencia.
# MAGIC
# MAGIC **Qué hace esta libreta.** Parte de los mejores modelos individuales del Avance 4, les ajusta hiperparámetros y
# MAGIC construye ensambles homogéneos y heterogéneos; elige el modelo final con criterios de negocio **antes** de ver los
# MAGIC datos de prueba y después lo evalúa una sola vez en septiembre (datos no vistos). Termina con intervalos de
# MAGIC pronóstico (P10/P50/P90), una compuerta de aceptación, el registro del modelo y la tabla de pronóstico operativo.
# MAGIC
# MAGIC | Clave | Modelo | Tipo | Estrategia |
# MAGIC | --- | --- | --- | --- |
# MAGIC | `B0_Ingenuo` | Ingenuo estacional | referencia | venta del mismo día de la semana más reciente ≤ T |
# MAGIC | `I1_LightGBM_Tweedie` | LightGBM Tweedie | individual (mejor del Avance 4) | boosting de árboles, **ajustado con Optuna** |
# MAGIC | `I2_CatBoost_Tweedie` | CatBoost Tweedie | individual | boosting de árboles simétricos |
# MAGIC | `I3_ElasticNet` | Elastic Net | individual | lineal regularizado sobre log(1 + venta) |
# MAGIC | `I4_XGBoost_Tweedie` | XGBoost Tweedie | individual (nuevo) | boosting de árboles, **ajustado con Optuna** |
# MAGIC | `I5_LightGBM_Log1p` | LightGBM log1p | individual (nuevo) | boosting sobre log(1 + venta) con corrección de escala |
# MAGIC | `I6_Theta` | Theta | individual (opcional) | estadístico por serie (apagado por defecto: costo de cómputo) |
# MAGIC | `E1_Bagging` | Bagging de LightGBM | ensamble homogéneo | 3 LightGBM en submuestras distintas, promediados |
# MAGIC | `E2_AdaBoost` | AdaBoost | ensamble homogéneo | boosting adaptativo de árboles de decisión |
# MAGIC | `E7_Bosque_Aleatorio` | Bosque aleatorio | ensamble homogéneo | bagging de árboles con submuestreo de variables |
# MAGIC | `E3_Blending` | Blending | ensamble heterogéneo | pesos convexos aprendidos en el bloque observado más reciente |
# MAGIC | `E4_Stacking_temporal` | Stacking temporal | ensamble heterogéneo | pesos no negativos por horizonte + corrección de nivel por tipo |
# MAGIC | `E5_Votacion` | Votación | ensamble heterogéneo | promedio de las 3 bases con menor error reciente |
# MAGIC | `E6_Stacking_LightGBM` | Stacking no lineal | ensamble heterogéneo | meta-modelo LightGBM sobre las predicciones de las bases |
# MAGIC | `D1_Stacking_sin_correccion` | Diagnóstico | diagnóstico | E4 sin la corrección de nivel: mide cuánto aporta esa corrección |
# MAGIC
# MAGIC Las bases de los ensambles heterogéneos son los individuales (`I1`–`I6`), entre ellos los mejores del Avance 4.
# MAGIC
# MAGIC **Protocolo temporal (sin fuga de información).** Cada modelo de un origen T se entrena sólo con filas cuyo
# MAGIC objetivo es ≤ T, y cada ensamble aprende sus pesos sólo con bloques que ya terminaron antes de T.
# MAGIC
# MAGIC | Bloque | Origen | Días pronosticados | Uso |
# MAGIC | --- | --- | --- | --- |
# MAGIC | `ajuste` | 20-may | 21-may a 03-jun | Optuna, número de árboles y primeros pesos de los ensambles |
# MAGIC | `corte_1` … `corte_4` | 03-jun, 17-jun, 01-jul, 17-ago | 4 bloques de 14 días | **desarrollo**: comparar y elegir (los mismos del Avance 4) |
# MAGIC | `prueba_1`, `prueba_2` | 31-ago, 14-sep | 01–14 y 15–28 de septiembre | **datos no vistos**: el modelo elegido, congelado con datos al 31-ago, pronostica septiembre |
# MAGIC | `pronostico` | último día con datos | 14 días siguientes | salida operativa (sin objetivo todavía) |
# MAGIC
# MAGIC **Dónde se cumple cada criterio de la rúbrica**
# MAGIC
# MAGIC | Criterio | Sección |
# MAGIC | --- | --- |
# MAGIC | Ensambles homogéneos y heterogéneos (≥ 4); stacking/blending con los mejores individuales; optimización de hiperparámetros | §2.4–2.7 (definición), §4 (ejecución), §5.4 (Optuna), §5.5 (pesos) |
# MAGIC | Tabla comparativa ordenada por la métrica principal, con otras métricas y tiempos de entrenamiento | §5.2 y §5.3 |
# MAGIC | Elección del modelo final alineada con el negocio | §5.7 (matriz de decisión) y §6.5 (compuerta) |
# MAGIC | Gráficos del modelo final con interpretación | §7 (siete gráficos, cada uno con su lectura calculada) |
# MAGIC | Objetivo 3.6: evaluación en datos no vistos | §5.3, §6 y §7 (septiembre) |
# MAGIC | Declaración sobre el uso de herramientas de IA | última celda |
# MAGIC
# MAGIC **Confidencialidad.** Con `MODO_PUBLICO = True` las salidas sólo muestran métricas relativas, tiempos,
# MAGIC hiperparámetros, pesos y ventas indexadas; nunca importes en pesos, volúmenes ni identificadores de tienda. Los datos
# MAGIC y las tablas con detalle permanecen en el catálogo privado de Databricks.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.features_origen` (versión Delta fija) · `silver.panel_categoria_dia` (sólo si `I6_Theta` está activo) · `mlops.validaciones` |
# MAGIC | **Escribe** | `mlops.av5_*` (resultados agregados por corrida) · `gold.pronostico_14d` · `mlops.monitoreo_modelo` · MLflow · `demanda_categoria` en Unity Catalog (alias `@challenger`) |
# MAGIC | **Tiempo** | `MODO_RAPIDO = True`: 20–40 min · completo: 3.5–5 h con 16 GB de RAM (conviene correrlo como Job) |
# MAGIC
# MAGIC **Cómo correrla.** (1) Con `MODO_RAPIDO = True`, «Run all»: comprueba que todo funciona. (2) Con
# MAGIC `MODO_RAPIDO = False`, como Job: son las cifras del reporte. (3) Si la corrida se interrumpe, escribe su
# MAGIC identificador en `RUN_ID_REANUDAR` y vuelve a correr: continúa desde el último bloque terminado. (4) Para volver a
# MAGIC ver resultados sin entrenar, `EJECUTAR_ENTRENAMIENTO = False` (y opcionalmente `RUN_ID_A_MOSTRAR`).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Bibliotecas
# MAGIC Serverless no trae LightGBM, XGBoost, CatBoost, Optuna ni SHAP. Cada una se instala en su propia celda (una
# MAGIC instalación con varias bibliotecas en una sola línea falló en el Avance 4) y al final se reinicia Python para que
# MAGIC queden disponibles. La sección 3 imprime las versiones efectivas y las registra en MLflow.

# COMMAND ----------

# MAGIC %pip install "lightgbm>=4.3,<5"

# COMMAND ----------

# MAGIC %pip install "xgboost>=2.0,<4"

# COMMAND ----------

# MAGIC %pip install "catboost>=1.2,<2"

# COMMAND ----------

# MAGIC %pip install "optuna>=3.6,<5"

# COMMAND ----------

# MAGIC %pip install "shap>=0.44,<1"

# COMMAND ----------

try:
    dbutils.library.restartPython()      # necesario después de %pip en Databricks; fuera de Databricks no existe y se ignora
except Exception:
    pass
