# Databricks notebook source
# MAGIC %md
# MAGIC # Pronóstico de demanda para Tiendas SIX · Equipo 43
# MAGIC **Proyecto Integrador · Maestría en Inteligencia Artificial Aplicada (MNA), Tecnológico de Monterrey**
# MAGIC
# MAGIC Daniel Juárez Ortiz · Kevin Alexis Valdez Pérez · Yael Ariel Márquez Mas
# MAGIC Asesora: Dra. Eduviges Ludivina Facundo · Patrocinador: Dr. Juan Arturo Nolazco Flores · Empresa: Bnext / Tiendas SIX
# MAGIC
# MAGIC **Objetivo.** Pronosticar la venta diaria por tienda y categoría de 1 a 14 días hacia adelante, con un rango
# MAGIC (P10–P50–P90) y no sólo un número, y operar el modelo con MLOps en Databricks: datos versionados, validación
# MAGIC automática, registro de modelos, compuerta de calidad, inferencia programada y monitoreo.
# MAGIC
# MAGIC **¿Cómo vamos?** Abre el notebook **`ESTADO_DEL_PROYECTO`** de esta carpeta: muestra qué etapas se han corrido,
# MAGIC las cifras clave y los hallazgos más recientes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Organización de esta carpeta
# MAGIC
# MAGIC ```
# MAGIC /Workspace/Shared/mna_equipo43/
# MAGIC ├── LEEME                      ← esta portada
# MAGIC ├── ESTADO_DEL_PROYECTO        ← avance, cifras clave y hallazgos (sólo lee)
# MAGIC ├── 1_datos/                   ← fase 1: entender, auditar y limpiar los datos
# MAGIC │   ├── 00_entorno_y_permisos
# MAGIC │   ├── 01_inventario_catalogo
# MAGIC │   ├── 02_carga_bronce
# MAGIC │   ├── 03_auditoria_trabajo_previo
# MAGIC │   ├── 04_eda_calidad
# MAGIC │   ├── 05_eda_tiempo_y_tiendas
# MAGIC │   ├── 06_eda_variables_y_promociones
# MAGIC │   ├── 07_eda_externas
# MAGIC │   ├── 08_limpieza_plata
# MAGIC │   ├── 09_panel_diario
# MAGIC │   ├── 10_eda_series_y_linea_base
# MAGIC │   ├── 11_validacion_calidad
# MAGIC │   └── 12_evidencia_calidad      ← parquets para Bnext en el volumen maestria.hallazgos_calidad
# MAGIC ├── 2_modelos/                 ← fase 2: variables por origen y modelos
# MAGIC │   ├── 20_features_oro           ← tabla Oro (serie × origen × horizonte) con prueba anti-fuga
# MAGIC │   ├── 21_avance4_modelos_alternativos ← Avance 4: torneo de modelos individuales, ajuste fino y modelo final
# MAGIC │   └── Avance5.43                ← Avance 5: ajuste fino, ensambles homogéneos y heterogéneos, prueba no vista (septiembre), intervalos, compuerta y pronóstico operativo
# MAGIC └── 3_mlops/                   ← fase 3: registro, compuerta, inferencia, monitoreo y jobs
# MAGIC ```
# MAGIC
# MAGIC Cada notebook trae su código completo (no depende de otros archivos), explica arriba qué preguntas responde, qué lee
# MAGIC y qué escribe, y en cada sección dice **qué hace** y **qué mirar**. Todos terminan con un **RESUMEN PARA REVISIÓN**
# MAGIC y guardan sus hallazgos en `mlops.hallazgos`. Los notebooks conservan sus resultados: se pueden consultar sin volver a correrlos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Datos
# MAGIC
# MAGIC **Fuente (Bnext, sólo lectura).** Catálogo `modeladoprescriptivoventas`, esquema `maestria`:
# MAGIC
# MAGIC | Volumen | Contenido |
# MAGIC | --- | --- |
# MAGIC | `/Volumes/modeladoprescriptivoventas/maestria/2026-03` … `2026-08` | Líneas de venta de cada mes (muchos parquet por mes) |
# MAGIC | `/Volumes/modeladoprescriptivoventas/maestria/processed` | Fuentes externas (tiendas, códigos postales, INEGI, clima, calendarios) y el ABT del equipo anterior |
# MAGIC | `/Volumes/modeladoprescriptivoventas/maestria/outputs` | Modelo campeón heredado y sus parámetros |
# MAGIC
# MAGIC **Tablas del equipo** (arquitectura medallón). Viven en `modeladoprescriptivoventas.maestria` con el prefijo de su capa,
# MAGIC porque el equipo puede crear tablas, modelos y volúmenes en `maestria` pero no esquemas nuevos:
# MAGIC
# MAGIC | Capa | Tabla en `maestria` | Qué es | La crea |
# MAGIC | --- | --- | --- | --- |
# MAGIC | Bronce | `bronze_ventas_lineas` | Los seis meses tal como llegan, con el archivo de origen de cada fila | `02` |
# MAGIC | Plata | `silver_ventas_lineas` | Líneas limpias, únicas y homologadas | `08` |
# MAGIC | Plata | `silver_catalogo_categorias`, `silver_catalogo_marcas`, `silver_catalogo_tiendas` | Dimensiones homologadas | `08` |
# MAGIC | Plata | `silver_panel_categoria_dia` | Una fila por tienda, categoría y día, con relleno acotado a la ventana de la tienda, días sin datos y atípicos marcados, tipo de categoría y promociones vigentes | `09` |
# MAGIC | Plata | `silver_promociones`, `silver_calendario_promociones` | Vigencia de cada promoción (inicio, fin, censura) y sus tiendas y categorías | `09` |
# MAGIC | Oro | `gold_*` | Variables para el modelo, predicciones a 14 días y escenarios (fase 2) | — |
# MAGIC | MLOps | `mlops_bronce_cargas`, `mlops_conciliacion_plata`, `mlops_validaciones` | Bitácoras de carga, conciliación Bronce → Plata y contratos de calidad | `02`, `08`, `11` |
# MAGIC | MLOps | `mlops_eda_*`, `mlops_auditoria_previo`, `mlops_linea_base_ingenua*`, `mlops_comparacion_modelo_heredado`, `mlops_hallazgos` | Exploración, auditoría, línea base y hallazgos | `03`–`10` |
# MAGIC | Volumen | `/Volumes/modeladoprescriptivoventas/maestria/hallazgos_calidad/` | Parquets con las líneas detrás de los hallazgos de calidad (importes imposibles, duplicados, venta en cero, % de descuento) y su `LEEME.md`, para que Bnext los revise | `12` |
# MAGIC | Modelos | `maestria.demanda_categoria` (registro de Unity Catalog) | Modelo campeón con alias `champion` / `challenger` (fase 3) | — |
# MAGIC
# MAGIC En el texto de cada notebook las tablas se nombran por capa (`bronze.ventas_lineas`); el nombre real es el de esta tabla.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fases del proyecto
# MAGIC
# MAGIC | Fase | Qué incluye | Estado |
# MAGIC | --- | --- | --- |
# MAGIC | **1 · Datos** | Inventario, carga de los seis meses, auditoría del trabajo previo, EDA, limpieza, panel diario, línea base ingenua y contratos de calidad | En ejecución |
# MAGIC | **2 · Modelos** | Variables sin fuga para h = 1…14, línea base LightGBM reentrenada con los seis meses, torneo de pérdidas (Tweedie, L1, Poisson), cuantiles P10/P90 calibrados, validación walk-forward | Siguiente |
# MAGIC | **3 · MLOps** | MLflow y registro en Unity Catalog (`champion` / `challenger`), compuerta de calidad, inferencia diaria a 14 días, monitoreo de deriva y error, jobs programados | Después |
# MAGIC | **4 · Consumo** | Tablero AI/BI y escenarios de precio y promoción (what-if) | Si alcanza el tiempo |
# MAGIC
# MAGIC **Trabajo previo.** El equipo anterior (modelo LightGBM, marzo–mayo, WMAPE 0.5002) y los Avances 2 y 3 se hicieron en
# MAGIC el servidor de Bnext y en computadoras locales. Aquí se usan sólo como referencia: el notebook `03` verifica sus cifras
# MAGIC contra los datos de Databricks y todo lo demás se rehace aquí con los seis meses.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cómo correr la fase 1
# MAGIC 1. Cómputo: **serverless** (el que usa el equipo) o un clúster con Unity Catalog.
# MAGIC 2. Abrir `1_datos/00_entorno_y_permisos` → **Run all**: confirma permisos, volúmenes y librerías.
# MAGIC 3. Correr en orden del `01` al `11`. Cada uno tarda de 5 a 60 minutos; el `02` y el `08` son los más pesados.
# MAGIC 4. Abrir `ESTADO_DEL_PROYECTO` → **Run all** para ver el avance.
# MAGIC
# MAGIC **Configuración.** La primera celda de código de cada notebook es idéntica en todos: catálogo, meses, esquemas y la
# MAGIC columna que define el día de la venta. Si algo cambia (por ejemplo, llega un mes nuevo), se cambia ahí en todos.
# MAGIC
# MAGIC **Privacidad (NDA).** Los notebooks sólo imprimen totales, porcentajes y distribuciones; nunca registros individuales
# MAGIC ni ubicaciones de tiendas.
