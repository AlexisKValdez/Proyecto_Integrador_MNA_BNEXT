# Pipeline en Azure Databricks — Equipo 43 (`mna_equipo43_avance6`)

Notebooks del proyecto en formato fuente de Databricks. Se importan en una carpeta del workspace (p. ej.
`/Workspace/Users/<usuario>/mna_equipo43_avance6/` o `/Workspace/Shared/mna_equipo43_avance6/`) y corren con cómputo
**serverless**. Cada notebook trae dentro su configuración y funciones de apoyo (no hay paquetes aparte). Las tablas del
equipo viven en `modeladoprescriptivoventas.maestria` con prefijo de capa (`bronze_*`, `silver_*`, `gold_*`, `mlops_*`).

## Notebooks

| # | Notebook | Qué hace | Parámetros de Job |
| --- | --- | --- | --- |
| 00 | `1_datos/00_entorno_y_permisos` | Revisa cómputo, bibliotecas y permisos de escritura | — |
| 01 | `1_datos/01_inventario_catalogo` | Inventario del catálogo: tablas, volúmenes, archivos por mes, experimentos | — |
| 02 | `1_datos/02_carga_bronce` | Carga los parquet mensuales a Bronce con trazabilidad | `MESES_A_CARGAR` |
| 04–07 | `1_datos/04…07_eda_*` | Análisis exploratorio: calidad, tiempo y tiendas, variables y promociones, externas | — |
| 08 | `1_datos/08_limpieza_plata` | Reglas de limpieza, deduplicación entre meses, homologación | `MESES_A_LIMPIAR` |
| 09 | `1_datos/09_panel_diario` | Panel tienda × categoría × día; promociones observadas | — |
| 10 | `1_datos/10_eda_series_y_linea_base` | Series y línea base ingenua en los cortes walk-forward fijos | — |
| 11 | `1_datos/11_validacion_calidad` | Reglas de calidad (`mlops_validaciones`); en Jobs detiene el pipeline si falla una crítica | `DETENER_SI_FALLA` |
| 13 | `1_datos/13_calendario_promociones_plan` | Excel/CSV editable del volumen → `silver_calendario_promociones_plan[_dia]` (futuras + histórico observado) | `ARCHIVO_CALENDARIO`, `INCLUIR_HISTORICO_OBSERVADO`, `FALLAR_SI_INVALIDO` |
| 20 | `2_modelos/20_features_oro` | Tabla Oro: serie × origen × horizonte, variables sólo con días ≤ origen, prueba anti-fuga, comentarios y constraints | `GENERAR_PRONOSTICO` |
| 21 | `2_modelos/21_avance4_modelos_alternativos` | Avance 4: torneo de modelos individuales (histórico) | — |
| A5 | `2_modelos/Avance5.43` | Avance 5: ajuste fino, ensambles, selección, prueba no vista, modelo final, `@challenger`, pronóstico | `MODO_RAPIDO`, `REENTRENAR_FINAL`, `RUN_ID_REANUDAR`, `USAR_CALENDARIO_PROMOS`, `REGISTRAR_EN_UC`, `INCLUIR_Q_LIGHTGBM`, … |
| 22 | `3_operacion/22_pronostico_operativo` | Inferencia con el modelo registrado → `gold_pronostico_14d` | `ALIAS_MODELO`, `PERMITIR_CHALLENGER`, `RUN_ID_MODELO`, `ORIGEN_PRONOSTICO` |
| 23 | `3_operacion/23_monitoreo` | Acierto en producción, PSI, frescura y alertas | `UMBRAL_ERROR_PP`, `UMBRAL_SESGO_PCT`, `UMBRAL_PSI_ALERTA`, `DIAS_FRESCURA` |
| 24 | `3_operacion/24_promocion_modelo` | `@challenger` → `@champion` con evidencia y aprobaciones de negocio | `ACCION`, `TOLERANCIA_WMAPE`, `FORZAR` |
| 30 | `3_operacion/30_simulacion_escenarios_whatif` | Escenarios de promoción re-puntuando con el modelo; uplift histórico | `ALCANCE_ESCENARIOS`, `DESCUENTOS_MALLA` |
| 31 | `3_operacion/31_desagregacion_predicciones_sku` | Reparto a producto (participación 28 d, fila OTROS, reconciliación) y vista semanal | `VENTANA_DIAS_SHARE`, `TOP_SKUS_POR_CATEGORIA` |
| 40 | `3_operacion/40_salidas_dashboard` | Tablas `gold_dash_*` para el tablero AI/BI | `DIAS_HISTORIA`, `UMBRAL_CAMBIO_SERIE` |
| — | `LEEME`, `ESTADO_DEL_PROYECTO` | Portada y página de estado (sólo lee) | — |

Los notebooks `03_auditoria_trabajo_previo` y `12_evidencia_calidad` contienen cifras del negocio y sólo viven en el
espacio privado de Databricks.

## Configuración común

La primera celda de cada notebook es la misma: catálogo y esquema, `MESES` (`2026-03` … `2026-09`), los cortes
walk-forward fijos y `parametro()`, que lee los parámetros de Job o widgets con el mismo nombre que las variables.

| Bloque | Origen | Uso |
| --- | --- | --- |
| `ajuste` | 2026-05-20 | ajuste de hiperparámetros |
| `corte_1` … `corte_4` | 06-03, 06-17, 07-01, 08-17 | desarrollo (los del Avance 4) |
| `prueba_1`, `prueba_2` | 08-31, 09-14 | datos no vistos (septiembre) |

## Orden

- **Mes nuevo**: J1 (02 → 08 → 09 → 10 → 11) → J2 (13 → 20) → J4 (22 → 30 → 31 → 40) → J5 (23 → 40).
- **Reentrenar**: J3 (`Avance5.43`) → 24 → J4.
- Definiciones de los Jobs y parámetros: `jobs/LEEME.md`. Operación paso a paso: `docs/RUNBOOK.md`.

## Pruebas sin datos de Bnext

`pruebas/` trae un generador de datos sintéticos y un harness que ejecuta los notebooks en Spark local simulando
serverless (`dbutils` y widgets emulados). Útil para probar cambios antes de subirlos.
