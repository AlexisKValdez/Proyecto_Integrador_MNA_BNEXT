# Auditoría de código · Pipeline en Databricks — Equipo 43 (noviembre de 2026)

Alcance: los 16 notebooks exportados de Databricks el 2-nov (`mna_equipo43_avances5y6`), las dos portadas (`LEEME`,
`ESTADO_DEL_PROYECTO`) y las celdas comunes que comparten. Para cada uno: qué hace, qué se encontró y qué se cambió en
esta versión (`mna_equipo43_avance6`). Severidad: **A** afecta resultados o datos · **B** operación/MLOps · **C** forma.

## 0. Comparación con lo que corre en Databricks

Los notebooks de Databricks son idénticos a los del repositorio (30-oct) salvo el encabezado del entorno serverless y
cuatro parámetros de `Avance5.43` (`MODO_RAPIDO = False`, `RUN_ID_REANUDAR` fijo, `REENTRENAR_FINAL = False`,
`INCLUIR_Q_LIGHTGBM = False`). No había cambios de código que rescatar; sí dos consecuencias de cómo se corrió:
sin reentrenamiento final no existen `gold.pronostico_14d` ni el modelo en Unity Catalog, y si septiembre no pasó por
08 → 20 la Oro no tiene los bloques de prueba. Ambas se resuelven con J1 → J2 → J3 (ver `jobs/LEEME.md`).

## 1. Celdas comunes (configuración, apoyo, columnas, series)

| Hallazgo | Sev. | Cambio |
| --- | --- | --- |
| Parámetros como constantes: para correr en Jobs había que editar celdas | B | `parametro(nombre, defecto)`: lee el widget/parámetro del Job con el mismo nombre; sin Job vale lo del código |
| No había bitácora de tiempos por notebook | B | `cerrar()` escribe `mlops.corridas` (inicio, fin, minutos, hallazgos, avisos) |
| Sin constraints ni comentarios en las tablas | B | `asegurar_restricciones()` (CHECK de Delta) y `comentar_columnas()` (diccionario en Catalog Explorer) |
| Escrituras `append` por `run_id` no eran idempotentes | A | `borrar_corrida()` antes de anexar (Avance5.43: 25 tablas, monitoreo e interpretaciones) |
| `Cronometro` sólo existía en Avance5.43 | C | Pasó a las celdas de apoyo |
| Textos «seis meses», «07_limpieza_plata», autoría distinta en LEEME | C | Corregidos (número de meses calculado desde `MESES`) |

## 2. Fase de datos (1_datos)

| Notebook | Qué hace | Hallazgos | Cambio |
| --- | --- | --- | --- |
| `00_entorno_y_permisos` | cómputo, librerías, permisos | Correcto. Texto «seis meses» | C: texto |
| `01_inventario_catalogo` | inventario de tablas, volúmenes, MLflow | Correcto | — |
| `02_carga_bronce` | parquet mensuales → Bronce con trazabilidad | Sólo se parametrizaba editando `MESES_A_CARGAR` | B: `parametro("MESES_A_CARGAR")` (Job J1 pasa el mes) |
| `04`–`07` EDA | calidad, tiempo/tiendas, variables/promociones, externas | Correctos; el 04 citaba `07_limpieza_plata` | C: referencia |
| `08_limpieza_plata` | reglas, deduplicación entre meses, homologación | `MESES_A_LIMPIAR` constante | B: parámetro |
| `09_panel_diario` | panel tienda × categoría × día, promociones observadas | Correcto. Escribe `silver.calendario_promociones` (observada); el 30 anterior la confundía con el plan | A (en el 30): el plan vive ahora en `silver.calendario_promociones_plan*` |
| `10_eda_series_y_linea_base` | series, línea base ingenua en cortes fijos | Correcto | C: textos |
| `11_validacion_calidad` | reglas de calidad → `mlops.validaciones` | No detenía el pipeline al fallar una regla crítica | B: `DETENER_SI_FALLA` como parámetro (J1 lo pone en True) |
| **`13_calendario_promociones_plan`** (nuevo) | Excel/CSV editable → `silver.calendario_promociones_plan` y `_plan_dia` | — | Validación (categorías, alcances, fechas, descuentos), `fecha_publicacion` para evitar fugas, histórico observado como plan, bitácora de versiones, catálogo de referencia junto al archivo |

## 3. Fase de modelos (2_modelos)

| Notebook | Hallazgos | Cambio |
| --- | --- | --- |
| `20_features_oro` | Correcto (anti-fuga, cortes fijos, propiedades). Sin comentarios por columna ni constraints | B: comentarios del diccionario en las 75 columnas; CHECK de `h`, `y ≥ 0`, `conjunto`; `GENERAR_PRONOSTICO` como parámetro |
| `21_avance4_modelos_alternativos` | Histórico del Avance 4; no se toca | — |
| `Avance5.43` | (a) publicación por `run_id` no idempotente → reanudar duplicaba tablas **A**; (b) firma del modelo con categóricas enteras y numéricas float32: la inferencia externa tenía que reproducir la codificación **A**; (c) calendario de promociones leído de la tabla observada y sin fecha de publicación (posible fuga en cortes) **A**; (d) parámetros sólo editando celdas **B**; (e) `psutil` ausente → excepción en serverless (corregido 30-oct) | (a) `borrar_corrida` antes de anexar; (b) firma con categóricas en **texto** y numéricas `double`: el pyfunc codifica con su vocabulario; (c) `f_*` desde `silver.calendario_promociones_plan_dia` con `fecha_publicacion ≤ origen`; `USAR_CALENDARIO_PROMOS = True` por defecto (si no hay tabla, sigue sin ellas y avisa); (d) widgets para los interruptores de 1.1, `REGISTRAR_EN_UC`, `INCLUIR_Q_LIGHTGBM`; §9 reescrita |

## 4. Operación y Avance 6 (3_operacion)

| Notebook | Hallazgos en la versión anterior | Versión nueva |
| --- | --- | --- |
| **`22_pronostico_operativo`** (nuevo) | No existía: pronosticar obligaba a reentrenar (4–6 h) | Carga `demanda_categoria@champion` (o `@challenger`, o el run de MLflow), puntúa el conjunto `pronostico` de la Oro por lotes, une el plan de promociones y escribe `gold.pronostico_14d` reemplazando sólo su origen; bitácora en `mlops.pronosticos_bitacora`; `fase = retrospectivo` para medir acierto en orígenes pasados |
| **`23_monitoreo`** (nuevo) | El monitoreo sólo existía como tabla de la fase de prueba | Acierto en producción (semana × región × tipo, por horizonte), PSI de 12 variables de la serie y cambio de media de 4 de contexto contra los bloques de evaluación recientes (8 semanas), frescura, pronóstico desactualizado, regla de reentrenar; `mlops.alertas` |
| **`24_promocion_modelo`** (nuevo) | Aprobaciones de negocio eran constantes (`EVIDENCIA_NEGOCIO`) y nadie movía el alias | Compuerta técnica + comparación con el campeón + `mlops.aprobaciones` (quién, cuándo) → alias `@champion` con bitácora |
| `30_simulacion_escenarios_whatif` | **A** Si no existía el pronóstico, fabricaba uno (muestra del torneo, mezcla 0.65/0.35, P10/P90 fijos, `run_id` inventado) y lo **escribía en la tabla oficial**. **A** Elasticidades constantes (1.30/0.80/0.40) y hallazgo con cifras fijas. **A** Duplicados por promociones traslapadas. **A** Leía el plan de la tabla observada. **B** Imprimía pesos. **C** No usaba las celdas comunes | Reescrito: escenarios = cambiar `f_*` y re-puntuar con el **modelo registrado** (S_PLAN, S0 sin promo, S1/S2/S3); `gold.uplift_promocional_historico` como referencia empírica (cota superior declarada) y respaldo si el modelo no usa `f_*`; sin respaldos que escriban pronósticos; `MODO_PUBLICO`; `gold.escenarios_whatif` + `_resumen` |
| `31_desagregacion_predicciones_sku` | **A** Mismo respaldo. **A** Top-15 renormalizado: la cola se asignaba a los líderes y «reconciliación perfecta» era texto fijo. **A** `inner join` dejaba fuera series sin historia. **B** Ventana de 29 días; filtro sin `anio_mes`; «surtido sugerido» sin inventario; pesos impresos | Reescrito: fila **OTROS** con el residual, reconciliación **calculada** y reportada, `SIN_HISTORIA_SKU` para series sin ventas en la ventana, 28 días exactos con poda por `anio_mes`, `techo_demanda_unidades_p90`, vista semanal `gold.surtido_sku_semana`, `MODO_PUBLICO` |
| **`40_salidas_dashboard`** (nuevo) | No había salidas para un tablero | Siete tablas `gold.dash_*` (resumen, serie de tiempo por nivel, precisión, promociones, alertas de negocio, estado del pipeline) + tablero AI/BI importable |

## 5. Portadas

`LEEME` y `ESTADO_DEL_PROYECTO` actualizados con la carpeta `3_operacion`, las tablas nuevas, la autoría uniforme y el
flujo `@challenger → @champion`.

## 6. Lo que queda igual a propósito

- Cortes walk-forward fijos, protocolo de selección antes de septiembre y métricas (comparables con los Avances 4 y 5).
- Nombres de variables y tablas que el equipo ya usa.
- Notebooks autocontenidos (las celdas comunes se copian al construir; `%run` queda como opción futura).
- `03_auditoria_trabajo_previo` y `12_evidencia_calidad` siguen fuera del repositorio público (contienen cifras del negocio).

## 7. Cambios del 4 de noviembre (octubre cargado y promociones con sólo fechas)

- `MESES` incluye `2026-10`; los bloques de prueba pasan a cuatro (`prueba_3` 28-sep, `prueba_4` 12-oct): la prueba no
  vista cubre del 1-sep al 26-oct con los modelos congelados al 31-ago (mide también cuánto envejece el modelo).
  Eventos de octubre a diciembre agregados (Halloween, Día de Muertos, Buen Fin, Revolución, Guadalupe, Navidad, fin de año).
- Notebook 13: `categoria_id = TODAS` (+ `tipo_categoria`) expande una promoción de la que sólo se conocen las fechas a
  todas las categorías del tipo; `descuento_planeado` vacío se imputa con el descuento histórico mediano de la categoría
  (`descuento_imputado = 1`); bloque **comentado** para cargar el catálogo de Bnext como tabla/archivo
  (`ORIGEN_CATALOGO_BNEXT`, `MAPEO_COLUMNAS_BNEXT`) cuando exista.
- Tablero: `gold.dash_serie_tiempo` trae `pct_series_con_promo` (observada antes del origen, planeada después) y el
  tablero lo grafica junto a la venta.

## 8. Pruebas realizadas

Toda la cadena 00 → 02 → 08 → 09 → 10 → 11 → 13 → 20 → `Avance5.43` → 22 → 23 → 24 → 30 → 31 → 40 corrió en Spark local
con datos sintéticos simulando serverless (sin `persist`, configuraciones restringidas, `dbutils.widgets` emulado con
parámetros de Job). Lo que sólo se puede comprobar en Databricks: registro en Unity Catalog (permisos), tiempos reales,
importación del tablero y creación de Jobs.
