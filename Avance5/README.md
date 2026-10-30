# Avance 5 · Modelo final — Equipo 43

**Proyecto Integrador · Maestría en Inteligencia Artificial Aplicada (MNA) · Tecnológico de Monterrey**
**Autores:** Daniela Juárez · Kevin Valdez · Yael Márquez
**Socio formador:** Tiendas SIX (Bnext) · **Entregable:** `Avance5.43`

Pronóstico de la venta diaria en pesos de cada **tienda × categoría** para los **14 días siguientes**, con ensambles de
modelos de aprendizaje automático, evaluación en datos no vistos (septiembre) y prácticas de MLOps en Azure Databricks.

---

## 1. Contenido de esta carpeta

| Archivo | Qué es |
| --- | --- |
| `Avance5.43.ipynb` | La libreta del avance. Se importa a Databricks, se ejecuta de principio a fin y se vuelve a exportar **con sus salidas** para la entrega (reemplaza a este archivo). |
| `Avance5.43.py` | La misma libreta en formato fuente de Databricks (útil para revisar diferencias en Git o importar como notebook). |
| `README.md` | Esta guía. |

La libreta **no contiene datos**: lee todo de las tablas del catálogo privado de Databricks
(`modeladoprescriptivoventas.maestria`). Con `MODO_PUBLICO = True` sus salidas sólo muestran métricas relativas,
tiempos, hiperparámetros y ventas indexadas, por lo que la versión ejecutada se puede publicar sin exponer información
de Bnext (acuerdo de confidencialidad).

---

## 2. Qué hace la libreta

1. **Parte de los mejores modelos individuales del Avance 4** (LightGBM Tweedie, CatBoost Tweedie, Elastic Net) y agrega
   XGBoost Tweedie, LightGBM sobre log(1 + venta) y, opcionalmente, Theta.
2. **Ajusta hiperparámetros** de los modelos más relevantes (LightGBM y XGBoost) con Optuna en un bloque reservado
   (20 de mayo), con paro temprano para fijar el número de árboles.
3. **Construye ensambles homogéneos** (bagging de LightGBM, AdaBoost, bosque aleatorio) y **heterogéneos** (blending,
   stacking temporal con corrección de nivel, votación y stacking no lineal), todos sin fuga de información.
4. **Compara** todo en los cuatro cortes de desarrollo del Avance 4 con WMAPE (métrica principal), sesgo, MAE y RMSE
   relativos, error en mercancía, error por semana del horizonte y tiempos de entrenamiento e inferencia.
5. **Elige el modelo final** con una matriz de decisión de negocio **antes** de ver septiembre.
6. **Evalúa en datos no vistos:** el modelo, congelado con datos al 31 de agosto, pronostica septiembre (1–28) sin
   reentrenar.
7. **Calcula intervalos P10/P50/P90** (conformal alrededor del modelo final y LightGBM cuantílico como comparación).
8. **Pasa una compuerta de aceptación**, registra el modelo en Unity Catalog (`demanda_categoria`, alias `@challenger`),
   guarda el **pronóstico operativo** de los 14 días siguientes y una tabla de **monitoreo**.
9. Genera **siete gráficos con interpretación calculada** y un resumen para el reporte.

### Modelos

| Clave | Modelo | Tipo | Estrategia |
| --- | --- | --- | --- |
| `B0_Ingenuo` | Ingenuo estacional | referencia | venta del mismo día de la semana más reciente |
| `I1_LightGBM_Tweedie` | LightGBM Tweedie | individual | boosting, ajustado con Optuna |
| `I2_CatBoost_Tweedie` | CatBoost Tweedie | individual | boosting de árboles simétricos |
| `I3_ElasticNet` | Elastic Net | individual | lineal regularizado sobre log(1 + venta) |
| `I4_XGBoost_Tweedie` | XGBoost Tweedie | individual (nuevo) | boosting, ajustado con Optuna |
| `I5_LightGBM_Log1p` | LightGBM log1p | individual (nuevo) | boosting sobre log(1 + venta) con corrección de escala |
| `I6_Theta` | Theta | individual (opcional) | estadístico por serie |
| `E1_Bagging` | Bagging de LightGBM | ensamble homogéneo | 3 LightGBM en submuestras distintas |
| `E2_AdaBoost` | AdaBoost | ensamble homogéneo | boosting adaptativo de árboles |
| `E7_Bosque_Aleatorio` | Bosque aleatorio | ensamble homogéneo | bagging de árboles con submuestreo de variables |
| `E3_Blending` | Blending | ensamble heterogéneo | pesos convexos del bloque observado más reciente |
| `E4_Stacking_temporal` | Stacking temporal | ensamble heterogéneo | pesos no negativos por horizonte + corrección de nivel |
| `E5_Votacion` | Votación | ensamble heterogéneo | promedio de las 3 mejores bases |
| `E6_Stacking_LightGBM` | Stacking no lineal | ensamble heterogéneo | meta-modelo LightGBM |
| `D1_Stacking_sin_correccion` | Diagnóstico | diagnóstico | E4 sin corrección de nivel (mide su aporte) |

### Protocolo temporal

```
 mar ─────────── may ── jun ──────── jul ─┐hueco┌─ ago ────────── sep ───────────── oct
                  │ajuste│c1│c2│      │c3│ 16–31 │     │c4│ p1 │ p2 │
                  20-may                         jul       17-ago 31-ago 14-sep  30-sep
  entrenamiento: sólo filas con objetivo ≤ origen de cada bloque      pronóstico → 1–14 oct
```

| Bloque | Origen | Uso |
| --- | --- | --- |
| `ajuste` | 20-may | Optuna, número de árboles y primeros pesos de los ensambles |
| `corte_1`…`corte_4` | 03-jun, 17-jun, 01-jul, 17-ago | desarrollo: comparar y elegir (los del Avance 4) |
| `prueba_1`, `prueba_2` | 31-ago, 14-sep | datos no vistos: el modelo elegido, congelado al 31-ago, pronostica septiembre |
| `pronostico` | último día con datos | salida operativa: 14 días siguientes |

---

## 3. Qué cambió respecto a la versión anterior de la libreta y por qué

| # | Cambio | Por qué |
| --- | --- | --- |
| 1 | Instalación de bibliotecas en celdas `%pip` (una por biblioteca) y verificación sin abortar | La versión anterior se detenía con «el compute no dispone de lightgbm, catboost, optuna» |
| 2 | Septiembre en todo el pipeline y **prueba en datos no vistos** | Los cuatro cortes ya se usaron para elegir modelos en el Avance 4; septiembre es la evaluación independiente (objetivo 3.6) |
| 3 | Optuna para LightGBM y XGBoost (30 pruebas, 8 aleatorias, paro temprano) | Antes los parámetros individuales estaban congelados y en el Avance 4 Optuna sólo alcanzó 7 pruebas aleatorias |
| 4 | Bases nuevas: XGBoost, LightGBM log1p, Theta opcional; Elastic Net con variables en log | Un ensamble sólo gana si sus bases se equivocan distinto; LightGBM y CatBoost se parecen mucho |
| 5 | Homogéneos fuertes: bagging de LightGBM y bosque aleatorio en modo `rf` | El bagging y AdaBoost de árboles con 350 mil filas eran mucho más débiles que el LightGBM al que acompañaban |
| 6 | Blending y stacking con pesos no negativos (sin intercepto ni malla en décimas) | Ridge con intercepto podía dar pesos negativos y un nivel aprendido en mayo que no vale en agosto; la malla obligaba a dar 10 % a bases que estorban |
| 7 | Corrección de nivel por tipo de categoría × bloque de horizonte (E4) y sesgo en la selección | El sesgo del ensamble anterior salió de ±3 % en dos cortes |
| 8 | Intervalos conformales alrededor del modelo final (P10 ≤ pronóstico ≤ P90 siempre) | Los cuantiles anteriores eran modelos aparte del pronóstico puntual |
| 9 | Bandera `ALCANCE_SERVICIOS` y WMAPE de mercancía en la comparativa | Los servicios elevan el error y no se surten físicamente |
| 10 | Apartado del **calendario de promociones futuras** y `s_promos_vigentes_t0` causal | El modelo sólo ve promociones pasadas; queda listo para usar las planeadas |
| 11 | Código comentado, variables explicadas (diccionario `DESCRIPCION_VARIABLES`) y validaciones concentradas | Que cualquier integrante pueda explicar cada parte |
| 12 | Resultados recuperables por `run_id` (tablas fijas `mlops.av5_*`), huella del código calculada, experimento MLflow propio | Antes la recuperación tenía identificadores escritos a mano y la huella fija |
| 13 | Siete gráficos con interpretación calculada (incluye importancia SHAP y calibración) | Las interpretaciones escritas a mano dejan de ser ciertas al reejecutar |
| 14 | Comparativa con tipo, estrategia, mejora contra el ingenuo y contra el mejor individual, y tiempos por corte | Criterio «selección» de la rúbrica |
| 15 | Matriz de decisión de negocio (error, estabilidad, sesgo, mercancía, costo, interpretabilidad, operabilidad) | Elegir el modelo alineado con el negocio, no sólo el de menor error |
| 16 | Muestreo por hash sin ordenar la tabla, float32, lotes de horizonte, puntos de control y `MODO_RAPIDO` | Cabe en 16 GB, se puede reanudar si se cae y se prueba en 20–40 min |
| 17 | Pronóstico operativo `gold.pronostico_14d` y tabla de monitoreo `mlops.monitoreo_modelo` | Lo que Bnext usará para surtir y para vigilar el modelo |

### Nombres que se conservan y nombres nuevos

Se conservan los nombres con los que ya trabajamos: `CONFIG_FUENTE`, `CATEGORICAS`, `NUMERICAS`, `METADATOS`,
`COLUMNAS_CARGA`, `CLAVES_SERIE`, `CLAVES_EJEMPLO`, `POINT_BASE_MODELS`, `POINT_HOMOGENEOUS`, `POINT_HETEROGENEOUS`,
`S6_POINT_PARAMETERS`, `S6_MODEL_ROW_BUDGETS`, `S7_HOMOGENEOUS_PARAMETERS`, `QUANTILE_LEVELS`, `QUANTILE_COLUMNS`,
`NOMINAL_COVERAGE`, `S6_QUANTILE_PARAMETERS`, `AGG_MODELS`, `AGG_LABELS`, `AGG_DIMENSIONS`, `AGG_POINT_SUMS`,
`AGG_QUANTILE_SUMS`, `FuenteAvance5`, `resolver_origenes`, `fecha_diaria`, `validar_config_fuente`, `normalizar_bloque`,
`runtime_threads`, `TreeEncoder`, `fit_tree_encoder`, `LinearEncoder`, `ModelBundle`, `fit_model`, `predict_bundle`,
`fit_individuals`, `motor_metrics`, `fit_homogeneous`, `MetaBundle`, `fit_meta`, `predict_ensembles`,
`finite_sample_cqr_offset`, `fit_quantiles`, `predict_quantiles`, `evaluate_gate`, `AcumuladorAvance5`,
`verificar_entorno_formal`, `RegistroFormal`, `dm_por_corte`, `preparar_compuerta`, `_flujo_train`, `_flujo_observed`,
`_flujo_wide`, `_flujo_costos`, `predecir_candidato`, `ejecutar_modelado`, las claves `B0`, `I1`–`I3`, `E1`–`E4` y las
variables `*_AV5` (`FUENTE_AV5`, `REGISTRO_AV5`, `COMPUERTA_AV5`, `TABLAS_SALIDA_AV5`, …).

| Antes | Ahora | Motivo |
| --- | --- | --- |
| `tune_homogeneous` (Optuna para Bagging y AdaBoost) | `tune_boosting` (Optuna para I1 e I4) | El ajuste fino se concentra en los modelos más relevantes |
| `S7_RIDGE_ALPHAS`, `MetaBundle.ridge/scaler` | `pesos_no_negativos`, `MetaBundle.pesos_stacking/factores_nivel` | Stacking con pesos no negativos y corrección de nivel |
| `preparar_corte` / `liberar_corte` | lectura por lotes de horizonte (`lotes_horizonte`) | Serverless no permite `persist` |
| `metrics_quantiles` | sumas de cuantiles dentro de `AcumuladorAvance5` | Una sola forma de evaluar, por bloques |
| `generar_analisis_agregado` | sección 7 (gráficos desde las tablas de la corrida) | Los gráficos se pueden regenerar sin entrenar |
| `HUELLA_IMPLEMENTACION_DISTRIBUIDA` fija | `huella_codigo()` calculada | La huella identifica de verdad el código ejecutado |
| Secciones de recuperación con `run_id` escrito a mano | `cargar_corrida(run_id)` automática | Ejecución secuencial coherente |

Nuevos: `I4_XGBoost_Tweedie`, `I5_LightGBM_Log1p`, `I6_Theta`, `E5_Votacion`, `E6_Stacking_LightGBM`,
`E7_Bosque_Aleatorio`, `D1_Stacking_sin_correccion`, `Q_Conformal`, `Q_LightGBM_CQR`.

---

## 4. Cómo correrla en Databricks

### 4.1 Actualizar el pipeline con septiembre (una vez)

Importar la carpeta `databricks/mna_equipo43` del repositorio (o `mna_equipo43.zip`) en
`/Workspace/Shared/mna_equipo43/` y correr, en este orden:

| Notebook | Ajuste antes de correr | Qué hace con septiembre |
| --- | --- | --- |
| `1_datos/02_carga_bronce` | `MESES_A_CARGAR = ["2026-09"]` | Carga sólo el volumen de septiembre en Bronce |
| `1_datos/08_limpieza_plata` | `MESES_A_LIMPIAR = MESES` (todos) | Vuelve a limpiar todos los meses para aplicar la deduplicación entre meses |
| `1_datos/09_panel_diario` | — | Rehace el panel tienda × categoría × día hasta el 30 de septiembre |
| `1_datos/10_eda_series_y_linea_base` | — | Línea base con los cortes fijos y los bloques de prueba |
| `1_datos/11_validacion_calidad` | — | Reglas de calidad (evidencia para la compuerta) |
| `2_modelos/20_features_oro` | — | Tabla Oro con los bloques de prueba, el conjunto de pronóstico y `s_promos_vigentes_t0` causal |

La configuración común (`MESES` con `2026-09`, cortes fijos del Avance 4 y bloques de prueba) ya viene en cada notebook.
Los cortes son **fijos**: un mes nuevo ya no los mueve, así que las cifras siguen siendo comparables con el Avance 4.

### 4.2 Prueba de humo (20–40 min)

1. Importar `Avance5.43` en `/Workspace/Shared/mna_equipo43/2_modelos/`.
2. Conectar a cómputo serverless y dejar `MODO_RAPIDO = True` (sección 1.1).
3. **Run all.** Debe terminar con «RESUMEN PARA REVISIÓN · Avance5.43». Sus cifras no van al reporte.

### 4.3 Corrida completa (3.5–5 h con 16 GB; menos con 32–64 GB)

1. Cambiar `MODO_RAPIDO = False`.
2. Crear un **Job** (Workflows → Create job → Notebook = `Avance5.43`, cómputo serverless, tiempo máximo 8 h, sin
   reintentos automáticos) y ejecutarlo. Así no depende de que la pestaña del navegador siga abierta.
3. Al terminar, abrir la libreta del Job, revisar las secciones 5–10 y exportarla:
   **File → Export → IPython Notebook**. Reemplazar `Avance5/Avance5.43.ipynb` en este repositorio con esa versión
   (con salidas) y completar la **declaración de uso de IA** de la última celda.

### 4.4 Si la corrida se interrumpe

Copiar el `run_id` que se imprime al inicio de la sección 4 (también está en `mlops.av5_ejecuciones`) en
`RUN_ID_REANUDAR` y volver a correr: la libreta carga el punto de control del volumen `outputs` y continúa en el
siguiente bloque.

### 4.5 Ver resultados sin volver a entrenar

`EJECUTAR_ENTRENAMIENTO = False` (y opcionalmente `RUN_ID_A_MOSTRAR = "<run_id>"`): las secciones 5–10 se construyen con
las tablas guardadas de la corrida.

---

## 5. Parámetros principales (sección 1)

| Parámetro | Valor por defecto | Para qué |
| --- | --- | --- |
| `MODO_RAPIDO` | `True` | Prueba de humo (5 % de las series, pocas pruebas de Optuna) |
| `MODO_PUBLICO` | `True` | Salidas sin importes ni volúmenes (aptas para GitHub) |
| `EJECUTAR_ENTRENAMIENTO` | `True` | `False` = sólo mostrar una corrida guardada |
| `RUN_ID_REANUDAR` / `RUN_ID_A_MOSTRAR` | `None` | Reanudar una corrida / elegir cuál mostrar |
| `REENTRENAR_FINAL` | `True` | Reentrenar el elegido con todos los datos y generar el pronóstico operativo |
| `CONFIG_FUENTE["max_filas_entrenamiento"]` | 3.5 M (16 GB), 7 M (32 GB), 14 M (64 GB) | Filas de entrenamiento en memoria; se ajusta solo a la RAM |
| `INCLUIR` | Theta apagado, el resto encendido | Qué candidatos se entrenan |
| `N_PRUEBAS_AJUSTE`, `TIEMPO_MAX_AJUSTE_MIN` | 30 pruebas, 25 min por modelo | Presupuesto de Optuna |
| `ALCANCE_SERVICIOS` | `"incluir"` | `"excluir"` si Bnext confirma que los servicios no se surten |
| `METODO_INTERVALOS` | `"automatico"` | El de menor pinball en septiembre dentro de la banda 75–85 % |
| `PESOS_DECISION` | 35/10/10/10/15/10/10 | Matriz de decisión (fijada antes de ver resultados) |
| `EVIDENCIA_NEGOCIO` | todo `False` | Cambiar a `True` cuando Bnext apruebe alcance, servicios y disponibilidad de atributos |
| `NIVEL_TECHO_OPERATIVO` | `"p90"` | Techo de surtido que se publica (P80 o P90, según decida Bnext) |
| `REGISTRAR_EN_UC` | `None` | `None` = registra en Unity Catalog sólo en modo completo y si pasan las comprobaciones técnicas; `True`/`False` lo fuerza |
| `USAR_CALENDARIO_PROMOS` | `False` | Activar cuando exista `silver.calendario_promociones` |

---

## 6. Qué escribe

| Destino | Contenido |
| --- | --- |
| `mlops.av5_ejecuciones` | Bitácora de eventos de cada corrida (iniciada, pasos, modelo elegido, finalizada o error) |
| `mlops.av5_*` | Resultados agregados con `run_id`: comparativas, por corte, por horizonte, diario, segmentos, calibración, residuos (muestra sin identificadores), cuantiles, Diebold–Mariano, decisión, pesos, ajuste fino, costos, auditoría temporal, parámetros, importancia, correlación de errores, compuerta, interpretaciones |
| `gold.pronostico_14d` | Pronóstico operativo por tienda × categoría × día: pronóstico, P10, P50, P90 y techo operativo (no se escribe en `MODO_RAPIDO`) |
| `mlops.monitoreo_modelo` | WMAPE, sesgo, cobertura y alertas por semana × región × tipo |
| Volumen `outputs/avance5_puntos_control/<run_id>` | Punto de control para reanudar |
| MLflow `/Users/<usuario>/mna_equipo43_avance5` | Parámetros, métricas, tablas, gráficas, interpretaciones y modelo final (`pyfunc`) |
| Unity Catalog `modeladoprescriptivoventas.maestria.demanda_categoria` | Nueva versión con alias `@challenger` si pasan las comprobaciones técnicas (sólo en modo completo) |

---

## 7. Dónde se cumple la rúbrica

| Criterio | Dónde |
| --- | --- |
| Ensambles homogéneos y heterogéneos (≥ 4) con los mejores individuales; optimización de hiperparámetros | §2.4–2.7, §4, §5.4, §5.5 |
| Comparativa ordenada por la métrica principal con otras métricas y tiempos de entrenamiento | §5.2, §5.3, §5.6 |
| Modelo final alineado con el negocio y argumentado | §5.7, §6.5 |
| ≥ 4 gráficos del modelo final con interpretación | §7 (siete) |
| Evaluación en datos no vistos (objetivo 3.6) | §5.3, §6, §7 |
| Declaración de uso de IA | última celda |

---

## 8. Promociones: lo que el modelo ve hoy

El modelo **sólo ve promociones pasadas**: las cuatro variables de promoción usan días anteriores o iguales al día del
pronóstico. Ninguna mira los 14 días que se pronostican, así que una promoción que empieza dentro de esa ventana no se
anticipa. La sección 9 deja listo el uso de un **calendario de promociones planeadas**: esquema, validación y las
variables `f_promos_planeadas`, `f_descuento_planeado` y `f_dias_promo_restantes`, que entran al modelo con
`USAR_CALENDARIO_PROMOS = True` en cuanto Bnext entregue el calendario.
