# Databricks notebook source
# MAGIC %md
# MAGIC # 22 · Pronóstico operativo (inferencia con el modelo registrado)
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC Separa **pronosticar** de **entrenar**: toma el modelo final del registro de Unity Catalog
# MAGIC (`demanda_categoria @champion`, o `@challenger` si todavía no hay campeón), puntúa el conjunto `pronostico` de la tabla
# MAGIC Oro (origen = último día con datos, h = 1…14) y escribe `gold.pronostico_14d` con pronóstico puntual, P10/P50/P90 y
# MAGIC techo operativo. Tarda minutos, no horas: es lo que corre cada vez que llegan datos nuevos (Job `J4_inferencia`),
# MAGIC sin reentrenar.
# MAGIC
# MAGIC **Promociones planeadas.** Si el modelo se entrenó con las variables `f_*`, aquí se unen desde
# MAGIC `silver.calendario_promociones_plan_dia` (13): cambiar el plan y volver a correr este notebook actualiza el pronóstico.
# MAGIC
# MAGIC **Otro origen.** Con `ORIGEN_PRONOSTICO = "AAAA-MM-DD"` se pronostica un origen distinto al último (por ejemplo, el
# MAGIC conjunto `pronostico` de una Oro anterior). El modelo rechaza orígenes anteriores a la calibración de sus intervalos
# MAGIC (protección anti-fuga): el acierto en producción se mide con `23_monitoreo` cuando llega la venta real de los días
# MAGIC pronosticados hacia adelante; mientras tanto la referencia es la prueba de septiembre del Avance 5.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.features_origen` (conjunto `pronostico`) · modelo `demanda_categoria` (UC o run de MLflow) · `silver.calendario_promociones_plan_dia` |
# MAGIC | **Escribe** | `gold.pronostico_14d` (reemplaza sólo su origen) · `mlops.pronosticos_bitacora` |
# MAGIC | **Parámetros** | `ALIAS_MODELO`, `PERMITIR_CHALLENGER`, `RUN_ID_MODELO`, `ORIGEN_PRONOSTICO`, `USAR_CALENDARIO_PROMOS`, `MODO_PUBLICO` |
# MAGIC | **Tiempo** | 10–30 min |
# MAGIC
# MAGIC **Requisito.** Una corrida de `Avance5.43` en modo completo que haya guardado el modelo final (compuerta técnica
# MAGIC aprobada o `REGISTRAR_EN_UC = True`). Si no hay modelo, este notebook se detiene y lo dice; **no** fabrica pronósticos.

# COMMAND ----------

# MAGIC %pip install "lightgbm>=4.3,<5"

# COMMAND ----------

# MAGIC %pip install "xgboost>=2.0,<4"

# COMMAND ----------

# MAGIC %pip install "catboost>=1.2,<2"

# COMMAND ----------

try:
    dbutils.library.restartPython()      # necesario después de %pip en Databricks; fuera de Databricks no existe y se ignora
except Exception:
    pass

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# @@PROMOCIONES

# COMMAND ----------

# @@MODELO

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook (también como parámetros de Job)
ALIAS_MODELO = parametro("ALIAS_MODELO", "champion")            # alias del registro: champion (producción) o challenger
PERMITIR_CHALLENGER = parametro("PERMITIR_CHALLENGER", True)    # si no hay @champion, usar @challenger
RUN_ID_MODELO = parametro("RUN_ID_MODELO", None)                # corrida del Avance 5 cuyo modelo usar si el registro no está disponible
ORIGEN_PRONOSTICO = parametro("ORIGEN_PRONOSTICO", None)        # None = último día con datos (operativo); una fecha = retrospectivo
USAR_CALENDARIO_PROMOS = parametro("USAR_CALENDARIO_PROMOS", True)
MODO_PUBLICO = parametro("MODO_PUBLICO", True)                  # True: no se imprimen importes
MIN_DIAS_CON_VENTA = 5                                          # mismas filas elegibles que en el entrenamiento (Avance 4 y 5)
TABLA_BITACORA = tabla("mlops", "pronosticos_bitacora")
RUN_ID_PRONOSTICO = "pron_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dt%H%M%S")
from uuid import uuid4
RUN_ID_PRONOSTICO += "_" + uuid4().hex[:6]
print("Corrida de pronóstico:", RUN_ID_PRONOSTICO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Modelo

# COMMAND ----------

titulo("1. MODELO REGISTRADO")
MODELO, INFO_MODELO = cargar_modelo_operativo(ALIAS_MODELO, RUN_ID_MODELO, PERMITIR_CHALLENGER)
ENTRADAS = columnas_modelo(MODELO) or []
USA_F = all(c in ENTRADAS for c in VARIABLES_PROMO_FUTURA)
mostrar(pd.DataFrame([{k: v for k, v in INFO_MODELO.items() if k != "etiquetas"}]), "Modelo cargado")
print(f"Variables de entrada del modelo: {len(ENTRADAS)} · usa promociones planeadas (f_*): {USA_F}")
hallazgo(f"Modelo {INFO_MODELO.get('modelo')} tomado de {INFO_MODELO['origen']} (alias {INFO_MODELO.get('alias')}, versión {INFO_MODELO.get('version')}), "
         f"entrenado en la corrida {INFO_MODELO.get('run_id_av5')}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Filas de la Oro a pronosticar

# COMMAND ----------

titulo("2. FILAS A PRONOSTICAR")
ORO = leer_tabla(TABLA_ORO, "20_features_oro")
try:
    VERSION_ORO = int(spark.sql(f"DESCRIBE HISTORY {TABLA_ORO} LIMIT 1").collect()[0]["version"])
except Exception:
    VERSION_ORO = None
if ORIGEN_PRONOSTICO is None:
    fila = ORO.where(F.col("conjunto") == "pronostico").agg(F.max("origen").alias("o")).collect()[0]
    if fila["o"] is None:
        raise RuntimeError("La Oro no tiene conjunto 'pronostico'. Corre 20_features_oro con GENERAR_PRONOSTICO = True.")
    ORIGEN, CONJUNTO, FASE = fila["o"], "pronostico", "operativo"
else:
    ORIGEN = pd.Timestamp(ORIGEN_PRONOSTICO).date()
    disponibles = {r["conjunto"] for r in ORO.where(F.col("origen") == F.lit(ORIGEN)).select("conjunto").distinct().collect()}
    if not disponibles:
        raise RuntimeError(f"La Oro no tiene filas con origen {ORIGEN}; orígenes de evaluación: ver mlops.oro_resumen.")
    CONJUNTO = "pronostico" if "pronostico" in disponibles else "evaluacion"
    FASE = "operativo" if CONJUNTO == "pronostico" else "retrospectivo"
elegible = ((F.col("dias_con_venta_hist") >= MIN_DIAS_CON_VENTA) & F.col("ingenuo_estacional").isNotNull() & (F.col("s_dias_activos_7") >= 5))
FILAS = ORO.where((F.col("conjunto") == CONJUNTO) & (F.col("origen") == F.lit(ORIGEN)) & elegible)
if USAR_CALENDARIO_PROMOS and USA_F:
    PLAN_DIA, INFO_CAL = cargar_variables_promo_futura(spark)
    if PLAN_DIA is None:
        aviso(f"El modelo usa promociones planeadas pero el calendario no está disponible ({INFO_CAL.get('motivo')}): se pronostica como si no hubiera promociones planeadas.")
    FILAS = unir_promociones_planeadas(FILAS, PLAN_DIA)
    print("Calendario de promociones:", INFO_CAL)
elif USAR_CALENDARIO_PROMOS and not USA_F:
    aviso("El modelo registrado no se entrenó con las variables f_* del calendario: el plan de promociones no influye en este pronóstico "
          "(reentrena con USAR_CALENDARIO_PROMOS = True en Avance5.43).")
    INFO_CAL = {"activo": False}
else:
    INFO_CAL = {"activo": False}
    FILAS = unir_promociones_planeadas(FILAS, None) if USA_F else FILAS
n = FILAS.agg(F.count("*").alias("filas"), F.countDistinct("tienda", "categoria_id").alias("series"), F.countDistinct("h").alias("h")).collect()[0]
print(f"Origen {ORIGEN} ({FASE}, conjunto {CONJUNTO}) · {n['filas']:,} filas · {n['series']:,} series · {n['h']} horizontes · Oro versión {VERSION_ORO}")
if n["filas"] == 0:
    raise RuntimeError("No hay filas elegibles para pronosticar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Puntuar y escribir `gold.pronostico_14d`

# COMMAND ----------

titulo("3. PRONÓSTICO")
with Cronometro() as reloj:
    try:
        PRON = puntuar_oro(FILAS, MODELO, por="h", extra=["f_promos_planeadas", "f_descuento_planeado"] if USA_F else [])
    except ValueError as e:
        if "calibraci" in str(e):
            raise RuntimeError(f"El modelo no puede pronosticar el origen {ORIGEN}: sus intervalos se calibraron con datos posteriores y la "
                               "protección anti-fuga lo impide. El acierto en producción se mide cuando llega la venta real de un origen "
                               "pronosticado hacia adelante (23_monitoreo); la referencia mientras tanto es la fase de prueba del Avance 5.") from e
        raise
print(f"{len(PRON):,} filas pronosticadas en {reloj.segundos / 60:.1f} min")
extra = {"modelo": INFO_MODELO.get("modelo") or "modelo_final", "run_id": INFO_MODELO.get("run_id_av5") or "",
         "version_oro": str(VERSION_ORO), "version_modelo": str(INFO_MODELO.get("version")), "alias_modelo": INFO_MODELO.get("alias"),
         "fase": FASE, "run_id_pronostico": RUN_ID_PRONOSTICO, "version_plan": str(INFO_CAL.get("version_plan", ""))}
DESTINO = escribir_pronostico(PRON, ORIGEN, extra)
hallazgo(f"Pronóstico {FASE} del origen {ORIGEN} escrito en {DESTINO}: {len(PRON):,} filas, {PRON[['tienda', 'categoria_id']].drop_duplicates().shape[0]:,} series, "
         f"{PRON['fecha_objetivo'].min()} → {PRON['fecha_objetivo'].max()}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Revisión (sin importes) y bitácora

# COMMAND ----------

titulo("4. REVISIÓN")
rev = pd.DataFrame([{
    "filas": len(PRON), "series": PRON[["tienda", "categoria_id"]].drop_duplicates().shape[0],
    "p10_le_pron_le_p90": float(((PRON["p10"] <= PRON["prediccion_puntual"]) & (PRON["prediccion_puntual"] <= PRON["p90"])).mean()),
    "ancho_relativo_mediano": float(((PRON["p90"] - PRON["p10"]) / PRON["prediccion_puntual"].clip(lower=1.0)).median()),
    "pronostico_cero": float((PRON["prediccion_puntual"] <= 0).mean()),
    "series_con_promo_planeada": float(PRON.groupby(["tienda", "categoria_id"])["f_promos_planeadas"].max().gt(0).mean()) if "f_promos_planeadas" in PRON else 0.0,
}])
mostrar(rev, "Revisión del pronóstico", formatos={"filas": "int", "series": "int", "p10_le_pron_le_p90": "pct", "ancho_relativo_mediano": "dec2",
                                                 "pronostico_cero": "pct", "series_con_promo_planeada": "pct"})
por_region = PRON.groupby(["region", "tipo_categoria"]).agg(series=("tienda", lambda s: s.nunique()), filas=("h", "size"),
                                                           ancho_rel=("p90", "size")).reset_index()
por_region["ancho_rel"] = PRON.assign(a=(PRON["p90"] - PRON["p10"]) / PRON["prediccion_puntual"].clip(lower=1.0)).groupby(["region", "tipo_categoria"])["a"].median().to_numpy()
mostrar(por_region, "Series y ancho relativo del intervalo por región × tipo", formatos={"series": "int", "filas": "int", "ancho_rel": "dec2"})
if not MODO_PUBLICO:
    tot = PRON.groupby("fecha_objetivo")[["prediccion_puntual", "p10", "p90"]].sum().reset_index()
    mostrar(tot, "Totales diarios (pesos)", formatos={"prediccion_puntual": "dinero", "p10": "dinero", "p90": "dinero"})
bit = pd.DataFrame([{"run_id_pronostico": RUN_ID_PRONOSTICO, "origen": ORIGEN, "fase": FASE, "modelo": extra["modelo"], "alias_modelo": extra["alias_modelo"],
                     "version_modelo": extra["version_modelo"], "run_id_entrenamiento": extra["run_id"], "version_oro": str(VERSION_ORO),
                     "version_plan": extra["version_plan"], "filas": int(len(PRON)), "series": int(rev["series"].iloc[0]),
                     "desde": PRON["fecha_objetivo"].min(), "hasta": PRON["fecha_objetivo"].max(), "minutos": float(reloj.segundos / 60),
                     "fecha": pd.Timestamp.now()}])
guardar(bit, "mlops", "pronosticos_bitacora", "Corridas de pronostico operativo: origen, modelo, version y filas (22)", modo="append")

# COMMAND ----------

cerrar("22_pronostico_operativo", siguiente="23_monitoreo · 30_simulacion_escenarios_whatif · 31_desagregacion_predicciones_sku · 40_salidas_dashboard")
