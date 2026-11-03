# Databricks notebook source
# MAGIC %md
# MAGIC # 30 · Simulación de escenarios promocionales (*what-if*) con el modelo
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Avance 6**
# MAGIC
# MAGIC Responde a los compradores y planeadores: *¿cuánto cambia la venta esperada de los próximos 14 días si activamos
# MAGIC (o no) una promoción, y con qué profundidad?* La respuesta sale del **mismo modelo registrado** que produce el
# MAGIC pronóstico operativo: cada escenario modifica las variables del calendario de promociones planeadas
# MAGIC (`f_promos_planeadas`, `f_descuento_planeado`, `f_dias_promo_restantes`) y vuelve a puntuar las mismas series.
# MAGIC Así el *uplift* es lo que el modelo aprendió de las promociones históricas, no una elasticidad escrita a mano.
# MAGIC
# MAGIC **Escenarios.** `S_PLAN` (el plan vigente del calendario: es el pronóstico operativo), `S0_SIN_PROMO` (se retiran las
# MAGIC promociones planeadas), `S1_10`, `S2_20`, `S3_30` (promoción en todo el horizonte con 10, 20 y 30 % de descuento en las
# MAGIC categorías del alcance). El valor del plan = `S_PLAN − S0_SIN_PROMO`.
# MAGIC
# MAGIC **Referencia empírica.** Además se calcula `gold.uplift_promocional_historico`: cuánto vendió cada categoría en días con
# MAGIC promoción respecto a días sin promoción (misma tienda, mes y día de la semana), por profundidad de descuento. Es
# MAGIC descriptivo (la promoción se programa cuando se espera vender más) y se declara como **cota superior**. Si el modelo
# MAGIC registrado no se entrenó con las variables `f_*`, los escenarios usan estos factores y quedan marcados `modo = empirico`.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.features_origen` (conjunto `pronostico`) · modelo registrado · `silver.calendario_promociones_plan_dia` · `silver.panel_categoria_dia` · `silver.catalogo_categorias` |
# MAGIC | **Escribe** | `gold.escenarios_whatif` (reemplaza su origen) · `gold.escenarios_whatif_resumen` · `gold.uplift_promocional_historico` |
# MAGIC | **Parámetros** | `ALIAS_MODELO`, `ALCANCE_ESCENARIOS`, `DESCUENTOS_MALLA`, `MODO_PUBLICO` |
# MAGIC | **Tiempo** | 15–40 min |
# MAGIC
# MAGIC **No fabrica pronósticos**: si no hay modelo registrado ni Oro con conjunto `pronostico`, se detiene y dice qué correr.

# COMMAND ----------

# MAGIC %pip install "lightgbm>=4.3,<5"

# COMMAND ----------

# MAGIC %pip install "xgboost>=2.0,<4"

# COMMAND ----------

# MAGIC %pip install "catboost>=1.2,<2"

# COMMAND ----------

try:
    dbutils.library.restartPython()
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
ALIAS_MODELO = parametro("ALIAS_MODELO", "champion")
PERMITIR_CHALLENGER = parametro("PERMITIR_CHALLENGER", True)
RUN_ID_MODELO = parametro("RUN_ID_MODELO", None)
ALCANCE_ESCENARIOS = parametro("ALCANCE_ESCENARIOS", ["mercancia"])   # tipos de categoría a los que se aplica la promoción simulada
DESCUENTOS_MALLA = parametro("DESCUENTOS_MALLA", [0.10, 0.20, 0.30])  # profundidades de la malla S1, S2, S3
MODO_PUBLICO = parametro("MODO_PUBLICO", True)
MIN_DIAS_CON_VENTA = 5
TABLA_ESCENARIOS = tabla("gold", "escenarios_whatif")
TABLA_ESCENARIOS_RESUMEN = tabla("gold", "escenarios_whatif_resumen")
TABLA_UPLIFT = tabla("gold", "uplift_promocional_historico")
TABLA_PANEL = tabla("silver", "panel_categoria_dia")
DESCUENTOS_MALLA = [float(d) for d in DESCUENTOS_MALLA]
ESCENARIOS = [("S_PLAN", "Plan vigente del calendario (= pronóstico operativo)", None),
              ("S0_SIN_PROMO", "Sin promociones planeadas en los 14 días", 0.0)] + \
             [(f"S{i}_{int(round(d * 100))}", f"Promoción en todo el horizonte con {d:.0%} de descuento", d) for i, d in enumerate(DESCUENTOS_MALLA, 1)]
print("Escenarios:", [e[0] for e in ESCENARIOS], "· alcance:", ALCANCE_ESCENARIOS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Referencia empírica: uplift histórico por categoría y profundidad de descuento
# MAGIC Para cada serie se compara la venta media de los días **con** promoción contra los días **sin** promoción del mismo
# MAGIC mes y día de la semana; el cociente se resume por categoría y por profundidad del descuento efectivo
# MAGIC (`tasa_descuento` del panel). Es una asociación, no un efecto causal: cota superior declarada.

# COMMAND ----------

titulo("1. UPLIFT HISTÓRICO (REFERENCIA)")
PANEL = leer_tabla(TABLA_PANEL, "09_panel_diario")
base = (PANEL.where((F.col("sin_datos") == 0) & (F.col("tienda_inactiva") == 0) & F.col("venta").isNotNull())
        .withColumn("promo", (F.coalesce(F.col("promos_vigentes"), F.lit(0)) > 0).cast("int"))
        .withColumn("dow", F.dayofweek("fecha"))
        .withColumn("profundidad", F.when(F.col("promo") == 0, "sin_promo")
                    .when(F.col("tasa_descuento") < 0.10, "menos_10").when(F.col("tasa_descuento") < 0.20, "10_a_20")
                    .when(F.col("tasa_descuento") < 0.30, "20_a_30").otherwise("30_o_mas")))
sin = (base.where(F.col("promo") == 0).groupBy("tienda", "categoria_id", "mes", "dow")
       .agg(F.avg("venta").alias("v_sin"), F.count("*").alias("n_sin")))
con = (base.where(F.col("promo") == 1).groupBy("tienda", "categoria_id", "mes", "dow", "profundidad")
       .agg(F.avg("venta").alias("v_con"), F.count("*").alias("n_con")))
pares = (con.join(sin, ["tienda", "categoria_id", "mes", "dow"]).where(F.col("v_sin") > 0)
         .withColumn("ratio", F.col("v_con") / F.col("v_sin")))
cat = leer_tabla(tabla("silver", "catalogo_categorias"), "08_limpieza_plata").select(F.col("categoria_id").cast("string").alias("categoria_id"), "tipo_categoria")
UPLIFT = (pares.withColumn("categoria_id", F.col("categoria_id").cast("string")).join(cat, "categoria_id", "left")
          .groupBy("categoria_id", "tipo_categoria", "profundidad")
          .agg(F.countDistinct("tienda").alias("tiendas"), F.sum("n_con").alias("dias_con_promo"),
               F.expr("percentile_approx(ratio, 0.5)").alias("uplift_mediano"),
               (F.sum(F.col("ratio") * F.col("n_con")) / F.sum("n_con")).alias("uplift_medio_ponderado"))
          .where(F.col("dias_con_promo") >= 20)
          .withColumn("uplift_mediano", F.least(F.greatest(F.col("uplift_mediano"), F.lit(0.5)), F.lit(5.0)))
          .withColumn("fecha_calculo", F.current_timestamp()))
guardar(UPLIFT, "gold", "uplift_promocional_historico",
        "Venta en dias con promocion / dias sin promocion por categoria y profundidad (misma tienda, mes y dia de semana); descriptivo, cota superior (30)")
comentar_columnas(TABLA_UPLIFT, {"profundidad": "Tramo del descuento efectivo del dia con promocion.", "uplift_mediano": "Mediana del cociente con/sin promocion (acotada 0.5-5).",
                                 "uplift_medio_ponderado": "Media del cociente ponderada por dias con promocion.", "dias_con_promo": "Dias con promocion que entran al calculo."})
UP = spark.table(TABLA_UPLIFT).toPandas()
por_tipo = (UP.groupby(["tipo_categoria", "profundidad"]).apply(lambda g: pd.Series({"categorias": g["categoria_id"].nunique(),
            "uplift_mediano": float(np.median(g["uplift_mediano"])), "uplift_ponderado": float(np.average(g["uplift_medio_ponderado"], weights=g["dias_con_promo"]))}))
            .reset_index()) if len(UP) else pd.DataFrame()
mostrar(por_tipo, "Uplift histórico por tipo de categoría y profundidad (venta con promo / sin promo)", formatos={"categorias": "int", "uplift_mediano": "dec2", "uplift_ponderado": "dec2"})
if len(por_tipo):
    merc = por_tipo[por_tipo["tipo_categoria"] == "mercancia"]
    hallazgo("Uplift histórico en mercancía (cota superior): " + "; ".join(f"{r.profundidad} ×{r.uplift_mediano:.2f}" for r in merc.itertuples()) + ".")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Modelo y filas a simular

# COMMAND ----------

titulo("2. MODELO Y FILAS")
MODELO, INFO_MODELO = cargar_modelo_operativo(ALIAS_MODELO, RUN_ID_MODELO, PERMITIR_CHALLENGER)
ENTRADAS = columnas_modelo(MODELO) or []
USA_F = all(c in ENTRADAS for c in VARIABLES_PROMO_FUTURA)
MODO_SIMULACION = "modelo" if USA_F else "empirico"
print(f"Modelo {INFO_MODELO.get('modelo')} ({INFO_MODELO['origen']}, alias {INFO_MODELO.get('alias')}, versión {INFO_MODELO.get('version')}) · "
      f"usa variables f_*: {USA_F} → simulación en modo {MODO_SIMULACION}")
if not USA_F:
    aviso("El modelo registrado no se entrenó con las variables del calendario (f_*): los escenarios usan los factores empíricos "
          "(cota superior). Para la simulación con el modelo, reentrena con USAR_CALENDARIO_PROMOS = True.")
ORO = leer_tabla(TABLA_ORO, "20_features_oro")
ORIGEN = ORO.where(F.col("conjunto") == "pronostico").agg(F.max("origen")).collect()[0][0]
if ORIGEN is None:
    raise RuntimeError("La Oro no tiene conjunto 'pronostico': corre 20_features_oro con GENERAR_PRONOSTICO = True.")
elegible = ((F.col("dias_con_venta_hist") >= MIN_DIAS_CON_VENTA) & F.col("ingenuo_estacional").isNotNull() & (F.col("s_dias_activos_7") >= 5))
FILAS = ORO.where((F.col("conjunto") == "pronostico") & (F.col("origen") == F.lit(ORIGEN)) & elegible)
PLAN_DIA, INFO_CAL = cargar_variables_promo_futura(spark)
FILAS = unir_promociones_planeadas(FILAS, PLAN_DIA)      # S_PLAN = el plan vigente (f_* = 0 si no hay calendario)
print("Calendario:", INFO_CAL)
n = FILAS.agg(F.count("*").alias("filas"), F.countDistinct("tienda", "categoria_id").alias("series")).collect()[0]
print(f"Origen {ORIGEN}: {n['filas']:,} filas · {n['series']:,} series")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Puntuar cada escenario
# MAGIC Por horizonte: se traen las filas una vez y se puntúan con las variables `f_*` de cada escenario (modo `modelo`), o se
# MAGIC multiplica el pronóstico base por el factor empírico de la categoría y la profundidad (modo `empirico`).

# COMMAND ----------

titulo("3. ESCENARIOS")
llaves = ["tienda", "categoria_id", "origen", "fecha_objetivo", "h", "tipo_categoria", "region"]
columnas = list(dict.fromkeys(llaves + VARIABLES_PROMO_FUTURA + ENTRADAS))
UP_IDX = UP.set_index(["categoria_id", "profundidad"])["uplift_mediano"].to_dict() if len(UP) else {}
UP_TIPO = {(r.tipo_categoria, r.profundidad): r.uplift_mediano for r in por_tipo.itertuples()} if len(por_tipo) else {}


def tramo(d):
    return "menos_10" if d < 0.10 else "10_a_20" if d < 0.20 else "20_a_30" if d < 0.30 else "30_o_mas"


def factor_empirico(cat, tipo, d):
    return float(UP_IDX.get((str(cat), tramo(d)), UP_TIPO.get((tipo, tramo(d)), 1.0)))


partes = []
with Cronometro() as reloj:
    for h in range(1, 15):
        pdf = FILAS.where(F.col("h") == h).select(*columnas).toPandas()
        if pdf.empty:
            continue
        en_alcance = pdf["tipo_categoria"].astype(str).isin(ALCANCE_ESCENARIOS).to_numpy()
        base_pred = None
        for id_esc, descripcion, descuento in ESCENARIOS:
            x = pdf.copy()
            if descuento is not None:
                if descuento == 0.0:                      # sin promociones planeadas (en todas las categorías)
                    for c in VARIABLES_PROMO_FUTURA:
                        x[c] = 0.0
                else:                                     # promoción en todo el horizonte para las categorías del alcance
                    x.loc[en_alcance, "f_promos_planeadas"] = np.maximum(x.loc[en_alcance, "f_promos_planeadas"].to_numpy(), 1.0)
                    x.loc[en_alcance, "f_descuento_planeado"] = descuento
                    x.loc[en_alcance, "f_dias_promo_restantes"] = 14 - h
            if MODO_SIMULACION == "modelo" or descuento is None or (descuento == 0.0 and not USA_F):
                pred = pd.DataFrame(MODELO.predict(_preparar_entrada(x[ENTRADAS].copy(), MODELO))).reset_index(drop=True)
                if descuento == 0.0 and not USA_F:
                    pred = pred.copy()                    # sin f_*: el modelo no distingue; S0 = base
            else:                                         # modo empírico: base × factor de la categoría
                f = np.where(en_alcance, [factor_empirico(c, t, descuento) for c, t in zip(x["categoria_id"], x["tipo_categoria"])], 1.0)
                pred = base_pred.copy()
                for c in ("prediccion_puntual", "p10", "p50", "p90", "techo_operativo"):
                    pred[c] = pred[c].to_numpy() * f
            if id_esc == "S_PLAN":
                base_pred = pred.copy()
            partes.append(pd.concat([pdf[llaves].reset_index(drop=True), pred[["prediccion_puntual", "p10", "p50", "p90"]],
                                     pd.DataFrame({"id_escenario": id_esc, "descripcion_escenario": descripcion,
                                                   "descuento_escenario": 0.0 if descuento is None else float(descuento),
                                                   "f_promos_planeadas_plan": pdf["f_promos_planeadas"].to_numpy(),
                                                   "f_descuento_planeado_plan": pdf["f_descuento_planeado"].to_numpy()})], axis=1))
        print(f"   h = {h}: {len(pdf):,} filas × {len(ESCENARIOS)} escenarios")
ESC = pd.concat(partes, ignore_index=True)
print(f"{len(ESC):,} filas simuladas en {reloj.segundos / 60:.1f} min")
base_cols = ESC[ESC["id_escenario"] == "S_PLAN"][llaves + ["prediccion_puntual", "p10", "p90"]].rename(
    columns={"prediccion_puntual": "venta_base_p50", "p10": "p10_base", "p90": "p90_base"})
ESC = ESC.merge(base_cols, on=llaves, how="left")
ESC["uplift_pesos"] = ESC["prediccion_puntual"] - ESC["venta_base_p50"]
ESC["uplift_pct"] = np.where(ESC["venta_base_p50"] > 0, ESC["uplift_pesos"] / ESC["venta_base_p50"] * 100, np.nan)
ESC = ESC.rename(columns={"prediccion_puntual": "venta_escenario_p50", "p10": "p10_escenario", "p90": "p90_escenario"}).drop(columns=["p50"])
for c in ("origen", "fecha_objetivo"):
    ESC[c] = pd.to_datetime(ESC[c]).dt.date
ESC["modo_simulacion"], ESC["modelo"], ESC["version_modelo"] = MODO_SIMULACION, str(INFO_MODELO.get("modelo")), str(INFO_MODELO.get("version"))
ESC["run_id"], ESC["fecha_simulacion"] = INFO_MODELO.get("run_id_av5") or "", pd.Timestamp.now()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Guardar y resumir (sin importes en modo público)

# COMMAND ----------

titulo("4. RESULTADOS")
limpio, esquema = _para_spark(ESC)
sdf = spark.createDataFrame(limpio, schema=esquema)
escritor = sdf.write.format("delta").mode("overwrite")
escritor = (escritor.option("replaceWhere", f"origen = DATE'{ORIGEN}'").option("mergeSchema", "true") if existe(TABLA_ESCENARIOS)
            else escritor.option("overwriteSchema", "true"))
escritor.partitionBy("id_escenario").saveAsTable(TABLA_ESCENARIOS)
try:
    spark.sql(f"COMMENT ON TABLE {TABLA_ESCENARIOS} IS 'Escenarios what-if de promociones por tienda x categoria x dia: venta base (plan), venta del escenario y uplift; modo modelo o empirico (30)'")
except Exception:
    pass
comentar_columnas(TABLA_ESCENARIOS, {"id_escenario": "S_PLAN plan vigente; S0_SIN_PROMO sin promociones; S1/S2/S3 promocion con 10/20/30 % de descuento en el alcance.",
                                     "venta_base_p50": "Pronostico con el plan vigente (= operativo).", "venta_escenario_p50": "Pronostico bajo el escenario.",
                                     "uplift_pct": "Cambio porcentual del escenario respecto al plan.", "modo_simulacion": "modelo (re-puntuado) o empirico (factores historicos).",
                                     "f_promos_planeadas_plan": "Promociones del plan vigente que cubren el dia.", "descuento_escenario": "Descuento aplicado en el escenario (0 en S_PLAN y S0)."})
print(f"   → guardada {TABLA_ESCENARIOS}")
RES = (ESC.groupby(["id_escenario", "descripcion_escenario", "descuento_escenario", "region", "tipo_categoria", "categoria_id"])
       .agg(series=("tienda", "nunique"), venta_base=("venta_base_p50", "sum"), venta_escenario=("venta_escenario_p50", "sum"),
            p10_escenario=("p10_escenario", "sum"), p90_escenario=("p90_escenario", "sum")).reset_index())
RES["uplift_pesos"] = RES["venta_escenario"] - RES["venta_base"]
RES["uplift_pct"] = np.where(RES["venta_base"] > 0, RES["uplift_pesos"] / RES["venta_base"] * 100, np.nan)
RES["origen"], RES["modo_simulacion"], RES["fecha_simulacion"] = ORIGEN, MODO_SIMULACION, pd.Timestamp.now()
guardar(RES, "gold", "escenarios_whatif_resumen", "Escenarios what-if agregados por escenario x region x tipo x categoria (30)")
por_esc = (ESC.groupby(["id_escenario", "tipo_categoria"]).agg(venta_base=("venta_base_p50", "sum"), venta_escenario=("venta_escenario_p50", "sum")).reset_index())
por_esc["uplift_pct"] = np.where(por_esc["venta_base"] > 0, (por_esc["venta_escenario"] / por_esc["venta_base"] - 1) * 100, np.nan)
vista = por_esc.pivot(index="id_escenario", columns="tipo_categoria", values="uplift_pct").reset_index()
mostrar(vista, "Uplift (%) por escenario y tipo de categoría respecto al plan vigente", formatos={c: "dec1" for c in vista.columns if c != "id_escenario"})
if not MODO_PUBLICO:
    mostrar(por_esc, "Totales por escenario (pesos)", formatos={"venta_base": "dinero", "venta_escenario": "dinero", "uplift_pct": "dec1"})
merc = por_esc[por_esc["tipo_categoria"] == "mercancia"].set_index("id_escenario")["uplift_pct"]
hallazgo(f"Simulación en modo {MODO_SIMULACION} para el origen {ORIGEN} ({n['series']:,} series): uplift en mercancía respecto al plan vigente → "
         + ", ".join(f"{k} {v:+.1f} %" for k, v in merc.items() if k != "S_PLAN") + ".")
if "S0_SIN_PROMO" in merc.index and abs(merc["S0_SIN_PROMO"]) < 0.05 and MODO_SIMULACION == "modelo":
    aviso("S0 ≈ S_PLAN: el plan vigente casi no tiene promociones en el horizonte (o el modelo les da poco peso); revisar el calendario.")

# COMMAND ----------

cerrar("30_simulacion_escenarios_whatif", siguiente="31_desagregacion_predicciones_sku · 40_salidas_dashboard")
