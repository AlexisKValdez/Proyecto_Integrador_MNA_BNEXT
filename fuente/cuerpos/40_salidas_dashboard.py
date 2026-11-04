# Databricks notebook source
# MAGIC %md
# MAGIC # 40 · Salidas para el tablero (tablas `gold.dash_*`)
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Avance 6**
# MAGIC
# MAGIC Prepara, ya calculadas, las tablas que lee el tablero AI/BI de Databricks (`dashboard/mna_equipo43_demanda.lvdash.json`).
# MAGIC El tablero sólo consulta y filtra: ningún cálculo pesado vive en él y los datos no salen de Databricks.
# MAGIC
# MAGIC | Tabla | Vista del tablero | Qué responde |
# MAGIC | --- | --- | --- |
# MAGIC | `gold.dash_resumen_ejecutivo` | Resumen | ¿Cuánto se espera vender en los próximos 14 días y cómo se compara con las dos semanas anteriores, por región y tipo? |
# MAGIC | `gold.dash_serie_tiempo` | Pronóstico | Historia reciente (8 semanas) + 14 días de pronóstico con banda P10–P90, desde nacional hasta tienda × categoría |
# MAGIC | `gold.dash_precision` y `gold.dash_precision_modelos` | Precisión | ¿Qué tan bien acierta el modelo (prueba y producción) y contra el ingenuo? |
# MAGIC | `gold.surtido_sku_semana` (31) | Surtido | ¿Qué productos y cuántas unidades cubren la demanda de cada tienda × categoría por semana? |
# MAGIC | `gold.dash_promociones` y `gold.escenarios_whatif_resumen` (30) | Promociones | ¿Qué promociones vienen, cuánto aportan y qué pasaría con otra profundidad? |
# MAGIC | `gold.dash_alertas` | Alertas | Series con cambio fuerte, alta incertidumbre, tiendas inactivas y alertas del monitoreo |
# MAGIC | `gold.dash_estado_pipeline` | Estado | Última corrida de cada notebook, frescura de datos, compuerta, modelo vigente |
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.pronostico_14d`, `silver.panel_categoria_dia`, `silver.catalogo_tiendas`, `mlops.monitoreo_modelo`, `mlops.av5_*`, `mlops.alertas`, `mlops.corridas`, `mlops.validaciones`, `silver.calendario_promociones_plan`, `gold.escenarios_whatif_resumen` |
# MAGIC | **Escribe** | `gold.dash_resumen_ejecutivo`, `gold.dash_serie_tiempo`, `gold.dash_precision`, `gold.dash_precision_modelos`, `gold.dash_promociones`, `gold.dash_alertas`, `gold.dash_estado_pipeline` |
# MAGIC | **Tiempo** | 5–15 min |

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook
DIAS_HISTORIA = parametro("DIAS_HISTORIA", 56)                 # semanas de historia real que muestra el tablero (8)
UMBRAL_CAMBIO_SERIE = parametro("UMBRAL_CAMBIO_SERIE", 0.30)   # alerta de serie si el pronóstico de 7 días difiere > 30 % de los 7 días reales previos
UMBRAL_INCERTIDUMBRE = parametro("UMBRAL_INCERTIDUMBRE", 3.0)  # alerta si P90 / P50 > 3 en la semana
TABLA_PRONOSTICO = tabla("gold", "pronostico_14d")
TABLA_PANEL = tabla("silver", "panel_categoria_dia")
AHORA = pd.Timestamp.now()
PRON = leer_tabla(TABLA_PRONOSTICO, "22_pronostico_operativo")
if "fase" in PRON.columns:
    PRON = PRON.where(F.coalesce(F.col("fase"), F.lit("operativo")) == "operativo")
ORIGEN = PRON.agg(F.max("origen")).collect()[0][0]
if ORIGEN is None:
    raise RuntimeError("No hay pronóstico operativo: corre 22_pronostico_operativo.")
P = (PRON.where(F.col("origen") == F.lit(ORIGEN))
     .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"), "origen", "fecha_objetivo", "h",
             "tipo_categoria", "region", "prediccion_puntual", "p10", "p90", "modelo", "run_id",
             *([F.col("f_promos_planeadas")] if "f_promos_planeadas" in PRON.columns else [F.lit(0.0).alias("f_promos_planeadas")])))
TIENDAS = leer_tabla(tabla("silver", "catalogo_tiendas"), "08_limpieza_plata").select(F.col("tienda").cast("string").alias("tienda"), F.col("zona").cast("string").alias("zona"),
                                                                                        F.col("jefatura").cast("string").alias("jefatura"), F.col("tipo_tienda").cast("string").alias("tipo_tienda"),
                                                                                        F.col("region").cast("string").alias("region"))
CAT = leer_tabla(tabla("silver", "catalogo_categorias"), "08_limpieza_plata").select(F.col("categoria_id").cast("string").alias("categoria_id"), F.col("nombre").alias("categoria_nombre"))
P = P.join(TIENDAS.drop("region"), "tienda", "left").join(CAT, "categoria_id", "left")
PANEL = leer_tabla(TABLA_PANEL, "09_panel_diario")
DESDE_HIST = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=DIAS_HISTORIA - 1)).date()
REAL = (PANEL.where((F.col("fecha") >= F.lit(DESDE_HIST)) & (F.col("fecha") <= F.lit(ORIGEN)) & (F.col("sin_datos") == 0))
        .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"), "fecha",
                F.col("venta").alias("real"), "tipo_categoria", F.col("tienda_inactiva"),
                (F.coalesce(F.col("promos_vigentes"), F.lit(0)) > 0).cast("double").alias("con_promo"))
        .join(TIENDAS, "tienda", "left").join(CAT, "categoria_id", "left"))
print(f"Origen {ORIGEN} · historia desde {DESDE_HIST}")
NIVELES = {"nacional": [], "region": ["region"], "tipo_categoria": ["tipo_categoria"], "region_tipo": ["region", "tipo_categoria"],
           "zona": ["zona"], "categoria": ["categoria_id", "categoria_nombre", "tipo_categoria"], "tienda": ["tienda", "region", "zona"],
           "serie": ["tienda", "categoria_id", "categoria_nombre", "region", "zona", "tipo_categoria"]}
LLAVES = ["tienda", "categoria_id", "categoria_nombre", "region", "zona", "tipo_categoria"]


def con_llaves(df, dims):
    for c in LLAVES:
        if c not in dims:
            df = df.withColumn(c, F.lit(None).cast("string"))
    return df.select(*LLAVES, *[c for c in df.columns if c not in LLAVES])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Serie de tiempo: historia + pronóstico con banda, en todos los niveles

# COMMAND ----------

titulo("1. SERIE DE TIEMPO")
partes = []
for nivel, dims in NIVELES.items():
    h = (REAL.groupBy(*dims, "fecha").agg(F.sum("real").alias("real"), F.countDistinct("tienda").alias("tiendas"),
                                          F.avg("con_promo").alias("pct_series_con_promo"))
         .withColumn("pronostico", F.lit(None).cast("double")).withColumn("p10", F.lit(None).cast("double")).withColumn("p90", F.lit(None).cast("double"))
         .withColumn("tipo_dato", F.lit("real")))
    p = (P.groupBy(*dims, F.col("fecha_objetivo").alias("fecha")).agg(F.sum("prediccion_puntual").alias("pronostico"), F.sum("p10").alias("p10"), F.sum("p90").alias("p90"),
                                                                     F.countDistinct("tienda").alias("tiendas"),
                                                                     F.avg((F.col("f_promos_planeadas") > 0).cast("double")).alias("pct_series_con_promo"))
         .withColumn("real", F.lit(None).cast("double")).withColumn("tipo_dato", F.lit("pronostico")))
    partes.append(con_llaves(h.unionByName(p, allowMissingColumns=True).withColumn("nivel", F.lit(nivel)), dims))
SERIE = partes[0]
for q in partes[1:]:
    SERIE = SERIE.unionByName(q)
SERIE = SERIE.withColumn("origen", F.lit(ORIGEN)).withColumn("fecha_calculo", F.lit(AHORA))
guardar(SERIE, "gold", "dash_serie_tiempo", "Historia reciente y pronostico con banda P10-P90 por nivel (nacional, region, zona, tipo, categoria, tienda, serie) (40)", particion="nivel")
comentar_columnas(tabla("gold", "dash_serie_tiempo"), {"nivel": "Nivel de agregacion: nacional, region, tipo_categoria, region_tipo, zona, categoria, tienda, serie.",
                                                       "tipo_dato": "real (venta observada) o pronostico.", "p10": "Suma de P10 (banda conservadora al agregar).",
                                                       "p90": "Suma de P90 (banda conservadora al agregar).", "tiendas": "Tiendas que aportan a la fila.",
                                                       "pct_series_con_promo": "Fraccion de series con promocion ese dia: observada (real) o planeada (pronostico)."})
n = spark.table(tabla("gold", "dash_serie_tiempo")).groupBy("nivel").count().toPandas()
mostrar(n, "Filas por nivel", formatos={"count": "int"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Resumen ejecutivo: 14 días pronosticados vs. 14 días reales anteriores

# COMMAND ----------

titulo("2. RESUMEN EJECUTIVO")
prev14 = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=13)).date()
prev28 = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=27)).date()
partes = []
for nivel, dims in NIVELES.items():
    if nivel in ("serie", "tienda"):
        continue
    f = (P.groupBy(*dims).agg(F.sum("prediccion_puntual").alias("venta_pronostico_14d"), F.sum("p10").alias("p10_14d"), F.sum("p90").alias("p90_14d"),
                              F.countDistinct("tienda", "categoria_id").alias("series"), F.countDistinct("tienda").alias("tiendas"),
                              F.avg((F.col("f_promos_planeadas") > 0).cast("double")).alias("pct_filas_con_promo_planeada")))
    r = (REAL.where(F.col("fecha") > F.lit(prev14)).groupBy(*dims).agg(F.sum("real").alias("venta_real_14d_previos")))
    r2 = (REAL.where((F.col("fecha") > F.lit(prev28)) & (F.col("fecha") <= F.lit(prev14))).groupBy(*dims).agg(F.sum("real").alias("venta_real_14d_anteriores")))
    j = f.join(r, dims, "left") if dims else f.crossJoin(r)
    j = j.join(r2, dims, "left") if dims else j.crossJoin(r2)
    partes.append(con_llaves(j.withColumn("nivel", F.lit(nivel)), dims))
RES = partes[0]
for q in partes[1:]:
    RES = RES.unionByName(q)
RES = (RES.withColumn("variacion_vs_14d_previos", F.col("venta_pronostico_14d") / F.col("venta_real_14d_previos") - 1)
       .withColumn("variacion_14d_previos_vs_anteriores", F.col("venta_real_14d_previos") / F.col("venta_real_14d_anteriores") - 1)
       .withColumn("ancho_relativo_banda", (F.col("p90_14d") - F.col("p10_14d")) / F.col("venta_pronostico_14d"))
       .withColumn("origen", F.lit(ORIGEN)).withColumn("desde", F.date_add(F.lit(ORIGEN), 1)).withColumn("hasta", F.date_add(F.lit(ORIGEN), 14))
       .withColumn("fecha_calculo", F.lit(AHORA)))
guardar(RES, "gold", "dash_resumen_ejecutivo", "Venta esperada en los 14 dias siguientes vs 14 dias reales previos por nivel (40)")
comentar_columnas(tabla("gold", "dash_resumen_ejecutivo"), {"variacion_vs_14d_previos": "Pronostico 14 dias / venta real de los 14 dias previos - 1.",
                                                            "ancho_relativo_banda": "(P90 - P10) / pronostico: incertidumbre relativa.",
                                                            "pct_filas_con_promo_planeada": "Fraccion de tienda x categoria x dia con promocion planeada en el horizonte."})
vista = spark.table(tabla("gold", "dash_resumen_ejecutivo")).where(F.col("nivel").isin("nacional", "region", "tipo_categoria")) \
    .select("nivel", "region", "tipo_categoria", "series", "variacion_vs_14d_previos", "ancho_relativo_banda", "pct_filas_con_promo_planeada").toPandas()
mostrar(vista, "Resumen (sin importes)", formatos={"series": "int", "variacion_vs_14d_previos": "pct", "ancho_relativo_banda": "dec2", "pct_filas_con_promo_planeada": "pct"})
nac = vista[vista["nivel"] == "nacional"].iloc[0]
hallazgo(f"Próximos 14 días ({ORIGEN} + 1…14): venta esperada {nac['variacion_vs_14d_previos']:+.1%} respecto a los 14 días reales previos, "
         f"en {int(nac['series']):,} series; {nac['pct_filas_con_promo_planeada']:.0%} de las filas con promoción planeada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Precisión: monitoreo (prueba y producción) y comparativa de modelos

# COMMAND ----------

titulo("3. PRECISIÓN")
if existe(tabla("mlops", "monitoreo_modelo")):
    M = spark.table(tabla("mlops", "monitoreo_modelo"))
    ult = M.groupBy("fase").agg(F.max("fecha_calculo").alias("f")).collect()
    M = M.join(spark.createDataFrame([(r["fase"], r["f"]) for r in ult], "fase string, f timestamp"), "fase").where(F.col("fecha_calculo") == F.col("f")).drop("f")
    guardar(M.withColumn("fecha_tablero", F.lit(AHORA)), "gold", "dash_precision", "WMAPE, sesgo y cobertura por semana x region x tipo en prueba y produccion (ultima corrida de cada fase) (40)")
    mostrar(M.groupBy("fase", "modelo").agg(F.count("*").alias("filas"), F.avg("wmape").alias("wmape_medio"), F.avg("cobertura").alias("cobertura_media")).toPandas(),
            "Monitoreo disponible", formatos={"filas": "int", "wmape_medio": "dec4", "cobertura_media": "pct"})
else:
    print("Sin tabla de monitoreo todavía (corre Avance5.43 completo y 23_monitoreo).")
if existe(tabla("mlops", "av5_comparativa")):
    C = spark.table(tabla("mlops", "av5_comparativa"))
    ultimo_run = C.orderBy(F.col("fecha_publicacion").desc()).select("run_id").limit(1).collect()[0][0]
    cols = [c for c in ("run_id", "modelo", "tipo", "estrategia", "wmape_media_cortes", "mejora_vs_ingenuo", "mejora_vs_mejor_individual", "sesgo_abs_medio_pct",
                        "wmape_mercancia", "segundos_entrenamiento", "fase") if c in C.columns]
    partes = [C.where(F.col("run_id") == ultimo_run).select(*cols)]
    if existe(tabla("mlops", "av5_comparativa_prueba")):
        CP = spark.table(tabla("mlops", "av5_comparativa_prueba"))
        partes.append(CP.where(F.col("run_id") == ultimo_run).select(*[c for c in cols if c in CP.columns]))
    CM = partes[0].unionByName(partes[1], allowMissingColumns=True) if len(partes) > 1 else partes[0]
    guardar(CM.withColumn("fecha_tablero", F.lit(AHORA)), "gold", "dash_precision_modelos", "Comparativa de modelos de la ultima corrida (desarrollo y prueba) (40)")
    mostrar(CM.orderBy("fase", "wmape_media_cortes").toPandas().head(12), "Comparativa (última corrida)", formatos={"wmape_media_cortes": "dec4", "mejora_vs_ingenuo": "pct"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Promociones próximas con el aporte estimado

# COMMAND ----------

titulo("4. PROMOCIONES")
if existe(tabla("silver", "calendario_promociones_plan")):
    PL = spark.table(tabla("silver", "calendario_promociones_plan")).where(F.col("fin_planeado") >= F.lit(ORIGEN)).join(CAT, "categoria_id", "left")
    if existe(tabla("gold", "escenarios_whatif_resumen")):
        E = spark.table(tabla("gold", "escenarios_whatif_resumen")).where(F.col("origen") == F.lit(ORIGEN))
        base = E.where(F.col("id_escenario") == "S_PLAN").groupBy("categoria_id").agg(F.sum("venta_escenario").alias("con_plan"))
        sinp = E.where(F.col("id_escenario") == "S0_SIN_PROMO").groupBy("categoria_id").agg(F.sum("venta_escenario").alias("sin_promo"))
        aporte = base.join(sinp, "categoria_id").withColumn("aporte_plan_pct", (F.col("con_plan") / F.col("sin_promo") - 1) * 100).select("categoria_id", "aporte_plan_pct")
        PL = PL.join(aporte, "categoria_id", "left")
    else:
        PL = PL.withColumn("aporte_plan_pct", F.lit(None).cast("double"))
    PL = (PL.withColumn("dias_para_inicio", F.datediff(F.col("inicio_planeado"), F.lit(ORIGEN)))
          .withColumn("en_horizonte", (F.col("inicio_planeado") <= F.date_add(F.lit(ORIGEN), 14)) & (F.col("fin_planeado") > F.lit(ORIGEN)))
          .withColumn("fecha_calculo", F.lit(AHORA)))
    guardar(PL, "gold", "dash_promociones", "Promociones planeadas vigentes o proximas con el aporte estimado del plan por categoria (40)")
    mostrar(PL.groupBy("fuente", "en_horizonte").agg(F.countDistinct("promo_id").alias("promociones"), F.avg("aporte_plan_pct").alias("aporte_medio_pct")).toPandas(),
            "Promociones por fuente", formatos={"promociones": "int", "aporte_medio_pct": "dec1"})
else:
    print("Sin calendario de promociones (corre 13).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Alertas para el negocio y del monitoreo

# COMMAND ----------

titulo("5. ALERTAS")
prev7 = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=6)).date()
real7 = REAL.where(F.col("fecha") >= F.lit(prev7)).groupBy("tienda", "categoria_id").agg(F.sum("real").alias("real_7d"), F.sum("tienda_inactiva").alias("dias_inactiva_7d"))
pron7 = (P.where(F.col("h") <= 7).groupBy("tienda", "categoria_id", "categoria_nombre", "region", "zona", "tipo_categoria")
         .agg(F.sum("prediccion_puntual").alias("pronostico_7d"), F.sum("p90").alias("p90_7d"), F.sum("prediccion_puntual").alias("p50_7d")))
S = (pron7.join(real7, ["tienda", "categoria_id"], "left")
     .withColumn("cambio", F.when(F.col("real_7d") > 0, F.col("pronostico_7d") / F.col("real_7d") - 1))
     .withColumn("incertidumbre", F.when(F.col("p50_7d") > 0, F.col("p90_7d") / F.col("p50_7d"))))
alertas_series = (S.select("tienda", "categoria_id", "categoria_nombre", "region", "zona", "tipo_categoria",
                           F.when(F.col("cambio") > UMBRAL_CAMBIO_SERIE, "subida_fuerte").when(F.col("cambio") < -UMBRAL_CAMBIO_SERIE, "caida_fuerte")
                           .when(F.col("incertidumbre") > UMBRAL_INCERTIDUMBRE, "alta_incertidumbre")
                           .when(F.col("dias_inactiva_7d") >= 3, "tienda_inactiva").alias("tipo"),
                           F.coalesce(F.col("cambio"), F.lit(0.0)).alias("cambio_7d"), F.col("incertidumbre"), F.col("dias_inactiva_7d"))
                  .where(F.col("tipo").isNotNull())
                  .withColumn("severidad", F.when(F.col("tipo").isin("caida_fuerte", "tienda_inactiva"), "alta").otherwise("media"))
                  .withColumn("mensaje", F.concat(F.lit("Serie "), F.col("categoria_nombre"), F.lit(": "), F.col("tipo"), F.lit(" ("), F.round(F.col("cambio_7d") * 100, 0).cast("int").cast("string"), F.lit(" % vs 7 dias previos)")))
                  .withColumn("origen", F.lit(ORIGEN)).withColumn("fuente", F.lit("series")).withColumn("fecha", F.lit(AHORA)))
if existe(tabla("mlops", "alertas")):
    A = (spark.table(tabla("mlops", "alertas")).where(~F.col("atendida"))
         .select(F.lit(None).cast("string").alias("tienda"), F.lit(None).cast("string").alias("categoria_id"), F.lit(None).cast("string").alias("categoria_nombre"),
                 F.lit(None).cast("string").alias("region"), F.lit(None).cast("string").alias("zona"), F.lit(None).cast("string").alias("tipo_categoria"),
                 "tipo", F.lit(None).cast("double").alias("cambio_7d"), F.lit(None).cast("double").alias("incertidumbre"), F.lit(None).cast("long").alias("dias_inactiva_7d"),
                 "severidad", "mensaje", F.col("origen").cast("date").alias("origen"), F.lit("monitoreo").alias("fuente"), F.col("fecha")))
    alertas_series = alertas_series.withColumn("dias_inactiva_7d", F.col("dias_inactiva_7d").cast("long")).unionByName(A)
guardar(alertas_series, "gold", "dash_alertas", "Alertas para el negocio (series con cambio fuerte, incertidumbre, tiendas inactivas) y del monitoreo (40)")
resumen_al = spark.table(tabla("gold", "dash_alertas")).groupBy("fuente", "tipo", "severidad").count().orderBy("fuente", "tipo").toPandas()
mostrar(resumen_al, "Alertas por tipo", formatos={"count": "int"})
hallazgo("Alertas del tablero: " + ", ".join(f"{t} {int(n):,}" for t, n in zip(resumen_al["tipo"], resumen_al["count"])) + ".")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Estado del pipeline

# COMMAND ----------

titulo("6. ESTADO DEL PIPELINE")
filas = []
def indicador(nombre, valor, detalle="", estado="ok"):
    filas.append({"indicador": nombre, "valor": str(valor), "detalle": str(detalle)[:300], "estado": estado, "fecha_calculo": AHORA})
ultimo_dia = PANEL.where(F.col("sin_datos") == 0).agg(F.max("fecha")).collect()[0][0]
indicador("ultima_venta_cargada", ultimo_dia, f"{(AHORA.date() - ultimo_dia).days} días antes de hoy", "ok" if (AHORA.date() - ultimo_dia).days <= 7 else "aviso")
indicador("origen_pronostico_operativo", ORIGEN, f"pronostica {pd.Timestamp(ORIGEN).date() + pd.Timedelta(days=1)} a {pd.Timestamp(ORIGEN).date() + pd.Timedelta(days=14)}",
          "ok" if ORIGEN >= ultimo_dia else "alerta")
m = P.agg(F.first("modelo").alias("m"), F.first("run_id").alias("r"), F.count("*").alias("n")).collect()[0]
indicador("modelo_en_uso", m["m"], f"corrida {m['r']} · {m['n']:,} filas pronosticadas")
if existe(tabla("mlops", "av5_compuerta")):
    g = spark.table(tabla("mlops", "av5_compuerta")).orderBy(F.col("fecha_publicacion").desc()).limit(1).toPandas().iloc[0]
    indicador("compuerta_tecnica", "cumple" if g["technical_pass"] else "no cumple", f"{g['estado_desarrollo']} · registro: {g['registry_action']}", "ok" if g["technical_pass"] else "alerta")
    indicador("produccion_bloqueada", bool(g["production_blocked"]), "faltan aprobaciones de negocio (mlops.aprobaciones)" if g["production_blocked"] else "", "aviso" if g["production_blocked"] else "ok")
if existe(tabla("mlops", "validaciones")):
    V = spark.table(tabla("mlops", "validaciones"))
    ult = V.agg(F.max("fecha_validacion")).collect()[0][0]
    V = V.where(F.col("fecha_validacion") == F.lit(ult))
    c = V.agg(F.sum((F.col("critica") & (F.col("estado") == "FALLA")).cast("int")).alias("cf"), F.sum((F.col("estado") == "FALLA").cast("int")).alias("f"), F.count("*").alias("n")).collect()[0]
    indicador("reglas_de_calidad", f"{c['n'] - c['f']}/{c['n']} pasan", f"{c['cf']} críticas fallan · validación del {ult}", "ok" if c["cf"] == 0 else "alerta")
if existe(tabla("mlops", "corridas")):
    R = spark.table(tabla("mlops", "corridas")).groupBy("notebook").agg(F.max("fin").alias("fin"), F.max_by("minutos", "fin").alias("minutos")).toPandas()
    for r in R.sort_values("notebook").itertuples():
        indicador(f"notebook {r.notebook}", pd.Timestamp(r.fin).strftime("%Y-%m-%d %H:%M"), f"{r.minutos:.1f} min")
if existe(tabla("mlops", "alertas")):
    na = spark.table(tabla("mlops", "alertas")).where(~F.col("atendida")).count()
    indicador("alertas_monitoreo_abiertas", na, "", "ok" if na == 0 else "aviso")
EST = pd.DataFrame(filas)
guardar(EST, "gold", "dash_estado_pipeline", "Indicadores de estado del pipeline y del modelo para el tablero (40)")
mostrar(EST, "Estado del pipeline")

# COMMAND ----------

cerrar("40_salidas_dashboard", siguiente="importar dashboard/mna_equipo43_demanda.lvdash.json en Databricks (ver dashboard/LEEME.md)")
