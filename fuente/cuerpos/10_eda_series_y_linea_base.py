# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · EDA de series y línea base ingenua sobre el panel limpio
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Preguntas que responde**
# MAGIC 1. Con ocho meses de historia (marzo–octubre), ¿a qué grano conviene pronosticar: categoría, marca (todas o las 300 líderes) o producto?
# MAGIC 2. ¿Qué tan intermitentes son las series tienda–categoría (Syntetos–Boylan) y cómo se reparten en ABC–XYZ?
# MAGIC 3. ¿Qué estacionalidad hay y qué efecto tienen los festivos y el Mundial?
# MAGIC 4. ¿Qué error tienen los modelos ingenuos? Es la **línea base nueva** que cualquier modelo debe superar, medida con
# MAGIC    cuatro cortes *walk-forward* de 14 días sobre el panel limpio (sustituye al 0.5002 del modelo heredado, que se
# MAGIC    midió sobre el ABT con duplicados y ceros inventados). También por tipo de categoría (mercancía, servicios, genérico).
# MAGIC 5. ¿El modelo heredado le gana a un ingenuo **en su propia ventana** (13–15 de julio, a un paso)? Se recalcula todo
# MAGIC    sobre las mismas tiendas, categorías y días de sus predicciones.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `silver.panel_categoria_dia` · `silver.ventas_lineas` (para marca y producto) · `outputs/preds_lightgbm_refinado_optuna.parquet` |
# MAGIC | **Escribe** | `mlops.eda_grano` · `eda_series_clasificacion` · `eda_estacionalidad` · `eda_eventos` · `eda_series_benchmark` · `linea_base_ingenua` · `linea_base_ingenua_horizonte` · `linea_base_por_tipo` · `comparacion_modelo_heredado` |
# MAGIC | **Decide** | Grano productivo · variables de calendario y eventos · cortes del walk-forward · métrica a superar |
# MAGIC | **Tiempo** | 15–40 min |

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# @@COLUMNAS

# COMMAND ----------

# @@ESTADISTICAS

# COMMAND ----------

# @@GRAFICAS

# COMMAND ----------

# @@EVENTOS

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook
MUESTRA_TIENDAS_PCT = 25     # % de tiendas (por hash, reproducible) para marca/producto, ingenuos por mes y autocorrelación
UMBRAL_VIABILIDAD = 0.50     # criterio del Avance 2: mediana de días con venta / días activos > 50 %
TOP_MARCAS, TOP_PRODUCTOS = 300, 100
MIN_DIAS = 5
N_CORTES = len(ORIGENES_CORTES)  # walk-forward: cortes de desarrollo fijos (configuración); DIAS_CORTE viene de la configuración
SERIES_ACF = 2000
INICIO = pd.Timestamp(min(MESES) + "-01")
FIN = pd.Timestamp(max(MESES) + "-01") + pd.offsets.MonthEnd(0)

P = leer_tabla(tabla("silver", "panel_categoria_dia"), "09_panel_diario")
L = leer_tabla(tabla("silver", "ventas_lineas"), "08_limpieza_plata")
en_muestra = lambda c: (F.abs(F.xxhash64(F.col(c).cast("string"))) % 100) < MUESTRA_TIENDAS_PCT
activos = P.where((F.col("tienda_inactiva") == 0) & (F.col("sin_datos") == 0) & F.col("venta").isNotNull())
act_tienda = activos.select("tienda", "fecha").distinct().groupBy("tienda").agg(F.count("*").alias("dias_activos"))
act_tienda = persistir(act_tienda)
print(f"Tiendas: {act_tienda.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ¿A qué grano conviene pronosticar?
# MAGIC **Qué hace.** Aplica el criterio del Avance 2 (fijado antes de ver resultados): un grano es viable si la **mediana**,
# MAGIC entre series con al menos 5 días de venta, de *días con venta / días activos de su tienda* supera el 50 %.
# MAGIC Categoría con todas las tiendas; marca y producto con una muestra de tiendas (la mediana no cambia con la muestra).
# MAGIC
# MAGIC **Qué mirar.** Con 92 días el Avance 2 obtuvo: categoría 53.7 % (viable), marca 31.7 % (no), 300 marcas 57.3 %
# MAGIC (viable), producto 11 % (no), 100 productos 41.5 % (piloto). Si con más meses cambia, se documenta la decisión.

# COMMAND ----------

titulo("1. DENSIDAD POR GRANO")
def densidad(series, nombre, act):
    s = series.where(F.col("dias_con_venta") >= MIN_DIAS).join(act, "tienda")
    s = s.withColumn("ratio", F.least(F.col("dias_con_venta") / F.col("dias_activos"), F.lit(1.0)))
    r = s.agg(F.count("*").alias("series"), F.sum("dias_activos").alias("filas_panel"), F.sum("dias_con_venta").alias("dias_venta"),
              F.sum("venta").alias("venta"), F.percentile_approx("ratio", [0.25, 0.5, 0.75], 10000).alias("q"),
              F.sum(F.when(F.col("ratio") > UMBRAL_VIABILIDAD, F.col("venta"))).alias("venta_densa"),
              F.sum(F.col("ratio") * F.col("venta")).alias("ratio_x_venta")).collect()[0]
    filas = r["filas_panel"] or 0
    v = r["venta"] or 0
    return {"grano": nombre, "series": r["series"], "filas_panel": filas,
            "pct_ceros": 1 - (r["dias_venta"] or 0) / filas if filas else np.nan,
            "ratio_p25": r["q"][0] if r["q"] else np.nan, "ratio_mediana": r["q"][1] if r["q"] else np.nan,
            "ratio_p75": r["q"][2] if r["q"] else np.nan, "venta": r["venta"],
            "ratio_ponderado": (r["ratio_x_venta"] or 0) / v if v else np.nan,
            "pct_venta_en_series_densas": (r["venta_densa"] or 0) / v if v else np.nan}

GRANO = []
s_cat = activos.where(F.col("venta") > 0).groupBy("tienda", "categoria_id").agg(
    F.countDistinct("fecha").alias("dias_con_venta"), F.sum("venta").alias("venta"))
GRANO.append(densidad(s_cat, "tienda × categoría (todas las tiendas)", act_tienda))
Lm = L.where(en_muestra("tienda") & (F.col("venta") > 0))
act_m = act_tienda.where(en_muestra("tienda"))
GRANO.append(densidad(s_cat.where(en_muestra("tienda")), f"tienda × categoría (muestra {MUESTRA_TIENDAS_PCT} %)", act_m))
s_m = Lm.groupBy("tienda", "categoria_id", "marca_id").agg(F.countDistinct("fecha_venta").alias("dias_con_venta"), F.sum("venta").alias("venta"))
top_m = (L.groupBy("marca_id").agg(F.sum("venta").alias("v")).orderBy(F.desc("v")).limit(TOP_MARCAS).select("marca_id"))
GRANO.append(densidad(s_m, "tienda × categoría × marca", act_m))
GRANO.append(densidad(s_m.join(top_m, "marca_id"), f"… sólo {TOP_MARCAS} marcas líderes", act_m))
s_p = Lm.groupBy("tienda", "producto_id").agg(F.countDistinct("fecha_venta").alias("dias_con_venta"), F.sum("venta").alias("venta"))
top_p = (L.groupBy("producto_id").agg(F.sum("venta").alias("v")).orderBy(F.desc("v")).limit(TOP_PRODUCTOS).select("producto_id"))
GRANO.append(densidad(s_p, "tienda × producto", act_m))
GRANO.append(densidad(s_p.join(top_p, "producto_id"), f"… sólo {TOP_PRODUCTOS} productos líderes", act_m))
GRANO = pd.DataFrame(GRANO)
GRANO["pct_venta_del_grano"] = GRANO["venta"] / np.where(GRANO.index == 0, GRANO.loc[0, "venta"], GRANO.loc[1, "venta"])
GRANO["viable"] = GRANO["ratio_mediana"] > UMBRAL_VIABILIDAD
GRANO["viable_por_venta"] = GRANO["pct_venta_en_series_densas"] > 0.5
mostrar(GRANO[["grano", "series", "filas_panel", "pct_ceros", "ratio_p25", "ratio_mediana", "ratio_p75", "ratio_ponderado",
               "pct_venta_en_series_densas", "pct_venta_del_grano", "viable", "viable_por_venta"]],
        f"Densidad por grano (series con ≥ {MIN_DIAS} días con venta; 'densa' = vende más de la mitad de sus días activos)",
        formatos={"series": "int", "filas_panel": "int", "pct_ceros": "pct", "ratio_p25": "pct", "ratio_mediana": "pct",
                  "ratio_p75": "pct", "ratio_ponderado": "pct", "pct_venta_en_series_densas": "pct", "pct_venta_del_grano": "pct"})
guardar(GRANO, "mlops", "eda_grano", f"Densidad de las series por grano con {len(MESES)} meses (EDA 10)")
for g in GRANO.itertuples():
    hallazgo(f"{g.grano}: {g.series:,} series, {g.pct_ceros:.1%} ceros, mediana de días con venta {g.ratio_mediana:.1%} "
             f"({'viable' if g.viable else 'no viable'} con el criterio del Avance 2); {g.pct_venta_en_series_densas:.1%} de su venta "
             f"está en series densas ({'viable' if g.viable_por_venta else 'no viable'} si se pondera por venta).")
g0 = GRANO.iloc[0]
hallazgo(f"Decisión de grano: con el criterio del Avance 2 (mediana de las series) tienda × categoría "
         f"{'es' if g0['viable'] else 'no es'} viable ({g0['ratio_mediana']:.1%}); ponderando por venta, "
         f"{g0['pct_venta_en_series_densas']:.1%} de la venta está en series que venden más de la mitad de sus días. "
         + ("Se propone pronosticar tienda × categoría para esas series densas y tratar la cola intermitente aparte (método para "
            "demanda intermitente o agregada a zona o categoría)." if g0["viable_por_venta"] else
            "Conviene un grano más agregado (zona × categoría o tienda × familia) para la mayor parte de la venta."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Intermitencia y segmentación (Syntetos–Boylan, ABC–XYZ)
# MAGIC **Qué hace.** Por serie: **ADI** = días activos / días con venta; **CV²** del tamaño de la venta en días con venta;
# MAGIC clase suave, errática, intermitente o grumosa; **ABC** por venta y **XYZ** por variabilidad diaria (contando ceros).
# MAGIC
# MAGIC **Qué mirar.** Qué parte de la *venta* está en series suaves o erráticas: es la que el modelo puede pronosticar bien.

# COMMAND ----------

titulo("2. INTERMITENCIA Y SEGMENTACIÓN")
ser = (activos.where(F.col("serie_relevante") == 1).groupBy("tienda", "categoria_id")
       .agg(F.count("*").alias("dias_activos"), F.sum((F.col("venta") > 0).cast("int")).alias("dias_con_venta"),
            F.sum("venta").alias("venta"), F.sum(F.col("venta") * F.col("venta")).alias("suma_cuadrados"),
            F.avg(F.when(F.col("venta") > 0, F.col("venta"))).alias("media_pos"),
            F.stddev_pop(F.when(F.col("venta") > 0, F.col("venta"))).alias("desv_pos"))
       .where(F.col("dias_con_venta") >= MIN_DIAS).toPandas())
ser["adi"] = ser["dias_activos"] / ser["dias_con_venta"]
ser["cv2"] = (ser["desv_pos"] / ser["media_pos"]) ** 2
media_all = ser["venta"] / ser["dias_activos"]
ser["cv_diario"] = np.sqrt((ser["suma_cuadrados"] / ser["dias_activos"] - media_all ** 2).clip(lower=0)) / media_all
ser["clase_sb"] = clasificar_sb(ser["adi"], ser["cv2"])
ser["abc"] = clase_abc(ser["venta"])
ser["xyz"] = clase_xyz(ser["cv_diario"])
guardar(ser[["tienda", "categoria_id", "dias_activos", "dias_con_venta", "venta", "adi", "cv2", "cv_diario", "clase_sb", "abc", "xyz"]],
        "mlops", "eda_series_clasificacion", "Una fila por serie tienda x categoria: ADI, CV2, Syntetos-Boylan, ABC y XYZ (EDA 10)")
SB = (ser.groupby("clase_sb").agg(series=("venta", "size"), venta=("venta", "sum"), adi_mediana=("adi", "median"),
                                  cv2_mediana=("cv2", "median")).reset_index())
SB["pct_series"] = SB["series"] / SB["series"].sum()
SB["pct_venta"] = SB["venta"] / SB["venta"].sum()
mostrar(SB[["clase_sb", "series", "pct_series", "pct_venta", "adi_mediana", "cv2_mediana"]], "Clases de Syntetos–Boylan",
        formatos={"series": "int", "pct_series": "pct", "pct_venta": "pct", "adi_mediana": "dec2", "cv2_mediana": "dec2"})
sb = SB.set_index("clase_sb")
hallazgo(f"{len(ser):,} series relevantes; suaves y erráticas: "
         f"{sum(sb.loc[c, 'pct_series'] for c in ('suave', 'errática') if c in sb.index):.1%} de las series y "
         f"{sum(sb.loc[c, 'pct_venta'] for c in ('suave', 'errática') if c in sb.index):.1%} de la venta.")
mat_s = pd.crosstab(ser["abc"], ser["xyz"], normalize="all")
mat_v = pd.crosstab(ser["abc"], ser["xyz"], values=ser["venta"], aggfunc="sum", normalize="all")
mostrar(mat_s.reset_index(), "ABC × XYZ: % de las series", formatos={c: "pct" for c in mat_s.columns})
mostrar(mat_v.reset_index(), "ABC × XYZ: % de la venta", formatos={c: "pct" for c in mat_v.columns})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Estacionalidad y eventos
# MAGIC **Qué hace.** Con la venta nacional diaria del panel: índice por día de la semana en cada mes, índice mensual y efecto
# MAGIC de cada evento (venta del día contra el mismo día de la semana ±4 semanas). Para el Mundial, cambio por categoría.

# COMMAND ----------

titulo("3. ESTACIONALIDAD Y EVENTOS")
nac = activos.groupBy("fecha").agg(F.sum("venta").alias("venta")).orderBy("fecha").toPandas()
nac["fecha"] = pd.to_datetime(nac["fecha"])
nac["mes"] = nac["fecha"].dt.strftime("%Y-%m")
nac["dow"] = nac["fecha"].dt.dayofweek + 1
dias = {1: "lun", 2: "mar", 3: "mié", 4: "jue", 5: "vie", 6: "sáb", 7: "dom"}
DOW_MES = pd.DataFrame([{"mes": m, **{dias[k]: v for k, v in indice_relativo(g["venta"], g["dow"]).items()}}
                        for m, g in nac.groupby("mes")])
mostrar(DOW_MES, "Índice por día de la semana en cada mes", formatos={d: "dec2" for d in dias.values()})
disp = DOW_MES[[d for d in dias.values() if d in DOW_MES]].std().mean()
hallazgo(f"Patrón semanal {'estable' if disp < 0.05 else 'variable'} entre meses (desviación media del índice: {disp:.3f}).")
mes_idx = nac.groupby("mes")["venta"].mean() / nac.groupby("mes")["venta"].mean().mean()
print("Índice mensual de venta diaria: " + ", ".join(f"{m} {v:.2f}" for m, v in mes_idx.items()))
guardar(DOW_MES.melt(id_vars="mes", var_name="dia", value_name="indice"), "mlops", "eda_estacionalidad",
        "Indice de venta por dia de la semana y mes (EDA 10)")
grafica_serie(nac["fecha"], nac["venta"] / 1e6, "Venta diaria nacional del panel limpio", "millones de MXN", EVENTOS)

serie = nac.set_index("fecha")["venta"]
fechas_evento = {pd.Timestamp(e["fecha"]) for e in EVENTOS}
EV = []
for e in EVENTOS:
    f0 = pd.Timestamp(e["fecha"])
    comp = [f for f in (f0 + pd.Timedelta(weeks=k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)) if f in serie.index and f not in fechas_evento]
    if f0 in serie.index and comp:
        b = serie.loc[comp].mean()
        EV.append({"fecha": f0.date(), "evento": e["nombre"], "tipo": e["tipo"], "venta": serie.loc[f0], "base_mismo_dia": b,
                   "efecto": serie.loc[f0] / b - 1})
EV = pd.DataFrame(EV)
if len(EV):
    mostrar(EV, "Efecto de los eventos", formatos={"venta": "dinero", "base_mismo_dia": "dinero", "efecto": "pct"})
    guardar(EV, "mlops", "eda_eventos", "Efecto de cada evento sobre la venta nacional (EDA 10)")
    g = EV.loc[EV["efecto"].abs().idxmax()]
    hallazgo(f"Evento con mayor efecto: {g['evento']} ({g['fecha']}), {g['efecto']:+.1%} contra el mismo día de la semana.")
cat = activos.groupBy("categoria_id", "fecha").agg(F.sum("venta").alias("venta")).toPandas()
cat["fecha"] = pd.to_datetime(cat["fecha"])
for p in PERIODOS:
    ini, fin = pd.Timestamp(p["inicio"]), min(pd.Timestamp(p["fin"]), FIN)
    if ini > FIN or ini - pd.Timedelta(days=28) < INICIO:
        continue
    durante = cat[(cat["fecha"] >= ini) & (cat["fecha"] <= fin)].groupby("categoria_id")["venta"].mean()
    antes = cat[(cat["fecha"] >= ini - pd.Timedelta(days=28)) & (cat["fecha"] < ini)].groupby("categoria_id")["venta"].mean()
    cambio = pd.DataFrame({"venta_dia_antes": antes, "venta_dia_durante": durante}).dropna()
    cambio["cambio"] = cambio["venta_dia_durante"] / cambio["venta_dia_antes"] - 1
    cambio = cambio[cambio["venta_dia_antes"] >= cambio["venta_dia_antes"].quantile(0.5)]
    nac_c = serie[(serie.index >= ini) & (serie.index <= fin)].mean() / serie[(serie.index >= ini - pd.Timedelta(days=28)) & (serie.index < ini)].mean() - 1
    try:
        nombres = spark.table(tabla("silver", "catalogo_categorias")).select("categoria_id", "nombre").toPandas().set_index("categoria_id")["nombre"]
        cambio["nombre"] = cambio.index.map(nombres)
    except Exception:
        pass
    mostrar(cambio.sort_values("cambio", ascending=False).head(10).reset_index(), f"Categorías que más suben durante {p['nombre']}",
            formatos={"venta_dia_antes": "dinero", "venta_dia_durante": "dinero", "cambio": "pct"})
    hallazgo(f"Durante {p['nombre']} la venta diaria nacional cambia {nac_c:+.1%} contra las cuatro semanas previas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Línea base ingenua con walk-forward (pronóstico a 14 días desde un origen)
# MAGIC **Qué hace.** Simula lo que hará el modelo: en cada corte se para en el **origen** (el día anterior al bloque) y
# MAGIC pronostica los 14 días siguientes **sin ver ningún dato del bloque**:
# MAGIC - **Ingenuo estacional**: para los días 1–7 del bloque, la venta del mismo día de la semana anterior al origen;
# MAGIC   para los días 8–14, la de dos semanas antes. Siempre son días ≤ origen.
# MAGIC - **Media móvil de 7 días**: el promedio de los 7 días que terminan en el origen, igual para los 14 días.
# MAGIC
# MAGIC Se evalúan cuatro cortes de 14 días, los más recientes cuyo bloque y los 14 días previos tienen datos completos
# MAGIC (se saltan huecos de entrega como julio 16–31), sobre series con al menos 5 días de venta **antes** del
# MAGIC corte, en días con la tienda activa. Métrica: **WMAPE** = Σ|real − pronóstico| / Σ real; sesgo = Σ pronóstico / Σ real − 1.
# MAGIC
# MAGIC **Qué mirar.** El WMAPE promedio de los cuatro cortes y su error por horizonte (h = 1 … 14): es la vara que el modelo
# MAGIC de la fase 2 debe superar con los mismos cortes. Al final, una tabla por mes con el ingenuo a un paso, sólo descriptiva.

# COMMAND ----------

titulo("4. LÍNEA BASE INGENUA (ORIGEN + 14 DÍAS)")
from pyspark import StorageLevel
claves = ["tienda", "categoria_id"]
panel = persistir(P.select(*claves, "fecha", "venta", "tienda_inactiva", "sin_datos", "en_warmup", "dia_atipico",
                           "tipo_categoria"))
# Cortes fijos de la configuración (los del Avance 4) más los bloques de prueba de septiembre y octubre cuando ya hay datos.
# Cada bloque de 14 días y los 14 días previos a su origen deben tener datos completos (así ningún corte cae en un
# hueco de entrega como julio 16–31). Los resúmenes de la línea base promedian sólo los cortes de desarrollo.
dias_con_datos = {r[0] for r in P.where(F.col("sin_datos") == 0).select("fecha").distinct().collect()}
CORTES = cortes_walk_forward(dias_con_datos)
CORTES_DESARROLLO = [c["corte"] for c in CORTES if c["fase"] == "desarrollo"]
mostrar(pd.DataFrame(CORTES)[["corte", "fase", "origen", "desde", "hasta"]],
        "Cortes walk-forward (desarrollo = Avance 4; prueba = septiembre y octubre, datos no vistos)")
if len(CORTES_DESARROLLO) < N_CORTES:
    aviso(f"Sólo {len(CORTES_DESARROLLO)} cortes de desarrollo tienen datos completos (se esperaban {N_CORTES}).")

partes = []
for c in CORTES:
    origen = c["origen"]
    historia = panel.where(F.col("fecha") <= F.lit(origen))
    relevantes = (historia.where(F.col("venta") > 0).groupBy(*claves).agg(F.count("*").alias("d"))
                  .where(F.col("d") >= MIN_DIAS).select(*claves))
    ma7 = (historia.where(F.col("fecha") > F.date_sub(F.lit(origen), 7))
           .groupBy(*claves).agg(F.avg("venta").alias("ma7"), F.count("venta").alias("n7")))
    bloque = (panel.where(F.col("fecha").between(c["desde"], c["hasta"]) & (F.col("tienda_inactiva") == 0)
                          & (F.col("sin_datos") == 0) & F.col("venta").isNotNull())
              .join(relevantes, claves)
              .withColumn("h", F.datediff("fecha", F.lit(origen)))
              .withColumn("fecha_ref", F.date_sub("fecha", F.when(F.col("h") <= 7, 7).otherwise(14))))
    ref = historia.select(*claves, F.col("fecha").alias("fecha_ref"), F.col("venta").alias("lag_estacional"))
    partes.append(bloque.join(ref, [*claves, "fecha_ref"], "left").join(ma7, claves, "left")
                  .where(F.col("lag_estacional").isNotNull() & (F.col("n7") >= 5))
                  .select(F.lit(c["corte"]).alias("corte"), *claves, "fecha", "h", "venta", "lag_estacional", "ma7",
                          "dia_atipico", "tipo_categoria"))
from functools import reduce
EV_TODOS = persistir(reduce(lambda a, b: a.unionByName(b), partes))
EV = EV_TODOS.where(F.col("corte").isin(CORTES_DESARROLLO))      # los resúmenes usan sólo los cortes de desarrollo


def metricas(df, por):
    return (df.groupBy(por).agg(F.count("*").alias("filas"), F.sum("venta").alias("real"),
                                F.sum(F.abs(F.col("venta") - F.col("lag_estacional"))).alias("e_lag"),
                                F.sum(F.abs(F.col("venta") - F.col("ma7"))).alias("e_ma7"),
                                F.sum("lag_estacional").alias("p_lag"), F.sum("ma7").alias("p_ma7"),
                                F.avg((F.col("venta") == 0).cast("double")).alias("pct_ceros"))
            .orderBy(por).toPandas()
            .assign(wmape_lag7=lambda d: d["e_lag"] / d["real"], wmape_ma7=lambda d: d["e_ma7"] / d["real"],
                    sesgo_lag7=lambda d: d["p_lag"] / d["real"] - 1, sesgo_ma7=lambda d: d["p_ma7"] / d["real"] - 1))


WF_TODOS = metricas(EV_TODOS, "corte").merge(pd.DataFrame(CORTES), on="corte")
WF = WF_TODOS[WF_TODOS["fase"] == "desarrollo"].reset_index(drop=True)
mostrar(WF_TODOS[["corte", "fase", "origen", "desde", "hasta", "filas", "pct_ceros", "wmape_lag7", "wmape_ma7", "sesgo_lag7", "sesgo_ma7"]],
        "Walk-forward: cortes de desarrollo y de prueba, 14 días desde cada origen",
        formatos={"filas": "int", "pct_ceros": "pct", "wmape_lag7": "dec4", "wmape_ma7": "dec4", "sesgo_lag7": "pct", "sesgo_ma7": "pct"})
POR_H = metricas(EV, "h")
mostrar(POR_H[["h", "filas", "wmape_lag7", "wmape_ma7"]], "Error por horizonte (días después del origen), cuatro cortes juntos",
        formatos={"filas": "int", "wmape_lag7": "dec4", "wmape_ma7": "dec4"})
RES = pd.DataFrame([{"modelo": "ingenuo_estacional", "wmape_media": WF["wmape_lag7"].mean(), "wmape_desv": WF["wmape_lag7"].std(),
                     "sesgo_medio": WF["sesgo_lag7"].mean()},
                    {"modelo": "media_movil_7d", "wmape_media": WF["wmape_ma7"].mean(), "wmape_desv": WF["wmape_ma7"].std(),
                     "sesgo_medio": WF["sesgo_ma7"].mean()}])
mostrar(RES, "Línea base (promedio de los cuatro cortes)", formatos={"wmape_media": "dec4", "wmape_desv": "dec4", "sesgo_medio": "pct"})
guardar(WF_TODOS.assign(fecha_calculo=pd.Timestamp.now()), "mlops", "linea_base_ingenua",
        "Linea base ingenua a 14 dias desde el origen (estacional y media movil 7 d), cortes de desarrollo y de prueba (EDA 10)")
if (WF_TODOS["fase"] == "prueba").any():
    prueba_lb = WF_TODOS[WF_TODOS["fase"] == "prueba"]
    hallazgo(f"Bloques de prueba (septiembre y octubre, datos no vistos): ingenuo estacional WMAPE {prueba_lb['wmape_lag7'].mean():.4f} "
             f"en {len(prueba_lb)} bloques (desarrollo: {WF['wmape_lag7'].mean():.4f}).")
guardar(POR_H.assign(fecha_calculo=pd.Timestamp.now()), "mlops", "linea_base_ingenua_horizonte",
        "WMAPE de la linea base ingenua por horizonte h = 1..14 (EDA 10)")
POR_TIPO = metricas(EV, "tipo_categoria")
POR_TIPO["pct_venta"] = POR_TIPO["real"] / POR_TIPO["real"].sum()
mostrar(POR_TIPO[["tipo_categoria", "filas", "pct_venta", "pct_ceros", "wmape_lag7", "wmape_ma7"]],
        "Línea base por tipo de categoría (cuatro cortes juntos)",
        formatos={"filas": "int", "pct_venta": "pct", "pct_ceros": "pct", "wmape_lag7": "dec4", "wmape_ma7": "dec4"})
guardar(POR_TIPO.assign(fecha_calculo=pd.Timestamp.now()), "mlops", "linea_base_por_tipo",
        "WMAPE de la linea base ingenua por tipo de categoria: mercancia, servicio, generico (EDA 10)")
merc = POR_TIPO[POR_TIPO["tipo_categoria"] == "mercancia"]
if len(merc):
    hallazgo(f"Sólo mercancía (sin recargas, pagos ni genérico): WMAPE ingenuo estacional {merc['wmape_lag7'].iloc[0]:.4f} y "
             f"media móvil {merc['wmape_ma7'].iloc[0]:.4f} ({merc['pct_venta'].iloc[0]:.1%} de la venta evaluada).")
sin_atip = metricas(EV.where(F.col("dia_atipico") == 0), "corte")
print(f"Sin días atípicos: WMAPE ingenuo estacional {sin_atip['wmape_lag7'].mean():.4f}, media móvil {sin_atip['wmape_ma7'].mean():.4f}")
mejor = RES.loc[RES["wmape_media"].idxmin()]
hallazgo(f"Línea base nueva (panel limpio, origen + 14 días, 4 cortes): ingenuo estacional WMAPE {RES.loc[0, 'wmape_media']:.4f}, "
         f"media móvil 7 d {RES.loc[1, 'wmape_media']:.4f}. El modelo debe superar {mejor['wmape_media']:.4f} ({mejor['modelo']}) "
         "en los mismos cortes.")
POR_DOW = metricas(EV.withColumn("dow", (F.dayofweek("fecha") + 5) % 7 + 1), "dow")      # 1 = lunes … 7 = domingo
POR_DOW["dia"] = POR_DOW["dow"].map({1: "lun", 2: "mar", 3: "mié", 4: "jue", 5: "vie", 6: "sáb", 7: "dom"})
mostrar(POR_DOW[["dia", "filas", "pct_ceros", "wmape_lag7", "wmape_ma7"]], "Error por día de la semana del día pronosticado",
        formatos={"filas": "int", "pct_ceros": "pct", "wmape_lag7": "dec4", "wmape_ma7": "dec4"})
rango_dow = POR_DOW["wmape_lag7"].max() - POR_DOW["wmape_lag7"].min()
rango_h = POR_H["wmape_lag7"].max() - POR_H["wmape_lag7"].min()
hallazgo((f"El error depende más del día de la semana que del horizonte" if rango_dow > rango_h else
          f"El error crece con el horizonte más de lo que cambia entre días de la semana")
         + f": el ingenuo estacional va de {POR_DOW['wmape_lag7'].min():.3f} ({POR_DOW.loc[POR_DOW['wmape_lag7'].idxmin(), 'dia']}) a "
         f"{POR_DOW['wmape_lag7'].max():.3f} ({POR_DOW.loc[POR_DOW['wmape_lag7'].idxmax(), 'dia']}) según el día pronosticado, y de "
         f"{POR_H['wmape_lag7'].min():.3f} a {POR_H['wmape_lag7'].max():.3f} según el horizonte "
         f"({POR_H['wmape_lag7'].iloc[0]:.3f} en h = 1, {POR_H['wmape_lag7'].iloc[-1]:.3f} en h = {int(POR_H['h'].iloc[-1])}).")

# descriptivo: ingenuo a un paso por mes (lag 7 con la ventana del panel), muestra de tiendas
w_lag = Window.partitionBy(*claves).orderBy("dia").rangeBetween(-7, -7)
mensual = (panel.where(en_muestra("tienda")).withColumn("dia", F.datediff("fecha", F.lit(INICIO.date())))
           .withColumn("lag_estacional", F.max("venta").over(w_lag))
           .withColumn("ma7", F.avg("venta").over(Window.partitionBy(*claves).orderBy("dia").rangeBetween(-7, -1)))
           .where((F.col("tienda_inactiva") == 0) & (F.col("sin_datos") == 0) & (F.col("en_warmup") == 0) & F.col("venta").isNotNull()
                  & F.col("lag_estacional").isNotNull() & F.col("ma7").isNotNull())
           .withColumn("mes", F.date_format("fecha", "yyyy-MM")))
POR_MES = metricas(mensual, "mes")
mostrar(POR_MES[["mes", "filas", "pct_ceros", "wmape_lag7", "wmape_ma7"]],
        f"Dificultad por mes: ingenuo a un paso (descriptivo, muestra {MUESTRA_TIENDAS_PCT} % de tiendas)",
        formatos={"filas": "int", "pct_ceros": "pct", "wmape_lag7": "dec4", "wmape_ma7": "dec4"})
guardar(POR_MES, "mlops", "eda_series_benchmark", "WMAPE de ingenuos a un paso por mes, muestra de tiendas (EDA 10, descriptivo)")
peor = POR_MES.loc[POR_MES["wmape_lag7"].idxmax()]
hallazgo(f"El mes más difícil es {peor['mes']} (WMAPE del ingenuo a un paso {peor['wmape_lag7']:.3f}); conviene que la validación lo incluya.")
liberar(EV_TODOS)



# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Autocorrelación
# MAGIC **Qué mirar.** Una autocorrelación alta en el rezago 7 confirma la estructura semanal que aprovechan los rezagos del modelo.

# COMMAND ----------

titulo("5. AUTOCORRELACIÓN")
largo_min = min(120, int(0.6 * ((FIN - INICIO).days + 1)))
muestra = (activos.where(en_muestra("tienda") & (F.col("serie_relevante") == 1)).groupBy("tienda", "categoria_id").count()
           .where(F.col("count") >= largo_min).orderBy(F.xxhash64("tienda", "categoria_id")).limit(SERIES_ACF))
CAL_ACF = pd.date_range(INICIO, FIN, freq="D")


def acf_de(claves_df, etiqueta):
    """Autocorrelación mediana de un grupo de series, sobre el calendario completo (los días inactivos o sin datos
    quedan como hueco, así un rezago de 7 siempre son 7 días)."""
    pdf = (activos.join(claves_df, ["tienda", "categoria_id"]).select("tienda", "categoria_id", "fecha", "venta").toPandas())
    pdf["fecha"] = pd.to_datetime(pdf["fecha"])
    A = pd.DataFrame([autocorrelacion(g.set_index("fecha")["venta"].reindex(CAL_ACF).to_numpy(), (1, 7, 14, 28))
                      for _, g in pdf.groupby(["tienda", "categoria_id"])])
    if A.empty:
        return None
    return {"grupo": etiqueta, "series": len(A), **{f"rezago_{k}": A[k].median() for k in (1, 7, 14, 28)},
            "p75_rezago_7": A[7].quantile(0.75)}


filas_acf = [acf_de(muestra.select("tienda", "categoria_id"), "series relevantes (muestra)")]
selA = ser.loc[ser["abc"] == "A", ["tienda", "categoria_id"]]
if len(selA):
    selA = selA.sample(min(SERIES_ACF, len(selA)), random_state=42)
    filas_acf.append(acf_de(spark.createDataFrame(selA), "clase A (las que hacen el 80 % de la venta)"))
ACF = pd.DataFrame([f for f in filas_acf if f])
if len(ACF):
    mostrar(ACF, "Autocorrelación mediana por grupo de series",
            formatos={c: "dec2" for c in ACF.columns if c.startswith(("rezago", "p75"))} | {"series": "int"})
    r7_tipica, r7_a = ACF["rezago_7"].iloc[0], (ACF["rezago_7"].iloc[-1] if len(ACF) > 1 else np.nan)
    lectura = ("la estructura semanal sí aparece en las series que pesan" if r7_a >= 0.2 else
               "ni siquiera en las series que pesan hay una estructura semanal fuerte serie por serie")
    hallazgo(f"Autocorrelación mediana en rezago 7: {r7_tipica:.2f} en la serie típica"
             + (f" y {r7_a:.2f} en las series clase A" if len(ACF) > 1 else "")
             + f"; {lectura}. El modelo debe apoyarse también en variables agregadas (patrón de la tienda, la categoría y la zona), "
               "no sólo en los rezagos de la propia serie.")
liberar(act_tienda)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. El modelo heredado contra un ingenuo, en su propia ventana
# MAGIC **Qué hace.** Lee las predicciones del equipo anterior (`outputs/preds_lightgbm_refinado_optuna.parquet`, 13–15 de
# MAGIC julio), las une con el panel limpio por tienda, categoría (homologada) y día, y en esas mismas filas calcula el WMAPE
# MAGIC del modelo y de tres ingenuos **a un paso** (el modelo heredado usaba la venta de ayer como variable): venta de ayer,
# MAGIC venta de hace 7 días y media de los 7 días previos. También compara su `y_real` con la venta del panel limpio.
# MAGIC
# MAGIC **Qué mirar.** Si el modelo no le gana claramente al ingenuo de ayer en sus propias filas, su 0.52 no es una mejora
# MAGIC real; y si `y_real` no coincide con la venta limpia, el modelo aprendió de datos distintos a los de Plata.

# COMMAND ----------

titulo("6. MODELO HEREDADO CONTRA INGENUOS (SU VENTANA)")
COMP = None
try:
    PR = leer_parquet(volumen("outputs", "preds_lightgbm_refinado_optuna.parquet"))
except Exception as e:
    PR = None
    aviso(f"No se pudieron leer las predicciones heredadas ({type(e).__name__}); se omite la comparación.")
if PR is not None:
    cols = {c.lower(): c for c in PR.columns}
    tp = tipos_de(PR)
    c_real = next((cols[k] for k in ("y_real", "venta_total", "y_true", "real", "actual") if k in cols), None)
    c_pred = next((c for c in PR.columns if re.match(r"pred", c, re.I)), None)
    c_fecha = next((cols[k] for k in ("fecha", "date", "fecha_venta") if k in cols), None)
    c_tienda, c_cat = cols.get("store_code"), cols.get("category_id")
    if not all([c_real, c_pred, c_fecha, c_tienda, c_cat]):
        aviso(f"Las predicciones heredadas no traen las columnas esperadas: {PR.columns}")
    else:
        mapa = (spark.table(tabla("silver", "catalogo_categorias"))
                .select(F.col("categoria_id").cast("string").alias("cat_canon"), F.explode(F.split("ids_origen", ",")).alias("origen")))
        pr = (PR.select(F.col(c_tienda).cast("string").alias("tienda_s"), F.col(c_cat).cast("string").alias("origen"),
                        F.to_date(a_timestamp(c_fecha, tp[c_fecha])).alias("fecha"),
                        F.col(c_real).cast("double").alias("y_real"), F.col(c_pred).cast("double").alias("pred"))
              .join(F.broadcast(mapa), "origen", "left")
              .withColumn("cat_s", F.coalesce("cat_canon", "origen")).drop("cat_canon", "origen"))
        rango = pr.agg(F.min("fecha"), F.max("fecha"), F.count("*")).collect()[0]
        desde_p, hasta_p, n_pred = rango[0], rango[1], rango[2]
        w = Window.partitionBy(*claves).orderBy("dia")
        base = (panel.where(F.col("fecha").between(F.date_sub(F.lit(desde_p), 8), F.lit(hasta_p)))
                .withColumn("dia", F.datediff("fecha", F.lit(INICIO.date())))
                .withColumn("lag1", F.max("venta").over(w.rangeBetween(-1, -1)))
                .withColumn("lag7", F.max("venta").over(w.rangeBetween(-7, -7)))
                .withColumn("ma7", F.avg("venta").over(w.rangeBetween(-7, -1)))
                .where(F.col("fecha").between(F.lit(desde_p), F.lit(hasta_p)))
                .select(F.col("tienda").cast("string").alias("tienda_s"), F.col("categoria_id").cast("string").alias("cat_s"),
                        "fecha", "venta", "tienda_inactiva", "sin_datos", "lag1", "lag7", "ma7"))
        J = pr.join(base, ["tienda_s", "cat_s", "fecha"], "left")
        cob = J.agg(F.avg(F.col("venta").isNotNull().cast("double")).alias("con_panel"),
                    F.avg(F.when(F.col("venta").isNotNull(), (F.abs(F.col("y_real") - F.col("venta"))
                                                              <= 0.01 * F.greatest(F.abs("venta"), F.lit(1.0))).cast("double"))).alias("igual"),
                    F.sum("y_real").alias("s_real"), F.sum("venta").alias("s_venta")).collect()[0]
        print(f"Predicciones heredadas: {n_pred:,} filas ({desde_p} a {hasta_p}); {cob['con_panel']:.1%} tienen fila en el panel "
              f"con tienda activa; y_real coincide (±1 %) con la venta limpia en {cob['igual'] or 0:.1%} de ellas.")
        Jv = J.where(F.col("venta").isNotNull() & F.col("lag1").isNotNull() & F.col("lag7").isNotNull() & F.col("ma7").isNotNull())
        m = Jv.agg(F.count("*").alias("filas"), F.sum("venta").alias("real"), F.sum("y_real").alias("real_heredado"),
                   F.sum(F.abs(F.col("y_real") - F.col("pred"))).alias("e_mod_suyo"),
                   F.sum(F.abs(F.col("venta") - F.col("pred"))).alias("e_mod"),
                   F.sum(F.abs(F.col("venta") - F.col("lag1"))).alias("e_lag1"),
                   F.sum(F.abs(F.col("venta") - F.col("lag7"))).alias("e_lag7"),
                   F.sum(F.abs(F.col("venta") - F.col("ma7"))).alias("e_ma7")).collect()[0].asDict()
        if m["filas"] and m["real"]:
            COMP = pd.DataFrame([
                {"pronostico": "modelo heredado contra su propio y_real", "wmape": m["e_mod_suyo"] / m["real_heredado"]},
                {"pronostico": "modelo heredado contra la venta limpia", "wmape": m["e_mod"] / m["real"]},
                {"pronostico": "ingenuo: venta de ayer", "wmape": m["e_lag1"] / m["real"]},
                {"pronostico": "ingenuo: mismo día de la semana anterior", "wmape": m["e_lag7"] / m["real"]},
                {"pronostico": "ingenuo: media de los 7 días previos", "wmape": m["e_ma7"] / m["real"]}])
            COMP["filas"] = m["filas"]
            COMP["ventana"] = f"{desde_p} a {hasta_p}"
            mostrar(COMP, "Mismas filas que las predicciones heredadas (a un paso)", formatos={"wmape": "dec4", "filas": "int"})
            guardar(COMP.assign(fecha_calculo=pd.Timestamp.now()), "mlops", "comparacion_modelo_heredado",
                    "Modelo heredado contra ingenuos a un paso en su ventana de validacion (EDA 10)")
            mejor_ing = COMP.iloc[2:].sort_values("wmape").iloc[0]
            mod = COMP.iloc[1]["wmape"]
            hallazgo(f"En su propia ventana ({desde_p} a {hasta_p}, {m['filas']:,} filas, a un paso) el modelo heredado tiene WMAPE "
                     f"{mod:.4f} contra la venta limpia; el mejor ingenuo ({mejor_ing['pronostico']}) tiene {mejor_ing['wmape']:.4f}: "
                     + ("el modelo sí le gana." if mod < mejor_ing["wmape"] else "el modelo NO le gana al ingenuo."))
liberar(panel)

# COMMAND ----------

cerrar("10_eda_series_y_linea_base", siguiente="11_validacion_calidad")
