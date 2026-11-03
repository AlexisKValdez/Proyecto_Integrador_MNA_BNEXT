# Databricks notebook source
# MAGIC %md
# MAGIC # 31 · Desagregación del pronóstico a producto (SKU) para el surtido en tienda
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Avance 6**
# MAGIC
# MAGIC El modelo pronostica **tienda × categoría** (el grano viable: el producto directo tiene 83 % de ceros). Para que el
# MAGIC pronóstico sirva al surtido, aquí se reparte a los productos de cada categoría en cada tienda con su **participación
# MAGIC móvil de 28 días** (*top-down*), anclada en el origen T del pronóstico (sin mirar días posteriores):
# MAGIC
# MAGIC 1. Participación de cada producto en la venta de su categoría × tienda en los 28 días que terminan en T.
# MAGIC 2. Se conservan los `TOP_SKUS_POR_CATEGORIA` productos con más venta y el resto se agrupa en una fila **OTROS** (así la
# MAGIC    suma de productos reproduce exactamente la categoría y la cola no se asigna en silencio a los líderes).
# MAGIC 3. P50, P10 y P90 de la categoría se reparten proporcionalmente (es una asignación consistente del intervalo, no el
# MAGIC    cuantil propio de cada producto) y se convierten a **unidades** con el precio medio observado del producto:
# MAGIC    `techo_demanda_unidades_p90`. Sin datos de inventario no es un pedido sugerido: es la demanda que cubre el P90.
# MAGIC 4. Se **verifica** la reconciliación (Σ productos = categoría) y se publica una vista **semanal** (h 1–7 y 8–14) que es
# MAGIC    lo que usa el tablero y lo que una tienda revisa al pedir.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.pronostico_14d` (origen operativo más reciente) · `silver.ventas_lineas` · `silver.catalogo_categorias` · `silver.catalogo_tiendas` |
# MAGIC | **Escribe** | `gold.predicciones_sku` (reemplaza su origen) · `gold.surtido_sku_semana` |
# MAGIC | **Parámetros** | `VENTANA_DIAS_SHARE`, `TOP_SKUS_POR_CATEGORIA`, `MODO_PUBLICO` |
# MAGIC | **Tiempo** | 10–30 min |
# MAGIC
# MAGIC **No fabrica pronósticos**: si `gold.pronostico_14d` no existe, se detiene y dice qué correr (`22_pronostico_operativo`).

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook (también como parámetros de Job)
VENTANA_DIAS_SHARE = parametro("VENTANA_DIAS_SHARE", 28)         # días que terminan en T para medir la participación
TOP_SKUS_POR_CATEGORIA = parametro("TOP_SKUS_POR_CATEGORIA", 15)  # productos líderes por categoría × tienda; el resto es OTROS
MODO_PUBLICO = parametro("MODO_PUBLICO", True)
TABLA_PRONOSTICO = tabla("gold", "pronostico_14d")
TABLA_LINEAS = tabla("silver", "ventas_lineas")
TABLA_SKU = tabla("gold", "predicciones_sku")
TABLA_SKU_SEMANA = tabla("gold", "surtido_sku_semana")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pronóstico de categoría y ventana de participación

# COMMAND ----------

titulo("1. PRONÓSTICO BASE")
PRON = leer_tabla(TABLA_PRONOSTICO, "22_pronostico_operativo (o Avance5.43 con REENTRENAR_FINAL = True)")
if "fase" in PRON.columns:
    PRON = PRON.where(F.coalesce(F.col("fase"), F.lit("operativo")) == "operativo")
ORIGEN = PRON.agg(F.max("origen")).collect()[0][0]
if ORIGEN is None:
    raise RuntimeError(f"{TABLA_PRONOSTICO} no tiene pronósticos operativos: corre 22_pronostico_operativo.")
CAT_RUN = (PRON.where(F.col("origen") == F.lit(ORIGEN))
           .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"),
                   "origen", "fecha_objetivo", "h", "tipo_categoria", "region", "prediccion_puntual", "p10", "p50", "p90", "techo_operativo",
                   "modelo", "run_id"))
DESDE = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=VENTANA_DIAS_SHARE - 1)).date()
MESES_VENTANA = sorted({pd.Timestamp(DESDE).strftime("%Y-%m"), pd.Timestamp(ORIGEN).strftime("%Y-%m")})
n = CAT_RUN.agg(F.count("*").alias("filas"), F.countDistinct("tienda", "categoria_id").alias("series")).collect()[0]
print(f"Origen {ORIGEN} · {n['filas']:,} filas de categoría · {n['series']:,} series · ventana de participación {DESDE} → {ORIGEN} ({VENTANA_DIAS_SHARE} días, meses {MESES_VENTANA})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Participación de cada producto (28 días que terminan en T) con fila OTROS

# COMMAND ----------

titulo("2. PARTICIPACIÓN POR PRODUCTO")
L = leer_tabla(TABLA_LINEAS, "08_limpieza_plata")
ventana = (L.where(F.col("anio_mes").isin(MESES_VENTANA) & (F.col("fecha_venta") >= F.lit(DESDE)) & (F.col("fecha_venta") <= F.lit(ORIGEN))
                   & (F.col("venta") > 0) & F.col("producto_id").isNotNull())
           .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"),
                   F.col("producto_id").cast("string").alias("producto_id"), "venta", "cantidad", "fecha_venta"))
sku = (ventana.groupBy("tienda", "categoria_id", "producto_id")
       .agg(F.sum("venta").alias("venta_28d"), F.sum("cantidad").alias("unidades_28d"),
            F.countDistinct("fecha_venta").alias("dias_con_venta_28d"), F.max("fecha_venta").alias("ultima_venta")))
w_cat = Window.partitionBy("tienda", "categoria_id")
w_rank = Window.partitionBy("tienda", "categoria_id").orderBy(F.desc("venta_28d"), F.desc("unidades_28d"), "producto_id")
sku = (sku.withColumn("total_cat_venta", F.sum("venta_28d").over(w_cat)).withColumn("total_cat_unidades", F.sum("unidades_28d").over(w_cat))
       .withColumn("rank_sku", F.row_number().over(w_rank)).withColumn("es_lider", F.col("rank_sku") <= TOP_SKUS_POR_CATEGORIA))
lideres = sku.where("es_lider").select("tienda", "categoria_id", "producto_id", "rank_sku", "venta_28d", "unidades_28d", "dias_con_venta_28d", "ultima_venta", "total_cat_venta")
otros = (sku.where(~F.col("es_lider")).groupBy("tienda", "categoria_id")
         .agg(F.lit("OTROS").alias("producto_id"), F.lit(TOP_SKUS_POR_CATEGORIA + 1).cast("int").alias("rank_sku"), F.sum("venta_28d").alias("venta_28d"),
              F.sum("unidades_28d").alias("unidades_28d"), F.max("dias_con_venta_28d").alias("dias_con_venta_28d"), F.max("ultima_venta").alias("ultima_venta"),
              F.max("total_cat_venta").alias("total_cat_venta"), F.countDistinct("producto_id").alias("productos_en_otros")))
SHARE = (lideres.withColumn("productos_en_otros", F.lit(0).cast("long")).unionByName(otros.withColumn("productos_en_otros", F.col("productos_en_otros").cast("long")))
         .withColumn("share_venta", F.col("venta_28d") / F.col("total_cat_venta"))
         .withColumn("precio_medio", F.col("venta_28d") / F.greatest(F.col("unidades_28d"), F.lit(1e-6))))
r = SHARE.agg(F.countDistinct("tienda", "categoria_id").alias("series"), F.count("*").alias("filas"),
              F.sum(F.when(F.col("producto_id") == "OTROS", F.col("share_venta"))).alias("share_otros_total"),
              F.sum(F.when(F.col("producto_id") == "OTROS", 1).otherwise(0)).alias("series_con_otros")).collect()[0].asDict()
share_otros = (SHARE.where(F.col("producto_id") == "OTROS").agg(F.avg("share_venta")).collect()[0][0]) or 0.0
mostrar(pd.DataFrame([{**r, "share_medio_otros": share_otros}]), "Participaciones calculadas",
        formatos={"series": "int", "filas": "int", "series_con_otros": "int", "share_medio_otros": "pct"})
hallazgo(f"Participación por producto en {r['series']:,} series (ventana de {VENTANA_DIAS_SHARE} días); los {TOP_SKUS_POR_CATEGORIA} líderes cubren en promedio "
         f"el {1 - share_otros:.0%} de la venta y la fila OTROS el {share_otros:.0%}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reparto del pronóstico y techo de demanda en unidades

# COMMAND ----------

titulo("3. REPARTO A PRODUCTO")
con_share = CAT_RUN.join(SHARE, ["tienda", "categoria_id"], "inner")
sin_share = (CAT_RUN.join(SHARE.select("tienda", "categoria_id").distinct(), ["tienda", "categoria_id"], "left_anti")
             .withColumn("producto_id", F.lit("SIN_HISTORIA_SKU")).withColumn("rank_sku", F.lit(0).cast("int")).withColumn("share_venta", F.lit(1.0))
             .withColumn("precio_medio", F.lit(None).cast("double")).withColumn("dias_con_venta_28d", F.lit(0).cast("long")).withColumn("ultima_venta", F.lit(None).cast("date"))
             .withColumn("productos_en_otros", F.lit(0).cast("long")))
SKU = (con_share.select(*CAT_RUN.columns, "producto_id", "rank_sku", "share_venta", "precio_medio", "dias_con_venta_28d", "ultima_venta", "productos_en_otros")
       .unionByName(sin_share.select(*CAT_RUN.columns, "producto_id", "rank_sku", "share_venta", "precio_medio", "dias_con_venta_28d", "ultima_venta", "productos_en_otros"))
       .withColumn("prediccion_sku_p50", F.col("prediccion_puntual") * F.col("share_venta"))
       .withColumn("p10_sku", F.col("p10") * F.col("share_venta")).withColumn("p90_sku", F.col("p90") * F.col("share_venta"))
       .withColumn("techo_demanda_unidades_p90", F.when(F.col("precio_medio") > 0, F.ceil(F.col("p90_sku") / F.col("precio_medio"))).cast("int"))
       .withColumn("demanda_unidades_p50", F.when(F.col("precio_medio") > 0, F.col("prediccion_sku_p50") / F.col("precio_medio")))
       .withColumn("semana_horizonte", F.when(F.col("h") <= 7, 1).otherwise(2))
       .withColumn("fecha_desagregacion", F.current_timestamp())
       .select("tienda", "categoria_id", "origen", "fecha_objetivo", "h", "semana_horizonte", "tipo_categoria", "region", "producto_id", "rank_sku",
               "share_venta", "precio_medio", "dias_con_venta_28d", "ultima_venta", "productos_en_otros",
               "prediccion_sku_p50", "p10_sku", "p90_sku", "demanda_unidades_p50", "techo_demanda_unidades_p90", "modelo", "run_id", "fecha_desagregacion"))
escritor = SKU.write.format("delta").mode("overwrite")
escritor = (escritor.option("replaceWhere", f"origen = DATE'{ORIGEN}'").option("mergeSchema", "true") if existe(TABLA_SKU)
            else escritor.option("overwriteSchema", "true"))
escritor.partitionBy("origen").saveAsTable(TABLA_SKU)
try:
    spark.sql(f"COMMENT ON TABLE {TABLA_SKU} IS 'Pronostico de categoria repartido a producto por participacion de 28 dias (top-down) con fila OTROS; techo de demanda en unidades al P90 (31)'")
except Exception:
    pass
comentar_columnas(TABLA_SKU, {"producto_id": "Producto; OTROS agrupa la cola fuera del top; SIN_HISTORIA_SKU = serie sin ventas en la ventana.",
                              "share_venta": "Participacion del producto en la venta de su categoria x tienda en la ventana.",
                              "prediccion_sku_p50": "Pronostico de la categoria x participacion (pesos).", "p10_sku": "P10 de la categoria x participacion.",
                              "p90_sku": "P90 de la categoria x participacion (asignacion proporcional, no el cuantil propio del producto).",
                              "techo_demanda_unidades_p90": "Unidades que cubren el P90 del producto (p90_sku / precio medio); no descuenta inventario.",
                              "demanda_unidades_p50": "Unidades esperadas (p50).", "semana_horizonte": "1 = h 1-7, 2 = h 8-14."})
print(f"   → guardada {TABLA_SKU}")
# ---- reconciliación verificada: la suma de productos debe igualar la categoría en cada tienda × categoría × día
rec = (spark.table(TABLA_SKU).where(F.col("origen") == F.lit(ORIGEN)).groupBy("tienda", "categoria_id", "fecha_objetivo")
       .agg(F.sum("prediccion_sku_p50").alias("suma_sku"))
       .join(CAT_RUN.select("tienda", "categoria_id", "fecha_objetivo", F.col("prediccion_puntual").alias("cat_pron")), ["tienda", "categoria_id", "fecha_objetivo"])
       .withColumn("dif_rel", F.abs(F.col("suma_sku") - F.col("cat_pron")) / F.greatest(F.col("cat_pron"), F.lit(1.0))))
rr = rec.agg(F.count("*").alias("grupos"), F.max("dif_rel").alias("dif_rel_max"), F.avg("dif_rel").alias("dif_rel_media"),
             F.sum((F.col("dif_rel") > 0.001).cast("int")).alias("grupos_fuera")).collect()[0].asDict()
mostrar(pd.DataFrame([rr]), "Reconciliación Σ productos vs categoría (pronóstico P50)", formatos={"grupos": "int", "dif_rel_max": "dec4", "dif_rel_media": "dec4", "grupos_fuera": "int"})
if rr["grupos_fuera"]:
    aviso(f"{rr['grupos_fuera']:,} de {rr['grupos']:,} tienda × categoría × día no reconcilian (diferencia máxima {rr['dif_rel_max']:.2%}).")
else:
    hallazgo(f"Reconciliación verificada: la suma de productos iguala la categoría en los {rr['grupos']:,} tienda × categoría × día (diferencia máxima {rr['dif_rel_max']:.4%}).")
stats = (spark.table(TABLA_SKU).where(F.col("origen") == F.lit(ORIGEN))
         .agg(F.count("*").alias("filas"), F.countDistinct("producto_id").alias("productos"),
              F.sum((F.col("producto_id") == "SIN_HISTORIA_SKU").cast("int")).alias("filas_sin_historia"),
              F.sum((F.col("producto_id") == "OTROS").cast("int")).alias("filas_otros")).collect()[0].asDict())
mostrar(pd.DataFrame([stats]), "Tabla por producto", formatos={k: "int" for k in stats})
hallazgo(f"{TABLA_SKU}: {stats['filas']:,} filas, {stats['productos']:,} productos distintos (+ OTROS); {stats['filas_sin_historia']:,} filas de series sin historia de producto.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vista semanal para el pedido (h 1–7 y h 8–14)

# COMMAND ----------

titulo("4. VISTA SEMANAL")
SEM = (spark.table(TABLA_SKU).where(F.col("origen") == F.lit(ORIGEN))
       .groupBy("tienda", "categoria_id", "tipo_categoria", "region", "producto_id", "rank_sku", "origen", "semana_horizonte")
       .agg(F.min("fecha_objetivo").alias("desde"), F.max("fecha_objetivo").alias("hasta"), F.max("share_venta").alias("share_venta"),
            F.max("precio_medio").alias("precio_medio"), F.sum("prediccion_sku_p50").alias("venta_p50_semana"), F.sum("p90_sku").alias("venta_p90_semana"),
            F.sum("demanda_unidades_p50").alias("unidades_p50_semana"), F.sum("techo_demanda_unidades_p90").alias("techo_unidades_p90_semana"),
            F.max("modelo").alias("modelo"), F.max("run_id").alias("run_id"))
       .withColumn("fecha_calculo", F.current_timestamp()))
guardar(SEM, "gold", "surtido_sku_semana", "Demanda por producto agregada a semana del horizonte (h 1-7 y 8-14): unidades P50 y techo P90 (31)")
comentar_columnas(TABLA_SKU_SEMANA, {"techo_unidades_p90_semana": "Suma de los techos diarios P90 (conservador: los cuantiles no son aditivos).",
                                     "unidades_p50_semana": "Unidades esperadas en la semana.", "semana_horizonte": "1 = dias 1-7, 2 = dias 8-14 desde el origen."})
top = (SEM.where(F.col("producto_id") != "OTROS").groupBy("tipo_categoria", "semana_horizonte")
       .agg(F.countDistinct("tienda").alias("tiendas"), F.countDistinct("producto_id").alias("productos"), F.avg("share_venta").alias("share_medio")).toPandas())
mostrar(top, "Cobertura de la vista semanal", formatos={"tiendas": "int", "productos": "int", "share_medio": "pct"})
if not MODO_PUBLICO:
    mostrar(SEM.groupBy("semana_horizonte").agg(F.sum("venta_p50_semana").alias("venta_p50"), F.sum("techo_unidades_p90_semana").alias("unidades_p90")).toPandas(),
            "Totales (pesos y unidades)", formatos={"venta_p50": "dinero", "unidades_p90": "int"})

# COMMAND ----------

cerrar("31_desagregacion_predicciones_sku", siguiente="40_salidas_dashboard")
