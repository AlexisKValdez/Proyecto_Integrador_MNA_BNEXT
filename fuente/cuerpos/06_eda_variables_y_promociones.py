# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · EDA de variables, tickets, catálogo y promociones
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Preguntas que responde**
# MAGIC 1. ¿Cómo se distribuyen cantidad, precio, costo e importe por línea, y dónde empiezan los errores de captura?
# MAGIC 2. ¿Cómo es un ticket típico?
# MAGIC 3. ¿Cuántas tiendas, categorías, marcas y productos entran y salen cada mes?
# MAGIC 4. ¿Qué tan concentrada está la venta (Gini, Pareto)?
# MAGIC 5. ¿El margen es confiable?
# MAGIC 6–9. ¿Cuánto se vende con promoción y descuento, qué tan profundo, en qué escala viene el porcentaje, cuánto duran
# MAGIC    las promociones y cómo se asocian con la venta por categoría?
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `bronze.ventas_lineas` (las sumas de venta excluyen las líneas fuera de rango POS) |
# MAGIC | **Escribe** | `mlops.eda_variables_cuantiles`, `eda_catalogo_mensual`, `eda_concentracion`, `eda_promociones_mensual`, `eda_promociones_catalogo`, `eda_promociones_categoria` |
# MAGIC | **Decide** | Límites de limpieza, segmentación de tiendas, grano del modelo, variables de promoción y el what-if |
# MAGIC | **Tiempo** | 15–30 min |

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
PROBABILIDADES = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
MUESTRA_TICKETS_PCT = 20        # % de tickets (por hash, reproducible) para la distribución de tickets
CORTES_PROFUNDIDAD = [0.0, 0.10, 0.25]   # niveles de descuento: ligero ≤ 10 %, medio 10–25 %, profundo > 25 %
TOP_CATEGORIAS = 15             # categorías principales para el análisis de promoción

df = leer_tabla(tabla("bronze", "ventas_lineas"), "02_carga_bronce")
C = resolver_columnas(df.columns)
TIPOS = tipos_de(df)
N = lambda k: num(C[k])
# venta para sumas (tickets, catálogo, concentración, promociones): sin las líneas fuera de rango POS, que con unos cuantos
# importes de cientos de millones deforman cualquier total; la sección 1 sí muestra la distribución cruda
FUERA_POS = fuera_de_rango_pos(C)
venta = F.when(~FUERA_POS, N("total_con_descuento")) if C["total_con_descuento"] else F.lit(None).cast("double")
sin_desc = F.when(~FUERA_POS, N("total_sin_descuento")) if C["total_sin_descuento"] else None

promo_id, tiene_promo = expr_promo(C, TIPOS)
tasa = (F.when(sin_desc > 0, F.least(F.greatest(1 - venta / sin_desc, F.lit(0.0)), F.lit(1.0)))
        if sin_desc is not None else F.lit(None).cast("double"))
base_promo = df.select(
    "anio_mes", fecha_venta(df).alias("fecha"), venta.alias("venta"),
    (sin_desc if sin_desc is not None else F.lit(None).cast("double")).alias("venta_sin_desc"),
    (num("order_line_subtotal") if "order_line_subtotal" in df.columns else F.lit(None).cast("double")).alias("subtotal"),
    tasa.alias("tasa"), tiene_promo.alias("tiene_promo"), promo_id.alias("promo_id"),
    (F.col(C["promo_nombre"]).cast("string") if C["promo_nombre"] else F.lit(None).cast("string")).alias("promo_nombre"),
    (N("pct_descuento") if C["pct_descuento"] else F.lit(None).cast("double")).alias("pct_desc"),
    (N("descuento_sumasix") if C["descuento_sumasix"] else F.lit(None).cast("double")).alias("sumasix"),
    (N("cantidad") if C["cantidad"] else F.lit(None).cast("double")).alias("unidades"),
    (F.col(C["tienda"]).cast("string") if C["tienda"] else F.lit(None)).alias("tienda"),
    (F.col(C["categoria"]).cast("string") if C["categoria"] else F.lit(None)).alias("categoria"),
    (F.col(C["marca"]).cast("string") if C["marca"] else F.lit(None)).alias("marca"))
con_desc = F.col("tasa") > 0.005

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Distribución de las variables numéricas
# MAGIC **Qué hace.** Cuantiles (del 0.1 % al 99.9 %), media, desviación y asimetría de cada variable numérica, para los seis
# MAGIC meses juntos y por mes. El descuento en pesos se calcula como `total_sin_descuento − total_con_descuento`.
# MAGIC
# MAGIC **Qué mirar.** La distancia entre P99 y el máximo (errores de captura: códigos de barras en el campo de costo) y si
# MAGIC la mediana o el P99 se mueven entre meses (cambio de precios o de captura).

# COMMAND ----------

titulo("1. DISTRIBUCIÓN DE LAS VARIABLES NUMÉRICAS")
variables = {k: N(k) for k in ("cantidad", "precio_unitario", "costo_unitario", "total_sin_descuento",
                               "total_con_descuento", "descuento_sumasix", "pago_puntos") if C[k]}
if C["total_sin_descuento"] and C["total_con_descuento"]:
    variables["descuento_pesos"] = N("total_sin_descuento") - N("total_con_descuento")
base = df.select("anio_mes", *[v.alias(k) for k, v in variables.items()])
aggs = []
for k in variables:
    aggs += [F.percentile_approx(k, PROBABILIDADES, 10000).alias(f"{k}__q"), F.mean(k).alias(f"{k}__media"),
             F.stddev(k).alias(f"{k}__desv"), F.skewness(k).alias(f"{k}__asimetria"), F.min(k).alias(f"{k}__minimo"),
             F.max(k).alias(f"{k}__maximo"), F.avg(F.col(k).isNull().cast("double")).alias(f"{k}__pct_nulos"),
             F.avg((F.col(k) == 0).cast("double")).alias(f"{k}__pct_ceros")]
resultados = pd.concat([base.agg(*aggs).withColumn("anio_mes", F.lit("todos")).toPandas(),   # una pasada: todo el periodo
                        base.groupBy("anio_mes").agg(*aggs).orderBy("anio_mes").toPandas()])  # una pasada: por mes
filas = []
for _, r in resultados.iterrows():
    for k in variables:
        fila = {"variable": k, "anio_mes": r["anio_mes"]}
        for m in ("media", "desv", "asimetria", "minimo", "maximo", "pct_nulos", "pct_ceros"):
            fila[m] = r[f"{k}__{m}"]
        qs = r[f"{k}__q"] if r[f"{k}__q"] is not None else [None] * len(PROBABILIDADES)
        for p, q in zip(PROBABILIDADES, qs):
            fila[f"p{p * 100:g}"] = q
        filas.append(fila)
CUANT = pd.DataFrame(filas)
guardar(CUANT, "mlops", "eda_variables_cuantiles", "Cuantiles y momentos de las variables numericas (EDA 06)")
todos = CUANT[CUANT["anio_mes"] == "todos"]
cols_q = ["variable", "minimo", "p1", "p25", "p50", "p75", "p99", "p99.9", "maximo", "media", "asimetria", "pct_ceros"]
mostrar(todos[[c for c in cols_q if c in todos.columns]], "Todo el periodo, sin limpiar",
          formatos={"pct_ceros": "pct", "asimetria": "dec2"})
mes = CUANT[CUANT["anio_mes"] != "todos"]
for col in ("p50", "p99"):
    piv = mes.pivot(index="variable", columns="anio_mes", values=col)
    mostrar(piv.reset_index(), f"{col.upper()} por mes")
for k in ("costo_unitario", "cantidad", "total_con_descuento"):
    if k in variables:
        r = todos.set_index("variable").loc[k]
        hallazgo(f"{k}: mediana {r['p50']:,.2f}, P99 {r['p99']:,.2f}, P99.9 {r['p99.9']:,.2f}, máximo {r['maximo']:,.0f} "
                     f"(asimetría {r['asimetria']:.1f}).")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tickets
# MAGIC **Qué hace.** Agrupa las líneas por `order_id` (sobre una muestra reproducible de tickets) y describe líneas,
# MAGIC unidades e importe por ticket, y tickets por tienda y día.
# MAGIC
# MAGIC **Qué mirar.** El ticket típico de conveniencia es chico (pocas líneas); tickets con cientos de líneas o importes muy
# MAGIC altos son compras mayoristas o errores, y conviene marcarlos antes de agregar a nivel tienda–categoría–día.

# COMMAND ----------

titulo("2. TICKETS")
if requiere(C, "id_ticket"):
    tk = (df.where((F.abs(F.xxhash64(F.col(C["id_ticket"]))) % 100) < MUESTRA_TICKETS_PCT)
          .groupBy(F.col(C["id_ticket"]).alias("ticket"))
          .agg(F.count("*").alias("lineas"),
               F.sum(N("cantidad") if C["cantidad"] else F.lit(None)).alias("unidades"),
               F.sum(venta).alias("importe")))
    q = tk.agg(*[F.percentile_approx(c, [0.5, 0.9, 0.99, 0.999], 10000).alias(c) for c in ("lineas", "unidades", "importe")],
               *[F.max(c).alias(f"max_{c}") for c in ("lineas", "unidades", "importe")],
               F.count("*").alias("tickets")).collect()[0]
    TK = pd.DataFrame([{"medida": c, "p50": q[c][0], "p90": q[c][1], "p99": q[c][2], "p99.9": q[c][3],
                        "maximo": q[f"max_{c}"]} for c in ("lineas", "unidades", "importe") if q[c] is not None])
    mostrar(TK, f"Por ticket ({q['tickets']:,} tickets de la muestra del {MUESTRA_TICKETS_PCT} %)")
    if len(TK):
        r = TK.set_index("medida")
        hallazgo(f"Ticket típico: {r.loc['lineas', 'p50']:.0f} líneas e importe mediano de ${r.loc['importe', 'p50']:,.2f}; "
                     f"el 1 % más grande pasa de {r.loc['lineas', 'p99']:.0f} líneas o ${r.loc['importe', 'p99']:,.0f}.")
    if C["tienda"]:
        try:
            fecha = fecha_venta(df)
            tpd = (df.groupBy(F.col(C["tienda"]).alias("tienda"), fecha.alias("fecha"))
                   .agg(F.approx_count_distinct(F.col(C["id_ticket"]), 0.02).alias("tickets"))
                   .agg(F.percentile_approx("tickets", [0.1, 0.5, 0.9], 1000).alias("q")).collect()[0]["q"])
            print(f"Tickets por tienda y día: P10 {tpd[0]:,}, mediana {tpd[1]:,}, P90 {tpd[2]:,}")
        except KeyError as e:
            aviso(str(e))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Catálogo por mes: entradas y salidas
# MAGIC **Qué hace.** Para tiendas, categorías, marcas y productos cuenta cuántos venden cada mes, cuántos son **nuevos**
# MAGIC (no vendían el mes anterior) y cuántos **desaparecen** (vendían el mes anterior y ya no), con la venta de los nuevos.
# MAGIC
# MAGIC **Qué mirar.** La rotación de productos justifica no modelar a nivel SKU (arranque en frío permanente); la de
# MAGIC tiendas define cuántas series nacen o mueren en la ventana de entrenamiento.

# COMMAND ----------

titulo("3. CATÁLOGO POR MES")
CAT = []
for k in ("tienda", "categoria", "marca", "producto"):
    if not C[k]:
        continue
    pm = (df.groupBy(F.col(C[k]).cast("string").alias("id"), "anio_mes").agg(F.sum(venta).alias("venta")).toPandas())
    meses = sorted(pm["anio_mes"].unique())
    conjuntos = {m: set(pm.loc[pm["anio_mes"] == m, "id"]) for m in meses}
    for i, m in enumerate(meses):
        act = conjuntos[m]
        prev = conjuntos[meses[i - 1]] if i else set()
        nuevos = act - prev if i else set()
        venta_mes = pm.loc[pm["anio_mes"] == m, "venta"].sum()
        venta_nuevos = pm.loc[(pm["anio_mes"] == m) & pm["id"].isin(nuevos), "venta"].sum()
        CAT.append({"entidad": k, "anio_mes": m, "activos": len(act), "nuevos": len(nuevos) if i else None,
                    "desaparecen": len(prev - act) if i else None,
                    "pct_venta_nuevos": venta_nuevos / venta_mes if (i and venta_mes) else None})
    presentes_todos = set.intersection(*conjuntos.values()) if conjuntos else set()
    print(f"{k}: {pm['id'].nunique():,} distintos en el periodo; presentes en los {len(meses)} meses: {len(presentes_todos):,}")
CAT = pd.DataFrame(CAT)
guardar(CAT, "mlops", "eda_catalogo_mensual", "Entidades activas, nuevas y que desaparecen por mes (EDA 06)")
mostrar(CAT, "Entradas y salidas del catálogo", formatos={"activos": "int", "nuevos": "int", "desaparecen": "int",
                                                             "pct_venta_nuevos": "pct"}, max_filas=40)
prod = CAT[CAT["entidad"] == "producto"].dropna(subset=["nuevos"])
if len(prod):
    hallazgo(f"Productos: en promedio {prod['nuevos'].mean():,.0f} nuevos y {prod['desaparecen'].mean():,.0f} que desaparecen "
                 f"cada mes (venta de los nuevos: {prod['pct_venta_nuevos'].mean():.1%} del mes).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Concentración de la venta
# MAGIC **Qué hace.** Índice de Gini, parte de la venta del 1 %, 10 % y 20 % más grandes y fracción de entidades necesaria
# MAGIC para acumular el 80 % de la venta, para tiendas, categorías, marcas y productos; curvas de Lorenz y las 20
# MAGIC categorías principales.
# MAGIC
# MAGIC **Qué mirar.** Una concentración alta entre tiendas justifica la segmentación por volumen (terciles A/B/C) y
# MAGIC reportar el error por segmento: un buen WMAPE global puede esconder un error alto en tiendas chicas.

# COMMAND ----------

titulo("4. CONCENTRACIÓN DE LA VENTA")
CONC, curvas, ginis, totales = [], {}, {}, {}
etiquetas = {"tienda": "Tiendas", "categoria": "Categorías", "marca": "Marcas", "producto": "Productos"}
for k in ("tienda", "categoria", "marca", "producto"):
    if not C[k]:
        continue
    t = df.groupBy(F.col(C[k]).cast("string").alias("id")).agg(F.sum(venta).alias("venta")).toPandas()
    t = t[t["venta"] > 0]
    totales[k] = t
    top = participacion_top(t["venta"], (0.01, 0.10, 0.20))
    g = gini(t["venta"])
    CONC.append({"entidad": k, "n": len(t), "gini": g, "top_1pct": top[0.01], "top_10pct": top[0.10],
                 "top_20pct": top[0.20], "fraccion_para_80pct": fraccion_para(t["venta"], 0.80)})
    curvas[etiquetas[k]] = curva_lorenz(t["venta"])
    ginis[etiquetas[k]] = g
CONC = pd.DataFrame(CONC)
guardar(CONC, "mlops", "eda_concentracion", "Concentracion de la venta por tipo de entidad (EDA 06)")
mostrar(CONC, "Concentración", formatos={"n": "int", "gini": "dec2", "top_1pct": "pct", "top_10pct": "pct",
                                           "top_20pct": "pct", "fraccion_para_80pct": "pct"})
for r in CONC.itertuples():
    hallazgo(f"{etiquetas[r.entidad]}: Gini {r.gini:.2f}; el 10 % más grande concentra {r.top_10pct:.1%} de la venta y "
                 f"{r.fraccion_para_80pct:.1%} de ellas acumulan el 80 %.")
if "categoria" in totales and C["categoria_nombre"]:
    nombres = (df.groupBy(F.col(C["categoria"]).cast("string").alias("id"))
               .agg(F.first(F.col(C["categoria_nombre"]), ignorenulls=True).alias("nombre")).toPandas())
    top = totales["categoria"].merge(nombres, on="id", how="left").sort_values("venta", ascending=False)
    top["pct"] = top["venta"] / top["venta"].sum()
    top["pct_acumulado"] = top["pct"].cumsum()
    mostrar(top.head(20)[["id", "nombre", "venta", "pct", "pct_acumulado"]], "20 categorías con más venta",
              formatos={"venta": "dinero", "pct": "pct", "pct_acumulado": "pct"})
    t20 = top.head(20)
    grafica_barras([str(n)[:28] for n in t20["nombre"].fillna(t20["id"])], t20["pct"] * 100,
                   "Participación en la venta, 20 categorías principales", "% de la venta del periodo",
                   formato="{:.1f} %", horizontal=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Precio y margen
# MAGIC **Qué hace.** Margen unitario `1 − costo / precio` en las líneas con ambos positivos.
# MAGIC
# MAGIC **Qué mirar.** Márgenes negativos o mayores a 90 % en una parte grande de las líneas indican que el costo no es
# MAGIC confiable y no conviene usarlo como variable.

# COMMAND ----------

titulo("5. PRECIO Y MARGEN")
if requiere(C, "precio_unitario", "costo_unitario"):
    m = (df.where((N("precio_unitario") > 0) & (N("costo_unitario") > 0))
         .select((1 - N("costo_unitario") / N("precio_unitario")).alias("margen")))
    r = m.agg(F.percentile_approx("margen", [0.01, 0.1, 0.5, 0.9, 0.99], 10000).alias("q"),
              F.avg((F.col("margen") < 0).cast("double")).alias("neg"),
              F.avg((F.col("margen") > 0.9).cast("double")).alias("alto")).collect()[0]
    print(f"Margen unitario: P1 {r['q'][0]:.2f}, P10 {r['q'][1]:.2f}, mediana {r['q'][2]:.2f}, P90 {r['q'][3]:.2f}, P99 {r['q'][4]:.2f}")
    hallazgo(f"Margen unitario mediano {r['q'][2]:.1%}; {r['neg']:.1%} de las líneas con margen negativo y {r['alto']:.1%} "
                 "con margen mayor a 90 %.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Promociones: incidencia por mes
# MAGIC **Qué hace.** Por mes: % de líneas y de venta con identificador de promoción, con descuento (reconstruido de los
# MAGIC importes), con descuento SumaSIX; número de promociones distintas y qué parte de los descuentos tiene promoción.
# MAGIC
# MAGIC **Qué mirar.** Si la incidencia cambia mucho entre meses (campañas de temporada, Mundial) y si los descuentos se
# MAGIC explican por promociones registradas o hay descuentos sin promoción.

# COMMAND ----------

titulo("6. PROMOCIONES: INCIDENCIA POR MES")
MENS = (base_promo.groupBy("anio_mes")
        .agg(F.count("*").alias("lineas"), F.sum("venta").alias("venta"),
             F.avg(F.col("tiene_promo").cast("double")).alias("pct_lineas_promo"),
             F.sum(F.when(F.col("tiene_promo"), F.col("venta"))).alias("venta_promo"),
             F.avg(con_desc.cast("double")).alias("pct_lineas_descuento"),
             F.sum(F.col("venta_sin_desc") - F.col("venta")).alias("descuento_pesos"),
             F.avg((F.col("sumasix") > 0).cast("double")).alias("pct_lineas_sumasix"),
             F.countDistinct(F.when(F.col("tiene_promo"), F.col("promo_id"))).alias("promociones"),
             F.avg(F.when(con_desc, F.col("tiene_promo").cast("double"))).alias("pct_descuentos_con_promo"))
        .orderBy("anio_mes").toPandas())
MENS["pct_venta_promo"] = MENS["venta_promo"] / MENS["venta"]
MENS["pct_descuento_sobre_venta"] = MENS["descuento_pesos"] / (MENS["venta"] + MENS["descuento_pesos"])
guardar(MENS, "mlops", "eda_promociones_mensual", "Incidencia de promociones y descuentos por mes (EDA 06)")
mostrar(MENS[["anio_mes", "pct_lineas_promo", "pct_venta_promo", "pct_lineas_descuento", "pct_descuento_sobre_venta",
                "pct_lineas_sumasix", "promociones", "pct_descuentos_con_promo"]], "Incidencia por mes",
          formatos={**{c: "pct" for c in ["pct_lineas_promo", "pct_venta_promo", "pct_lineas_descuento",
                                          "pct_descuento_sobre_venta", "pct_lineas_sumasix", "pct_descuentos_con_promo"]},
                    "promociones": "int"})
tot_l = MENS["lineas"].sum()
hallazgo(f"{(MENS['pct_lineas_promo'] * MENS['lineas']).sum() / tot_l:.1%} de las líneas llevan identificador de promoción "
             f"({MENS['venta_promo'].sum() / MENS['venta'].sum():.1%} de la venta); "
             f"{(MENS['pct_lineas_descuento'] * MENS['lineas']).sum() / tot_l:.1%} tienen descuento en importe. "
             f"Promociones distintas por mes: {MENS['promociones'].min():,} a {MENS['promociones'].max():,}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Profundidad del descuento y escala del porcentaje
# MAGIC **Qué hace.** Calcula la tasa efectiva por línea, `1 − total_con_descuento / total_sin_descuento`, la agrupa en
# MAGIC niveles (sin, ligero, medio, profundo) y compara contra `order_line_discount_percent` para descubrir su escala:
# MAGIC si coincide con la tasa × 100 es un porcentaje; si coincide con la tasa sin multiplicar es una fracción; si coincide
# MAGIC con la tasa × 10,000 tiene un error de escala.
# MAGIC
# MAGIC **Qué mirar.** Qué escala explica la mayoría de las líneas: define cómo se corrige el campo en Plata (o si se
# MAGIC descarta y se usa sólo la tasa reconstruida, como en el Avance 2).

# COMMAND ----------

titulo("7. PROFUNDIDAD DEL DESCUENTO Y ESCALA DEL PORCENTAJE")
nivel = (F.when(F.col("tasa").isNull(), "sin dato").when(F.col("tasa") <= 0.005, "sin descuento")
         .when(F.col("tasa") <= CORTES_PROFUNDIDAD[1], "ligero (≤10 %)")
         .when(F.col("tasa") <= CORTES_PROFUNDIDAD[2], "medio (10–25 %)").otherwise("profundo (>25 %)"))
NIV = (base_promo.groupBy("anio_mes", nivel.alias("nivel")).agg(F.count("*").alias("lineas"), F.sum("venta").alias("venta"))
       .toPandas())
NIV["pct_lineas"] = NIV["lineas"] / NIV.groupby("anio_mes")["lineas"].transform("sum")
piv = NIV.pivot(index="nivel", columns="anio_mes", values="pct_lineas").reset_index()
mostrar(piv, "% de líneas por nivel de descuento y mes", formatos={c: "pct" for c in piv.columns if c != "nivel"})
q = base_promo.where(con_desc).agg(F.percentile_approx("tasa", [0.1, 0.25, 0.5, 0.75, 0.9, 0.99], 10000).alias("q")).collect()[0]["q"]
if q:
    hallazgo(f"En las líneas con descuento la tasa efectiva tiene mediana {q[2]:.1%} (P25 {q[1]:.1%}, P75 {q[3]:.1%}, "
                 f"P90 {q[4]:.1%}, P99 {q[5]:.1%}).")
if C["pct_descuento"]:
    esc = (base_promo.where(con_desc & (F.col("pct_desc") > 0))
           .select(F.when(F.abs(F.col("pct_desc") / 100 - F.col("tasa")) < 0.01, "porcentaje (×100)")
                   .when(F.abs(F.col("pct_desc") - F.col("tasa")) < 0.01, "fracción (×1)")
                   .when(F.abs(F.col("pct_desc") / 10000 - F.col("tasa")) < 0.01, "error de escala (×10,000)")
                   .when(F.abs(F.col("pct_desc") - (F.col("venta_sin_desc") - F.col("venta"))) < 0.05,
                         "monto en pesos del descuento")
                   .when((F.col("subtotal") > 0) &
                         (F.abs(F.col("pct_desc") / 100 - F.col("tasa") * F.col("venta_sin_desc")
                                / F.when(F.col("subtotal") > 0, F.col("subtotal"))) < 0.01),
                         "porcentaje sobre el subtotal sin impuestos")
                   .otherwise("no coincide").alias("escala"))
           .groupBy("escala").count().toPandas())
    esc["pct"] = esc["count"] / esc["count"].sum()
    mostrar(esc.sort_values("count", ascending=False), "Escala de order_line_discount_percent frente a la tasa reconstruida",
              formatos={"count": "int", "pct": "pct"})
    # cuando no coincide: razón entre el campo y la tasa reconstruida (×100); una razón constante delata otra base de cálculo
    rz = (base_promo.where(con_desc & (F.col("pct_desc") > 0))
          .select((F.col("pct_desc") / F.when(F.col("tasa") > 0, F.col("tasa") * 100)).alias("razon"))
          .agg(F.percentile_approx("razon", [0.05, 0.25, 0.5, 0.75, 0.95], 10000).alias("q")).collect()[0]["q"])
    if rz:
        print("Razón order_line_discount_percent / (tasa reconstruida × 100): "
              + ", ".join(f"P{int(p * 100)} {v:,.3f}" for p, v in zip([0.05, 0.25, 0.5, 0.75, 0.95], rz)))
    sin_pct = base_promo.where(con_desc).agg(F.avg((F.coalesce(F.col("pct_desc"), F.lit(0)) == 0).cast("double"))).collect()[0][0]
    hallazgo(f"order_line_discount_percent: escala dominante «{esc.sort_values('count').iloc[-1]['escala']}» "
                 f"({esc['pct'].max():.0%} de las líneas comparables); {sin_pct:.1%} de las líneas con descuento lo tienen en cero.")
orden = ["sin descuento", "ligero (≤10 %)", "medio (10–25 %)", "profundo (>25 %)"]
tot_niv = NIV.groupby("nivel")["lineas"].sum().reindex(orden).fillna(0)
grafica_barras(orden, tot_niv / tot_niv.sum() * 100, "Líneas por nivel de descuento", "% de líneas", formato="{:.1f} %")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Ciclo de vida de las promociones
# MAGIC **Qué hace.** Una fila por promoción: primera y última fecha, días activos, duración, tiendas, categorías y marcas
# MAGIC que alcanza, líneas, venta y descuento. Luego resume duración, alcance y en qué día de la semana arrancan.
# MAGIC
# MAGIC **Qué mirar.** Si la mayoría de la venta en promoción viene de promociones con fechas definidas y de varios días,
# MAGIC su calendario se puede reconstruir del histórico y usar como **variable conocida a futuro** (lo que se planea antes
# MAGIC de que ocurra). Si son promociones de un día o sin patrón, la promoción sólo puede entrar rezagada.

# COMMAND ----------

titulo("8. CICLO DE VIDA DE LAS PROMOCIONES")
if C["promo_id"]:
    total_tiendas = base_promo.select(F.countDistinct("tienda")).collect()[0][0]
    PROMOS = (base_promo.where(F.col("tiene_promo") & F.col("fecha").isNotNull())
              .groupBy("promo_id")
              .agg(F.first("promo_nombre", ignorenulls=True).alias("nombre"),
                   F.min("fecha").alias("inicio"), F.max("fecha").alias("fin"),
                   F.countDistinct("fecha").alias("dias_activos"), F.countDistinct("tienda").alias("tiendas"),
                   F.countDistinct("categoria").alias("categorias"), F.countDistinct("marca").alias("marcas"),
                   F.count("*").alias("lineas"), F.sum("venta").alias("venta"),
                   F.sum(F.col("venta_sin_desc") - F.col("venta")).alias("descuento_pesos"),
                   F.avg("tasa").alias("tasa_media"))
              .withColumn("duracion", F.datediff("fin", "inicio") + 1)
              .withColumn("dia_semana_inicio", F.dayofweek("inicio"))      # 1 = domingo en Spark
              .withColumn("dia_mes_inicio", F.dayofmonth("inicio"))
              .toPandas())
    PROMOS["pct_tiendas"] = PROMOS["tiendas"] / total_tiendas if total_tiendas else np.nan
    PROMOS["continuidad"] = PROMOS["dias_activos"] / PROMOS["duracion"]
    guardar(PROMOS, "mlops", "eda_promociones_catalogo", "Una fila por promocion: fechas, alcance y venta (EDA 06)")
    v = PROMOS["venta"].sum()
    largas = PROMOS["duracion"] >= 7
    nacionales = PROMOS["pct_tiendas"] >= 0.5
    print(f"Promociones distintas: {len(PROMOS):,}; duración mediana {PROMOS['duracion'].median():.0f} días; "
              f"continuidad mediana {PROMOS['continuidad'].median():.0%} (días activos / duración)")
    resumen = pd.DataFrame([
        {"grupo": "duración ≥ 7 días", "promociones": int(largas.sum()), "pct_venta_promo": PROMOS.loc[largas, "venta"].sum() / v},
        {"grupo": "duración 1 día", "promociones": int((PROMOS["duracion"] == 1).sum()),
         "pct_venta_promo": PROMOS.loc[PROMOS["duracion"] == 1, "venta"].sum() / v},
        {"grupo": "en ≥ 50 % de las tiendas", "promociones": int(nacionales.sum()), "pct_venta_promo": PROMOS.loc[nacionales, "venta"].sum() / v},
        {"grupo": "continuidad ≥ 90 %", "promociones": int((PROMOS["continuidad"] >= 0.9).sum()),
         "pct_venta_promo": PROMOS.loc[PROMOS["continuidad"] >= 0.9, "venta"].sum() / v},
    ])
    mostrar(resumen, "Estructura de las promociones", formatos={"promociones": "int", "pct_venta_promo": "pct"})
    pct_largas = PROMOS.loc[largas, "venta"].sum() / v
    pct_continuas = PROMOS.loc[PROMOS["continuidad"] >= 0.9, "venta"].sum() / v
    if pct_largas >= 0.6 and pct_continuas >= 0.5:
        lectura = ("las promociones tienen vigencias largas y continuas: su calendario se puede reconstruir con inicio y fin "
                   "y usar como variable conocida a futuro")
    elif pct_largas >= 0.6:
        lectura = (f"las promociones duran semanas pero aparecen de forma intermitente dentro de su vigencia (continuidad mediana "
                   f"{PROMOS['continuidad'].median():.0%}): el calendario debe reconstruirse por día de presencia, no sólo con inicio y fin")
    else:
        lectura = "predominan promociones cortas: la promoción sólo puede entrar al modelo rezagada o con un calendario de Bnext"
    hallazgo(f"{pct_largas:.1%} de la venta en promoción viene de promociones de 7 días o más y {pct_continuas:.1%} de "
                 f"promociones activas casi todos los días de su vigencia; {lectura}.")
    dias = {1: "dom", 2: "lun", 3: "mar", 4: "mié", 5: "jue", 6: "vie", 7: "sáb"}
    arranque = PROMOS.groupby("dia_semana_inicio")["venta"].sum() / v
    print("Día de la semana en que arrancan (por venta): "
              + ", ".join(f"{dias[k]} {x:.0%}" for k, x in arranque.sort_values(ascending=False).items()))
    mostrar(PROMOS.sort_values("venta", ascending=False).head(20)[
        ["promo_id", "nombre", "inicio", "fin", "dias_activos", "tiendas", "categorias", "venta", "tasa_media"]],
        "20 promociones con más venta", formatos={"tiendas": "int", "categorias": "int", "venta": "dinero", "tasa_media": "pct"})
    activas = (base_promo.where(F.col("tiene_promo") & F.col("fecha").isNotNull()).groupBy("fecha")
               .agg(F.countDistinct("promo_id").alias("promociones_activas"),
                    F.sum("venta").alias("venta_promo")).orderBy("fecha").toPandas())
    total_dia = base_promo.where(F.col("fecha").isNotNull()).groupBy("fecha").agg(F.sum("venta").alias("venta")).toPandas()
    activas = activas.merge(total_dia, on="fecha", how="right").fillna(0).sort_values("fecha")
    activas["pct_venta_promo"] = activas["venta_promo"] / activas["venta"]
    activas = activas[activas["fecha"] >= pd.Timestamp(min(MESES) + "-01").date()]
    activas["fecha"] = pd.to_datetime(activas["fecha"])
    cal = pd.date_range(activas["fecha"].min(), activas["fecha"].max(), freq="D")
    act_cal = activas.set_index("fecha").reindex(cal)          # los días sin datos quedan como hueco
    grafica_serie(act_cal.index, act_cal["pct_venta_promo"] * 100, "Venta en promoción", "% de la venta del día", EVENTOS)
    grafica_serie(act_cal.index, act_cal["promociones_activas"], "Promociones activas por día", "promociones", EVENTOS)
    fin_datos = activas["fecha"].max()
    corte = PROMOS["fin"].astype(str).value_counts()
    if len(corte):
        print(f"Fecha de fin más frecuente: {corte.index[0]} ({corte.iloc[0]:,} promociones). Las promociones que terminan el último "
              "día con datos de un mes incompleto (como el 15 de julio) pueden seguir vigentes: su duración está censurada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Promoción por categoría y asociación con la venta
# MAGIC **Qué hace.** Para las categorías principales: % de la venta en promoción y profundidad media. Después, con la serie
# MAGIC nacional diaria de cada categoría, correlaciona la intensidad promocional del día con la venta relativa a su nivel
# MAGIC normal (venta del día entre el promedio del mismo día de la semana en las cuatro semanas previas).
# MAGIC
# MAGIC **Qué mirar.** Categorías con correlación positiva alta son candidatas a mostrar efecto promocional en el modelo y en
# MAGIC el simulador. Es una asociación descriptiva: la promoción también se programa cuando se espera más venta.

# COMMAND ----------

titulo("9. PROMOCIÓN POR CATEGORÍA")
if C["categoria"]:
    PC = (base_promo.groupBy("categoria").agg(F.sum("venta").alias("venta"),
                                        F.sum(F.when(F.col("tiene_promo"), F.col("venta"))).alias("venta_promo"),
                                        F.avg(F.when(con_desc, F.col("tasa"))).alias("tasa_media_con_desc"))
          .toPandas())
    PC["pct_venta_promo"] = PC["venta_promo"] / PC["venta"]
    nombres = None
    if C["categoria_nombre"]:
        nombres = (df.groupBy(F.col(C["categoria"]).cast("string").alias("categoria"))
                   .agg(F.first(F.col(C["categoria_nombre"]), ignorenulls=True).alias("nombre")).toPandas())
        PC = PC.merge(nombres, on="categoria", how="left")
    PC = PC.sort_values("venta", ascending=False)
    guardar(PC, "mlops", "eda_promociones_categoria", "Promocion por categoria (EDA 06)")
    cols = ["categoria"] + (["nombre"] if "nombre" in PC else []) + ["venta", "pct_venta_promo", "tasa_media_con_desc"]
    mostrar(PC.head(TOP_CATEGORIAS)[cols], f"{TOP_CATEGORIAS} categorías con más venta",
              formatos={"venta": "dinero", "pct_venta_promo": "pct", "tasa_media_con_desc": "pct"})
    top_ids = PC.head(TOP_CATEGORIAS)["categoria"].tolist()
    serie = (base_promo.where(F.col("categoria").isin(top_ids) & F.col("fecha").isNotNull())
             .groupBy("categoria", "fecha")
             .agg(F.sum("venta").alias("venta"), F.sum(F.when(F.col("tiene_promo"), F.col("venta"))).alias("venta_promo"))
             .toPandas().fillna(0))
    serie["fecha"] = pd.to_datetime(serie["fecha"])
    serie = serie.sort_values(["categoria", "fecha"])
    serie["intensidad"] = serie["venta_promo"] / serie["venta"].replace(0, np.nan)
    serie["dow"] = serie["fecha"].dt.dayofweek
    serie["base_4s"] = (serie.groupby(["categoria", "dow"])["venta"]
                        .transform(lambda s: s.shift(1).rolling(4, min_periods=3).mean()))
    serie["venta_relativa"] = serie["venta"] / serie["base_4s"]
    validas = serie.dropna(subset=["intensidad", "venta_relativa"])
    corr = pd.DataFrame([{"categoria": k, "dias": len(g),
                          "corr_intensidad_venta": (g["intensidad"].corr(g["venta_relativa"])
                                                    if g["intensidad"].std() > 0 and g["venta_relativa"].std() > 0 else np.nan),
                          "intensidad_media": g["intensidad"].mean()} for k, g in validas.groupby("categoria")])
    if nombres is not None:
        corr = corr.merge(nombres, on="categoria", how="left")
    mostrar(corr.sort_values("corr_intensidad_venta", ascending=False),
              "Correlación entre intensidad promocional del día y venta relativa (serie nacional)",
              formatos={"dias": "int", "corr_intensidad_venta": "dec2", "intensidad_media": "pct"})
    pos = corr[corr["corr_intensidad_venta"] > 0.3]
    hallazgo(f"{len(pos)} de {len(corr)} categorías principales muestran correlación > 0.3 entre la intensidad promocional "
                 "y la venta relativa del día (asociación descriptiva, no efecto causal).")

# COMMAND ----------

cerrar("06_eda_variables_y_promociones", siguiente="07_eda_externas")
