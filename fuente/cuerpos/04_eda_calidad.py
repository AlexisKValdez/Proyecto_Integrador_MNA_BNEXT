# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · EDA de calidad de los datos, mes por mes
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Preguntas que responde**
# MAGIC 1. ¿Qué columnas vienen vacías, constantes o cambian su forma de captura entre meses?
# MAGIC 2. ¿Cuántas líneas están duplicadas, cuánto inflan la venta y qué cambia entre versiones de una misma línea?
# MAGIC 3. ¿Cuántas líneas violan las reglas de rango de Bnext y las de consistencia entre importes?
# MAGIC 4. ¿Hay estatus de cancelación o devolución que deban excluirse?
# MAGIC 5. ¿El catálogo de categorías y marcas es consistente (un nombre por identificador)?
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `bronze.ventas_lineas` |
# MAGIC | **Escribe** | `mlops.eda_calidad_columnas`, `eda_calidad_duplicados`, `eda_calidad_reglas`, `eda_calidad_estatus` |
# MAGIC | **Decide** | Las reglas de limpieza del notebook `08_limpieza_plata` |
# MAGIC | **Tiempo** | 15–30 min |

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# @@COLUMNAS

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook
MUESTRA_IDS_VERSIONES = 2000    # identificadores con versiones distintas que se inspeccionan columna por columna
UMBRAL_SALTO_NULOS = 0.10       # una columna cuyo % de nulos cambia más que esto entre meses se reporta
COSTO_MAX, CANTIDAD_MAX, VENTA_MIN, VENTA_MAX = 5_000, 1_000, 0, 15_000   # reglas de rango POS de Bnext

df = leer_tabla(tabla("bronze", "ventas_lineas"), "02_carga_bronce")
C = resolver_columnas(df.columns)
TIPOS = tipos_de(df)
NEGOCIO = [c for c in df.columns if c not in TRAZA]
mostrar(pd.DataFrame([{"nombre_logico": k, "columna": v or "— no encontrada —"} for k, v in C.items()]),
        "Columnas del proyecto encontradas en Bronce", max_filas=40)
N = lambda k: num(C[k])
venta = N("total_con_descuento") if C["total_con_descuento"] else F.lit(0.0)
print(f"{len(NEGOCIO)} columnas de negocio en Bronce")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Perfil de cada columna por mes
# MAGIC **Qué hace.** Para cada columna y mes: % de nulos, % de texto vacío, valores distintos (aproximado) y, si es numérica
# MAGIC o fecha, mínimo y máximo. Todo en una sola pasada por mes.
# MAGIC
# MAGIC **Qué mirar.** Columnas 100 % nulas o constantes (no aportan nada al modelo), columnas cuyo % de nulos salta entre
# MAGIC meses (cambió la forma de capturar) y columnas que existen sólo en algunos meses.

# COMMAND ----------

titulo("1. PERFIL POR COLUMNA Y MES")
numericas = [c for c in NEGOCIO if TIPOS[c] in ("bigint", "int", "double", "float", "date", "timestamp") or TIPOS[c].startswith("decimal")]
textos = [c for c in NEGOCIO if TIPOS[c] == "string"]
aggs = [F.count("*").alias("__n")]
for i, c in enumerate(NEGOCIO):
    aggs.append(F.sum(F.col(f"`{c}`").isNull().cast("int")).alias(f"nul_{i}"))
    aggs.append(F.approx_count_distinct(f"`{c}`", 0.05).alias(f"dis_{i}"))
    if c in textos:
        aggs.append(F.sum((F.trim(F.col(f"`{c}`")) == "").cast("int")).alias(f"vac_{i}"))
    if c in numericas:
        aggs.append(F.min(f"`{c}`").cast("string").alias(f"min_{i}"))
        aggs.append(F.max(f"`{c}`").cast("string").alias(f"max_{i}"))
perfil_raw = df.groupBy("anio_mes").agg(*aggs).toPandas()
filas = []
for _, r in perfil_raw.iterrows():
    n = r["__n"]
    for i, c in enumerate(NEGOCIO):
        filas.append({"anio_mes": r["anio_mes"], "columna": c, "tipo": TIPOS[c], "filas": n,
                      "pct_nulos": r[f"nul_{i}"] / n if n else np.nan,
                      "pct_vacios": (r[f"vac_{i}"] / n) if c in textos and n else 0.0,
                      "distintos": int(r[f"dis_{i}"]),
                      "minimo": r.get(f"min_{i}") if c in numericas else None,
                      "maximo": r.get(f"max_{i}") if c in numericas else None})
PERFIL = pd.DataFrame(filas).sort_values(["columna", "anio_mes"])
guardar(PERFIL, "mlops", "eda_calidad_columnas", "Perfil de cada columna de Bronce por mes (EDA 04)")

agg = (PERFIL.groupby("columna")
       .agg(tipo=("tipo", "first"), pct_nulos_min=("pct_nulos", "min"), pct_nulos_max=("pct_nulos", "max"),
            pct_vacios_max=("pct_vacios", "max"), distintos_max=("distintos", "max")).reset_index())
agg["salto_nulos"] = agg["pct_nulos_max"] - agg["pct_nulos_min"]
vacias = agg[agg["pct_nulos_min"] >= 0.9999]
constantes = agg[(agg["distintos_max"] <= 1) & (agg["pct_nulos_min"] < 0.9999)]
saltos = agg[agg["salto_nulos"] > UMBRAL_SALTO_NULOS].sort_values("salto_nulos", ascending=False)
mostrar(agg.sort_values("pct_nulos_max", ascending=False)[["columna", "tipo", "pct_nulos_min", "pct_nulos_max",
                                                              "pct_vacios_max", "distintos_max"]],
          "Resumen por columna (todo el periodo)", formatos={"pct_nulos_min": "pct2", "pct_nulos_max": "pct2",
                                                            "pct_vacios_max": "pct2", "distintos_max": "int"}, max_filas=90)
casi = agg[(agg["pct_nulos_min"] >= 0.99) & (agg["pct_nulos_min"] < 0.9999)]
if len(casi):
    hallazgo(f"{len(casi)} columnas casi vacías (≥ 99 % nulos): "
             + ", ".join(f"{r.columna} ({1 - r.pct_nulos_min:.3%} con dato)" for r in casi.itertuples()) + ".")
hallazgo(f"{len(vacias)} columnas vacías en todos los meses ({', '.join(vacias['columna'].head(10))}) y "
             f"{len(constantes)} constantes ({', '.join(constantes['columna'].head(10))}): no aportan información.")
if len(saltos):
    piv = PERFIL[PERFIL["columna"].isin(saltos["columna"])].pivot(index="columna", columns="anio_mes", values="pct_nulos")
    mostrar(piv.reset_index(), "Columnas cuyo % de nulos cambia entre meses", formatos={c: "pct" for c in piv.columns})
    hallazgo(f"{len(saltos)} columnas cambian su % de nulos más de {UMBRAL_SALTO_NULOS:.0%} entre meses "
                 f"({', '.join(saltos['columna'].head(8))}): cambió la captura; no deben usarse sin tratarlas por mes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Duplicados de `order_line_id`
# MAGIC **Qué hace.** Cuenta los identificadores de línea que aparecen más de una vez y los separa en dos casos: **copias
# MAGIC exactas** (todas las columnas iguales: reproceso de carga) y **versiones distintas** (mismo identificador, contenido
# MAGIC distinto: correcciones posteriores). Calcula cuánto inflan la venta si no se eliminan, si un mismo identificador
# MAGIC llega en dos meses distintos y qué columnas cambian entre versiones.
# MAGIC
# MAGIC **Qué mirar.** El importe inflado (el trabajo previo no deduplicaba y sobrestimaba la venta) y qué columnas cambian
# MAGIC en las versiones: si sólo cambia `sync_date` es reproceso; si cambian importes, la regla de conservación debe tomar la
# MAGIC versión más reciente.

# COMMAND ----------

titulo("2. DUPLICADOS DE order_line_id")
if requiere(C, "id_linea"):
    idl = C["id_linea"]
    con_hash = df.select(idl, "anio_mes", venta.alias("__venta"),
                         (a_timestamp(C["fecha_sync"], TIPOS[C["fecha_sync"]]) if C["fecha_sync"] else F.lit(None).cast("timestamp")).alias("__sync"),
                         F.xxhash64(*[F.col(f"`{c}`") for c in NEGOCIO]).alias("__hash"))
    por_id = (con_hash.groupBy(idl)
              .agg(F.count("*").alias("n"), F.size(F.collect_set("__hash")).alias("versiones"),
                   F.size(F.collect_set("anio_mes")).alias("meses"), F.sum("__venta").alias("venta_total")))
    from pyspark import StorageLevel
    por_id = persistir(por_id)
    rep_ids = persistir(por_id.where("n > 1"))
    resumen = rep_ids.agg(F.count("*").alias("ids_repetidos"), F.sum(F.col("n") - 1).alias("filas_extra"),
                          F.sum((F.col("versiones") == 1).cast("int")).alias("ids_copias_exactas"),
                          F.sum((F.col("versiones") > 1).cast("int")).alias("ids_con_versiones"),
                          F.sum((F.col("meses") > 1).cast("int")).alias("ids_en_varios_meses")).collect()[0].asDict()
    tot = por_id.agg(F.sum("n").alias("filas"), F.count("*").alias("ids")).collect()[0]
    total_filas, total_ids = int(tot["filas"]), int(tot["ids"])
    # importe que sobra: todas las filas de ids repetidos menos la versión que se conservaría (la de sync más reciente)
    w = Window.partitionBy(idl).orderBy(F.col("__sync").desc_nulls_last(), F.col("__hash"))
    conservada = (con_hash.join(rep_ids.select(idl), idl, "inner")
                  .withColumn("__rn", F.row_number().over(w))
                  .groupBy("anio_mes")
                  .agg(F.sum("__venta").alias("venta_ids_repetidos"),
                       F.sum(F.when(F.col("__rn") == 1, F.col("__venta"))).alias("venta_conservada"),
                       F.sum((F.col("__rn") > 1).cast("int")).alias("filas_extra")))
    por_mes = (con_hash.groupBy("anio_mes").agg(F.count("*").alias("filas"), F.countDistinct(idl).alias("ids_unicos"),
                                                 F.sum("__venta").alias("venta"))
               .join(conservada, "anio_mes", "left").fillna(0).orderBy("anio_mes").toPandas())
    por_mes["venta_inflada"] = por_mes["venta_ids_repetidos"] - por_mes["venta_conservada"]
    por_mes["pct_filas_extra"] = por_mes["filas_extra"] / por_mes["filas"]
    por_mes["pct_venta_inflada"] = por_mes["venta_inflada"] / por_mes["venta"]
    mostrar(por_mes[["anio_mes", "filas", "ids_unicos", "filas_extra", "pct_filas_extra", "venta_inflada",
                       "pct_venta_inflada"]], "Duplicados por mes",
              formatos={"filas": "int", "ids_unicos": "int", "filas_extra": "int", "pct_filas_extra": "pct2",
                        "venta_inflada": "dinero", "pct_venta_inflada": "pct2"})
    print(f"Total: {total_filas:,} filas, {total_ids:,} identificadores únicos; repetidos: {resumen['ids_repetidos'] or 0:,} "
              f"(copias exactas: {resumen['ids_copias_exactas'] or 0:,}; con versiones distintas: {resumen['ids_con_versiones'] or 0:,}; "
              f"en más de un mes: {resumen['ids_en_varios_meses'] or 0:,})")
    hallazgo(f"{int(por_mes['filas_extra'].sum()):,} filas sobran por duplicados ({por_mes['filas_extra'].sum() / total_filas:.2%}); "
                 f"inflan la venta en ${por_mes['venta_inflada'].sum():,.0f} ({por_mes['venta_inflada'].sum() / por_mes['venta'].sum():.2%}). "
                 f"{resumen['ids_en_varios_meses'] or 0:,} identificadores aparecen en dos o más meses.")
    guardar(por_mes, "mlops", "eda_calidad_duplicados", "Duplicados de order_line_id por mes (EDA 04)")

    # qué columnas cambian entre versiones de un mismo identificador
    if (resumen["ids_con_versiones"] or 0) > 0:
        ids_v = rep_ids.where("versiones > 1").select(idl).limit(MUESTRA_IDS_VERSIONES)
        cambio = (df.join(ids_v, idl, "inner").groupBy(idl)
                  .agg(*[F.countDistinct(F.col(f"`{c}`")).alias(c) for c in NEGOCIO if c != idl]))
        frec = cambio.select(*[F.avg((F.col(f"`{c}`") > 1).cast("double")).alias(c)
                               for c in NEGOCIO if c != idl]).toPandas().T.reset_index()
        frec.columns = ["columna", "pct_ids_donde_cambia"]
        frec = frec[frec["pct_ids_donde_cambia"] > 0].sort_values("pct_ids_donde_cambia", ascending=False)
        mostrar(frec, f"Columnas que cambian entre versiones (muestra de {MUESTRA_IDS_VERSIONES:,} identificadores)",
                  formatos={"pct_ids_donde_cambia": "pct"})
        if len(frec):
            hallazgo("Entre versiones de una misma línea cambian sobre todo: "
                         + ", ".join(f"{r.columna} ({r.pct_ids_donde_cambia:.0%})" for r in frec.head(6).itertuples()) + ".")
    liberar(rep_ids)
    liberar(por_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reglas de rango y de consistencia
# MAGIC **Qué hace.** Evalúa por mes cada regla: las cuatro de rango acordadas con Bnext (parámetros arriba) y
# MAGIC reglas de consistencia entre importes. Para cada una reporta filas, % y venta afectada.
# MAGIC
# MAGIC **Qué mirar.** Las reglas de rango eliminaban 0.0097 % de las filas en el trabajo previo; si con los meses nuevos
# MAGIC el porcentaje cambia mucho, hay un problema nuevo de captura. `descuento_pct_mayor_100` confirma el error de escala
# MAGIC del porcentaje de descuento; `total_inconsistente` dice si `total_sin_descuento = cantidad × precio`.

# COMMAND ----------

titulo("3. REGLAS DE RANGO Y CONSISTENCIA")
reglas = {}
if C["cantidad"]:
    reglas["cantidad_no_positiva"] = N("cantidad") <= 0
    reglas["cantidad_mayor_max"] = N("cantidad") > CANTIDAD_MAX
if C["costo_unitario"]:
    reglas["costo_mayor_max"] = N("costo_unitario") > COSTO_MAX
if C["precio_unitario"]:
    reglas["precio_no_positivo"] = N("precio_unitario") <= 0
if C["total_con_descuento"]:
    reglas["venta_negativa"] = venta < VENTA_MIN
    reglas["venta_cero"] = venta == 0
    reglas["venta_mayor_max"] = venta > VENTA_MAX
    reglas["venta_nula"] = venta.isNull()
if C["pct_descuento"]:
    reglas["descuento_pct_mayor_100"] = N("pct_descuento") > 100
    reglas["descuento_pct_negativo"] = N("pct_descuento") < 0
if C["total_sin_descuento"] and C["total_con_descuento"]:
    reglas["con_descuento_mayor_que_sin"] = venta > N("total_sin_descuento") + 0.01
    reglas["sin_descuento_cero_con_venta"] = (N("total_sin_descuento") == 0) & (venta > 0)
if C["total_sin_descuento"] and C["cantidad"] and C["precio_unitario"]:
    esperado = N("cantidad") * N("precio_unitario")
    reglas["total_inconsistente"] = F.abs(esperado - N("total_sin_descuento")) > F.greatest(F.lit(1.0), 0.05 * F.abs(N("total_sin_descuento")))
for k in ("tienda", "categoria", "producto"):
    if C[k]:
        reglas[f"{k}_nula"] = F.col(C[k]).isNull()
if C["fecha_ticket"]:
    reglas["fecha_venta_nula"] = a_timestamp(C["fecha_ticket"], TIPOS[C["fecha_ticket"]]).isNull()
rango_bnext = [r for r in ("costo_mayor_max", "cantidad_mayor_max", "venta_negativa", "venta_mayor_max") if r in reglas]
if rango_bnext:
    combinada = reglas[rango_bnext[0]]
    for r in rango_bnext[1:]:
        combinada = combinada | reglas[r]
    reglas["rango_bnext_cualquiera"] = combinada

aggs = [F.count("*").alias("filas"), F.sum(venta).alias("venta")]
for nombre, cond in reglas.items():
    cond = F.coalesce(cond, F.lit(False))
    aggs += [F.sum(cond.cast("int")).alias(f"f_{nombre}"), F.sum(F.when(cond, venta)).alias(f"v_{nombre}")]
crudo = df.groupBy("anio_mes").agg(*aggs).orderBy("anio_mes").toPandas()
filas = []
for _, r in crudo.iterrows():
    for nombre in reglas:
        filas.append({"anio_mes": r["anio_mes"], "regla": nombre, "filas": int(r[f"f_{nombre}"] or 0),
                      "pct_filas": (r[f"f_{nombre}"] or 0) / r["filas"], "venta_afectada": r[f"v_{nombre}"] or 0.0,
                      "pct_venta": (r[f"v_{nombre}"] or 0.0) / r["venta"] if r["venta"] else np.nan})
REGLAS = pd.DataFrame(filas)
guardar(REGLAS, "mlops", "eda_calidad_reglas", "Filas y venta que violan cada regla, por mes (EDA 04)")
tot = (REGLAS.groupby("regla").agg(filas=("filas", "sum"), venta_afectada=("venta_afectada", "sum")).reset_index())
tot["pct_filas"] = tot["filas"] / crudo["filas"].sum()
mostrar(tot.sort_values("filas", ascending=False), "Reglas en todo el periodo",
          formatos={"filas": "int", "pct_filas": "pct2", "venta_afectada": "dinero"})
piv = REGLAS.pivot(index="regla", columns="anio_mes", values="pct_filas")
mostrar(piv.reset_index(), "% de filas que viola cada regla, por mes", formatos={c: "pct2" for c in piv.columns})
if "rango_bnext_cualquiera" in reglas:
    t = tot.set_index("regla").loc["rango_bnext_cualquiera"]
    hallazgo(f"Las reglas de rango de Bnext marcan {int(t['filas']):,} filas ({t['pct_filas']:.4%}) en todo el periodo "
                 "(compárese con el Avance 2, marzo a mayo).")
for r in ("descuento_pct_mayor_100", "total_inconsistente", "venta_cero"):
    if r in reglas:
        t = tot.set_index("regla").loc[r]
        hallazgo(f"{r}: {int(t['filas']):,} filas ({t['pct_filas']:.2%}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Estatus, cancelaciones y devoluciones
# MAGIC **Qué hace.** Busca columnas de estatus (`order_status_id`, `order_line_status_id` y cualquier columna cuyo nombre
# MAGIC sugiera estatus, cancelación o devolución) y muestra sus valores por mes con la venta asociada.
# MAGIC
# MAGIC **Qué mirar.** En el trabajo previo estas columnas eran constantes. Si en los meses nuevos aparecen otros valores,
# MAGIC hay que saber con Bnext qué significa cada uno y excluir las líneas canceladas en Plata.

# COMMAND ----------

titulo("4. ESTATUS, CANCELACIONES Y DEVOLUCIONES")
patron = re.compile(r"status|estatus|estado_orden|cancel|devol|refund|return", re.I)
cols_estatus = sorted({C.get("estatus_linea"), C.get("estatus_orden")} - {None} |
                      {c for c in NEGOCIO if patron.search(c) and TIPOS[c] in ("string", "bigint", "int", "boolean")})
EST = []
for c in cols_estatus:
    distintos = df.select(F.approx_count_distinct(f"`{c}`")).collect()[0][0]
    if distintos > 50:
        print(f"{c}: {distintos:,} valores distintos; no parece un estatus, se omite")
        continue
    v = (df.groupBy("anio_mes", F.col(f"`{c}`").cast("string").alias("valor"))
         .agg(F.count("*").alias("filas"), F.sum(venta).alias("venta")).toPandas())
    v["columna"] = c
    EST.append(v)
if EST:
    EST = pd.concat(EST)
    guardar(EST, "mlops", "eda_calidad_estatus", "Valores de columnas de estatus por mes (EDA 04)")
    for c, g in EST.groupby("columna"):
        piv = g.pivot_table(index="valor", columns="anio_mes", values="filas", aggfunc="sum", fill_value=0).reset_index()
        mostrar(piv, f"{c}: filas por valor y mes")
        if g["valor"].nunique() > 1:
            hallazgo(f"{c} tiene {g['valor'].nunique()} valores distintos: confirmar con Bnext cuáles son ventas válidas.")
        else:
            print(f"{c}: un solo valor ({g['valor'].iloc[0]}) en todos los meses")
else:
    print("No se encontraron columnas de estatus")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Consistencia del catálogo
# MAGIC **Qué hace.** Revisa que cada `category_id` tenga un solo nombre y cada nombre un solo identificador (en el trabajo
# MAGIC previo: la categoría 617 sin nombre y YOGURT / YOGURTH duplicadas), lo mismo para marcas, y cuenta tiendas,
# MAGIC categorías, marcas y productos por mes.
# MAGIC
# MAGIC **Qué mirar.** Las categorías con nombres repetidos o identificadores sin nombre: son las reglas de homologación de
# MAGIC Plata (notebook 07). Siempre se trabaja con los identificadores numéricos, nunca con los nombres.

# COMMAND ----------

titulo("5. CONSISTENCIA DEL CATÁLOGO")
for ident, nombre, etiqueta in (("categoria", "categoria_nombre", "categoría"), ("marca", "marca_nombre", "marca")):
    if not (C[ident] and C[nombre]):
        aviso(f"Sin columnas de {etiqueta} (identificador y nombre)")
        continue
    pares = (df.groupBy(F.col(C[ident]).alias("id"), F.trim(F.upper(F.col(C[nombre]))).alias("nombre"))
             .agg(F.count("*").alias("filas"), F.sum(venta).alias("venta")).toPandas())
    nombres_por_id = pares.groupby("id")["nombre"].nunique()
    ids_por_nombre = pares.dropna(subset=["nombre"]).groupby("nombre")["id"].nunique()
    sin_nombre = pares[pares["nombre"].isna() | (pares["nombre"] == "")]
    print(f"{etiqueta}: {pares['id'].nunique():,} identificadores, {pares['nombre'].nunique():,} nombres; "
              f"identificadores con más de un nombre: {int((nombres_por_id > 1).sum())}; "
              f"nombres con más de un identificador: {int((ids_por_nombre > 1).sum())}; "
              f"identificadores con filas sin nombre: {sin_nombre['id'].nunique()}")
    if etiqueta == "categoría":
        multi = pares[pares["id"].isin(nombres_por_id[nombres_por_id > 1].index)].sort_values(["id", "filas"], ascending=[True, False])
        mostrar(multi.head(30), "Categorías con más de un nombre", formatos={"filas": "int", "venta": "dinero"})
        dup = pares[pares["nombre"].isin(ids_por_nombre[ids_por_nombre > 1].index)].sort_values(["nombre", "filas"], ascending=[True, False])
        mostrar(dup.head(30), "Nombres de categoría con más de un identificador", formatos={"filas": "int", "venta": "dinero"})
        mostrar(sin_nombre.sort_values("filas", ascending=False).head(15), "Categorías con filas sin nombre",
                  formatos={"filas": "int", "venta": "dinero"})
        # nombres casi iguales (YOGURT / YOGURTH): mismo nombre sin la última letra o sin espacios
        base = pares.dropna(subset=["nombre"]).drop_duplicates(["id", "nombre"]).copy()
        base["clave"] = base["nombre"].str.replace(r"[^A-Z0-9]", "", regex=True).str.rstrip("SH")
        parecidos = base.groupby("clave").filter(lambda g: g["id"].nunique() > 1).sort_values("clave")
        mostrar(parecidos[["clave", "id", "nombre", "filas"]].head(30), "Categorías con nombre casi igual y distinto identificador",
                  formatos={"filas": "int"})
        if len(parecidos) or (nombres_por_id > 1).any():
            hallazgo(f"Catálogo de categorías por homologar: {int((nombres_por_id > 1).sum())} con varios nombres, "
                         f"{parecidos['clave'].nunique()} grupos de nombres casi iguales, {sin_nombre['id'].nunique()} sin nombre.")

if C["tienda"]:
    card = (df.groupBy("anio_mes").agg(
        *[F.approx_count_distinct(F.col(C[k]), 0.01).alias(k) for k in ("tienda", "categoria", "marca", "producto") if C[k]])
        .orderBy("anio_mes").toPandas())
    mostrar(card, "Entidades distintas por mes (aproximado ±1 %)", formatos={c: "int" for c in card.columns if c != "anio_mes"})

# COMMAND ----------

cerrar("04_eda_calidad", siguiente="05_eda_tiempo_y_tiendas")
