# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Limpieza: de Bronce a Plata
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Para qué sirve.** Convierte las líneas crudas de Bronce en líneas **limpias, únicas y homologadas**, y deja una
# MAGIC conciliación paso a paso: cuántas filas, identificadores y pesos quita cada regla, mes por mes. Así cualquiera puede
# MAGIC verificar que Plata = Bronce − lo que quitó cada regla.
# MAGIC
# MAGIC **Reglas, en este orden** (todas son parámetros abajo). Primero se decide **cuál es la versión vigente** de cada línea
# MAGIC y después se valida: si la versión más reciente de una línea es una cancelación o está fuera de rango, se quita la
# MAGIC línea, en lugar de rescatar una versión anterior (el Avance 2 filtraba primero; el notebook 03 replica su orden).
# MAGIC 1. **Identificador nulo**: líneas sin `order_line_id` (no se pueden deduplicar).
# MAGIC 2. **Copias exactas**: misma línea repetida con idéntico contenido (reproceso de carga) → se queda una.
# MAGIC 3. **Versiones previas**: mismo `order_line_id` con contenido distinto → se queda la de `sync_date` más reciente.
# MAGIC 4. **Duplicado en otro mes**: el mismo `order_line_id` llegó en dos volúmenes → se queda el de sincronización más reciente.
# MAGIC 5. **Fecha nula**: la fecha de venta no se puede convertir.
# MAGIC 6. **Venta nula**: la línea no trae importe con descuento (muy pocas en todo el periodo). El Avance 2 también las quitaba,
# MAGIC    dentro de su filtro de rango (en pandas un valor vacío no pasa la comparación).
# MAGIC 7. **Rango POS** acordado con Bnext: costo > 5,000, cantidad > 1,000, venta < 0 o > 15,000.
# MAGIC 8. **Estatus excluidos** (cancelaciones o devoluciones), si el EDA 04 y Bnext lo confirman.
# MAGIC 9. **Líneas huérfanas** de la categoría 617 sin producto (pruebas del sistema).
# MAGIC 10. **Tienda o categoría nula**: no se pueden asignar a ninguna serie.
# MAGIC
# MAGIC **Homologación** (con lo que encontró el EDA 04): las ocho parejas de categorías con el mismo nombre escrito distinto
# MAGIC se unen en un identificador (YOGURTH 715 → YOGURT 548 como en el Avance 2; AGUA 6 → AGUAS 661; REFRESCO 758 →
# MAGIC REFRESCOS 5; CAFE 10020 → CAFES 238; PAN Y GALLETAS 716 → PAN Y GALLETA 709; y los segundos identificadores de
# MAGIC CACAHUATE Y SEMILLAS, CHICHARRONES y PALOMITAS); la sección 5 muestra la evidencia de cada unión (meses y productos en
# MAGIC común). Nombre canónico para la 617; marca nula → marca genérica `-1`; tasa de descuento reconstruida de los importes
# MAGIC (`1 − venta / venta sin descuento`), porque el campo de porcentaje no corresponde a la tasa en 89 % de las líneas (06).
# MAGIC
# MAGIC **Importes que se conservan.** `venta` (con descuento e impuestos, la del Avance 2 y el objetivo por defecto),
# MAGIC `venta_neta` (= venta − impuestos), `subtotal` e `impuestos`, porque el 03 mostró que el importe del Avance 2 no
# MAGIC corresponde exactamente a ninguna columna; así la fase 2 puede elegir el objetivo sin volver a limpiar.
# MAGIC
# MAGIC **Catálogos enriquecidos.** Categorías con su tipo (`mercancia`, `servicio` —recargas TAE, pagos de servicios y
# MAGIC tarjetas de regalo de Arcadial— o `generico`); tiendas con tipo, zona, región (completada con la región más común de
# MAGIC su zona cuando falta), jefatura y estatus de tienda.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `bronze.ventas_lineas` |
# MAGIC | **Escribe** | `silver.ventas_lineas` (partición `anio_mes`) · `silver.catalogo_categorias` (con `tipo_categoria`) · `silver.catalogo_marcas` · `silver.catalogo_tiendas` (con región completada) · `mlops.conciliacion_plata` |
# MAGIC | **Tiempo** | 20–60 min |

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# @@COLUMNAS

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook (las reglas de limpieza)
COSTO_MAX, CANTIDAD_MAX, VENTA_MIN, VENTA_MAX = 5_000, 1_000, 0, 15_000   # rango POS de Bnext (Avance 1)
ESTATUS_EXCLUIDOS = {}                 # p. ej. {"estatus_orden": ["3"]} cuando Bnext confirme qué valor es cancelación (ver 04 §4)
# identificador de origen → identificador canónico. 715→548 es la del Avance 2; las demás son las parejas de nombre casi
# igual que encontró el EDA 04 (el canónico es el de más líneas, salvo YOGURT, que conserva la decisión del Avance 2)
UNIFICAR_CATEGORIAS = {715: 548, 6: 661, 758: 5, 10020: 238, 716: 709, 761: 667, 763: 496, 764: 465}
NOMBRES_CATEGORIA = {617: "PROMOCIONALES_ESTACIONALES_617", 548: "YOGURT", 661: "AGUAS", 5: "REFRESCOS", 238: "CAFES",
                     709: "PAN Y GALLETAS"}   # nombre canónico (si no, el más frecuente)
PATRON_SERVICIOS = r"ARCADIAL|\bTAE\b|GIFT ?CARD|RECARGA|PAGO DE SERVICIO|SERVICIOS"   # categorías que no son mercancía
PATRON_GENERICO = r"GENERIC"
EXCLUIR_HUERFANAS_617 = True           # quita líneas de la 617 sin producto (pruebas)
MARCA_GENERICA = -1                    # marca nula → esta marca
MESES_A_LIMPIAR = parametro("MESES_A_LIMPIAR", MESES)   # o una lista de meses para reprocesar sólo esos (también como parámetro de Job)
RECARGA_TOTAL = False                  # True = reescribe silver.ventas_lineas completa
TABLA = tabla("silver", "ventas_lineas")

B = leer_tabla(tabla("bronze", "ventas_lineas"), "02_carga_bronce")
C = resolver_columnas(B.columns)
TIPOS = tipos_de(B)
NEGOCIO = [c for c in B.columns if c not in TRAZA]
if not requiere(C, "id_linea", "tienda", "categoria", "total_con_descuento"):
    raise RuntimeError("Faltan columnas indispensables para limpiar (ver aviso arriba).")
print("Fecha de venta:", COLUMNA_FECHA_VENTA, "· horas en UTC:", HORAS_EN_UTC)
print("Destino:", TABLA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Líneas con columnas canónicas
# MAGIC **Qué hace.** Lleva Bronce a nombres en español y tipos fijos, calcula la fecha de venta, la huella de cada línea
# MAGIC (para detectar copias exactas) y la fecha de sincronización (para elegir la versión más reciente).

# COMMAND ----------

titulo("1. COLUMNAS CANÓNICAS")


def col(logico, tipo="string"):
    """Columna lógica con tipo fijo; nula si Bnext no la manda."""
    if not C.get(logico):
        return F.lit(None).cast(tipo)
    return num(C[logico]).cast(tipo) if tipo in ("double", "bigint", "int") else F.col(C[logico]).cast(tipo)


ENTEROS = ("bigint", "int", "smallint", "tinyint")
def col_id(logico):
    """Identificador sin perder precisión: entero si en Bronce es entero; si es texto, texto sin espacios.
    (Pasarlo por número convertiría en nulo un código con letras y redondearía valores muy grandes.)"""
    if not C.get(logico):
        return F.lit(None).cast("bigint")
    t = TIPOS[C[logico]]
    if t in ENTEROS:
        return F.col(C[logico]).cast("bigint")
    if t in ("double", "float") or t.startswith("decimal"):
        return F.expr(f"try_cast(try_cast(`{C[logico]}` as decimal(38,0)) as string)")
    return F.nullif(F.trim(F.col(C[logico]).cast("string")), F.lit(""))


def literal_como(valor, columna_logica):
    """Constante con el mismo tipo que el identificador (para comparar o mapear categorías y marcas)."""
    return F.lit(valor).cast("bigint") if C.get(columna_logica) and TIPOS[C[columna_logica]] in ENTEROS else F.lit(str(valor))


ts_venta = a_timestamp(COLUMNA_FECHA_VENTA, TIPOS[COLUMNA_FECHA_VENTA])
if HORAS_EN_UTC:
    ts_venta = F.from_utc_timestamp(ts_venta, "America/Mexico_City")
ts_sync = a_timestamp(C["fecha_sync"], TIPOS[C["fecha_sync"]]) if C.get("fecha_sync") else F.lit(None).cast("timestamp")
categoria_origen = col_id("categoria")
categoria = categoria_origen
for origen_id, destino_id in UNIFICAR_CATEGORIAS.items():       # 715 → 548, etc.
    categoria = F.when(categoria_origen == literal_como(origen_id, "categoria"), literal_como(destino_id, "categoria")).otherwise(categoria)
marca_origen = col_id("marca")
venta = col("total_con_descuento", "double")
venta_sin = col("total_sin_descuento", "double")
promo, tiene_promo = expr_promo(C, TIPOS)
LINEAS = B.select(
    col_id("id_linea").alias("id_linea"), col_id("id_ticket").alias("id_ticket"),
    col_id("tienda").alias("tienda"), col("tipo_tienda").alias("tipo_tienda"), col("zona").alias("zona"),
    col("region").alias("region"),
    categoria.alias("categoria_id"), categoria_origen.alias("categoria_id_origen"),
    F.coalesce(marca_origen, literal_como(MARCA_GENERICA, "marca")).alias("marca_id"),
    marca_origen.isNull().alias("marca_era_nula"),
    col_id("producto").alias("producto_id"),
    col("cantidad", "double").alias("cantidad"), col("precio_unitario", "double").alias("precio_unitario"),
    col("costo_unitario", "double").alias("costo_unitario"),
    venta.alias("venta"), venta_sin.alias("venta_sin_descuento"),
    col("impuestos", "double").alias("impuestos"), col("subtotal", "double").alias("subtotal"),
    (venta - F.coalesce(col("impuestos", "double"), F.lit(0.0))).alias("venta_neta"),
    F.when(venta_sin > 0, F.least(F.greatest(1 - venta / venta_sin, F.lit(0.0)), F.lit(1.0))).otherwise(F.lit(0.0)).alias("tasa_descuento"),
    F.greatest(F.coalesce(venta_sin - venta, F.lit(0.0)), F.lit(0.0)).alias("descuento_pesos"),
    col("pct_descuento", "double").alias("pct_descuento_origen"),
    col("descuento_sumasix", "double").alias("descuento_sumasix"), col("pago_puntos", "double").alias("pago_puntos"),
    promo.alias("promo_id"), tiene_promo.alias("tiene_promo"),
    F.when(tiene_promo, col("promo_nombre")).alias("promo_nombre"),
    col("estatus_tienda").alias("estatus_tienda"), col("jefatura").alias("jefatura"),
    ts_venta.alias("ts_venta"), F.to_date(ts_venta).alias("fecha_venta"), F.hour(ts_venta).alias("hora_venta"),
    ts_sync.alias("ts_sync"),
    col("estatus_orden").alias("estatus_orden"), col("estatus_linea").alias("estatus_linea"),
    F.xxhash64(*[F.col(f"`{c}`") for c in NEGOCIO]).alias("huella"),
    F.col("anio_mes"), F.col("_archivo"), F.col("_fecha_carga").alias("_fecha_carga_bronce"))
print("Columnas de Plata:", ", ".join(c for c in LINEAS.columns if c != "huella"))
print("Tipo de los identificadores:", {k: TIPOS[C[k]] for k in ("id_linea", "tienda", "categoria", "marca", "producto") if C.get(k)})

# reglas de validez (5-9): se evalúan sobre la versión vigente; una línea se atribuye a la primera que no cumple
fuera_rango = (F.coalesce(F.col("costo_unitario") > COSTO_MAX, F.lit(False)) |
               F.coalesce(F.col("cantidad") > CANTIDAD_MAX, F.lit(False)) |
               F.coalesce((F.col("venta") < VENTA_MIN) | (F.col("venta") > VENTA_MAX), F.lit(False)))
estatus_malo = F.lit(False)
for logico, valores in ESTATUS_EXCLUIDOS.items():
    estatus_malo = estatus_malo | F.coalesce(F.col(logico).cast("string").isin([str(v) for v in valores]), F.lit(False))
huerfana = ((F.col("categoria_id_origen") == literal_como(617, "categoria")) & F.col("producto_id").isNull()
            if EXCLUIR_HUERFANAS_617 else F.lit(False))
REGLA_VALIDEZ = (F.when(F.col("fecha_venta").isNull(), "5_fecha_nula")
                 .when(F.col("venta").isNull(), "6_venta_nula")
                 .when(fuera_rango, "7_rango_pos")
                 .when(estatus_malo, "8_estatus_excluido")
                 .when(huerfana, "9_huerfana_617")
                 .when(F.col("tienda").isNull() | F.col("categoria_id").isNull(), "10_tienda_o_categoria_nula"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Identificadores que llegaron en más de un mes
# MAGIC **Qué hace.** Busca `order_line_id` presentes en dos o más volúmenes mensuales y decide en cuál se conservan
# MAGIC (el de sincronización más reciente). **Qué mirar.** Si son muchos, Bnext reentregó días de un mes dentro de otro.

# COMMAND ----------

titulo("2. IDENTIFICADORES EN MÁS DE UN MES")
from pyspark import StorageLevel
orden_version = F.struct(F.coalesce(F.col("ts_sync"), F.lit("1900-01-01").cast("timestamp")),
                         F.col("_fecha_carga_bronce"), F.col("anio_mes"))
MULTIMES = (LINEAS.where(F.col("id_linea").isNotNull())
            .groupBy("id_linea")
            .agg(F.size(F.collect_set("anio_mes")).alias("meses"), F.max_by("anio_mes", orden_version).alias("mes_que_se_queda"))
            .where("meses > 1"))
MULTIMES = persistir(MULTIMES)
n_multimes = MULTIMES.count()
print(f"Identificadores en más de un mes: {n_multimes:,}")
if n_multimes:
    mostrar(MULTIMES.groupBy("mes_que_se_queda").count().orderBy("mes_que_se_queda"), "¿En qué mes se conservan?",
            formatos={"count": "int"})
    hallazgo(f"{n_multimes:,} order_line_id llegaron en más de un volumen mensual; se conserva la versión más reciente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Limpieza mes por mes y conciliación
# MAGIC **Qué mirar.** En la bitácora, `cuadra = sí` en todos los meses (filas de Bronce = filas de Plata + filas quitadas)
# MAGIC y que ninguna regla quite una proporción muy distinta entre meses.

# COMMAND ----------

titulo("3. LIMPIEZA Y CONCILIACIÓN")
PASOS = ["0_bronce", "1_id_nulo", "2_copia_exacta", "3_version_previa", "4_duplicado_en_otro_mes", "5_fecha_nula",
         "6_venta_nula", "7_rango_pos", "8_estatus_excluido", "9_huerfana_617", "10_tienda_o_categoria_nula", "11_plata"]
multimes = (F.broadcast(MULTIMES.select("id_linea", "mes_que_se_queda")) if n_multimes < 5_000_000
            else MULTIMES.select("id_linea", "mes_que_se_queda"))
conciliacion, primera, dup_dias = [], True, []
DUPLICADO = ["2_copia_exacta", "3_version_previa"]
for mes in MESES_A_LIMPIAR:
    t0 = time.time()
    d = LINEAS.where(F.col("anio_mes") == mes)
    con_id = d.where(F.col("id_linea").isNotNull())
    con_id = con_id.withColumn("rn_copia", F.row_number().over(Window.partitionBy("id_linea", "huella").orderBy("_archivo")))
    unica = F.col("rn_copia") == 1
    con_id = con_id.withColumn("rn_version", F.when(unica, F.row_number().over(
        Window.partitionBy("id_linea", unica).orderBy(F.col("ts_sync").desc_nulls_last(), F.col("_fecha_carga_bronce").desc(),
                                                     F.col("_archivo").desc()))))
    con_id = (con_id.join(multimes, "id_linea", "left")
              .withColumn("regla", F.when(F.col("rn_copia") > 1, "2_copia_exacta")
                                   .when(F.col("rn_version") > 1, "3_version_previa")
                                   .when(F.col("mes_que_se_queda").isNotNull() & (F.col("mes_que_se_queda") != mes),
                                         "4_duplicado_en_otro_mes")
                                   .otherwise(REGLA_VALIDEZ))
              .drop("rn_copia", "rn_version", "mes_que_se_queda"))
    sin_id = d.where(F.col("id_linea").isNull()).withColumn("regla", F.lit("1_id_nulo"))
    todo = persistir(con_id.unionByName(sin_id))
    resumen = {r["regla"]: r for r in todo.groupBy("regla").agg(F.count("*").alias("filas"), F.sum("venta").alias("venta")).collect()}
    total_filas = sum(r["filas"] for r in resumen.values())
    if not total_filas:
        aviso(f"{mes}: sin líneas en Bronce; se omite")
        liberar(todo)
        continue
    total = todo.agg(F.countDistinct("id_linea").alias("ids")).collect()[0]
    es_dup = F.col("regla").isin(*DUPLICADO)
    for r in (todo.groupBy("fecha_venta")
              .agg(F.count("*").alias("filas"), F.sum(es_dup.cast("int")).alias("filas_duplicadas"),
                   F.sum(F.when(es_dup, F.col("venta"))).alias("venta_duplicada"),
                   F.concat_ws(",", F.array_sort(F.collect_set(F.when(es_dup, F.element_at(F.split("_archivo", "/"), -1)))))
                   .alias("archivos"))
              .where("filas_duplicadas > 0").collect()):
        dup_dias.append({"anio_mes": mes, **r.asDict()})
    filas = [{"anio_mes": mes, "paso": "0_bronce", "filas": total_filas, "ids": total["ids"],
              "venta": float(sum((r["venta"] or 0.0) for r in resumen.values()))}]
    for paso in PASOS[1:-1]:
        q = resumen.get(paso)
        filas.append({"anio_mes": mes, "paso": paso, "filas": -(q["filas"] if q else 0), "ids": None,
                      "venta": -((q["venta"] or 0.0) if q else 0.0)})
    plata = todo.where(F.col("regla").isNull()).drop("regla", "huella")
    plata = plata.withColumn("mes_venta", F.date_format("fecha_venta", "yyyy-MM")).withColumn("_fecha_limpieza", F.current_timestamp())
    escritor = plata.write.format("delta")
    if (RECARGA_TOTAL and primera) or not existe(TABLA):
        escritor.mode("overwrite").option("overwriteSchema", "true").partitionBy("anio_mes").saveAsTable(TABLA)
    else:
        escritor.mode("overwrite").option("replaceWhere", f"anio_mes = '{mes}'").option("mergeSchema", "true").saveAsTable(TABLA)
    primera = False
    escrita = (spark.table(TABLA).where(F.col("anio_mes") == mes)
               .agg(F.count("*").alias("filas"), F.countDistinct("id_linea").alias("ids"), F.sum("venta").alias("venta")).collect()[0])
    filas.append({"anio_mes": mes, "paso": "11_plata", "filas": escrita["filas"], "ids": escrita["ids"], "venta": escrita["venta"] or 0.0})
    cuadra = total_filas + sum(f["filas"] for f in filas if f["paso"] not in ("0_bronce", "11_plata")) == escrita["filas"]
    unicas = escrita["filas"] == escrita["ids"]
    for f in filas:
        f["cuadra"] = "sí" if cuadra else "NO"
    conciliacion += filas
    print(f"{mes}: {total_filas:,} → {escrita['filas']:,} líneas · cuadra: {'sí' if cuadra else 'NO'} · "
          f"ids únicos: {'sí' if unicas else 'NO'} · {time.time() - t0:.0f} s")
    if not cuadra or not unicas:
        aviso(f"{mes}: la conciliación no cuadra o quedaron identificadores repetidos; revisar.")
    liberar(todo)
liberar(MULTIMES)

if dup_dias:
    DD = pd.DataFrame(dup_dias)
    DD["pct_del_dia"] = DD["filas_duplicadas"] / DD["filas"]
    DD = DD.sort_values("filas_duplicadas", ascending=False)
    mostrar(DD[["anio_mes", "fecha_venta", "filas", "filas_duplicadas", "pct_del_dia", "venta_duplicada", "archivos"]].head(12),
            "Días con más líneas duplicadas (copias y versiones) y el archivo de donde vienen",
            formatos={"filas": "int", "filas_duplicadas": "int", "pct_del_dia": "pct", "venta_duplicada": "dinero"})
    top = DD.head(3)
    peso_top = top["filas_duplicadas"].sum() / DD["filas_duplicadas"].sum()
    hallazgo(f"{DD['fecha_venta'].nunique()} días tienen líneas duplicadas; los tres con más concentran {peso_top:.0%} de ellas: "
             + "; ".join(f"{r.fecha_venta} ({r.filas_duplicadas:,} líneas, {r.pct_del_dia:.0%} del día, archivo {r.archivos})"
                         for r in top.itertuples())
             + (". Parece un archivo diario cargado dos veces, no duplicados dispersos." if peso_top >= 0.5 else
                ". Están repartidos en muchos días."))

CONC = pd.DataFrame(conciliacion)
CONC["fecha_limpieza"] = pd.Timestamp.now()
guardar(CONC, "mlops", "conciliacion_plata", "Conciliacion Bronce a Plata por mes y regla: filas, ids y venta",
        modo="overwrite" if set(MESES_A_LIMPIAR) == set(MESES) else "append")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Conciliación en el formato del Avance 2
# MAGIC **Qué mirar.** El total de la columna `filas` de cada regla y su importe; y el subtotal de marzo a mayo para
# MAGIC compararlo con la conciliación que reportó el Avance 2.

# COMMAND ----------

titulo("4. CONCILIACIÓN TOTAL")
tot = CONC.groupby("paso", sort=False).agg(filas=("filas", "sum"), venta=("venta", "sum")).reset_index()
base_filas, base_venta = tot.loc[tot["paso"] == "0_bronce", ["filas", "venta"]].iloc[0]
tot["pct_filas"] = tot["filas"] / base_filas
tot["pct_venta"] = tot["venta"] / base_venta
mostrar(tot, f"Bronce → Plata, {len(MESES)} meses", formatos={"filas": "int", "venta": "dinero", "pct_filas": "pct2", "pct_venta": "pct2"})
piv = CONC[~CONC["paso"].isin(["0_bronce", "11_plata"])].pivot(index="paso", columns="anio_mes", values="filas").reindex(PASOS[1:-1])
base_mes = CONC[CONC["paso"] == "0_bronce"].set_index("anio_mes")["filas"]
mostrar((-piv / base_mes).reset_index(), "% de líneas que quita cada regla, por mes", formatos={c: "pct2" for c in piv.columns})
plata_tot = tot.loc[tot["paso"] == "11_plata"].iloc[0]
hallazgo(f"Plata: {int(plata_tot['filas']):,} líneas limpias de {int(base_filas):,} en Bronce "
         f"({1 - plata_tot['filas'] / base_filas:.2%} quitadas); venta limpia ${plata_tot['venta']:,.0f} "
         f"({plata_tot['venta'] / base_venta - 1:+.2%} contra Bronce).")
for paso in PASOS[1:-1]:
    t = tot[tot["paso"] == paso]
    if len(t) and t["filas"].iloc[0] != 0:
        hallazgo(f"{paso}: {int(-t['filas'].iloc[0]):,} líneas ({-t['pct_filas'].iloc[0]:.3%}) y ${-t['venta'].iloc[0]:,.0f}.")
m3 = CONC[CONC["anio_mes"].isin(["2026-03", "2026-04", "2026-05"]) & (CONC["paso"] == "11_plata")]
if len(m3) == 3:
    print(f"\nMarzo a mayo en Plata: {int(m3['filas'].sum()):,} líneas y ${m3['venta'].sum():,.2f} "
          "(compárese con la conciliación del Avance 2; aquí la versión vigente se elige antes de validar)")
IMPORTES = (spark.table(TABLA).groupBy("anio_mes")
            .agg(F.sum("venta").alias("venta"), F.sum("venta_neta").alias("venta_neta"), F.sum("subtotal").alias("subtotal"),
                 F.sum("impuestos").alias("impuestos"), F.sum((F.col("venta") == 0).cast("int")).alias("lineas_venta_cero"))
            .orderBy("anio_mes").toPandas())
mostrar(IMPORTES, "Importes de Plata por mes (columnas candidatas a objetivo)",
        formatos={"venta": "dinero", "venta_neta": "dinero", "subtotal": "dinero", "impuestos": "dinero", "lineas_venta_cero": "int"})
i3 = IMPORTES[IMPORTES["anio_mes"].isin(["2026-03", "2026-04", "2026-05"])]
if len(i3) == 3:
    hallazgo(f"Marzo–mayo limpio: venta ${i3['venta'].sum():,.0f}, venta neta de impuestos ${i3['venta_neta'].sum():,.0f} y "
             f"subtotal ${i3['subtotal'].sum():,.0f} (compárense con la conciliación del Avance 2).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Catálogos homologados
# MAGIC **Qué hace.** Una fila por categoría canónica (con su nombre más frecuente y los identificadores de origen que
# MAGIC agrupa), por marca y por tienda (con su ventana activa: primera y última venta). Son las dimensiones del modelo.

# COMMAND ----------

titulo("5. CATÁLOGOS")
P = spark.table(TABLA)
if C.get("categoria_nombre"):
    nombres = (B.select(categoria.alias("categoria_id"), F.upper(F.trim(F.col(C["categoria_nombre"]))).alias("nombre"))
               .where(F.col("nombre").isNotNull() & (F.col("nombre") != ""))
               .groupBy("categoria_id", "nombre").count()
               .withColumn("rn", F.row_number().over(Window.partitionBy("categoria_id").orderBy(F.desc("count"))))
               .where("rn = 1").select("categoria_id", "nombre"))
else:
    nombres = P.select("categoria_id").distinct().withColumn("nombre", F.lit(None).cast("string"))
CAT = (P.groupBy("categoria_id")
       .agg(F.collect_set("categoria_id_origen").alias("ids_origen"), F.count("*").alias("lineas"),
            F.sum("venta").alias("venta"), F.min("fecha_venta").alias("primera_venta"), F.max("fecha_venta").alias("ultima_venta"))
       .join(nombres, "categoria_id", "left"))
for cat_id, nombre_canonico in NOMBRES_CATEGORIA.items():
    CAT = CAT.withColumn("nombre", F.when(F.col("categoria_id") == literal_como(cat_id, "categoria"), F.lit(nombre_canonico))
                         .otherwise(F.col("nombre")))
CAT = CAT.withColumn("ids_origen", F.concat_ws(",", F.array_sort(F.transform("ids_origen", lambda x: x.cast("string")))))
CAT = CAT.withColumn("tipo_categoria",
                     F.when(F.upper(F.col("nombre")).rlike(PATRON_SERVICIOS), "servicio")
                     .when(F.upper(F.col("nombre")).rlike(PATRON_GENERICO), "generico").otherwise("mercancia"))
guardar(CAT, "silver", "catalogo_categorias", "Categorias canonicas: nombre, identificadores de origen, lineas y venta")
cat_pd = CAT.orderBy(F.desc("venta")).toPandas()
mostrar(cat_pd.head(25), f"Categorías canónicas ({len(cat_pd):,}); las 25 con más venta",
        formatos={"lineas": "int", "venta": "dinero"})
unificadas = cat_pd[cat_pd["ids_origen"].str.contains(",")]
if len(unificadas):
    hallazgo(f"{len(cat_pd):,} categorías canónicas; {len(unificadas)} agrupan varios identificadores de origen "
             f"({', '.join(unificadas['ids_origen'].head(8))}).")
# evidencia de cada unión: en qué meses vende cada identificador y cuántos productos comparten (si comparten productos o
# uno sustituye al otro en el tiempo, es la misma categoría con otro código)
if UNIFICAR_CATEGORIAS:
    entero = TIPOS[C["categoria"]] in ENTEROS
    ids_u = sorted({(int(k) if entero else str(k)) for pareja in UNIFICAR_CATEGORIAS.items() for k in pareja})
    ev = (P.where(F.col("categoria_id_origen").isin(*ids_u) | F.col("categoria_id").isin(*ids_u))
          .groupBy("categoria_id", "categoria_id_origen")
          .agg(F.count("*").alias("lineas"), F.sum("venta").alias("venta"),
               F.concat_ws(",", F.array_sort(F.collect_set("anio_mes"))).alias("meses"),
               F.collect_set("producto_id").alias("productos")))
    evp = ev.toPandas()
    filas_ev = []
    for canon, g in evp.groupby("categoria_id"):
        prod = {r.categoria_id_origen: set(r.productos) for r in g.itertuples()}
        for r in g.itertuples():
            otros = set().union(*[v for k, v in prod.items() if k != r.categoria_id_origen]) if len(prod) > 1 else set()
            filas_ev.append({"canonica": canon, "origen": r.categoria_id_origen, "lineas": r.lineas, "venta": r.venta,
                             "meses": r.meses, "productos": len(r.productos),
                             "productos_en_comun": len(set(r.productos) & otros)})
    EVID = pd.DataFrame(filas_ev).sort_values(["canonica", "lineas"], ascending=[True, False])
    if "group" in B.columns:            # grupo (nivel superior del catálogo) de cada identificador de origen
        grp = (B.where(F.col(C["categoria"]).isin(*ids_u))
               .groupBy(F.col(C["categoria"]).alias("origen"), F.col("group").alias("grupo")).count()
               .withColumn("rn", F.row_number().over(Window.partitionBy("origen").orderBy(F.desc("count"))))
               .where("rn = 1").select("origen", "grupo").toPandas())
        EVID = EVID.merge(grp, on="origen", how="left")
    mostrar(EVID, "Evidencia de las uniones de categorías (meses con venta y productos en común)",
            formatos={"lineas": "int", "venta": "dinero", "productos": "int", "productos_en_comun": "int"})
    sin_comun = int((EVID.groupby("canonica")["productos_en_comun"].sum() == 0).sum())
    hallazgo(f"De las {EVID['canonica'].nunique()} uniones de categorías, {sin_comun} no comparten productos entre sus "
             "identificadores y los dos venden casi todos los meses: no es un cambio de código en el tiempo, sino dos ramas del "
             "catálogo con el mismo nombre. Unirlas es una decisión de negocio (misma categoría para el cliente); si Bnext prefiere "
             "separarlas, basta quitarlas de UNIFICAR_CATEGORIAS.")
TIPOS_CAT = cat_pd.groupby("tipo_categoria").agg(categorias=("categoria_id", "count"), venta=("venta", "sum")).reset_index()
TIPOS_CAT["pct_venta"] = TIPOS_CAT["venta"] / TIPOS_CAT["venta"].sum()
mostrar(TIPOS_CAT, "Venta limpia por tipo de categoría", formatos={"categorias": "int", "venta": "dinero", "pct_venta": "pct"})
tc = TIPOS_CAT.set_index("tipo_categoria")["pct_venta"]
hallazgo(f"Venta limpia por tipo de categoría: mercancía {tc.get('mercancia', 0):.1%}, servicios (recargas, pagos, tarjetas) "
         f"{tc.get('servicio', 0):.1%}, genérico {tc.get('generico', 0):.1%}. Los servicios no son demanda de producto: el panel los "
         "marca para poder separarlos al modelar.")

if C.get("marca_nombre"):
    nombres_m = (B.select(F.coalesce(marca_origen, literal_como(MARCA_GENERICA, "marca")).alias("marca_id"),
                          F.upper(F.trim(F.col(C["marca_nombre"]))).alias("nombre"))
                 .where(F.col("nombre").isNotNull()).groupBy("marca_id", "nombre").count()
                 .withColumn("rn", F.row_number().over(Window.partitionBy("marca_id").orderBy(F.desc("count"))))
                 .where("rn = 1").select("marca_id", "nombre"))
else:
    nombres_m = P.select("marca_id").distinct().withColumn("nombre", F.lit(None).cast("string"))
MAR = (P.groupBy("marca_id").agg(F.count("*").alias("lineas"), F.sum("venta").alias("venta"),
                                 F.countDistinct("categoria_id").alias("categorias"))
       .join(nombres_m, "marca_id", "left")
       .withColumn("nombre", F.when(F.col("marca_id") == literal_como(MARCA_GENERICA, "marca"), F.lit("GENERICA_SIN_MARCA"))
                   .otherwise(F.col("nombre"))))
guardar(MAR, "silver", "catalogo_marcas", "Marcas: nombre mas frecuente, lineas, venta y categorias")
nulas = P.agg(F.avg(F.col("marca_era_nula").cast("double"))).collect()[0][0] or 0
hallazgo(f"{MAR.count():,} marcas; {nulas:.2%} de las líneas no traían marca y quedaron como genérica ({MARCA_GENERICA}) "
         "(Avance 2: 0.40 %).")

ATRIB_TIENDA = ["tipo_tienda", "zona", "region", "jefatura", "estatus_tienda"]
ultimo = lambda c: F.max_by(c, F.when(F.col(c).isNotNull(), F.col("ts_venta")))     # último valor no nulo
tienda_dia = (P.groupBy("tienda", "fecha_venta")
              .agg(F.size(F.collect_set("id_ticket")).alias("tickets"), F.sum("venta").alias("venta"),
                   *[ultimo(c).alias(c) for c in ATRIB_TIENDA], F.max("ts_venta").alias("ts_venta")))
TIE = (tienda_dia.groupBy("tienda")
       .agg(*[ultimo(c).alias(c) for c in ATRIB_TIENDA],
            F.min("fecha_venta").alias("primera_venta"), F.max("fecha_venta").alias("ultima_venta"),
            F.count("*").alias("dias_con_venta"), F.sum("venta").alias("venta"), F.sum("tickets").alias("tickets"))
       .withColumn("dias_ventana", F.datediff("ultima_venta", "primera_venta") + 1))
# región faltante: la más común entre las tiendas de su zona (Bronce trae región sólo en ~35 % de las líneas)
moda_zona = (TIE.where(F.col("region").isNotNull()).groupBy("zona", "region").count()
             .withColumn("rn", F.row_number().over(Window.partitionBy("zona").orderBy(F.desc("count"))))
             .where("rn = 1").select("zona", F.col("region").alias("region_zona")))
TIE = (TIE.join(moda_zona, "zona", "left")
       .withColumn("region_imputada", F.col("region").isNull() & F.col("region_zona").isNotNull())
       .withColumn("region", F.coalesce("region", "region_zona")).drop("region_zona"))
guardar(TIE, "silver", "catalogo_tiendas", "Tiendas: ventana activa (primera y ultima venta), dias con venta, venta y tickets")
r = TIE.agg(F.count("*").alias("n"), F.sum((F.col("dias_con_venta") >= 5).cast("int")).alias("n5"),
            F.sum(F.col("region").isNull().cast("int")).alias("sin_region"),
            F.sum(F.col("region_imputada").cast("int")).alias("imputadas")).collect()[0]
hallazgo(f"{r['n']:,} tiendas con venta limpia; {r['n5']:,} con al menos 5 días de venta (compárese con el Avance 2, mar–may). "
         f"Región: {r['imputadas']:,} completadas con la de su zona y {r['sin_region']:,} siguen sin región.")
for c in ("tipo_tienda", "region", "estatus_tienda"):
    mostrar(spark.table(tabla("silver", "catalogo_tiendas")).groupBy(c).agg(F.count("*").alias("tiendas"), F.sum("venta").alias("venta"))
            .orderBy(F.desc("tiendas")), f"Tiendas por {c}", formatos={"tiendas": "int", "venta": "dinero"}, max_filas=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verificación de Plata

# COMMAND ----------

titulo("6. VERIFICACIÓN")
v = P.agg(F.count("*").alias("filas"), F.countDistinct("id_linea").alias("ids"),
          F.min("fecha_venta").alias("desde"), F.max("fecha_venta").alias("hasta"),
          F.sum(F.col("venta").isNull().cast("int")).alias("venta_nula"),
          F.sum(((F.col("tasa_descuento") < 0) | (F.col("tasa_descuento") > 1)).cast("int")).alias("tasa_fuera")).collect()[0]
print(f"Filas {v['filas']:,} · ids únicos {v['ids']:,} · {v['desde']} → {v['hasta']} · venta nula {v['venta_nula']:,} · "
      f"tasa fuera de [0, 1] {v['tasa_fuera']:,}")
if v["filas"] != v["ids"]:
    aviso("Plata tiene order_line_id repetidos.")
try:
    spark.sql(f"OPTIMIZE {TABLA} ZORDER BY (tienda, fecha_venta)")
    print("OPTIMIZE ZORDER BY (tienda, fecha_venta) ejecutado")
except Exception as e:
    print(f"(OPTIMIZE no disponible: {type(e).__name__})")
try:
    spark.sql(f"COMMENT ON TABLE {TABLA} IS 'Lineas de venta limpias, unicas y homologadas (08_limpieza_plata). Ver mlops.conciliacion_plata'")
except Exception:
    pass

# COMMAND ----------

cerrar("08_limpieza_plata", siguiente="09_panel_diario")
