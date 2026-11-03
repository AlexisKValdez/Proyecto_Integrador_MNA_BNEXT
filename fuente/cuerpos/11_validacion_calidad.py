# Databricks notebook source
# MAGIC %md
# MAGIC # 11 · Validación de calidad (contratos de datos)
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Para qué sirve.** Es la compuerta antes de construir variables y entrenar: revisa reglas que **siempre** deben
# MAGIC cumplirse en Bronce, Plata y el panel. En la etapa de MLOps este notebook corre automáticamente antes de cada
# MAGIC reentrenamiento; si una regla falla, el pipeline se detiene.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `bronze.ventas_lineas` · `silver.ventas_lineas` · catálogos de Plata · `silver.panel_categoria_dia` · `mlops.bronce_cargas` · `mlops.conciliacion_plata` · `mlops.linea_base_ingenua` |
# MAGIC | **Escribe** | `mlops.validaciones` (una fila por regla y corrida) |
# MAGIC | **Tiempo** | 5–15 min |

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook
DETENER_SI_FALLA = parametro("DETENER_SI_FALLA", False)
# True en el pipeline automático (Job J1): el notebook termina con error si una regla crítica falla y así el Job se
# detiene antes de construir la Oro o entrenar con datos que no pasan las reglas.
BANDA_CEROS = (0.30, 0.75)   # proporción de ceros comerciales esperada por mes en el panel
MIN_FILAS_DIA = 0.50         # un día con menos de esta fracción de las líneas de la mediana de su mes es sospechoso
VENTA_MAX_LINEA, CANTIDAD_MAX_LINEA, COSTO_MAX_LINEA = 15_000, 1_000, 5_000   # rango POS de Bnext
RESULTADOS = []


def regla(capa, nombre, valor, cumple, critica=True, detalle=""):
    RESULTADOS.append({"capa": capa, "regla": nombre, "valor": valor, "estado": "PASA" if cumple else "FALLA",
                       "critica": critica, "detalle": detalle})
    print(f"{'✓' if cumple else '✗'} [{capa}] {nombre}: {valor}" + (f" — {detalle}" if detalle else ""))

# COMMAND ----------

titulo("1. BRONCE")
B = leer_tabla(tabla("bronze", "ventas_lineas"), "02_carga_bronce")
meses_b = sorted(r[0] for r in B.select("anio_mes").distinct().collect())
regla("bronze", "están todos los meses", ", ".join(meses_b), set(MESES) <= set(meses_b))
try:
    cargas = spark.table(tabla("mlops", "bronce_cargas")).toPandas().sort_values("fecha_carga").groupby("anio_mes").tail(1)
    malos = cargas[cargas["estado"] != "ok"]
    regla("bronze", "conciliación de filas contra metadatos", f"{len(cargas) - len(malos)}/{len(cargas)} meses ok", len(malos) == 0)
except Exception as e:
    regla("bronze", "conciliación de filas contra metadatos", f"sin bitácora ({type(e).__name__})", False)

# COMMAND ----------

titulo("2. PLATA: LÍNEAS")
S = leer_tabla(tabla("silver", "ventas_lineas"), "08_limpieza_plata")
r = S.agg(F.count("*").alias("filas"), F.countDistinct("id_linea").alias("ids"),
          F.sum(F.col("venta").isNull().cast("int")).alias("venta_nula"),
          F.sum(((F.col("venta") < 0) | (F.col("venta") > VENTA_MAX_LINEA)).cast("int")).alias("venta_fuera"),
          F.sum(((F.col("cantidad") > CANTIDAD_MAX_LINEA) | (F.col("costo_unitario") > COSTO_MAX_LINEA)).cast("int")).alias("pos_fuera"),
          F.sum((F.col("venta_neta") > F.col("venta") + 0.01).cast("int")).alias("neta_mayor"),
          F.sum(F.col("fecha_venta").isNull().cast("int")).alias("fecha_nula"),
          F.sum(((F.col("tasa_descuento") < 0) | (F.col("tasa_descuento") > 1)).cast("int")).alias("tasa_fuera"),
          F.sum(F.col("tienda").isNull().cast("int")).alias("tienda_nula"),
          F.sum(F.col("categoria_id").isNull().cast("int")).alias("categoria_nula"),
          F.min("fecha_venta").alias("desde"), F.max("fecha_venta").alias("hasta")).collect()[0]
regla("silver", "order_line_id único", f"{r['filas']:,} filas · {r['ids']:,} ids", r["filas"] == r["ids"])
regla("silver", "venta no nula", f"{r['venta_nula']:,}", r["venta_nula"] == 0)
regla("silver", f"venta de línea entre 0 y {VENTA_MAX_LINEA:,}", f"{r['venta_fuera']:,} fuera", r["venta_fuera"] == 0)
regla("silver", f"cantidad ≤ {CANTIDAD_MAX_LINEA:,} y costo ≤ {COSTO_MAX_LINEA:,} (rango POS)", f"{r['pos_fuera']:,} fuera", r["pos_fuera"] == 0)
regla("silver", "venta neta de impuestos ≤ venta", f"{r['neta_mayor']:,} líneas con impuestos negativos", r["neta_mayor"] == 0,
      critica=False)
regla("silver", "fecha de venta no nula", f"{r['fecha_nula']:,}", r["fecha_nula"] == 0)
regla("silver", "tasa de descuento en [0, 1]", f"{r['tasa_fuera']:,} fuera", r["tasa_fuera"] == 0)
regla("silver", "tienda y categoría no nulas", f"{r['tienda_nula']:,} / {r['categoria_nula']:,}", r["tienda_nula"] + r["categoria_nula"] == 0)
inicio, fin = pd.Timestamp(min(MESES) + "-01").date(), (pd.Timestamp(max(MESES) + "-01") + pd.offsets.MonthEnd(0)).date()
regla("silver", "fechas dentro del periodo", f"{r['desde']} → {r['hasta']}", r["desde"] >= inicio and r["hasta"] <= fin, critica=False)
dia = S.groupBy("fecha_venta").count().withColumn("mes", F.date_format("fecha_venta", "yyyy-MM")).toPandas()
dia["mediana_mes"] = dia.groupby("mes")["count"].transform("median")
bajos = dia[dia["count"] < MIN_FILAS_DIA * dia["mediana_mes"]]
regla("silver", f"ningún día con menos del {MIN_FILAS_DIA:.0%} de las líneas de la mediana de su mes", f"{len(bajos)} días",
      len(bajos) == 0, critica=False, detalle=", ".join(str(d) for d in bajos["fecha_venta"].head(10)))
try:
    conc = spark.table(tabla("mlops", "conciliacion_plata")).toPandas()
    ok = (conc.sort_values("fecha_limpieza", ascending=False).drop_duplicates("anio_mes")["cuadra"] == "sí").all()
    regla("silver", "Plata = Bronce − filas quitadas (cada mes)", "sí" if ok else "no", ok)
except Exception as e:
    regla("silver", "Plata = Bronce − filas quitadas (cada mes)", f"sin conciliación ({type(e).__name__})", False)
CATALOGO_CAT = spark.table(tabla("silver", "catalogo_categorias"))
cat = CATALOGO_CAT.select("categoria_id")
sin_cat = S.select("categoria_id").distinct().join(cat, "categoria_id", "left_anti").count()
regla("silver", "toda categoría está en el catálogo", f"{sin_cat} sin catálogo", sin_cat == 0)
cp = CATALOGO_CAT.select("categoria_id", "nombre", *(["tipo_categoria"] if "tipo_categoria" in CATALOGO_CAT.columns else [])).toPandas()
cp["clave"] = cp["nombre"].fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.rstrip("SH")
dup = cp[cp["clave"] != ""].groupby("clave").filter(lambda g: g["categoria_id"].nunique() > 1)
regla("silver", "sin categorías con el mismo nombre en dos identificadores", f"{dup['clave'].nunique()} grupos",
      dup.empty, critica=False, detalle=", ".join(f"{c}: {sorted(g['categoria_id'].tolist())}" for c, g in dup.groupby("clave"))[:300])
sin_nombre = int(cp["nombre"].isna().sum() + (cp["nombre"] == "").sum())
regla("silver", "toda categoría tiene nombre", f"{sin_nombre} sin nombre", sin_nombre == 0, critica=False)
if "tipo_categoria" in cp:
    tipos = cp["tipo_categoria"].value_counts().to_dict()
    regla("silver", "toda categoría tiene tipo (mercancía, servicio o genérico)", str(tipos), cp["tipo_categoria"].notna().all(),
          critica=False)
tiendas_cat = spark.table(tabla("silver", "catalogo_tiendas")).select("tienda")
sin_tienda = S.select("tienda").distinct().join(tiendas_cat, "tienda", "left_anti").count()
regla("silver", "toda tienda está en el catálogo de tiendas", f"{sin_tienda} sin catálogo", sin_tienda == 0)

# COMMAND ----------

titulo("3. PLATA: PANEL")
P = leer_tabla(tabla("silver", "panel_categoria_dia"), "09_panel_diario")
r = P.agg(F.count("*").alias("filas"), F.countDistinct("tienda", "categoria_id", "fecha").alias("llaves"),
          F.sum((F.col("venta") < 0).cast("int")).alias("negativa"),
          F.sum((F.col("venta").isNull() & (F.col("tienda_inactiva") == 0) & (F.col("sin_datos") == 0)).cast("int")).alias("nula_activa"),
          F.countDistinct(F.when(F.col("sin_datos") == 1, F.col("fecha"))).alias("dias_sin_datos"),
          F.sum((F.col("venta").isNotNull() & (F.col("tienda_inactiva") == 1)).cast("int")).alias("valor_inactiva"),
          F.sum((F.col("promos_vigentes") < 0).cast("int")).alias("promos_neg"),
          F.sum(F.col("tipo_categoria").isNull().cast("int")).alias("sin_tipo"),
          F.countDistinct(F.when(F.col("dia_atipico") == 1, F.col("fecha"))).alias("dias_atipicos")).collect()[0]
regla("panel", "llave tienda-categoría-fecha única", f"{r['filas']:,} filas · {r['llaves']:,} llaves", r["filas"] == r["llaves"])
regla("panel", "venta ≥ 0", f"{r['negativa']:,} negativas", r["negativa"] == 0)
regla("panel", "venta nula sólo si la tienda está inactiva o el día no tiene datos", f"{r['nula_activa']:,} / {r['valor_inactiva']:,}",
      r["nula_activa"] == 0 and r["valor_inactiva"] == 0)
dias_sd = [x[0] for x in P.where(F.col("sin_datos") == 1).select("fecha").distinct().collect()]
regla("panel", "todos los días del periodo tienen datos", f"{r['dias_sin_datos']} días sin datos", r["dias_sin_datos"] == 0,
      critica=False, detalle=(f"{resumir_fechas(dias_sd)}: pedir a Bnext la entrega faltante" if dias_sd else ""))
dias_at = [x[0] for x in P.where(F.col("dia_atipico") == 1).select("fecha").distinct().collect()]
regla("panel", "días atípicos (con datos pero menos tiendas) marcados", f"{r['dias_atipicos']} días", True, critica=False,
      detalle=resumir_fechas(dias_at) if dias_at else "")
regla("panel", "promociones vigentes ≥ 0", f"{r['promos_neg']:,} negativas", r["promos_neg"] == 0)
regla("panel", "toda fila tiene tipo de categoría", f"{r['sin_tipo']:,} sin tipo", r["sin_tipo"] == 0, critica=False)
T = (S.where(F.col("fecha_venta").between(inicio, fin)).groupBy("tienda")
     .agg(F.min("fecha_venta").alias("primera_venta"), F.max("fecha_venta").alias("ultima_venta")))
fuera = P.join(T, "tienda").where((F.col("fecha") < F.col("primera_venta")) | (F.col("fecha") > F.col("ultima_venta"))).count()
regla("panel", "ninguna fila fuera de la ventana de su tienda", f"{fuera:,}", fuera == 0)
ceros = (P.where("tienda_inactiva = 0 AND sin_datos = 0 AND en_warmup = 0 AND serie_relevante = 1").groupBy("mes")
         .agg(F.avg((F.col("venta") == 0).cast("double")).alias("p")).orderBy("mes").toPandas())
fuera_banda = ceros[(ceros["p"] < BANDA_CEROS[0]) | (ceros["p"] > BANDA_CEROS[1])]
regla("panel", f"ceros comerciales por mes entre {BANDA_CEROS[0]:.0%} y {BANDA_CEROS[1]:.0%}",
      ", ".join(f"{m} {p:.1%}" for m, p in zip(ceros["mes"], ceros["p"])), len(fuera_banda) == 0, critica=False)
v_s = S.where(F.col("fecha_venta").between(inicio, fin)).agg(F.sum("venta")).collect()[0][0] or 0.0
v_p = P.agg(F.sum("venta")).collect()[0][0] or 0.0
regla("panel", "venta del panel = venta de Plata", f"${v_p:,.2f} vs ${v_s:,.2f}", abs(v_p - v_s) <= max(1.0, 1e-6 * v_s))

# COMMAND ----------

titulo("4. LÍNEA BASE")
try:
    LB = spark.table(tabla("mlops", "linea_base_ingenua")).toPandas()
    LB = LB[LB["fecha_calculo"] == LB["fecha_calculo"].max()].sort_values("corte")
    sd = set(dias_sd)
    toca = []
    for c in LB.itertuples():
        origen = pd.Timestamp(c.origen).date()
        necesarios = {origen - dt.timedelta(days=k) for k in range(14)} | \
                     {d.date() for d in pd.date_range(c.desde, c.hasta, freq="D")}
        if necesarios & sd:
            toca.append(c.corte)
    n_dev = int(LB["corte"].astype(str).str.startswith("corte_").sum())
    n_prueba = int(LB["corte"].astype(str).str.startswith("prueba_").sum())
    regla("línea base", "cuatro cortes walk-forward de desarrollo", f"{n_dev} cortes de desarrollo · {n_prueba} de prueba",
          n_dev == len(ORIGENES_CORTES), critica=False,
          detalle=", ".join(f"{c.corte}: {c.desde} a {c.hasta}" for c in LB.itertuples()))
    regla("línea base", "ningún corte ni sus 14 días previos tocan días sin datos", f"{len(toca)} cortes", not toca,
          detalle=", ".join(toca))
    regla("línea base", "WMAPE de la línea base calculado", f"estacional {LB['wmape_lag7'].mean():.4f} · media móvil {LB['wmape_ma7'].mean():.4f}",
          LB[["wmape_lag7", "wmape_ma7"]].notna().all().all())
except Exception as e:
    regla("línea base", "tabla de la línea base", f"no disponible ({type(e).__name__}); corre el 10", False)

# COMMAND ----------

titulo("5. RESULTADO")
RES = pd.DataFrame(RESULTADOS)
mostrar(RES, "Reglas", max_filas=100)
guardar(RES.assign(fecha_validacion=pd.Timestamp.now()), "mlops", "validaciones", "Resultado de las reglas de calidad por corrida", modo="append")
criticas = RES[(RES["estado"] == "FALLA") & RES["critica"]]
avisos_ = RES[(RES["estado"] == "FALLA") & ~RES["critica"]]
if len(criticas):
    aviso(f"{len(criticas)} reglas críticas fallan: {', '.join(criticas['regla'])}.")
else:
    hallazgo(f"Las {int(RES['critica'].sum())} reglas críticas pasan; {len(avisos_)} reglas de aviso por revisar.")
for a in avisos_.itertuples():
    aviso(f"{a.regla}: {a.valor}")
if DETENER_SI_FALLA and len(criticas):
    raise AssertionError("Validación de calidad fallida: " + "; ".join(criticas["regla"]))

# COMMAND ----------

cerrar("11_validacion_calidad", siguiente="ESTADO_DEL_PROYECTO (en la carpeta principal)")
