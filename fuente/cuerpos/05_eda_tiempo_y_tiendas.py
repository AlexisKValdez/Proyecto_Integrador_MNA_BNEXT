# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · EDA del tiempo y de las tiendas
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Preguntas que responde**
# MAGIC 1. ¿Qué columna define el día de la venta y en qué zona horaria vienen las horas?
# MAGIC 2. ¿Hay días faltantes o con carga incompleta en los meses cargados? ¿Cómo cambia la venta por día de la semana y del mes?
# MAGIC 3. ¿Cuántas tiendas operan todo el periodo, cuántas abren o cierran a la mitad y cuántas tienen huecos?
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `bronze.ventas_lineas` (sin limpiar; la venta excluye las líneas fuera de rango POS y los duplicados suman alrededor de 1 % de más, sin cambiar los patrones) |
# MAGIC | **Escribe** | `mlops.eda_cobertura_diaria` · `mlops.eda_tienda_dia` · `mlops.eda_tiendas_actividad` |
# MAGIC | **Decide** | `COLUMNA_FECHA_VENTA` y `HORAS_EN_UTC` · días a excluir · relleno de ceros sólo dentro de la ventana de cada tienda |
# MAGIC | **Tiempo** | 10–20 min |

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
FECHAS_CANDIDATAS = ["finish_order", "date_orders", "date_ordered", "sync_date"]
MUESTRA_FECHAS = 0.05      # fracción de líneas para comparar las columnas de fecha
UMBRAL_DIA_BAJO = 0.50     # un día con menos de este % de las líneas de la mediana de su mes es sospechoso
UMBRAL_TIENDAS = 0.95      # un día con menos de este % de las tiendas de la mediana es posible apagón

df = leer_tabla(tabla("bronze", "ventas_lineas"), "02_carga_bronce")
C = resolver_columnas(df.columns)
TIPOS = tipos_de(df)
mostrar(pd.DataFrame([{"nombre_logico": k, "columna": v or "— no encontrada —", "tipo": TIPOS.get(v, "") if v else ""}
                      for k, v in C.items()]), "Columnas del proyecto encontradas en Bronce", max_filas=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ¿Qué columna define el día de la venta?
# MAGIC **Qué hace.** Sobre una muestra de líneas compara cada columna candidata (`finish_order`, `date_orders`,
# MAGIC `sync_date`…): qué porcentaje se puede convertir a fecha, qué porcentaje cae dentro del mes de su volumen y, si el
# MAGIC nombre del archivo indica el día, cuántas coinciden con él. También muestra la distribución por hora del día.
# MAGIC
# MAGIC **Qué mirar.** La columna correcta se convierte casi al 100 % y cae en el mes de su volumen. En la distribución por
# MAGIC hora, una tienda de conveniencia vende sobre todo entre las 7 y las 22 h; si el pico aparece entre 0 y 5 h, las horas
# MAGIC vienen en UTC y hay que poner `HORAS_EN_UTC = True` en la configuración, o el día de la venta quedará corrido.

# COMMAND ----------

titulo("1. ¿QUÉ COLUMNA DEFINE EL DÍA DE LA VENTA?")
candidatas = [c for c in FECHAS_CANDIDATAS if c in TIPOS]
try:   # tipo con el que venía cada columna en los parquet (02_carga_bronce): un timestamp sin zona se lee como hora local
    plan = spark.table(tabla("mlops", "bronce_columnas")).select("columna_bronce", "tipo_bronce", "tipos_origen").toPandas()
    mostrar(plan[plan["columna_bronce"].isin(candidatas)], "Tipo de origen de las columnas de fecha")
except Exception:
    pass
print("Candidatas presentes:", candidatas, "· configurada:", COLUMNA_FECHA_VENTA)
muestra = (df.select(*candidatas, "anio_mes", "_fecha_archivo")
           .sample(fraction=MUESTRA_FECHAS, seed=42))
muestra = persistir(muestra)
print(f"Muestra para comparar fechas: {muestra.count():,} líneas")
filas_cand, horas = [], {}
for c in candidatas:
    ts = a_timestamp(c, TIPOS[c])
    m = muestra.select(ts.alias("ts"), a_timestamp(c, TIPOS[c], acotar=False).alias("ts_crudo"), "anio_mes", "_fecha_archivo")
    r = m.agg(F.count("*").alias("n"),
              F.avg(F.col("ts").isNotNull().cast("double")).alias("pct_convertible"),
              F.avg((F.col("ts_crudo").isNotNull() & F.col("ts").isNull()).cast("double")).alias("pct_centinela"),
              F.date_format(F.min("ts_crudo"), "yyyy-MM-dd HH:mm").alias("minimo_crudo"),
              F.date_format(F.max("ts_crudo"), "yyyy-MM-dd HH:mm").alias("maximo_crudo"),
              F.avg((F.date_format("ts", "yyyy-MM") == F.col("anio_mes")).cast("double")).alias("pct_en_su_mes"),
              F.avg(F.when(F.col("_fecha_archivo").isNotNull(),
                           (F.to_date("ts") == F.col("_fecha_archivo")).cast("double"))).alias("pct_igual_fecha_archivo"),
              F.avg(F.when(F.col("_fecha_archivo").isNotNull(),
                           (F.datediff(F.to_date("ts"), F.col("_fecha_archivo")) == 1).cast("double"))).alias("pct_un_dia_despues"),
              F.date_format(F.min("ts"), "yyyy-MM-dd HH:mm").alias("minimo"),
              F.date_format(F.max("ts"), "yyyy-MM-dd HH:mm").alias("maximo")).collect()[0].asDict()
    r["columna"], r["tipo"] = c, TIPOS[c]
    filas_cand.append(r)
    hh = m.where(F.col("ts").isNotNull()).groupBy(F.hour("ts").alias("hora")).count().toPandas()
    horas[c] = hh.set_index("hora")["count"].reindex(range(24), fill_value=0)
liberar(muestra)
CAND = pd.DataFrame(filas_cand)[["columna", "tipo", "pct_convertible", "pct_centinela", "pct_en_su_mes",
                                 "pct_igual_fecha_archivo", "pct_un_dia_despues", "minimo", "maximo"]]
mostrar(CAND, "Calidad de cada columna candidata (muestra; fechas fuera de 2000–2099 cuentan como centinela)",
          formatos={k: "pct" for k in ["pct_convertible", "pct_centinela", "pct_en_su_mes", "pct_igual_fecha_archivo",
                                       "pct_un_dia_despues"]})
for r in filas_cand:
    if (r.get("pct_centinela") or 0) > 0:
        hallazgo(f"{r['columna']}: {r['pct_centinela']:.2%} de la muestra trae fechas centinela fuera de 2000–2099 "
                 f"(de {r['minimo_crudo']} a {r['maximo_crudo']}); se tratan como nulas.")
if horas:
    HORAS = pd.DataFrame({c: s / s.sum() for c, s in horas.items()})
    HORAS.index.name = "hora"
    mostrar(HORAS.reset_index(), "Distribución de líneas por hora del día (% de cada columna)",
              formatos={c: "pct" for c in horas}, max_filas=24)
    for c, s in horas.items():
        pico = int(s.idxmax())
        madrugada = float(s.loc[0:5].sum() / s.sum()) if s.sum() else 0.0
        if c == COLUMNA_FECHA_VENTA:
            hallazgo(f"{c}: hora pico {pico}:00; {madrugada:.1%} de las líneas entre 0 y 5 h.")
            if 0 <= pico <= 5 or madrugada > 0.25:
                aviso(f"{c} parece venir en UTC (pico de madrugada). Si se confirma, pon HORAS_EN_UTC = True "
                      "en la configuración de todos los notebooks y vuelve a correr este.")
if len(CAND):
    puntaje = CAND["pct_convertible"].fillna(0) * CAND["pct_en_su_mes"].fillna(0)
    mejor = CAND.loc[puntaje.idxmax(), "columna"]
    hallazgo(f"Columna recomendada para el día de la venta: {mejor} "
                 f"({CAND.loc[puntaje.idxmax(), 'pct_convertible']:.1%} convertible, "
                 f"{CAND.loc[puntaje.idxmax(), 'pct_en_su_mes']:.1%} dentro del mes de su volumen).")
    if mejor != COLUMNA_FECHA_VENTA:
        aviso(f"La configuración usa {COLUMNA_FECHA_VENTA} pero la evidencia favorece {mejor}: "
              "si se confirma, cambia COLUMNA_FECHA_VENTA en la configuración de todos los notebooks.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Serie diaria: días faltantes, días incompletos y patrón semanal
# MAGIC **Qué hace.** Con la columna configurada calcula por día: líneas, tickets, tiendas que reportan, unidades y venta
# MAGIC (`order_line_total_with_discount`). La venta se mide **sin las líneas fuera de rango POS** (importes, cantidades o
# MAGIC costos imposibles): unas cuantas líneas de cientos de millones de pesos bastan para fabricar un pico diario o un
# MAGIC "jueves fuerte" que no existen. La venta cruda se muestra al lado como referencia. Resume por mes, busca días sin datos
# MAGIC o con menos de la mitad de las líneas de la mediana del mes, y mide el patrón por día de la semana y por día del mes.
# MAGIC
# MAGIC **Qué mirar.** Días faltantes o caídas bruscas de tiendas que reportan (cargas incompletas o apagones, que no son
# MAGIC demanda cero); líneas cuya fecha cae fuera del mes de su volumen (llegaron tarde); el índice del sábado frente al lunes.

# COMMAND ----------

titulo("2. SERIE DIARIA")
d = df.withColumn("fecha", fecha_venta(df))
venta_cruda = num(C["total_con_descuento"]) if C["total_con_descuento"] else F.lit(None).cast("double")
fuera_pos = fuera_de_rango_pos(C)
venta = F.when(~fuera_pos, venta_cruda)                      # venta sin líneas fuera de rango POS
unidades = num(C["cantidad"]) if C["cantidad"] else F.lit(None).cast("double")
ticket = F.col(C["id_ticket"]) if C["id_ticket"] else F.lit(None)
tienda = F.col(C["tienda"]) if C["tienda"] else F.lit(None)

cruce = (d.groupBy("anio_mes", F.date_format("fecha", "yyyy-MM").alias("mes_de_la_fecha")).count()
         .orderBy("anio_mes", "mes_de_la_fecha").toPandas())
fuera = cruce[cruce["anio_mes"] != cruce["mes_de_la_fecha"]]
mostrar(cruce.pivot_table(index="anio_mes", columns="mes_de_la_fecha", values="count", fill_value=0,
                            aggfunc="sum").reset_index(),
          "Líneas por volumen (filas) y mes de su fecha (columnas)")
pct_fuera = fuera["count"].sum() / cruce["count"].sum() if len(cruce) else 0
hallazgo(f"{fuera['count'].sum():,} líneas ({pct_fuera:.2%}) tienen una fecha fuera del mes de su volumen "
             "(llegada tardía, fecha nula o mal capturada).")

DIARIA = (d.where(F.col("fecha").isNotNull())
          .groupBy("fecha")
          .agg(F.count("*").alias("lineas"),
               F.approx_count_distinct(ticket, 0.01).alias("tickets"),
               F.countDistinct(tienda).alias("tiendas"),
               F.sum(F.when(~fuera_pos, unidades)).alias("unidades"),
               F.sum(venta).alias("venta"),
               F.sum(venta_cruda).alias("venta_cruda"),
               F.sum(fuera_pos.cast("int")).alias("lineas_fuera_rango"))
          .orderBy("fecha").toPandas())
DIARIA["fecha"] = pd.to_datetime(DIARIA["fecha"])
INICIO_PERIODO = pd.Timestamp(min(MESES) + "-01")
FIN_PERIODO = (pd.Timestamp(max(MESES) + "-01") + pd.offsets.MonthEnd(0))
atipicas = DIARIA[(DIARIA["fecha"] < INICIO_PERIODO) | (DIARIA["fecha"] > FIN_PERIODO)]
if len(atipicas):
    hallazgo(f"{int(atipicas['lineas'].sum()):,} líneas con fecha fuera del periodo {INICIO_PERIODO.date()}–{FIN_PERIODO.date()} "
                 f"(de {atipicas['fecha'].min().date()} a {atipicas['fecha'].max().date()}); se excluyen del análisis diario.")
DIARIA = DIARIA[(DIARIA["fecha"] >= INICIO_PERIODO) & (DIARIA["fecha"] <= FIN_PERIODO)].reset_index(drop=True)
DIARIA["anio_mes"] = DIARIA["fecha"].dt.strftime("%Y-%m")
DIARIA["dia_semana"] = DIARIA["fecha"].dt.dayofweek + 1        # 1 = lunes
DIARIA["dia_mes"] = DIARIA["fecha"].dt.day

calendario = pd.date_range(INICIO_PERIODO, FIN_PERIODO, freq="D")
faltan = sorted(set(calendario) - set(DIARIA["fecha"]))
med_mes = DIARIA.groupby("anio_mes")["lineas"].transform("median")
med_tiendas = DIARIA["tiendas"].median()
DIARIA["dia_bajo"] = DIARIA["lineas"] < UMBRAL_DIA_BAJO * med_mes
DIARIA["pocas_tiendas"] = DIARIA["tiendas"] < UMBRAL_TIENDAS * med_tiendas

MENSUAL = (DIARIA.groupby("anio_mes")
           .agg(dias=("fecha", "count"), lineas=("lineas", "sum"), tickets=("tickets", "sum"),
                venta=("venta", "sum"), venta_cruda=("venta_cruda", "sum"), lineas_fuera_rango=("lineas_fuera_rango", "sum"),
                venta_dia=("venta", "mean"), tiendas_dia=("tiendas", "median"),
                dias_bajos=("dia_bajo", "sum"), dias_pocas_tiendas=("pocas_tiendas", "sum"))
           .reset_index())
MENSUAL["dias_calendario"] = [pd.Period(m).days_in_month for m in MENSUAL["anio_mes"]]
MENSUAL["ticket_promedio"] = MENSUAL["venta"] / MENSUAL["tickets"]
MENSUAL["pct_venta_fuera_rango"] = 1 - MENSUAL["venta"] / MENSUAL["venta_cruda"]
mostrar(MENSUAL[["anio_mes", "dias", "dias_calendario", "lineas", "tickets", "venta", "venta_dia",
                   "ticket_promedio", "tiendas_dia", "dias_bajos", "dias_pocas_tiendas", "venta_cruda",
                   "lineas_fuera_rango", "pct_venta_fuera_rango"]],
          "Resumen mensual (venta sin líneas fuera de rango POS; la cruda, al final)",
          formatos={"lineas": "int", "tickets": "int", "venta": "dinero", "venta_dia": "dinero",
                    "ticket_promedio": "dec2", "tiendas_dia": "int", "dias_bajos": "int", "dias_pocas_tiendas": "int",
                    "venta_cruda": "dinero", "lineas_fuera_rango": "int", "pct_venta_fuera_rango": "pct"})
hallazgo(f"Las líneas fuera de rango POS ({int(MENSUAL['lineas_fuera_rango'].sum()):,}) concentran "
         f"{1 - MENSUAL['venta'].sum() / MENSUAL['venta_cruda'].sum():.1%} de la venta cruda del periodo; el mes más afectado es "
         f"{MENSUAL.loc[MENSUAL['pct_venta_fuera_rango'].idxmax(), 'anio_mes']} "
         f"({MENSUAL['pct_venta_fuera_rango'].max():.1%}). Todo el análisis de venta de aquí en adelante las excluye.")
picos = DIARIA.assign(pct=1 - DIARIA["venta"] / DIARIA["venta_cruda"]).sort_values("pct", ascending=False).head(5)
mostrar(picos[["fecha", "venta", "venta_cruda", "lineas_fuera_rango", "pct"]],
        "Días donde las líneas fuera de rango pesan más (picos falsos de la venta cruda)",
        formatos={"venta": "dinero", "venta_cruda": "dinero", "lineas_fuera_rango": "int", "pct": "pct"})
hallazgo(f"Periodo con datos: {DIARIA['fecha'].min().date()} a {DIARIA['fecha'].max().date()} "
             f"({len(DIARIA)} días con ventas de {len(calendario)} del calendario); faltan "
             f"{len(faltan)} días" + (f": {resumir_fechas(faltan)}." if faltan else "."))
if faltan:
    aviso(f"Faltan {len(faltan)} días sin ninguna venta en todo el país ({faltan[0].date()} a {faltan[-1].date()}): "
          "es una entrega incompleta, no demanda cero. El panel (09) los marca como 'sin datos'.")
sospechosos = DIARIA[DIARIA["dia_bajo"] | DIARIA["pocas_tiendas"]]
mostrar(sospechosos[["fecha", "lineas", "tiendas", "venta", "dia_bajo", "pocas_tiendas"]],
          "Días sospechosos de carga incompleta o apagón", formatos={"lineas": "int", "tiendas": "int", "venta": "dinero"})
if len(sospechosos):
    hallazgo(f"{len(sospechosos)} días sospechosos (menos del {UMBRAL_DIA_BAJO:.0%} de las líneas de la mediana del mes "
                 f"o menos del {UMBRAL_TIENDAS:.0%} de las tiendas); candidatos a excluir o a marcar en Plata.")

# patrón semanal y mensual sólo con días normales
normales = DIARIA[~(DIARIA["dia_bajo"] | DIARIA["pocas_tiendas"])]
NOMBRES_DIA = {1: "lun", 2: "mar", 3: "mié", 4: "jue", 5: "vie", 6: "sáb", 7: "dom"}
idx_dow = indice_relativo(normales["venta"], normales["dia_semana"])
idx_dow_lin = indice_relativo(normales["lineas"], normales["dia_semana"])
DOW = pd.DataFrame({"dia": [NOMBRES_DIA[k] for k in idx_dow], "indice_venta": list(idx_dow.values()),
                    "indice_lineas": [idx_dow_lin[k] for k in idx_dow]})
mostrar(DOW, "Índice por día de la semana (1.00 = día promedio)", formatos={"indice_venta": "dec2", "indice_lineas": "dec2"})
hallazgo(f"Día más fuerte: {DOW.loc[DOW['indice_venta'].idxmax(), 'dia']} "
             f"({DOW['indice_venta'].max():.2f}); más débil: {DOW.loc[DOW['indice_venta'].idxmin(), 'dia']} "
             f"({DOW['indice_venta'].min():.2f}); razón máx./mín. {DOW['indice_venta'].max() / DOW['indice_venta'].min():.2f}.")
idx_dom = indice_relativo(normales["venta"], normales["dia_mes"])
# el día del mes se confunde con el de la semana (un 31 que cae en sábado vende más por ser sábado): se ajusta dividiendo
# cada día entre el promedio de su mismo día de la semana en su mes
normales = normales.assign(ajustada=normales["venta"] / normales.groupby(["anio_mes", "dia_semana"])["venta"].transform("mean"))
aj = normales.groupby("dia_mes").agg(indice_ajustado=("ajustada", "mean"), dias=("ajustada", "size"))
DOM = pd.DataFrame({"dia_mes": list(idx_dom.keys()), "indice_venta": list(idx_dom.values())}).merge(
    aj.reset_index(), on="dia_mes", how="left")
mostrar(DOM.sort_values("indice_ajustado", ascending=False).head(10),
        "Días del mes con mayor venta (índice crudo y ajustado por día de la semana)",
        formatos={"indice_venta": "dec2", "indice_ajustado": "dec2", "dias": "int"})
QUINCENA = [1, 2, 15, 16, 30, 31]
quincena = DOM[DOM["dia_mes"].isin(QUINCENA)]["indice_ajustado"].mean()
resto = DOM[~DOM["dia_mes"].isin(QUINCENA)]["indice_ajustado"].mean()
hallazgo(f"Efecto quincena (ajustado por día de la semana): los días 1, 2, 15, 16, 30 y 31 venden en promedio "
         f"{quincena / resto - 1:+.1%} que el resto (índice {quincena:.2f} contra {resto:.2f}).")

guardar(DIARIA, "mlops", "eda_cobertura_diaria", "Serie diaria nacional: lineas, tickets, tiendas, unidades y venta por dia (EDA 05)")

CAL = DIARIA.set_index("fecha").reindex(calendario)          # los días sin datos quedan como hueco en la gráfica
grafica_serie(CAL.index, CAL["venta"] / 1e6, "Venta diaria sin líneas fuera de rango (todas las tiendas)",
              "millones de MXN", EVENTOS)
grafica_serie(CAL.index, CAL["tiendas"], "Tiendas que reportan ventas cada día", "tiendas", EVENTOS)
grafica_barras(DOW["dia"], DOW["indice_venta"], "Venta por día de la semana (días normales)", "índice (1.00 = día promedio)",
               resaltar=[DOW.loc[DOW["indice_venta"].idxmax(), "dia"]], referencia=1.0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Actividad de las tiendas
# MAGIC **Qué hace.** Para cada tienda calcula su primer y último día con venta, los días con venta y los huecos dentro de
# MAGIC su ventana, y la clasifica: **estable** (todo el periodo, casi sin huecos), **alta** (abre después del inicio),
# MAGIC **baja** (deja de vender antes del final), **con huecos** (más del 5 % de días sin venta dentro de su ventana).
# MAGIC También cuenta, por día, cuántas tiendas que estaban operando no reportaron nada.
# MAGIC
# MAGIC **Qué mirar.** Este resultado define el relleno de ceros: sólo se rellena dentro de la ventana de cada tienda y
# MAGIC los días en que una tienda no emitió ningún ticket son *inactividad*, no demanda cero. Un día con muchas tiendas
# MAGIC ausentes a la vez apunta a un problema de carga.

# COMMAND ----------

titulo("3. ACTIVIDAD DE LAS TIENDAS")
if requiere(C, "tienda"):
    TIENDA_DIA = (d.where(F.col("fecha").between(INICIO_PERIODO.date(), FIN_PERIODO.date()))
                  .groupBy(tienda.alias("tienda"), "fecha")
                  .agg(F.count("*").alias("lineas"), F.approx_count_distinct(ticket, 0.02).alias("tickets"),
                       F.sum(venta).alias("venta")))
    td = spark.table(guardar(TIENDA_DIA, "mlops", "eda_tienda_dia",
                             "Lineas, tickets y venta por tienda y dia (EDA 05); base de la ventana activa de cada tienda"))
    d_min, d_max = DIARIA["fecha"].min().date(), DIARIA["fecha"].max().date()
    ACT = (td.groupBy("tienda")
           .agg(F.min("fecha").alias("primera"), F.max("fecha").alias("ultima"),
                F.countDistinct("fecha").alias("dias_con_venta"), F.sum("venta").alias("venta"),
                F.sum("lineas").alias("lineas"))
           .withColumn("dias_ventana", F.datediff("ultima", "primera") + 1)
           .withColumn("cobertura", F.col("dias_con_venta") / F.col("dias_ventana"))
           .toPandas())
    # la ventana se mide sólo con días que tienen datos en todo el país (un hueco de entrega no es un hueco de la tienda)
    dias_datos = np.array(sorted(DIARIA["fecha"].dt.date.unique()))
    ACT["dias_ventana"] = (np.searchsorted(dias_datos, pd.to_datetime(ACT["ultima"]).dt.date.to_numpy(), side="right")
                           - np.searchsorted(dias_datos, pd.to_datetime(ACT["primera"]).dt.date.to_numpy(), side="left"))
    ACT["cobertura"] = ACT["dias_con_venta"] / ACT["dias_ventana"].clip(lower=1)
    margen = 7
    ACT["abre_en_periodo"] = pd.to_datetime(ACT["primera"]) > pd.Timestamp(d_min) + pd.Timedelta(days=margen)
    ACT["cierra_en_periodo"] = pd.to_datetime(ACT["ultima"]) < pd.Timestamp(d_max) - pd.Timedelta(days=margen)
    ACT["clase"] = np.select(
        [ACT["abre_en_periodo"] & ACT["cierra_en_periodo"], ACT["abre_en_periodo"], ACT["cierra_en_periodo"],
         ACT["cobertura"] < 0.95],
        ["alta y baja", "alta", "baja", "con huecos"], default="estable")
    RES = (ACT.groupby("clase").agg(tiendas=("tienda", "count"), venta=("venta", "sum"),
                                     dias_mediana=("dias_con_venta", "median"),
                                     cobertura_mediana=("cobertura", "median")).reset_index())
    RES["pct_tiendas"] = RES["tiendas"] / RES["tiendas"].sum()
    RES["pct_venta"] = RES["venta"] / RES["venta"].sum()
    mostrar(RES.sort_values("tiendas", ascending=False)[["clase", "tiendas", "pct_tiendas", "pct_venta",
                                                            "dias_mediana", "cobertura_mediana"]],
              "Tiendas por tipo de actividad", formatos={"tiendas": "int", "pct_tiendas": "pct", "pct_venta": "pct",
                                                         "dias_mediana": "int", "cobertura_mediana": "pct"})
    estables = RES.loc[RES["clase"] == "estable", "pct_tiendas"].sum()
    hallazgo(f"{len(ACT):,} tiendas con venta; {estables:.1%} estables todo el periodo. "
                 f"Abren en el periodo: {int(ACT['abre_en_periodo'].sum()):,}; dejan de vender: {int(ACT['cierra_en_periodo'].sum()):,}; "
                 f"mediana de días con venta: {ACT['dias_con_venta'].median():.0f} de {len(DIARIA)}.")
    guardar(ACT, "mlops", "eda_tiendas_actividad",
            "Ventana activa por tienda: primera y ultima venta, dias con venta, cobertura y clase (EDA 05)")

    # tiendas en su ventana que no reportan cada día
    ventana = spark.createDataFrame(ACT[["tienda", "primera", "ultima"]])
    cal = spark.createDataFrame(pd.DataFrame({"fecha": list(dias_datos)}))      # sólo días con datos
    en_ventana = (ventana.crossJoin(cal).where(F.col("fecha").between(F.col("primera"), F.col("ultima")))
                  .groupBy("fecha").agg(F.count("*").alias("tiendas_en_ventana")))
    reportan = td.groupBy("fecha").agg(F.countDistinct("tienda").alias("tiendas_reportan"))
    AUS = en_ventana.join(reportan, "fecha", "left").fillna(0).orderBy("fecha").toPandas()
    AUS["ausentes"] = AUS["tiendas_en_ventana"] - AUS["tiendas_reportan"]
    AUS["pct_ausentes"] = AUS["ausentes"] / AUS["tiendas_en_ventana"]
    mostrar(AUS.sort_values("pct_ausentes", ascending=False).head(10),
              "Días con más tiendas ausentes dentro de su ventana",
              formatos={"tiendas_en_ventana": "int", "tiendas_reportan": "int", "ausentes": "int", "pct_ausentes": "pct"})
    # cerca de los extremos el % baja por construcción (la ventana empieza en la primera venta y acaba en la última),
    # así que el día típico y el peor se miden lejos de los bordes
    BORDE = 21
    AUS["fecha"] = pd.to_datetime(AUS["fecha"])
    interior = AUS[(AUS["fecha"] >= pd.Timestamp(d_min) + pd.Timedelta(days=BORDE)) &
                   (AUS["fecha"] <= pd.Timestamp(d_max) - pd.Timedelta(days=BORDE))]
    interior = interior if len(interior) else AUS
    hallazgo(f"En un día típico {interior['pct_ausentes'].median():.1%} de las tiendas en operación no reporta ventas "
                 f"(medido a más de {BORDE} días de los extremos); el peor día "
                 f"({pd.Timestamp(interior.loc[interior['pct_ausentes'].idxmax(), 'fecha']).date()}) llega a "
                 f"{interior['pct_ausentes'].max():.1%}. Esos días-tienda son inactividad, no demanda cero.")
    AUS_CAL = AUS.set_index("fecha").reindex(calendario)
    grafica_serie(AUS_CAL.index, AUS_CAL["pct_ausentes"] * 100,
                  f"Tiendas en operación que no reportan ventas (los primeros y últimos {BORDE} días bajan por construcción)",
                  "% de tiendas en su ventana", EVENTOS, color=NARANJA)

# COMMAND ----------

cerrar("05_eda_tiempo_y_tiendas", siguiente="06_eda_variables_y_promociones")
