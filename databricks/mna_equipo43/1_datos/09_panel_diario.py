# Databricks notebook source
# MAGIC %md
# MAGIC # 09 · Panel diario tienda × categoría
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Para qué sirve.** Arma la tabla que usan los modelos: una fila por **tienda, categoría y día**, con la venta del día
# MAGIC y sus componentes. El punto delicado son los días sin venta:
# MAGIC
# MAGIC | Situación | Cómo se detecta | Qué se hace |
# MAGIC | --- | --- | --- |
# MAGIC | **Cero comercial** | La tienda emitió tickets ese día, pero ninguno de la categoría | Fila con `venta = 0`: demanda nula observada |
# MAGIC | **Tienda inactiva** | La tienda no emitió ningún ticket ese día (cierre o apagón) | Fila con `venta` nula y `tienda_inactiva = 1`: no es demanda cero |
# MAGIC | **Día sin datos** | Casi ninguna tienda del país tiene ventas ese día (falta la entrega, como julio 16–31) | `sin_datos = 1`; la venta queda nula salvo en las tiendas que sí reportaron |
# MAGIC | **Fuera de ventana** | Antes de la primera venta o después de la última de la tienda | No se genera la fila (no se inventan ceros antes de abrir) |
# MAGIC
# MAGIC Además marca `en_warmup` (primeros 7 días de la tienda: los rezagos aún no tienen historia), `serie_relevante`
# MAGIC (la serie tiene al menos 5 días con venta), `dia_atipico` (días con datos pero con menos del 95 % de las tiendas que
# MAGIC reportan normalmente, como el 17 de mayo y el 6–7 de junio que encontró el 05) y `tipo_categoria` (mercancía,
# MAGIC servicio o genérico, del catálogo del 08). Es el mismo diseño del Avance 2, ahora con los seis meses.
# MAGIC
# MAGIC **Calendario de promociones.** El 06 mostró que 95 % de la venta en promoción viene de promociones largas y
# MAGIC continuas, así que su vigencia se puede reconstruir: cada promoción tiene un inicio y un fin (primera y última venta
# MAGIC con ese identificador en todo el país) y un conjunto de tiendas y categorías. `promos_vigentes` cuenta, para cada
# MAGIC tienda, categoría y día, cuántas promociones estaban vigentes. Es la versión observada del plan comercial: para usarla
# MAGIC como variable conocida a futuro en la fase 2 hay que calcularla sólo con datos hasta el origen o pedir el plan a Bnext.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `silver.ventas_lineas` |
# MAGIC | **Escribe** | `silver.panel_categoria_dia` (partición `mes`) · `silver.promociones` · `silver.calendario_promociones` · `mlops.panel_resumen` |
# MAGIC | **Tiempo** | 10–30 min |

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL PROYECTO · es la misma en los 11 notebooks (si cambias algo, cámbialo en todos)
# ════════════════════════════════════════════════════════════════════════════════════════════════
CATALOGO       = "modeladoprescriptivoventas"     # catálogo de Unity Catalog del proyecto
ESQUEMA_DATOS  = "maestria"                       # esquema donde están los volúmenes de Bnext
RAIZ_VOLUMENES = "/Volumes"
MESES = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]   # un volumen por mes

# Cortes del walk-forward. Son FIJOS (los del Avance 4): así, al llegar un mes nuevo, los modelos se siguen comparando
# en las mismas ventanas. Los bloques de prueba son datos no vistos: no intervienen en el ajuste ni en la elección del
# modelo y sólo existen cuando ya están cargados sus días (septiembre).
DIAS_CORTE = 14                                           # días que se pronostican desde cada origen
ORIGEN_AJUSTE = "2026-05-20"                              # bloque 21-may…03-jun: sólo para afinar hiperparámetros
ORIGENES_CORTES = {"corte_1": "2026-06-03", "corte_2": "2026-06-17",      # desarrollo: con estos se compara y se elige
                   "corte_3": "2026-07-01", "corte_4": "2026-08-17"}
ORIGENES_PRUEBA = {"prueba_1": "2026-08-31", "prueba_2": "2026-09-14"}    # prueba no vista: 01–14 y 15–28 de septiembre

# Dónde se guardan las tablas del equipo:
#   False → dentro de maestria con prefijo de capa: modeladoprescriptivoventas.maestria.bronze_ventas_lineas,
#           maestria.silver_ventas_lineas, maestria.mlops_hallazgos, … (es lo que permiten hoy los permisos del equipo)
#   True  → esquemas propios bronze / silver / gold / mlops, si algún día Bnext da permiso de CREATE SCHEMA en el catálogo
ESQUEMAS_PROPIOS = False

# Qué columna define el día de la venta y en qué zona horaria vienen sus horas. El notebook 05 compara las candidatas;
# si recomienda otra columna o detecta horas en UTC, cámbialo aquí en todos los notebooks y vuelve a correr del 05 en adelante.
COLUMNA_FECHA_VENTA = "finish_order"
HORAS_EN_UTC = False

VOLUMENES = f"{RAIZ_VOLUMENES}/{CATALOGO}/{ESQUEMA_DATOS}"      # /Volumes/modeladoprescriptivoventas/maestria

if ESQUEMAS_PROPIOS:      # si los esquemas propios no existen, se usa maestria (así nunca falla por esquema inexistente)
    try:
        _existentes = {r[0] for r in spark.sql(f"SHOW SCHEMAS IN {CATALOGO}").collect()}
    except Exception:
        _existentes = set()
    if not {"bronze", "silver", "gold", "mlops"} <= _existentes:
        ESQUEMAS_PROPIOS = False
        print("(los esquemas bronze/silver/gold/mlops no existen: las tablas se guardan en maestria con prefijo)")


def tabla(capa: str, nombre: str) -> str:
    """Nombre completo de una tabla del equipo. capa = bronze, silver, gold o mlops.
    tabla("bronze", "ventas_lineas") → modeladoprescriptivoventas.bronze.ventas_lineas"""
    if ESQUEMAS_PROPIOS:
        return f"{CATALOGO}.{capa}.{nombre}"
    return f"{CATALOGO}.{ESQUEMA_DATOS}.{capa}_{nombre}"


def volumen(*partes: str) -> str:
    """Ruta dentro de los volúmenes de maestria. volumen("2026-03") → /Volumes/modeladoprescriptivoventas/maestria/2026-03"""
    return "/".join([VOLUMENES, *partes])


def cortes_walk_forward(dias_con_datos, incluir_prueba=True):
    """Cortes configurados arriba como lista de {corte, origen, desde, hasta, fase}. Un corte sólo se usa si su bloque de
    14 días y los 14 días previos a su origen tienen datos (así ninguno cae en un hueco de entrega como julio 16–31).
    Los de prueba aparecen únicamente cuando sus días ya están cargados."""
    import pandas as pd
    dias = {pd.Timestamp(d).date() for d in dias_con_datos}
    fuentes = list(ORIGENES_CORTES.items()) + (list(ORIGENES_PRUEBA.items()) if incluir_prueba else [])
    salida = []
    for nombre, origen in fuentes:
        o = pd.Timestamp(origen)
        necesarios = {(o + pd.Timedelta(days=k)).date() for k in range(-DIAS_CORTE + 1, DIAS_CORTE + 1)}
        if not necesarios <= dias:
            if nombre.startswith("corte_"):
                print(f"⚠ {nombre} (origen {o.date()}): faltan días con datos en su bloque o en los 14 días previos; se omite")
            continue
        salida.append({"corte": nombre, "origen": o.date(), "desde": (o + pd.Timedelta(days=1)).date(),
                       "hasta": (o + pd.Timedelta(days=DIAS_CORTE)).date(),
                       "fase": "desarrollo" if nombre.startswith("corte_") else "prueba"})
    return salida


print("Catálogo         :", CATALOGO)
print("Datos crudos     :", ", ".join(volumen(m) for m in MESES[:2]), "…", volumen(MESES[-1]))
print("Tablas del equipo:", tabla("bronze", "…"), "|", tabla("silver", "…"), "|", tabla("mlops", "…"))

# COMMAND ----------

# ─────────────────────── funciones de apoyo (iguales en todos los notebooks; no hace falta editarlas) ───────────────────────
import datetime as dt
import json
import os
import re
import time

import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

import logging
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message=".*No Partition Defined for Window.*")   # ventanas globales pequeñas (ABC, rangos)
logging.getLogger("pyspark.sql.connect.logging").setLevel(logging.CRITICAL)   # serverless: no repetir errores ya manejados
logging.getLogger("py4j").setLevel(logging.CRITICAL)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 80)
def conf_set(clave, valor):
    """Cambia una configuración de Spark si el cómputo lo permite (serverless sólo acepta algunas). Devuelve True/False."""
    try:
        spark.conf.set(clave, valor)
        return True
    except Exception:
        return False


def persistir(df):
    """Guarda en memoria/disco un resultado que se usa varias veces. En serverless no está permitido: se sigue sin guardar."""
    try:
        from pyspark import StorageLevel
        return df.persist(StorageLevel.MEMORY_AND_DISK)
    except Exception:
        return df


def liberar(df):
    try:
        df.unpersist()
    except Exception:
        pass


conf_set("spark.sql.session.timeZone", "America/Mexico_City")
_INICIO_NOTEBOOK = time.time()
HALLAZGOS = []          # lo que este notebook encontró; se imprime al final y se guarda en mlops.hallazgos


def titulo(texto):
    print("\n" + "═" * 110 + f"\n{texto}\n" + "═" * 110)


def subtitulo(texto):
    print(f"\n── {texto} " + "─" * max(0, 104 - len(texto)))


def hallazgo(texto):
    """Un resultado que importa para decidir algo. Se repite en el resumen final."""
    HALLAZGOS.append(texto)
    print(f"► {texto}")


def aviso(texto):
    """Algo que hay que revisar o decidir antes de seguir."""
    HALLAZGOS.append(f"REVISAR: {texto}")
    print(f"⚠ {texto}")


_FMT = {"int": lambda v: f"{v:,.0f}", "pct": lambda v: f"{v:.1%}", "pct2": lambda v: f"{v:.2%}",
        "dinero": lambda v: f"${v:,.0f}", "dec1": lambda v: f"{v:,.1f}", "dec2": lambda v: f"{v:,.2f}",
        "dec3": lambda v: f"{v:,.3f}", "dec4": lambda v: f"{v:,.4f}",
        "auto": lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.4f}"}


def _vacio(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (v is pd.NaT)


def mostrar(df, titulo_tabla=None, formatos=None, max_filas=60):
    """Imprime una tabla (pandas o Spark) completa y alineada, lista para copiar.
    formatos = {"columna": "int" | "pct" | "pct2" | "dinero" | "dec1" | "dec2" | "dec3" | "dec4" | "auto"}"""
    if titulo_tabla:
        subtitulo(titulo_tabla)
    if df is None:
        print("(sin datos)")
        return
    if not isinstance(df, pd.DataFrame):
        df = df.limit(max_filas + 1).toPandas()
    if df.empty:
        print("(sin filas)")
        return
    vista = df.head(max_filas).copy()
    for col, f in (formatos or {}).items():
        if col in vista.columns:
            vista[col] = vista[col].map(lambda v: "" if _vacio(v) else _FMT[f](float(v)))
    print(vista.to_string(index=False))
    if len(df) > max_filas:
        print(f"… y {len(df) - max_filas:,} filas más")


def _para_spark(pdf):
    """pandas → (pandas limpio, esquema de Spark): texto mixto a str, nulos de pandas a None, tipos explícitos
    (así una columna toda nula no rompe la creación de la tabla)."""
    from pyspark.sql import types as T
    pdf = pdf.copy()
    pdf.columns = [str(c) for c in pdf.columns]
    campos = []
    for c in pdf.columns:
        s = pdf[c]
        d = str(s.dtype)
        if d.startswith("datetime64"):
            if getattr(s.dt, "tz", None) is not None:
                pdf[c] = s.dt.tz_localize(None)
            tipo = T.TimestampType()
        elif d.startswith(("int", "uint")):
            tipo = T.LongType()
        elif d.startswith(("float", "Int", "Float", "UInt")):
            pdf[c] = s.astype("float64")
            tipo = T.DoubleType()
        elif d == "bool":
            tipo = T.BooleanType()
        elif d == "object" and len(s.dropna()) and all(isinstance(v, dt.date) and not isinstance(v, dt.datetime)
                                                       for v in s.dropna().head(200)):
            pdf[c] = pd.Series([None if _vacio(v) else v for v in s], index=s.index, dtype=object)
            tipo = T.DateType()
        else:
            pdf[c] = pd.Series([None if _vacio(v) or v is pd.NA else str(v) for v in s], index=s.index, dtype=object)
            tipo = T.StringType()
        campos.append(T.StructField(c, tipo, True))
    return pdf, T.StructType(campos)


def guardar(df, capa, nombre, comentario=None, modo="overwrite", particion=None):
    """Guarda un DataFrame (pandas o Spark) como tabla Delta del equipo y devuelve su nombre completo."""
    nombre_tabla = tabla(capa, nombre)
    if isinstance(df, pd.DataFrame):
        limpio, esquema = _para_spark(df)
        sdf = spark.createDataFrame(limpio, schema=esquema)
    else:
        sdf = df
    w = sdf.write.format("delta").mode(modo)
    w = w.option("overwriteSchema", "true") if modo == "overwrite" else w.option("mergeSchema", "true")
    if particion:
        w = w.partitionBy(particion)
    w.saveAsTable(nombre_tabla)
    if comentario:
        try:
            spark.sql(f"COMMENT ON TABLE {nombre_tabla} IS '{comentario.replace(chr(39), '')}'")
        except Exception:
            pass
    print(f"   → guardada {nombre_tabla}")
    return nombre_tabla


def tipos_de(df):
    """{columna: tipo de Spark} de un DataFrame."""
    return {f.name: f.dataType.simpleString() for f in df.schema.fields}


def leer_parquet(ruta):
    """Lee un parquet (archivo o carpeta de un volumen) con Spark, aunque traiga timestamps en nanosegundos
    (Spark no los lee directo): los convierte a timestamp conservando la hora del reloj."""
    import pyarrow.parquet as pq
    muestra = ruta
    if os.path.isdir(ruta):
        partes = sorted(os.path.join(b, a) for b, _, fs in os.walk(ruta) for a in fs if a.endswith(".parquet"))
        muestra = partes[0] if partes else ruta
    try:
        nanos = {f.name: f.type.tz is not None for f in pq.read_schema(muestra) if str(f.type).startswith("timestamp[ns")}
    except Exception:
        nanos = {}
    conf_set("spark.sql.parquet.inferTimestampNTZ.enabled", "true")   # horas sin zona = hora del reloj
    if nanos and not conf_set("spark.sql.legacy.parquet.nanosAsLong", "true"):
        # serverless no permite esa configuración: esquema explícito con los nanosegundos como número (sí se puede leer así)
        from pyspark.sql.pandas.types import from_arrow_type
        from pyspark.sql import types as T
        import pyarrow as pa
        campos = []
        for f in pq.read_schema(muestra):
            t = f.type.value_type if pa.types.is_dictionary(f.type) else f.type
            if pa.types.is_null(t):
                continue
            try:
                campos.append(T.StructField(f.name, T.LongType() if f.name in nanos else from_arrow_type(t), True))
            except Exception:
                pass
        df = spark.read.schema(T.StructType(campos)).parquet(ruta)
    else:
        df = spark.read.parquet(ruta)
    leidos = dict(df.dtypes)
    for c, con_zona in nanos.items():
        if leidos.get(c) != "bigint":        # INT96 u otro: Spark ya lo leyó como timestamp
            continue
        ts = F.expr(f"timestamp_micros(cast(`{c}` / 1000 as bigint))")
        df = df.withColumn(c, ts if con_zona else F.to_utc_timestamp(ts, spark.conf.get("spark.sql.session.timeZone")))
    return df


SENSIBLE = re.compile(r"lat|lon|geo|direcc|address|calle|tel|phone|correo|email|rfc|curp|cliente|customer|usuario|user_|nombre_", re.I)


def json_seguro(obj, prof=0):
    """Copia de un JSON sin llaves sensibles (ubicación, contacto, personas) y con listas recortadas, para imprimir."""
    if prof > 4:
        return "…"
    if isinstance(obj, dict):
        return {k: json_seguro(v, prof + 1) for k, v in obj.items() if not SENSIBLE.search(str(k))}
    if isinstance(obj, list):
        return [json_seguro(v, prof + 1) for v in obj[:10]] + (["…"] if len(obj) > 10 else [])
    return obj


def existe(nombre_tabla):
    try:
        return spark.catalog.tableExists(nombre_tabla)
    except Exception:
        try:
            spark.table(nombre_tabla).schema
            return True
        except Exception:
            return False


def leer_tabla(nombre_tabla, notebook_que_la_crea):
    """spark.table con un mensaje claro si la tabla todavía no existe."""
    if not existe(nombre_tabla):
        raise RuntimeError(f"No existe {nombre_tabla}. Corre antes el notebook {notebook_que_la_crea}.")
    return spark.table(nombre_tabla)


def cerrar(notebook, siguiente=None):
    """Imprime el resumen para revisión y guarda los hallazgos en mlops.hallazgos."""
    titulo(f"RESUMEN PARA REVISIÓN · {notebook} · {dt.datetime.now():%Y-%m-%d %H:%M} · "
           f"{(time.time() - _INICIO_NOTEBOOK) / 60:.1f} min")
    for i, h in enumerate(HALLAZGOS, 1):
        print(f"{i:>2}. {h}")
    if not HALLAZGOS:
        print("(sin hallazgos registrados)")
    try:
        if HALLAZGOS:
            ahora = dt.datetime.now()
            filas = [(notebook, ahora, i, h) for i, h in enumerate(HALLAZGOS, 1)]
            (spark.createDataFrame(filas, "notebook string, fecha_corrida timestamp, orden int, hallazgo string")
             .write.format("delta").mode("append").option("mergeSchema", "true")
             .saveAsTable(tabla("mlops", "hallazgos")))
            print(f"\n(hallazgos guardados en {tabla('mlops', 'hallazgos')})")
    except Exception as e:
        print(f"\n(no se pudieron guardar los hallazgos: {type(e).__name__}: {str(e)[:120]})")
    if siguiente:
        print(f"\nSiguiente notebook: {siguiente}")


def resumir_fechas(fechas, max_items=6):
    """Lista de fechas como rangos consecutivos: '2026-07-16 a 2026-07-31' en lugar de 16 fechas sueltas."""
    fechas = sorted(pd.Timestamp(f).date() for f in fechas)
    rangos, ini, ant = [], None, None
    for f in fechas:
        if ini is None:
            ini = ant = f
        elif (f - ant).days == 1:
            ant = f
        else:
            rangos.append((ini, ant)); ini = ant = f
    if ini is not None:
        rangos.append((ini, ant))
    txt = [f"{a}" if a == b else f"{a} a {b}" for a, b in rangos]
    return ", ".join(txt[:max_items]) + (f" y {len(txt) - max_items} rangos más" if len(txt) > max_items else "")

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook
DIAS_WARMUP = 7          # primeros días de cada tienda marcados como calentamiento
UMBRAL_SIN_DATOS = 0.50  # un día con menos de esta fracción de las tiendas de la mediana se marca "sin datos" (entrega incompleta)
MIN_DIAS_CON_VENTA = 5   # una serie es relevante si vendió al menos estos días
UMBRAL_ATIPICO = 0.95    # un día con datos pero con menos de esta fracción de las tiendas de la mediana es "atípico"
TABLA = tabla("silver", "panel_categoria_dia")
INICIO = pd.Timestamp(min(MESES) + "-01").date()
FIN = (pd.Timestamp(max(MESES) + "-01") + pd.offsets.MonthEnd(0)).date()
L = leer_tabla(tabla("silver", "ventas_lineas"), "08_limpieza_plata")
print("Periodo:", INICIO, "→", FIN, "· destino:", TABLA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Agregado diario y actividad de cada tienda

# COMMAND ----------

titulo("1. AGREGADO DIARIO")
lineas = L.where(F.col("fecha_venta").between(INICIO, FIN))
fuera = L.where(~F.col("fecha_venta").between(INICIO, FIN)).count()
if fuera:
    aviso(f"{fuera:,} líneas de Plata tienen fecha fuera de {INICIO} a {FIN}; no entran al panel.")
AGG = (lineas.groupBy("tienda", "categoria_id", F.col("fecha_venta").alias("fecha"))
       .agg(F.sum("venta").alias("venta"), F.sum("venta_sin_descuento").alias("venta_sin_descuento"),
            F.sum("descuento_pesos").alias("descuento_pesos"), F.sum("cantidad").alias("unidades"),
            F.count("*").alias("lineas"), F.size(F.collect_set("id_ticket")).alias("tickets"),
            F.avg(F.col("tiene_promo").cast("double")).alias("intensidad_promo"),
            F.sum("venta_neta").alias("venta_neta"),
            F.size(F.collect_set(F.when(F.col("tiene_promo"), F.col("promo_id")))).alias("promos_en_venta"),
            F.size(F.collect_set("marca_id")).alias("marcas"), F.size(F.collect_set("producto_id")).alias("productos")))
AGG = AGG.withColumn("tasa_descuento", F.when(F.col("venta_sin_descuento") > 0,
                                              F.least(F.greatest(1 - F.col("venta") / F.col("venta_sin_descuento"), F.lit(0.0)), F.lit(1.0)))
                     .otherwise(F.lit(0.0)))
ACTIVA = (lineas.groupBy("tienda", F.col("fecha_venta").alias("fecha"))
          .agg(F.size(F.collect_set("id_ticket")).alias("tickets_tienda")))
ACTIVA = persistir(ACTIVA)
# ventana de cada tienda = primer y último día con algún ticket dentro del periodo
VENTANA = ACTIVA.groupBy("tienda").agg(F.min("fecha").alias("primera_venta"), F.max("fecha").alias("ultima_venta"))
# días sin datos: el país entero no tiene ventas (o casi): falta la entrega de ese día, no es demanda cero
COBERTURA = ACTIVA.groupBy("fecha").agg(F.count("*").alias("tiendas")).toPandas()
mediana_tiendas = COBERTURA["tiendas"].median() if len(COBERTURA) else 0
calendario = pd.date_range(INICIO, FIN, freq="D").date
con_datos = set(COBERTURA.loc[COBERTURA["tiendas"] >= UMBRAL_SIN_DATOS * mediana_tiendas, "fecha"])
DIAS_SIN_DATOS = sorted(set(calendario) - con_datos)
if DIAS_SIN_DATOS:
    aviso(f"{len(DIAS_SIN_DATOS)} días sin datos (menos del {UMBRAL_SIN_DATOS:.0%} de las tiendas de un día típico): "
          f"{resumir_fechas(DIAS_SIN_DATOS)}. Se marcan sin_datos = 1, con venta nula, y no cuentan como tienda inactiva.")
DIAS_ATIPICOS = sorted(set(COBERTURA.loc[(COBERTURA["tiendas"] < UMBRAL_ATIPICO * mediana_tiendas)
                                         & (COBERTURA["tiendas"] >= UMBRAL_SIN_DATOS * mediana_tiendas), "fecha"]))
if DIAS_ATIPICOS:
    hallazgo(f"{len(DIAS_ATIPICOS)} días atípicos (con datos, pero menos del {UMBRAL_ATIPICO:.0%} de las tiendas de un día típico): "
             f"{resumir_fechas(DIAS_ATIPICOS)}. Se marcan dia_atipico = 1 (las tiendas que no vendieron quedan como inactivas).")
if DIAS_ATIPICOS and existe(tabla("silver", "catalogo_tiendas")):
    # ¿dónde faltan las tiendas en los días atípicos? (un evento regional, como una ley seca, se concentra en una región)
    REG = (spark.table(tabla("silver", "catalogo_tiendas"))
           .select("tienda", F.coalesce(F.col("region"), F.lit("SIN REGIÓN")).alias("region")))
    dias_df = spark.createDataFrame(pd.DataFrame({"fecha": sorted(con_datos)}))
    AUS_REG = (VENTANA.join(REG, "tienda", "left").crossJoin(dias_df)
               .where(F.col("fecha").between(F.col("primera_venta"), F.col("ultima_venta")))
               .join(ACTIVA.select("tienda", "fecha").withColumn("reporta", F.lit(1)), ["tienda", "fecha"], "left")
               .groupBy("region", "fecha").agg(F.count("*").alias("en_ventana"),
                                               F.sum(F.coalesce(F.col("reporta"), F.lit(0))).alias("reportan"))
               .toPandas())
    AUS_REG["pct_ausentes"] = 1 - AUS_REG["reportan"] / AUS_REG["en_ventana"]
    tipico = (AUS_REG[~AUS_REG["fecha"].isin(DIAS_ATIPICOS)].groupby("region")["pct_ausentes"].median().rename("dia_tipico"))
    piv = (AUS_REG[AUS_REG["fecha"].isin(DIAS_ATIPICOS)].pivot_table(index="region", columns="fecha", values="pct_ausentes")
           .join(tipico).reset_index())
    piv.columns = [str(c) for c in piv.columns]
    mostrar(piv, "Tiendas ausentes por región en los días atípicos (contra un día típico)",
            formatos={c: "pct" for c in piv.columns if c != "region"})
    for d in DIAS_ATIPICOS:
        x = AUS_REG[AUS_REG["fecha"] == d].set_index("region")["pct_ausentes"] - tipico
        if len(x.dropna()):
            hallazgo(f"{d}: la región con más tiendas ausentes de lo normal es {x.idxmax()} ({x.max():+.1%} sobre su día típico).")
SERIES = AGG.where(F.col("venta") > 0).groupBy("tienda", "categoria_id").agg(F.countDistinct("fecha").alias("dias_con_venta"))
print(f"Series tienda × categoría con al menos una venta: {SERIES.count():,}")
try:
    TIPO_CAT = F.broadcast(spark.table(tabla("silver", "catalogo_categorias")).select("categoria_id", "tipo_categoria"))
except Exception:
    TIPO_CAT = None
    aviso("No se encontró silver.catalogo_categorias con tipo_categoria; corre el 08 de nuevo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Calendario de promociones
# MAGIC **Qué hace.** Una fila por promoción (inicio, fin, alcance y venta) y una por promoción × tienda × categoría.
# MAGIC Una promoción cuyo último día coincide con el último día antes de un hueco de entrega (15 de julio) o con el final
# MAGIC del periodo queda con `fin_censurado = 1`: pudo seguir vigente.

# COMMAND ----------

titulo("2. CALENDARIO DE PROMOCIONES")
pl = lineas.where(F.col("tiene_promo") & F.col("promo_id").isNotNull())
primer_dia, ultimo_dia = min(con_datos), max(con_datos)
antes_de_hueco = sorted({d - dt.timedelta(days=1) for d in DIAS_SIN_DATOS} - set(DIAS_SIN_DATOS))
PROMO = (pl.groupBy("promo_id")
         .agg(F.min("fecha_venta").alias("inicio"), F.max("fecha_venta").alias("fin"),
              F.first("promo_nombre", ignorenulls=True).alias("nombre"), F.countDistinct("fecha_venta").alias("dias_con_venta"),
              F.countDistinct("tienda").alias("tiendas"), F.countDistinct("categoria_id").alias("categorias"),
              F.sum("venta").alias("venta"), F.avg("tasa_descuento").alias("tasa_media"))
         .withColumn("duracion", F.datediff("fin", "inicio") + 1)
         .withColumn("inicio_censurado", (F.col("inicio") <= F.lit(primer_dia)).cast("int"))
         .withColumn("fin_censurado", ((F.col("fin") >= F.lit(ultimo_dia)) |
                                       (F.col("fin").isin(antes_de_hueco) if antes_de_hueco else F.lit(False))).cast("int")))
guardar(PROMO, "silver", "promociones", "Una fila por promocion: inicio, fin, alcance, venta y censura (09_panel_diario)")
COMBOS = (pl.select("promo_id", "tienda", "categoria_id").distinct()
          .join(spark.table(tabla("silver", "promociones")).select("promo_id", "inicio", "fin", "fin_censurado"), "promo_id"))
guardar(COMBOS, "silver", "calendario_promociones",
        "Promocion x tienda x categoria con la vigencia de la promocion (09_panel_diario)")
rp = (spark.table(tabla("silver", "promociones"))
      .agg(F.count("*").alias("promos"), F.percentile_approx("duracion", 0.5).alias("dur_med"),
           F.sum("fin_censurado").alias("censuradas"), F.sum("venta").alias("venta")).collect()[0])
hallazgo(f"Calendario de promociones: {rp['promos']:,} promociones, duración mediana {rp['dur_med']} días; "
         f"{rp['censuradas']:,} con fin censurado (terminan justo antes de un hueco de entrega o al final del periodo).")
# vigencia como eventos +1 (inicio) y −1 (día siguiente al fin) acumulados por serie; el inicio se acota a la ventana de la tienda
EVT = (spark.table(tabla("silver", "calendario_promociones")).join(VENTANA, "tienda")
       .select("tienda", "categoria_id", F.greatest("inicio", "primera_venta").alias("d_ini"), F.date_add("fin", 1).alias("d_fin")))
EVT = (EVT.select("tienda", "categoria_id", F.col("d_ini").alias("fecha"), F.lit(1).alias("delta"))
       .unionByName(EVT.select("tienda", "categoria_id", F.col("d_fin").alias("fecha"), F.lit(-1).alias("delta")))
       .groupBy("tienda", "categoria_id", "fecha").agg(F.sum("delta").alias("delta_promos")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Relleno acotado a la ventana de cada tienda
# MAGIC **Qué hace.** Para cada serie genera un día por cada fecha entre la primera y la última venta de su tienda, une la venta
# MAGIC del día y decide: cero comercial (la tienda vendió ese día) o tienda inactiva (no vendió nada).

# COMMAND ----------

titulo("3. RELLENO ACOTADO")
CAL = (SERIES.join(VENTANA, "tienda")
       .select("tienda", "categoria_id", "dias_con_venta", "primera_venta",
               F.explode(F.sequence("primera_venta", "ultima_venta")).alias("fecha")))
w_serie = Window.partitionBy("tienda", "categoria_id").orderBy("fecha").rowsBetween(Window.unboundedPreceding, 0)
PANEL = (CAL.join(AGG, ["tienda", "categoria_id", "fecha"], "left")
         .join(ACTIVA, ["tienda", "fecha"], "left")
         .join(EVT, ["tienda", "categoria_id", "fecha"], "left")
         .withColumn("promos_vigentes", F.sum(F.coalesce(F.col("delta_promos"), F.lit(0))).over(w_serie).cast("int"))
         .withColumn("sin_datos", F.col("fecha").isin(DIAS_SIN_DATOS).cast("int") if DIAS_SIN_DATOS else F.lit(0))
         .withColumn("dia_atipico", F.col("fecha").isin(DIAS_ATIPICOS).cast("int") if DIAS_ATIPICOS else F.lit(0))
         .withColumn("tienda_inactiva", (F.col("tickets_tienda").isNull() & (F.col("sin_datos") == 0)).cast("int"))
         .withColumn("venta", F.when(F.col("tickets_tienda").isNull(), F.lit(None).cast("double"))
                     .otherwise(F.coalesce(F.col("venta"), F.lit(0.0))))
         .fillna(0, subset=["unidades", "lineas", "tickets", "descuento_pesos", "marcas", "productos", "promos_en_venta"])
         .withColumn("venta_neta", F.when(F.col("venta").isNotNull(), F.coalesce(F.col("venta_neta"), F.lit(0.0))))
         .withColumn("venta_sin_descuento", F.when(F.col("venta").isNotNull(), F.coalesce(F.col("venta_sin_descuento"), F.lit(0.0))))
         .withColumn("tasa_descuento", F.when(F.col("venta").isNotNull(), F.coalesce(F.col("tasa_descuento"), F.lit(0.0))))
         .withColumn("intensidad_promo", F.when(F.col("venta").isNotNull(), F.coalesce(F.col("intensidad_promo"), F.lit(0.0))))
         .withColumn("en_warmup", (F.datediff("fecha", "primera_venta") < DIAS_WARMUP).cast("int"))
         .withColumn("serie_relevante", (F.col("dias_con_venta") >= MIN_DIAS_CON_VENTA).cast("int"))
         .withColumn("mes", F.date_format("fecha", "yyyy-MM"))
         .select("tienda", "categoria_id", "fecha", "mes", "venta", "venta_neta", "venta_sin_descuento", "descuento_pesos",
                 "tasa_descuento", "unidades", "lineas", "tickets", "intensidad_promo", "promos_en_venta", "promos_vigentes",
                 "marcas", "productos", F.coalesce(F.col("tickets_tienda"), F.lit(0)).alias("tickets_tienda"),
                 "tienda_inactiva", "sin_datos", "dia_atipico", "en_warmup", "serie_relevante", "dias_con_venta"))
if TIPO_CAT is not None:
    PANEL = PANEL.join(TIPO_CAT, "categoria_id", "left")
else:
    PANEL = PANEL.withColumn("tipo_categoria", F.lit(None).cast("string"))
guardar(PANEL, "silver", "panel_categoria_dia",
        "Panel tienda x categoria x dia con relleno acotado a la ventana de la tienda (09_panel_diario)", particion="mes")
P = spark.table(TABLA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Resumen del panel
# MAGIC **Qué mirar.** Proporción de ceros comerciales, celdas de tienda inactiva y la mediana de días con venta por serie.
# MAGIC Para marzo–mayo se compara con el panel del Avance 2 (filas, porcentaje de ceros y celdas inactivas),
# MAGIC sabiendo que aquí la ventana de cada tienda se mide con seis meses.

# COMMAND ----------

titulo("4. RESUMEN DEL PANEL")
def resumen(df, etiqueta):
    r = df.agg(F.count("*").alias("filas"), F.countDistinct("tienda", "categoria_id").alias("series"),
               F.sum("tienda_inactiva").alias("celdas_inactivas"), F.sum("sin_datos").alias("celdas_sin_datos"),
               F.sum("en_warmup").alias("filas_warmup"),
               F.sum((F.col("venta") == 0).cast("int")).alias("ceros"), F.sum("venta").alias("venta")).collect()[0].asDict()
    utiles = df.where((F.col("tienda_inactiva") == 0) & (F.col("sin_datos") == 0) & (F.col("en_warmup") == 0)
                      & (F.col("serie_relevante") == 1))
    u = utiles.agg(F.count("*").alias("filas_modelables"), F.avg((F.col("venta") == 0).cast("double")).alias("pct_ceros"),
                   F.countDistinct("tienda", "categoria_id").alias("series_relevantes")).collect()[0].asDict()
    dens = (utiles.groupBy("tienda", "categoria_id").agg(F.avg((F.col("venta") > 0).cast("double")).alias("r"))
            .agg(F.percentile_approx("r", 0.5).alias("mediana_dias_con_venta")).collect()[0][0])
    return {"periodo": etiqueta, **r, **u, "mediana_dias_con_venta": dens}

filas = [resumen(P, f"{MESES[0]} a {MESES[-1]}")]
p3 = P.where(F.col("mes").isin("2026-03", "2026-04", "2026-05"))
if p3.limit(1).count():
    filas.append(resumen(p3, "2026-03 a 2026-05"))
RES = pd.DataFrame(filas)
mostrar(RES, "Panel", formatos={"filas": "int", "series": "int", "celdas_inactivas": "int", "celdas_sin_datos": "int",
                                "filas_warmup": "int", "ceros": "int",
                                "venta": "dinero", "filas_modelables": "int", "pct_ceros": "pct2", "series_relevantes": "int",
                                "mediana_dias_con_venta": "pct2"})
guardar(RES.assign(fecha=pd.Timestamp.now()), "mlops", "panel_resumen", "Resumen del panel diario por periodo", modo="append")
r = RES.iloc[0]
hallazgo(f"Panel de seis meses: {int(r['filas']):,} filas, {int(r['series']):,} series; {int(r['filas_modelables']):,} filas "
         f"modelables (tienda activa, fuera de warm-up, serie relevante) con {r['pct_ceros']:.2%} de ceros comerciales; "
         f"{int(r['celdas_inactivas']):,} celdas de tienda inactiva y {int(r['celdas_sin_datos']):,} de días sin datos; "
         f"mediana de días con venta por serie {r['mediana_dias_con_venta']:.1%}.")
if len(RES) > 1:
    r3 = RES.iloc[1]
    hallazgo(f"Marzo–mayo: {int(r3['filas_modelables']):,} filas modelables y {r3['pct_ceros']:.2%} de ceros "
             "(compárese con el panel del Avance 2).")

por_mes = (P.groupBy("mes").agg(F.count("*").alias("filas"), F.sum("tienda_inactiva").alias("inactivas"),
                                 F.sum("sin_datos").alias("sin_datos"),
                                 F.avg(F.when(F.col("venta").isNotNull(), (F.col("venta") == 0).cast("double"))).alias("pct_ceros"),
                                 F.sum("venta").alias("venta"))
           .orderBy("mes").toPandas())
mostrar(por_mes, "Por mes", formatos={"filas": "int", "inactivas": "int", "sin_datos": "int", "pct_ceros": "pct2", "venta": "dinero"})
por_tipo = (P.where(F.col("venta").isNotNull()).groupBy("tipo_categoria")
            .agg(F.countDistinct("tienda", "categoria_id").alias("series"), F.sum("venta").alias("venta"),
                 F.avg((F.col("venta") == 0).cast("double")).alias("pct_ceros"))
            .orderBy(F.desc("venta")).toPandas())
por_tipo["pct_venta"] = por_tipo["venta"] / por_tipo["venta"].sum()
mostrar(por_tipo, "Por tipo de categoría", formatos={"series": "int", "venta": "dinero", "pct_ceros": "pct2", "pct_venta": "pct"})
pv = (P.where(F.col("venta").isNotNull())
      .agg(F.avg((F.col("promos_vigentes") > 0).cast("double")).alias("con_promo"),
           F.sum((F.col("promos_vigentes") < 0).cast("int")).alias("negativos"),
           F.avg(F.when(F.col("promos_vigentes") > 0, (F.col("venta") > 0).cast("double"))).alias("venta_con"),
           F.avg(F.when(F.col("promos_vigentes") == 0, (F.col("venta") > 0).cast("double"))).alias("venta_sin")).collect()[0])
hallazgo(f"{pv['con_promo']:.1%} de las celdas con dato tienen al menos una promoción vigente; con promoción la serie vende "
         f"{pv['venta_con'] or 0:.1%} de los días y sin promoción {pv['venta_sin'] or 0:.1%} (asociación, no efecto causal).")
if pv["negativos"]:
    aviso(f"{pv['negativos']:,} celdas con promos_vigentes negativo: revisar el calendario de promociones.")

# cuadre: la venta del panel es la venta de Plata dentro del periodo
v_plata = lineas.agg(F.sum("venta")).collect()[0][0] or 0.0
v_panel = P.agg(F.sum("venta")).collect()[0][0] or 0.0
print(f"Venta en Plata (periodo): ${v_plata:,.2f} · en el panel: ${v_panel:,.2f} · diferencia: ${v_panel - v_plata:,.2f}")
if abs(v_panel - v_plata) > max(1.0, 1e-6 * v_plata):
    aviso("La venta del panel no cuadra con la de Plata (líneas de tiendas sin ventana o fuera del periodo).")
else:
    hallazgo("La venta del panel cuadra con la de Plata al centavo.")
try:
    spark.sql(f"OPTIMIZE {TABLA} ZORDER BY (tienda, categoria_id)")
except Exception:
    pass
liberar(ACTIVA)

# COMMAND ----------

cerrar("09_panel_diario", siguiente="10_eda_series_y_linea_base")
