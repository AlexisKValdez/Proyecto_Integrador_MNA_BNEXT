# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · EDA de series y línea base ingenua sobre el panel limpio
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Preguntas que responde**
# MAGIC 1. Con seis meses de historia, ¿a qué grano conviene pronosticar: categoría, marca (todas o las 300 líderes) o producto?
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

# ─────────────────────── columnas del proyecto (iguales en todos los notebooks) ───────────────────────
# Nombre lógico → nombres posibles en los parquet de Bnext. Se usa el primero que exista en la tabla.
# Si Bnext cambia un nombre, basta con agregarlo aquí (en todos los notebooks).
CANDIDATAS = {
    "id_linea": ["order_line_id"], "id_ticket": ["order_id"],
    "tienda": ["store_code", "store_id"], "tipo_tienda": ["store_type"], "zona": ["zone_id", "zone"], "region": ["region"],
    "categoria": ["category_id"], "categoria_nombre": ["category"],
    "marca": ["brand_id"], "marca_nombre": ["brand"],
    "producto": ["product_code", "product_id"], "producto_nombre": ["product_name"],
    "cantidad": ["order_line_qty_ordered"], "precio_unitario": ["order_line_unit_price"],
    "costo_unitario": ["order_line_unit_cost"],
    "total_con_descuento": ["order_line_total_with_discount"],
    "total_sin_descuento": ["order_line_total_without_discount"],
    "pct_descuento": ["order_line_discount_percent"], "descuento_sumasix": ["order_line_sumasix_discount"],
    "pago_puntos": ["pago_con_puntos"], "promo_id": ["order_line_promo_id"], "promo_nombre": ["order_line_promo_name"],
    "fecha_ticket": ["finish_order"], "fecha_orden": ["date_orders", "date_ordered"], "fecha_sync": ["sync_date"],
    "estatus_linea": ["order_line_status_id"], "estatus_orden": ["order_status_id"],
    "impuestos": ["order_line_total_taxes"], "subtotal": ["order_line_subtotal"],
    "estatus_tienda": ["store_status_id"], "jefatura": ["jefatura_six"],
}
TRAZA = ["_archivo", "_fecha_archivo", "_fecha_carga", "anio_mes"]     # columnas que agrega 02_carga_bronce


def _sanear(nombre):
    import unicodedata
    t = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", t.strip().lower()).strip("_")) or "columna"
    return f"c_{t}" if t[0].isdigit() else t


def resolver_columnas(columnas):
    """{nombre_lógico: columna real o None}"""
    disponibles = {_sanear(c): c for c in columnas}
    return {k: next((disponibles[_sanear(o)] for o in v if _sanear(o) in disponibles), None)
            for k, v in CANDIDATAS.items()}


def requiere(C, *logicos):
    """True si existen todas las columnas lógicas; si falta alguna lo avisa y devuelve False."""
    faltan = [l for l in logicos if not C.get(l)]
    if faltan:
        aviso(f"faltan columnas {faltan}; se omite esta parte (agrega su nombre real en CANDIDATAS)")
        return False
    return True


def num(columna):
    """Columna como número; un valor mal capturado se vuelve nulo en lugar de romper la consulta."""
    return F.expr(f"try_cast(`{columna}` as double)")


FORMATOS_FECHA = ["yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd HH:mm:ss.SSS", "yyyy-MM-dd\\'T\\'HH:mm:ss",
                  "dd/MM/yyyy HH:mm:ss", "dd/MM/yyyy HH:mm", "dd/MM/yyyy", "yyyy-MM-dd", "yyyyMMdd"]


FECHA_MINIMA, FECHA_MAXIMA = "2000-01-01", "2100-01-01"   # fuera de este rango la fecha es un valor centinela


def a_timestamp(columna, tipo_spark, acotar=True):
    """Convierte una columna de fecha (texto, número o timestamp) en timestamp sin fallar.
    Con acotar=True, las fechas fuera de 2000–2099 (centinelas como 0001-01-01 o años de cinco cifras, que además no caben
    en una fecha de Python) quedan nulas."""
    t = tipo_spark.lower()
    c = f"`{columna}`"
    if t.startswith("timestamp"):
        ts = F.col(columna)
    elif t == "date":
        ts = F.col(columna).cast("timestamp")
    elif t in ("bigint", "int", "integer", "smallint", "tinyint", "double", "float") or t.startswith("decimal"):
        ts = F.expr(f"CASE WHEN abs({c}) < 1e13 THEN timestamp_seconds(CASE WHEN {c} > 1e11 THEN {c} / 1000 ELSE {c} END) END")
    else:
        intentos = [f"try_to_timestamp(cast({c} as string))"] + \
                   [f"try_to_timestamp(cast({c} as string), '{fmt}')" for fmt in FORMATOS_FECHA]
        ts = F.expr(f"coalesce({', '.join(intentos)})")
    if not acotar:
        return ts
    return F.when((ts >= F.lit(FECHA_MINIMA).cast("timestamp")) & (ts < F.lit(FECHA_MAXIMA).cast("timestamp")), ts)


RANGO_POS = {"costo_max": 5_000, "cantidad_max": 1_000, "venta_min": 0, "venta_max": 15_000}   # reglas de rango POS de Bnext


def fuera_de_rango_pos(C):
    """True si la línea viola alguna regla de rango POS de Bnext (costo, cantidad o importe imposibles): son errores de
    captura, como una línea de mil millones de pesos, que deforman cualquier suma de venta."""
    fuera = F.lit(False)
    if C.get("costo_unitario"):
        fuera = fuera | F.coalesce(num(C["costo_unitario"]) > RANGO_POS["costo_max"], F.lit(False))
    if C.get("cantidad"):
        fuera = fuera | F.coalesce(num(C["cantidad"]) > RANGO_POS["cantidad_max"], F.lit(False))
    if C.get("total_con_descuento"):
        v = num(C["total_con_descuento"])
        fuera = fuera | F.coalesce((v < RANGO_POS["venta_min"]) | (v > RANGO_POS["venta_max"]), F.lit(False))
    return fuera


def fecha_venta(df):
    """Fecha (DATE) de la venta: COLUMNA_FECHA_VENTA convertida sin fallar y, si HORAS_EN_UTC, pasada a hora de México."""
    tipos = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    if COLUMNA_FECHA_VENTA not in tipos:
        raise KeyError(f"No existe la columna de fecha {COLUMNA_FECHA_VENTA!r}; revisa COLUMNA_FECHA_VENTA en la configuración")
    ts = a_timestamp(COLUMNA_FECHA_VENTA, tipos[COLUMNA_FECHA_VENTA])
    if HORAS_EN_UTC:
        ts = F.from_utc_timestamp(ts, "America/Mexico_City")
    return F.to_date(ts)


def expr_promo(C, tipos):
    """(identificador de promoción como texto, ¿la línea tiene promoción?). Bnext manda el identificador como número
    (double con nulos); un nulo, NaN o 0 significa sin promoción."""
    if not C.get("promo_id"):
        return F.lit(None).cast("string"), F.lit(False)
    c = C["promo_id"]
    t = tipos[c]
    if t in ("double", "float") or t.startswith("decimal") or t in ("bigint", "int", "smallint", "tinyint"):
        valor = F.expr(f"try_cast(`{c}` as double)")
        tiene = valor.isNotNull() & ~F.isnan(valor) & (valor != 0)
        return F.when(tiene, F.expr(f"try_cast(try_cast(`{c}` as decimal(38,0)) as string)")), tiene
    texto = F.trim(F.col(c).cast("string"))
    tiene = texto.isNotNull() & (texto != "") & ~texto.isin("0", "0.0", "null", "None", "NaN", "nan")
    return F.when(tiene, texto), tiene

# COMMAND ----------

# ─────────────────────── estadísticas del EDA (concentración, intermitencia, segmentos) ───────────────────────
def _arr(x):
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


# ----------------------------------------------------------------------------- concentración

def gini(valores):
    """Índice de Gini de valores no negativos: 0 = todos iguales, cercano a 1 = uno concentra todo."""
    x = np.sort(_arr(valores))
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    n = x.size
    i = np.arange(1, n + 1)
    return float((2 * np.sum(i * x)) / (n * x.sum()) - (n + 1) / n)


def curva_lorenz(valores, puntos=101):
    """(fracción acumulada de entidades, fracción acumulada del valor), de menor a mayor."""
    x = np.sort(_arr(valores))
    if x.size == 0 or x.sum() == 0:
        return np.linspace(0, 1, puntos), np.linspace(0, 1, puntos)
    acumulado = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    fraccion = np.linspace(0, 1, acumulado.size)
    malla = np.linspace(0, 1, puntos)
    return malla, np.interp(malla, fraccion, acumulado)


def participacion_top(valores, fracciones=(0.01, 0.05, 0.10, 0.20, 0.50)):
    """Parte del total que concentra el ``f`` % de entidades más grandes, para cada ``f``."""
    x = np.sort(_arr(valores))[::-1]
    total = x.sum()
    salida = {}
    for f in fracciones:
        k = max(1, int(np.ceil(f * x.size))) if x.size else 0
        salida[f] = float(x[:k].sum() / total) if total else float("nan")
    return salida


def fraccion_para(valores, objetivo=0.80):
    """Fracción de entidades (de mayor a menor) necesaria para acumular ``objetivo`` del total."""
    x = np.sort(_arr(valores))[::-1]
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    acumulado = np.cumsum(x) / x.sum()
    k = int(np.searchsorted(acumulado, objetivo - 1e-12) + 1)
    return float(min(k, x.size) / x.size)


# ----------------------------------------------------------------------------- intermitencia y segmentos

ADI_CORTE = 1.32
CV2_CORTE = 0.49
CLASES_SB = ("suave", "errática", "intermitente", "grumosa")


def clasificar_sb(adi, cv2, adi_corte=ADI_CORTE, cv2_corte=CV2_CORTE):
    """Clasificación de Syntetos y Boylan (2005) de una serie de demanda.

    * ``adi``: intervalo medio entre días con venta (días activos / días con venta).
    * ``cv2``: coeficiente de variación al cuadrado del tamaño de la venta en los días con venta.

    suave (ADI < 1.32, CV² < 0.49) · errática (ADI < 1.32, CV² ≥ 0.49) ·
    intermitente (ADI ≥ 1.32, CV² < 0.49) · grumosa (ADI ≥ 1.32, CV² ≥ 0.49).
    """
    adi = np.asarray(adi, dtype=float)
    cv2 = np.asarray(cv2, dtype=float)
    clase = np.where(adi < adi_corte,
                     np.where(cv2 < cv2_corte, CLASES_SB[0], CLASES_SB[1]),
                     np.where(cv2 < cv2_corte, CLASES_SB[2], CLASES_SB[3]))
    return np.where(np.isnan(adi) | np.isnan(cv2), "sin_datos", clase)


def clase_abc(valores, cortes=(0.80, 0.95)):
    """A = el conjunto mínimo de entidades (de mayor a menor) que alcanza el 80 % del valor;
    B = las siguientes hasta alcanzar el 95 %; C = el resto."""
    x = np.asarray(valores, dtype=float)
    orden = np.argsort(-x, kind="stable")
    acumulado = np.cumsum(x[orden]) / x.sum() if x.sum() else np.zeros_like(x)
    anterior = np.concatenate([[0.0], acumulado[:-1]])
    etiquetas = np.where(anterior < cortes[0], "A", np.where(anterior < cortes[1], "B", "C"))
    salida = np.empty(x.size, dtype=object)
    salida[orden] = etiquetas
    return salida.astype(str)


def clase_xyz(cv, cortes=(0.5, 1.0)):
    """X = estable (CV < 0.5), Y = variable (0.5–1.0), Z = muy variable (> 1.0); CV de la venta diaria."""
    cv = np.asarray(cv, dtype=float)
    return np.where(np.isnan(cv), "sin_datos", np.where(cv < cortes[0], "X", np.where(cv <= cortes[1], "Y", "Z")))


# ----------------------------------------------------------------------------- series

def autocorrelacion(x, rezagos=(1, 7, 14, 28)):
    """Autocorrelación muestral de una serie en los rezagos pedidos (NaN si la serie es constante)."""
    a = np.asarray(x, dtype=float)
    a = a - np.nanmean(a)
    den = np.nansum(a * a)
    salida = {}
    for k in rezagos:
        if den == 0 or k >= a.size:
            salida[k] = float("nan")
        else:
            salida[k] = float(np.nansum(a[k:] * a[:-k]) / den)
    return salida


def indice_relativo(valores, grupos):
    """Media por grupo dividida entre la media de las medias de grupo (1.0 = día promedio)."""
    v = np.asarray(valores, dtype=float)
    g = np.asarray(grupos)
    medias = {k: float(np.nanmean(v[g == k])) for k in sorted(set(g.tolist()))}
    base = np.nanmean(list(medias.values()))
    return {k: (m / base if base else float("nan")) for k, m in medias.items()}

# COMMAND ----------

# ─────────────────────── gráficas (un solo estilo para todo el proyecto) ───────────────────────
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

AZUL, GRIS, NARANJA, TINTA, SUAVE = "#2a78d6", "#8a8f98", "#d9822b", "#1f2328", "#5b616b"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c9ccd1",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e8eb",
                     "grid.linewidth": 0.6, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
                     "axes.labelcolor": SUAVE, "xtick.color": SUAVE, "ytick.color": SUAVE, "font.size": 9,
                     "axes.titlelocation": "left", "legend.frameon": False})


def grafica_serie(fechas, valores, titulo, unidad, eventos=None, color=AZUL, alto=2.8):
    """Serie diaria en una línea delgada, un solo eje; las fechas de eventos se marcan con líneas punteadas."""
    fig, ax = plt.subplots(figsize=(11, alto))
    ax.plot(pd.to_datetime(fechas), valores, color=color, linewidth=1.6)
    lo, hi = pd.to_datetime(fechas).min(), pd.to_datetime(fechas).max()
    for e in (eventos or []):
        f = pd.Timestamp(e["fecha"])
        if lo <= f <= hi:
            ax.axvline(f, color=NARANJA if e.get("tipo") == "futbol" else GRIS, linewidth=0.8, linestyle=":")
    ax.set_title(titulo)
    ax.set_ylabel(unidad)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.set_xlim(lo, hi)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def grafica_barras(etiquetas, valores, titulo, unidad, resaltar=None, formato="{:.2f}", referencia=None, horizontal=False):
    """Barras de una sola serie; la barra resaltada va en azul y las demás en gris; valor escrito en cada barra."""
    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.28 * len(etiquetas)) if horizontal else 3))
    colores = [AZUL if (resaltar is None or e in resaltar) else "#b8c4d6" for e in etiquetas]
    if horizontal:
        ax.barh(list(etiquetas)[::-1], list(valores)[::-1], color=colores[::-1], height=0.7)
        for i, v in enumerate(list(valores)[::-1]):
            ax.text(v, i, " " + formato.format(v), va="center", color=TINTA, fontsize=8)
        ax.set_xlabel(unidad)
    else:
        ax.bar(list(etiquetas), list(valores), color=colores, width=0.7)
        for i, v in enumerate(valores):
            ax.text(i, v, formato.format(v), ha="center", va="bottom", color=TINTA, fontsize=8)
        ax.set_ylabel(unidad)
    if referencia is not None:
        (ax.axvline if horizontal else ax.axhline)(referencia, color=GRIS, linewidth=0.8, linestyle="--")
    ax.set_title(titulo)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

# COMMAND ----------

# ─────────────────────── calendario de eventos del periodo (agrega aquí los que falten) ───────────────────────
EVENTOS = [
    {"fecha": "2026-03-16", "nombre": "Natalicio de Benito Juárez (descanso)", "tipo": "festivo"},
    {"fecha": "2026-04-02", "nombre": "Jueves Santo", "tipo": "comercial"},
    {"fecha": "2026-04-03", "nombre": "Viernes Santo", "tipo": "comercial"},
    {"fecha": "2026-05-01", "nombre": "Día del Trabajo", "tipo": "festivo"},
    {"fecha": "2026-05-10", "nombre": "Día de las Madres", "tipo": "comercial"},
    {"fecha": "2026-06-11", "nombre": "Mundial: México vs Sudáfrica (inauguración)", "tipo": "futbol"},
    {"fecha": "2026-06-18", "nombre": "Mundial: México vs Corea del Sur", "tipo": "futbol"},
    {"fecha": "2026-06-21", "nombre": "Día del Padre", "tipo": "comercial"},
    {"fecha": "2026-06-24", "nombre": "Mundial: México vs Chequia", "tipo": "futbol"},
    {"fecha": "2026-09-15", "nombre": "Noche mexicana (víspera de la Independencia)", "tipo": "comercial"},
    {"fecha": "2026-09-16", "nombre": "Día de la Independencia", "tipo": "festivo"},
]
PERIODOS = [{"inicio": "2026-06-11", "fin": "2026-07-19", "nombre": "Copa Mundial 2026"}]

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
# MAGIC (viable), producto 11 % (no), 100 productos 41.5 % (piloto). Si con seis meses cambia, se documenta la decisión.

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
guardar(GRANO, "mlops", "eda_grano", "Densidad de las series por grano con seis meses (EDA 10)")
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
# Cortes fijos de la configuración (los del Avance 4) más los bloques de prueba de septiembre cuando ya hay datos.
# Cada bloque de 14 días y los 14 días previos a su origen deben tener datos completos (así ningún corte cae en un
# hueco de entrega como julio 16–31). Los resúmenes de la línea base promedian sólo los cortes de desarrollo.
dias_con_datos = {r[0] for r in P.where(F.col("sin_datos") == 0).select("fecha").distinct().collect()}
CORTES = cortes_walk_forward(dias_con_datos)
CORTES_DESARROLLO = [c["corte"] for c in CORTES if c["fase"] == "desarrollo"]
mostrar(pd.DataFrame(CORTES)[["corte", "fase", "origen", "desde", "hasta"]],
        "Cortes walk-forward (desarrollo = Avance 4; prueba = septiembre, datos no vistos)")
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
    hallazgo(f"Bloques de prueba (septiembre, datos no vistos): ingenuo estacional WMAPE {prueba_lb['wmape_lag7'].mean():.4f} "
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
