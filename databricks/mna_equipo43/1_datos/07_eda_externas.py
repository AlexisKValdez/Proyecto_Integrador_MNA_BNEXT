# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · EDA de fuentes externas y archivos heredados
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Preguntas que responde**
# MAGIC 1. ¿Qué archivos hay en `processed` y `outputs` (catálogo de tiendas, códigos postales, INEGI, clima, calendarios
# MAGIC    deportivos, ABT y modelo heredados)?
# MAGIC 2. ¿Cuántas filas y columnas tiene cada uno y qué periodo cubre?
# MAGIC 3. ¿Cuántas tiendas con venta están en el catálogo de tiendas?
# MAGIC 4. ¿Qué fuentes no llegan hasta agosto y hay que actualizar?
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `/Volumes/modeladoprescriptivoventas/maestria/processed` y `/outputs` · `bronze.ventas_lineas` |
# MAGIC | **Escribe** | `mlops.eda_externas_archivos` |
# MAGIC | **Tiempo** | 5–10 min |
# MAGIC | **Privacidad** | Las columnas de ubicación (latitud, longitud, dirección) sólo se cuentan, nunca se imprimen |

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

# ----------------------------------------------------------------- parámetros de este notebook
import csv
import io

MAX_MB_PANDAS = 400          # archivos de texto más grandes no se perfilan aquí
FILAS_PERFIL = 200_000       # filas que se leen de cada archivo de texto para perfilarlo
INICIO = pd.Timestamp(min(MESES) + "-01")
FIN = pd.Timestamp(max(MESES) + "-01") + pd.offsets.MonthEnd(0)


def ruta_local(ruta):
    for prefijo in ("dbfs:", "file:"):
        if ruta.startswith(prefijo):
            return ruta[len(prefijo):]
    return ruta

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Inventario de `processed` y `outputs`

# COMMAND ----------

titulo("1. ARCHIVOS EN processed Y outputs")
def listar(ruta, prof=0):
    salida = []
    try:
        for f in dbutils.fs.ls(ruta):
            if f.name.endswith("/"):
                if prof < 4:
                    salida += listar(f.path, prof + 1)
            elif not f.name.startswith(("_", ".")):
                salida.append({"volumen": ruta.rstrip("/").split("/")[-1] if prof == 0 else "", "ruta": f.path,
                               "archivo": f.name, "mb": f.size / 1024**2,
                               "tipo": f.name.rsplit(".", 1)[-1].lower() if "." in f.name else "(sin extensión)"})
    except Exception as e:
        aviso(f"No se pudo listar {ruta}: {type(e).__name__}")
    return salida

INV = []
for vol in ("processed", "outputs"):
    for a in listar(volumen(vol)):
        a["volumen"] = vol
        INV.append(a)
INV = pd.DataFrame(INV)
if len(INV):
    mostrar(INV[["volumen", "archivo", "tipo", "mb"]].sort_values(["volumen", "archivo"]), "Archivos",
              formatos={"mb": "dec2"}, max_filas=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Perfil de cada archivo tabular
# MAGIC **Qué hace.** Lee cada CSV, TXT o parquet (los de texto detectando separador, codificación y renglones de
# MAGIC encabezado) y reporta filas, columnas, % de nulos y, si tiene columnas de fecha, su rango.
# MAGIC
# MAGIC **Qué mirar.** El rango de fechas del clima y de los calendarios deportivos: si terminan en mayo, hay que
# MAGIC descargar junio a agosto (y agregar el Mundial) antes de construir las variables externas.

# COMMAND ----------

titulo("2. PERFIL DE LOS ARCHIVOS TABULARES")

def leer_texto(ruta: str, mb: float) -> pd.DataFrame:
    """Lee un CSV/TXT detectando codificación, separador y renglones previos al encabezado."""
    with open(ruta_local(ruta), "rb") as f:
        crudo = f.read(256 * 1024)
    codificacion = "utf-8"
    try:
        crudo.decode("utf-8")
    except UnicodeDecodeError:
        codificacion = "latin-1"
    texto = crudo.decode(codificacion, errors="replace")
    lineas = texto.splitlines()[:30]
    mejor = (",", 0, 0)
    for sep in [",", "|", ";", "\t"]:
        for salto in range(0, 3):
            conteos = [len(next(csv.reader([l], delimiter=sep))) for l in lineas[salto:salto + 10] if l.strip()]   # respeta comillas
            if conteos and min(conteos) == max(conteos) and conteos[0] > mejor[1]:
                mejor = (sep, conteos[0], salto)
    sep, _, salto = mejor
    return pd.read_csv(ruta_local(ruta), sep=sep, skiprows=salto, encoding=codificacion, nrows=FILAS_PERFIL,
                       low_memory=False, on_bad_lines="skip", dtype=str)

def rango_fechas(d: pd.DataFrame) -> tuple:
    for c in d.columns:
        if re.search(r"fecha|date|time", str(c), re.I) and not re.search(r"dia_|day_|_dia$|_day$", str(c), re.I):
            f = pd.to_datetime(d[c], errors="coerce", dayfirst=False)
            if f.notna().mean() > 0.8:
                return c, f.min(), f.max()
    return None, None, None

PERF, TABLAS = [], {}
for a in INV.to_dict("records") if len(INV) else []:
    fila = {"volumen": a["volumen"], "archivo": a["archivo"], "tipo": a["tipo"], "mb": a["mb"]}
    try:
        if re.search(r"lgbm|champion|(^|_)model(\.|_|$)", a["archivo"], re.I) and a["tipo"] != "json":
            fila["nota"] = "artefacto de modelo (no se perfila)"
        elif a["tipo"] in ("csv", "txt") and a["mb"] <= MAX_MB_PANDAS:
            d = leer_texto(a["ruta"], a["mb"])
            TABLAS[a["archivo"]] = d
            fila.update(filas_leidas=len(d), columnas=d.shape[1])
        elif a["tipo"] == "parquet":
            import pyarrow.parquet as pq
            meta = pq.read_metadata(ruta_local(a["ruta"]))
            fila.update(filas_leidas=meta.num_rows, columnas=meta.num_columns)
            nombres = [c.name for c in meta.schema.to_arrow_schema()]
            col_f = next((c for c in nombres if re.search(r"^fecha$|^date$|fecha", c, re.I)), None)
            if col_f:
                import pyarrow.compute as pc
                columna = pq.read_table(ruta_local(a["ruta"]), columns=[col_f]).column(col_f)
                mm = pc.min_max(columna).as_py()
                fila.update(columna_fecha=col_f, desde=str(mm["min"])[:10], hasta=str(mm["max"])[:10])
            TABLAS[a["archivo"]] = None
            fila["columnas_lista"] = ", ".join(nombres[:40]) + (" …" if len(nombres) > 40 else "")
        elif a["tipo"] == "json" and a["mb"] < 20:
            with open(ruta_local(a["ruta"]), encoding="utf-8") as f:
                contenido = json.load(f)
            fila["contenido"] = ", ".join(list(contenido)[:20]) if isinstance(contenido, dict) else type(contenido).__name__
            TABLAS[a["archivo"]] = contenido
        else:
            fila["nota"] = "no se perfila (formato o tamaño)"
        d = TABLAS.get(a["archivo"])
        if isinstance(d, pd.DataFrame):
            c, mn, mx = rango_fechas(d)
            if c:
                fila.update(columna_fecha=c, desde=str(mn)[:10], hasta=str(mx)[:10])
            fila["pct_nulos_medio"] = float(d.isna().mean().mean())
            fila["columnas_lista"] = ", ".join(map(str, d.columns[:40])) + (" …" if d.shape[1] > 40 else "")
    except Exception as e:
        fila["nota"] = f"error al leer: {type(e).__name__}: {str(e)[:120]}"
    PERF.append(fila)
PERF = pd.DataFrame(PERF)
if len(PERF):
    guardar(PERF, "mlops", "eda_externas_archivos", "Perfil de los archivos externos y heredados (EDA 07)")
    cols = [c for c in ["volumen", "archivo", "filas_leidas", "columnas", "columna_fecha", "desde", "hasta", "pct_nulos_medio", "nota"]
            if c in PERF]
    mostrar(PERF[cols], "Perfil por archivo", formatos={"filas_leidas": "int", "columnas": "int", "pct_nulos_medio": "pct"},
              max_filas=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Catálogo de tiendas: cobertura y atributos
# MAGIC **Qué hace.** Busca el archivo de tiendas, detecta qué columna es el código de tienda comparando sus valores con los
# MAGIC `store_code` de Bronce, y mide cuántas tiendas con venta están en el catálogo y qué atributos tienen completos.
# MAGIC
# MAGIC **Qué mirar.** En el trabajo previo 8.37 % de las tiendas quedó sin geografía (cruce exacto por código postal). Si
# MAGIC el catálogo trae coordenadas, el cruce por coordenadas recupera casi todas.

# COMMAND ----------

titulo("3. CATÁLOGO DE TIENDAS")
cand = [k for k, v in TABLAS.items() if isinstance(v, pd.DataFrame) and re.search(r"store|tienda", k, re.I)]
df_b = leer_tabla(tabla("bronze", "ventas_lineas"), "02_carga_bronce")
C = resolver_columnas(df_b.columns)
if cand and C["tienda"]:
    cat = TABLAS[cand[0]]
    tiendas_venta = set(df_b.select(F.col(C["tienda"]).cast("string")).distinct().toPandas().iloc[:, 0].dropna()
                        .str.strip().str.lstrip("0"))
    mejor, cobertura = None, 0.0
    for c in cat.columns:
        valores = set(cat[c].dropna().astype(str).str.strip().str.lstrip("0"))
        cob = len(tiendas_venta & valores) / len(tiendas_venta) if tiendas_venta else 0
        if cob > cobertura:
            mejor, cobertura = c, cob
    print(f"Archivo: {cand[0]} · {len(cat):,} filas · columna llave detectada: {mejor}")
    hallazgo(f"{cobertura:.1%} de las {len(tiendas_venta):,} tiendas con venta están en el catálogo {cand[0]} (llave {mejor}).")
    atributos = pd.DataFrame({"columna": cat.columns, "pct_completo": cat.notna().mean().values,
                              "distintos": [cat[c].nunique() for c in cat.columns]})
    atributos["sensible"] = atributos["columna"].astype(str).str.contains(SENSIBLE)
    mostrar(atributos, "Atributos del catálogo de tiendas (las columnas sensibles sólo se cuentan)",
              formatos={"pct_completo": "pct", "distintos": "int"}, max_filas=60)
else:
    aviso("No se encontró un catálogo de tiendas (archivo con 'store' o 'tienda' en el nombre) en processed/outputs. "
          "El 08 arma uno desde Bronce con los atributos que trae cada línea (tipo de tienda, zona, región, jefatura).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Artefactos del modelo heredado
# MAGIC **Qué hace.** Muestra los parámetros guardados (`best_params_busqueda.json`, `inference_metadata_v1.json`) y el tamaño
# MAGIC del modelo, para poder reproducir la línea base heredada sobre el panel nuevo.

# COMMAND ----------

titulo("4. ARTEFACTOS DEL MODELO HEREDADO")
for nombre, contenido in TABLAS.items():
    if isinstance(contenido, dict):
        print(f"{nombre}:")
        texto = json.dumps(json_seguro(contenido), ensure_ascii=False, indent=1, default=str)
        print(texto[:2500] + (" …" if len(texto) > 2500 else ""))
modelos = INV[INV["tipo"].isin(["txt", "pkl", "joblib", "model", "json"]) & INV["archivo"].str.contains(r"lgbm|champion|(?:^|_)model(?:\.|_|$)", case=False, regex=True)] \
    if len(INV) else pd.DataFrame()
if len(modelos):
    mostrar(modelos[["archivo", "mb"]], "Archivos de modelo", formatos={"mb": "dec2"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ¿Qué falta para las variables externas?
# MAGIC **Qué mirar.** Cada fuente externa debe cubrir hasta el último día de ventas (y los 14 días que se pronostican).
# MAGIC Lo que no llegue hasta agosto hay que descargarlo y cargarlo a `processed/` antes de construir variables.

# COMMAND ----------

titulo("5. COBERTURA CONTRA EL PERIODO DE VENTAS")
if len(PERF) and "hasta" in PERF:
    heredados = (r"dataset_modelado|last_state|lgbm|inference|best_params|features_categoria|avance|ventas_consolidado|"
                 r"preds_|anomalias|auditoria_columnas")
    cob = PERF.dropna(subset=["hasta"])[["volumen", "archivo", "columna_fecha", "desde", "hasta"]].copy()
    cob = cob[~cob["archivo"].str.contains(heredados, case=False, regex=True)]          # sólo fuentes externas
    cob["cubre_hasta_fin"] = pd.to_datetime(cob["hasta"], errors="coerce") >= FIN
    cob["dias_que_faltan"] = (FIN - pd.to_datetime(cob["hasta"], errors="coerce")).dt.days.clip(lower=0)
    faltan = cob[~cob["cubre_hasta_fin"]]
    if not len(cob):
        aviso("En Databricks no hay fuentes externas con fecha (clima, INEGI, códigos postales, calendarios): sólo están "
              "las ventas y los archivos del equipo anterior. Para usar variables externas hay que cargarlas a processed/ "
              "o generarlas en un notebook (el calendario de festivos y del Mundial ya viene en la configuración).")
    else:
        mostrar(cob, f"Fuentes con fecha contra el fin del periodo de ventas ({FIN.date()})", formatos={"dias_que_faltan": "int"})
    if len(faltan):
        aviso(f"{len(faltan)} fuentes no llegan al {FIN.date()}: {', '.join(faltan['archivo'])}. "
              "Hay que actualizarlas (clima, calendarios deportivos, Mundial) antes de construir variables externas.")
    else:
        hallazgo("Todas las fuentes con fecha cubren el periodo de ventas completo.")

# COMMAND ----------

cerrar("07_eda_externas", siguiente="08_limpieza_plata")
