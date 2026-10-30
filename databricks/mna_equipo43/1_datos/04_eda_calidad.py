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
# MAGIC | **Decide** | Las reglas de limpieza del notebook `07_limpieza_plata` |
# MAGIC | **Tiempo** | 15–30 min |

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
