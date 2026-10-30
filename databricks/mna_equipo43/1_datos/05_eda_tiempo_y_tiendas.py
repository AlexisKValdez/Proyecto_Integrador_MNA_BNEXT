# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · EDA del tiempo y de las tiendas
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Preguntas que responde**
# MAGIC 1. ¿Qué columna define el día de la venta y en qué zona horaria vienen las horas?
# MAGIC 2. ¿Hay días faltantes o con carga incompleta en los seis meses? ¿Cómo cambia la venta por día de la semana y del mes?
# MAGIC 3. ¿Cuántas tiendas operan todo el periodo, cuántas abren o cierran a la mitad y cuántas tienen huecos?
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `bronze.ventas_lineas` (sin limpiar; la venta excluye las líneas fuera de rango POS y los duplicados suman alrededor de 1 % de más, sin cambiar los patrones) |
# MAGIC | **Escribe** | `mlops.eda_cobertura_diaria` · `mlops.eda_tienda_dia` · `mlops.eda_tiendas_actividad` |
# MAGIC | **Decide** | `COLUMNA_FECHA_VENTA` y `HORAS_EN_UTC` · días a excluir · relleno de ceros sólo dentro de la ventana de cada tienda |
# MAGIC | **Tiempo** | 10–20 min |

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
