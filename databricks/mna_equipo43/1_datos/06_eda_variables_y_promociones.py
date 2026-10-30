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
