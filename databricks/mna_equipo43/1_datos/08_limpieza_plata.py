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
# MAGIC 6. **Venta nula**: la línea no trae importe con descuento (unas 600 en seis meses). El Avance 2 también las quitaba,
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
MESES_A_LIMPIAR = MESES                # o una lista de meses para reprocesar sólo esos
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
mostrar(tot, "Bronce → Plata, seis meses", formatos={"filas": "int", "venta": "dinero", "pct_filas": "pct2", "pct_venta": "pct2"})
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
