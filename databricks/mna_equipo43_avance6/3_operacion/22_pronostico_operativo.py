# Databricks notebook source
# MAGIC %md
# MAGIC # 22 · Pronóstico operativo (inferencia con el modelo registrado)
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC Separa **pronosticar** de **entrenar**: toma el modelo final del registro de Unity Catalog
# MAGIC (`demanda_categoria @champion`, o `@challenger` si todavía no hay campeón), puntúa el conjunto `pronostico` de la tabla
# MAGIC Oro (origen = último día con datos, h = 1…14) y escribe `gold.pronostico_14d` con pronóstico puntual, P10/P50/P90 y
# MAGIC techo operativo. Tarda minutos, no horas: es lo que corre cada vez que llegan datos nuevos (Job `J4_inferencia`),
# MAGIC sin reentrenar.
# MAGIC
# MAGIC **Promociones planeadas.** Si el modelo se entrenó con las variables `f_*`, aquí se unen desde
# MAGIC `silver.calendario_promociones_plan_dia` (13): cambiar el plan y volver a correr este notebook actualiza el pronóstico.
# MAGIC
# MAGIC **Otro origen.** Con `ORIGEN_PRONOSTICO = "AAAA-MM-DD"` se pronostica un origen distinto al último (por ejemplo, el
# MAGIC conjunto `pronostico` de una Oro anterior). El modelo rechaza orígenes anteriores a la calibración de sus intervalos
# MAGIC (protección anti-fuga): el acierto en producción se mide con `23_monitoreo` cuando llega la venta real de los días
# MAGIC pronosticados hacia adelante; mientras tanto la referencia es la prueba de septiembre del Avance 5.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.features_origen` (conjunto `pronostico`) · modelo `demanda_categoria` (UC o run de MLflow) · `silver.calendario_promociones_plan_dia` |
# MAGIC | **Escribe** | `gold.pronostico_14d` (reemplaza sólo su origen) · `mlops.pronosticos_bitacora` |
# MAGIC | **Parámetros** | `ALIAS_MODELO`, `PERMITIR_CHALLENGER`, `RUN_ID_MODELO`, `ORIGEN_PRONOSTICO`, `USAR_CALENDARIO_PROMOS`, `MODO_PUBLICO` |
# MAGIC | **Tiempo** | 10–30 min |
# MAGIC
# MAGIC **Requisito.** Una corrida de `Avance5.43` en modo completo que haya guardado el modelo final (compuerta técnica
# MAGIC aprobada o `REGISTRAR_EN_UC = True`). Si no hay modelo, este notebook se detiene y lo dice; **no** fabrica pronósticos.

# COMMAND ----------

# MAGIC %pip install "lightgbm>=4.3,<5"

# COMMAND ----------

# MAGIC %pip install "xgboost>=2.0,<4"

# COMMAND ----------

# MAGIC %pip install "catboost>=1.2,<2"

# COMMAND ----------

try:
    dbutils.library.restartPython()      # necesario después de %pip en Databricks; fuera de Databricks no existe y se ignora
except Exception:
    pass

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL PROYECTO · es la misma en los 11 notebooks (si cambias algo, cámbialo en todos)
# ════════════════════════════════════════════════════════════════════════════════════════════════
CATALOGO       = "modeladoprescriptivoventas"     # catálogo de Unity Catalog del proyecto
ESQUEMA_DATOS  = "maestria"                       # esquema donde están los volúmenes de Bnext
RAIZ_VOLUMENES = "/Volumes"
MESES = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]   # un volumen por mes

# Cortes del walk-forward. Son FIJOS (los del Avance 4): así, al llegar un mes nuevo, los modelos se siguen comparando
# en las mismas ventanas. Los bloques de prueba son datos no vistos: no intervienen en el ajuste ni en la elección del
# modelo y sólo existen cuando ya están cargados sus días (septiembre y octubre).
DIAS_CORTE = 14                                           # días que se pronostican desde cada origen
ORIGEN_AJUSTE = "2026-05-20"                              # bloque 21-may…03-jun: sólo para afinar hiperparámetros
ORIGENES_CORTES = {"corte_1": "2026-06-03", "corte_2": "2026-06-17",      # desarrollo: con estos se compara y se elige
                   "corte_3": "2026-07-01", "corte_4": "2026-08-17"}
ORIGENES_PRUEBA = {"prueba_1": "2026-08-31", "prueba_2": "2026-09-14",    # prueba no vista: 01–14 y 15–28 de septiembre
                   "prueba_3": "2026-09-28", "prueba_4": "2026-10-12"}    # y 29-sep–12-oct, 13–26 de octubre (modelos congelados al 31-ago)

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


def _parametro_a_texto(valor):
    if valor is None:
        return ""
    if isinstance(valor, bool):
        return "True" if valor else "False"
    if isinstance(valor, (list, tuple)):
        return ",".join(str(v) for v in valor)
    return str(valor)


def _parametro_desde_texto(texto, defecto):
    """Convierte el texto del widget al tipo del valor por defecto. Vacío = valor por defecto (o None si el defecto es None)."""
    texto = (texto or "").strip()
    if texto == "" or texto.lower() in ("none", "null"):
        return defecto if defecto is not None else None
    if isinstance(defecto, bool):
        if texto.lower() in ("true", "1", "sí", "si", "yes", "verdadero"):
            return True
        if texto.lower() in ("false", "0", "no", "falso"):
            return False
        raise ValueError(f"El parámetro esperaba True/False y recibió {texto!r}")
    if isinstance(defecto, int) and not isinstance(defecto, bool):
        return int(float(texto))
    if isinstance(defecto, float):
        return float(texto)
    if isinstance(defecto, (list, tuple)):
        if texto.startswith("["):
            import json
            return list(json.loads(texto))
        return [v.strip() for v in texto.split(",") if v.strip()]
    return texto


def parametro(nombre, defecto, descripcion=None):
    """Parámetro del notebook con el mismo nombre que la variable. Si el notebook corre en un Job (o se escribió el valor
    en el widget de la parte superior), toma ese valor; si no, usa el valor por defecto escrito en el código. El tipo
    (True/False, número, lista separada por comas, texto) se deduce del valor por defecto. Así el mismo notebook sirve en
    interactivo y en Workflows sin editar celdas."""
    try:
        dbutils.widgets.text(nombre, _parametro_a_texto(defecto), descripcion or nombre)
        crudo = dbutils.widgets.get(nombre)
    except Exception:
        return defecto
    valor = _parametro_desde_texto(crudo, defecto)
    if _parametro_a_texto(valor) != _parametro_a_texto(defecto):
        print(f"(parámetro {nombre} = {valor!r}, tomado del widget o del Job; por defecto {defecto!r})")
    return valor

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


def comentar_columnas(nombre_tabla, descripciones):
    """Escribe el comentario de cada columna de una tabla (queda visible en Catalog Explorer y lo usa Genie)."""
    hechos = 0
    for col, texto in (descripciones or {}).items():
        try:
            spark.sql(f"ALTER TABLE {nombre_tabla} ALTER COLUMN `{col}` COMMENT '{str(texto).replace(chr(39), '')}'")
            hechos += 1
        except Exception:
            pass
    return hechos


def asegurar_restricciones(nombre_tabla, restricciones):
    """Agrega restricciones CHECK de Delta que aún no existan. restricciones = {"nombre": "expresión SQL"}. Delta valida
    la regla sobre los datos existentes: si no se cumple, la rechaza y aquí queda como aviso (la tabla sigue sin ella)."""
    try:
        existentes = {r["key"].replace("delta.constraints.", "") for r in spark.sql(f"SHOW TBLPROPERTIES {nombre_tabla}").collect()
                      if str(r["key"]).startswith("delta.constraints.")}
    except Exception:
        existentes = set()
    for nombre, expresion in (restricciones or {}).items():
        if nombre in existentes:
            continue
        try:
            spark.sql(f"ALTER TABLE {nombre_tabla} ADD CONSTRAINT {nombre} CHECK ({expresion})")
        except Exception as e:
            aviso(f"la restricción {nombre} ({expresion}) no se pudo agregar a {nombre_tabla}: {str(e)[:140]}")


def borrar_corrida(nombre_tabla, valor, columna="run_id"):
    """Borra de una tabla las filas de una corrida antes de volver a escribirlas (idempotencia: repetir o reanudar una
    corrida no duplica resultados). Devuelve True si borró o no había nada que borrar."""
    if not existe(nombre_tabla):
        return True
    try:
        if columna in spark.table(nombre_tabla).columns:
            spark.sql(f"DELETE FROM {nombre_tabla} WHERE {columna} = '{str(valor).replace(chr(39), '')}'")
        return True
    except Exception as e:
        print(f"(no se pudo borrar {columna} = {valor} de {nombre_tabla}: {type(e).__name__}: {str(e)[:100]})")
        return False


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
    try:      # bitácora de corridas: qué notebook corrió, cuándo y cuánto tardó (la lee el tablero de estado del pipeline)
        fila = [(notebook, dt.datetime.fromtimestamp(_INICIO_NOTEBOOK), dt.datetime.now(),
                 float((time.time() - _INICIO_NOTEBOOK) / 60), len(HALLAZGOS),
                 sum(1 for h in HALLAZGOS if h.startswith("REVISAR:")), "OK")]
        (spark.createDataFrame(fila, "notebook string, inicio timestamp, fin timestamp, minutos double, hallazgos int, avisos int, estado string")
         .write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(tabla("mlops", "corridas")))
    except Exception as e:
        print(f"(no se pudo registrar la corrida en mlops.corridas: {type(e).__name__})")
    if siguiente:
        print(f"\nSiguiente notebook: {siguiente}")


class Cronometro:
    """Mide los segundos de un bloque de código: `with Cronometro() as reloj: ...` y después `reloj.segundos`."""

    def __enter__(self):
        self._inicio = time.perf_counter()
        self.segundos = 0.0
        return self

    def __exit__(self, *exc):
        self.segundos = time.perf_counter() - self._inicio
        return False


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

# ─────────────────────── calendario de promociones planeadas (contrato compartido por 13, 20, Avance5.43, 22 y 30) ───────────────────────
TABLA_CALENDARIO_PLAN = tabla("silver", "calendario_promociones_plan")          # una fila por promoción × alcance (lo que se edita)
TABLA_CALENDARIO_PLAN_DIA = tabla("silver", "calendario_promociones_plan_dia")  # tienda × categoría × día con las promociones que lo cubren
VARIABLES_PROMO_FUTURA = ["f_promos_planeadas", "f_descuento_planeado", "f_dias_promo_restantes"]
ESQUEMA_CALENDARIO_PROMOS = {
    "promo_id": "texto · identificador de la promoción (obligatorio)",
    "nombre": "texto · nombre comercial (opcional)",
    "inicio_planeado": "fecha · primer día planeado (obligatorio)",
    "fin_planeado": "fecha · último día planeado (obligatorio)",
    "categoria_id": "texto · categoría canónica; vacío si se da categoria_nombre; TODAS = todas las categorías (o las del tipo_categoria indicado)",
    "categoria_nombre": "texto · nombre de la categoría tal como está en silver.catalogo_categorias (se traduce a categoria_id); TODAS = todas",
    "tipo_categoria": "texto · mercancia / generico / servicio; con TODAS acota a ese tipo (vacío = todos los tipos)",
    "tienda": "texto · tienda; vacío = todas las tiendas del alcance indicado en zona o región",
    "zona": "texto · zona; vacío = no se acota por zona",
    "region": "texto · región; vacío = no se acota por región",
    "descuento_planeado": "decimal 0–1 · descuento planeado (0.15 = 15 %); vacío = desconocido (se imputa el descuento histórico de la categoría)",
    "tipo_mecanica": "texto · 'descuento', 'lleva N paga M', 'precio fijo', 'regalo', … (opcional)",
    "fecha_publicacion": "fecha · desde cuándo se conoce el plan; vacío = inicio_planeado − DIAS_ANTICIPACION_DEFECTO",
    "fuente": "texto · 'bnext' (plan real), 'placeholder' (inventada, sustituir) o 'historico_observado' (derivada del 09)",
    "comentario": "texto · libre (opcional)",
}


def validar_calendario_promociones(calendario) -> dict:
    """Revisa un calendario (Spark) contra ESQUEMA_CALENDARIO_PROMOS. Regresa conteos; no imprime filas."""
    faltan = [c for c in ("promo_id", "inicio_planeado", "fin_planeado", "categoria_id") if c not in calendario.columns]
    if faltan:
        return {"valido": False, "motivo": f"faltan columnas obligatorias: {faltan}"}
    r = calendario.agg(F.count("*").alias("filas"),
                       F.sum(F.when(F.col("fin_planeado") < F.col("inicio_planeado"), 1).otherwise(0)).alias("fechas_invertidas"),
                       F.sum(F.when(F.col("categoria_id").isNull(), 1).otherwise(0)).alias("sin_categoria"),
                       F.min("inicio_planeado").alias("primer_dia"), F.max("fin_planeado").alias("ultimo_dia")).collect()[0].asDict()
    r["valido"] = bool(r["filas"]) and not r["fechas_invertidas"] and not r["sin_categoria"]
    return r


def cargar_variables_promo_futura(spark):
    """Tabla tienda × categoría × día con las promociones planeadas que lo cubren (la escribe 13_calendario_promociones_plan).
    Regresa (DataFrame listo para unir, información) o (None, motivo) si no existe o no es válida."""
    if not existe(TABLA_CALENDARIO_PLAN_DIA):
        return None, {"activo": False, "motivo": f"no existe {TABLA_CALENDARIO_PLAN_DIA}: corre 13_calendario_promociones_plan"}
    dia = spark.table(TABLA_CALENDARIO_PLAN_DIA)
    faltan = {"tienda", "categoria_id", "fecha", "promos"} - set(dia.columns)
    if faltan:
        return None, {"activo": False, "motivo": f"{TABLA_CALENDARIO_PLAN_DIA} no tiene las columnas {sorted(faltan)}"}
    resumen = dia.agg(F.count("*").alias("filas"), F.min("fecha").alias("primer_dia"), F.max("fecha").alias("ultimo_dia"),
                      F.max("version_plan").alias("version_plan")).collect()[0].asDict()
    if not resumen["filas"]:
        return None, {"activo": False, "motivo": f"{TABLA_CALENDARIO_PLAN_DIA} está vacía"}
    listo = dia.select(F.col("tienda").cast("string").alias("_t"), F.col("categoria_id").cast("string").alias("_c"),
                       F.col("fecha").alias("fecha_objetivo"), F.col("promos"))
    return listo, {"activo": True, **{k: (str(v) if k != "filas" else int(v)) for k, v in resumen.items()}}


def unir_promociones_planeadas(df, dia):
    """Agrega a un DataFrame de la Oro (tienda, categoria_id, origen, fecha_objetivo) las variables f_*: sólo cuentan las
    promociones cuya fecha_publicacion es ≤ origen (lo que se sabía al pronosticar), por eso no hay fuga en los cortes."""
    if dia is None:
        for c in VARIABLES_PROMO_FUTURA:
            df = df.withColumn(c, F.lit(0.0).cast("float"))
        return df
    unido = (df.withColumn("_t", F.col("tienda").cast("string")).withColumn("_c", F.col("categoria_id").cast("string"))
             .join(dia, ["_t", "_c", "fecha_objetivo"], "left").drop("_t", "_c")
             .withColumn("_vis", F.expr("filter(promos, p -> p.fecha_publicacion <= origen)")))
    sin = F.col("promos").isNull()
    return (unido.withColumn("f_promos_planeadas", F.when(sin, F.lit(0)).otherwise(F.size("_vis")).cast("float"))
            .withColumn("f_descuento_planeado", F.coalesce(F.when(sin, F.lit(0.0)).otherwise(F.expr("array_max(transform(_vis, p -> p.descuento))")), F.lit(0.0)).cast("float"))
            .withColumn("f_dias_promo_restantes", F.coalesce(F.when(sin, F.lit(0)).otherwise(F.expr("array_min(transform(_vis, p -> p.restantes))")), F.lit(0)).cast("float"))
            .drop("promos", "_vis"))

# COMMAND ----------

# ─────────────────────── modelo registrado: cargarlo y puntuar la Oro con él (22_pronostico_operativo, 30_simulacion_escenarios_whatif) ───────────────────────
NOMBRE_MODELO_UC = f"{CATALOGO}.{ESQUEMA_DATOS}.demanda_categoria"
TABLA_ORO = tabla("gold", "features_origen")
TABLA_PRONOSTICO = tabla("gold", "pronostico_14d")
TABLA_RESUMEN_AV5 = tabla("mlops", "av5_resumen_corrida")
COLUMNAS_PRONOSTICO = ["tienda", "origen", "fecha_objetivo", "h", "categoria_id", "tipo_categoria", "region",
                       "prediccion_puntual", "p10", "p50", "p90", "techo_operativo"]
DESCRIPCION_PRONOSTICO = {
    "tienda": "Identificador de la tienda.", "origen": "Día T con cuyos datos se hizo el pronóstico.",
    "fecha_objetivo": "Día pronosticado (T + h).", "h": "Horizonte en días (1–14).",
    "categoria_id": "Categoría canónica.", "tipo_categoria": "mercancia, servicio o generico.", "region": "Región de la tienda.",
    "prediccion_puntual": "Pronóstico de venta del día (pesos).", "p10": "Límite inferior del intervalo (10 %).",
    "p50": "Mediana del intervalo.", "p90": "Límite superior del intervalo (90 %).",
    "techo_operativo": "Cantidad que cubre la demanda con el nivel elegido (P90 por defecto): techo de surtido.",
    "modelo": "Modelo que produjo el pronóstico.", "run_id": "Corrida de entrenamiento (mlops.av5_ejecuciones).",
    "version_oro": "Versión Delta de la tabla Oro usada.", "fecha_generacion": "Cuándo se calculó.",
    "version_modelo": "Versión del modelo en Unity Catalog (o run de MLflow).", "alias_modelo": "champion / challenger / run_mlflow.",
    "fase": "operativo (futuro) o retrospectivo (origen pasado, para medir el acierto).",
    "f_promos_planeadas": "Promociones planeadas que cubren el día (según el calendario vigente al pronosticar).",
}


def cargar_modelo_operativo(alias="champion", run_id_av5=None, permitir_challenger=True):
    """Carga el modelo final como pyfunc de MLflow. Orden: Unity Catalog @alias → @challenger (si se permite) → el run
    de MLflow de la corrida (run_id_av5 o la última corrida completa con modelo guardado). Regresa (modelo, info)."""
    import mlflow
    import mlflow.pyfunc
    intentos, info = [], {"origen": None, "alias": None, "version": None, "run_id_av5": run_id_av5, "modelo": None}
    try:
        mlflow.set_registry_uri("databricks-uc")
        from mlflow import MlflowClient
        cliente = MlflowClient()
        aliases = [alias] + (["challenger"] if permitir_challenger and alias != "challenger" else [])
        for a in aliases:
            try:
                mv = cliente.get_model_version_by_alias(NOMBRE_MODELO_UC, a)
                modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO_UC}@{a}")
                etiquetas = dict(mv.tags or {})
                info.update(origen="unity_catalog", alias=a, version=str(mv.version), run_id_mlflow=mv.run_id,
                            run_id_av5=etiquetas.get("run_id_av5", run_id_av5), modelo=etiquetas.get("modelo"), etiquetas=etiquetas)
                if a != alias:
                    aviso(f"No existe @{alias} en {NOMBRE_MODELO_UC}: se usa @{a} (versión {mv.version}).")
                return modelo, info
            except Exception as e:
                intentos.append(f"@{a}: {type(e).__name__}: {str(e)[:100]}")
    except Exception as e:
        intentos.append(f"registro: {type(e).__name__}: {str(e)[:100]}")
    # respaldo: el run de MLflow de la corrida del Avance 5 (el modelo se guarda ahí aunque el registro en UC falle)
    try:
        resumen = spark.table(TABLA_RESUMEN_AV5).where(F.col("mlflow_run_id").isNotNull())
        if run_id_av5:
            resumen = resumen.where(F.col("run_id") == run_id_av5)
        fila = resumen.where(F.col("modo") == "completo").orderBy(F.col("fecha_publicacion").desc()).limit(1).collect()
        if not fila:
            fila = resumen.orderBy(F.col("fecha_publicacion").desc()).limit(1).collect()
            if fila:
                aviso("No hay corridas en modo completo: se usa el modelo de una corrida en modo rápido (sólo sirve para probar el flujo).")
        if not fila:
            raise RuntimeError("no hay corridas con modelo guardado en " + TABLA_RESUMEN_AV5)
        fila = fila[0].asDict()
        modelo = mlflow.pyfunc.load_model(f"runs:/{fila['mlflow_run_id']}/modelo_final")
        info.update(origen="run_mlflow", alias="run_mlflow", version=str(fila["mlflow_run_id"])[:8], run_id_mlflow=fila["mlflow_run_id"],
                    run_id_av5=fila["run_id"], modelo=fila["modelo_final"])
        aviso(f"El modelo no se pudo tomar de Unity Catalog ({'; '.join(intentos)[:200]}); se usa el modelo guardado en el run de MLflow "
              f"de la corrida {fila['run_id']}.")
        return modelo, info
    except Exception as e:
        intentos.append(f"run de MLflow: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError("No hay modelo disponible. Corre Avance5.43 en modo completo (o con REGISTRAR_EN_UC = True) para que "
                       "guarde el modelo final. Intentos: " + " | ".join(intentos))


def columnas_modelo(modelo):
    """Columnas de entrada que pide la firma del modelo (None si no tiene firma)."""
    esquema = modelo.metadata.get_input_schema()
    return list(esquema.input_names()) if esquema is not None else None


def _preparar_entrada(pdf, modelo):
    """Ajusta tipos a la firma: fechas a datetime, enteros, decimales a double y categóricas a texto."""
    esquema = modelo.metadata.get_input_schema()
    if esquema is None:
        return pdf
    for col in esquema.inputs:
        n, t = col.name, getattr(col.type, "name", str(col.type)).replace("DataType.", "")
        if n not in pdf.columns:
            continue
        if "datetime" in t:
            pdf[n] = pd.to_datetime(pdf[n])
        elif t in ("long", "integer"):
            pdf[n] = pd.to_numeric(pdf[n], errors="coerce").fillna(0).astype("int64")
        elif t in ("double", "float"):
            pdf[n] = pd.to_numeric(pdf[n], errors="coerce").astype("float64")
        elif t == "string":
            pdf[n] = pdf[n].map(lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)).astype(object)
    return pdf


def puntuar_oro(filas, modelo, por="h", extra=()):
    """Puntúa un DataFrame de Spark con filas de la Oro (tienda, categoria_id, origen, fecha_objetivo, h, variables) por
    lotes en el driver y regresa pandas con las llaves, tipo_categoria, region, las columnas `extra` que existan y las
    columnas del pronóstico."""
    entradas = columnas_modelo(modelo) or []
    llaves = ["tienda", "categoria_id", "origen", "fecha_objetivo", "h", "tipo_categoria", "region"] + [c for c in extra if c in filas.columns]
    faltan = [c for c in entradas if c not in filas.columns]
    if faltan:
        raise ValueError(f"A la Oro le faltan columnas que pide el modelo: {faltan}")
    columnas = list(dict.fromkeys(llaves + entradas))
    valores = [r[0] for r in filas.select(por).distinct().orderBy(por).collect()] if por else [None]
    partes = []
    for v in valores:
        lote = filas.where(F.col(por) == v) if por else filas
        pdf = lote.select(*columnas).toPandas()
        if pdf.empty:
            continue
        salida = modelo.predict(_preparar_entrada(pdf[entradas].copy(), modelo))
        salida = pd.DataFrame(salida).reset_index(drop=True)
        partes.append(pd.concat([pdf[llaves].reset_index(drop=True), salida], axis=1))
        print(f"   {por} = {v}: {len(pdf):,} filas puntuadas")
    if not partes:
        return pd.DataFrame(columns=COLUMNAS_PRONOSTICO)
    salida = pd.concat(partes, ignore_index=True)
    for c in ("tienda", "categoria_id", "tipo_categoria", "region"):
        salida[c] = salida[c].astype(str)
    for c in ("origen", "fecha_objetivo"):
        salida[c] = pd.to_datetime(salida[c]).dt.date
    salida["h"] = salida["h"].astype("int32")
    return salida[COLUMNAS_PRONOSTICO + [c for c in llaves if c not in COLUMNAS_PRONOSTICO]]


def escribir_pronostico(pdf, origen, extra):
    """Escribe el pronóstico en gold.pronostico_14d reemplazando sólo ese origen (los otros orígenes se conservan) y deja
    restricciones y comentarios en la tabla."""
    datos = pdf.copy()
    for k, v in extra.items():
        datos[k] = v
    datos["fecha_generacion"] = pd.Timestamp.now()
    limpio, esquema = _para_spark(datos)
    sdf = spark.createDataFrame(limpio, schema=esquema)
    escritor = sdf.write.format("delta").mode("overwrite")
    if existe(TABLA_PRONOSTICO):
        escritor = escritor.option("replaceWhere", f"origen = DATE'{origen}'").option("mergeSchema", "true")
    else:
        escritor = escritor.option("overwriteSchema", "true")
    escritor.saveAsTable(TABLA_PRONOSTICO)
    asegurar_restricciones(TABLA_PRONOSTICO, {"pron_h_valido": "h BETWEEN 1 AND 14",
                                              "pron_no_negativo": "prediccion_puntual >= 0 AND p10 >= 0",
                                              "pron_intervalo_ordenado": "p10 <= p50 AND p50 <= p90"})
    comentar_columnas(TABLA_PRONOSTICO, {k: v for k, v in DESCRIPCION_PRONOSTICO.items() if k in datos.columns})
    try:
        spark.sql(f"COMMENT ON TABLE {TABLA_PRONOSTICO} IS 'Pronostico operativo a 14 dias por tienda x categoria x dia con intervalo P10-P90 y techo de surtido; cada corrida reemplaza solo su origen'")
    except Exception:
        pass
    return TABLA_PRONOSTICO

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook (también como parámetros de Job)
ALIAS_MODELO = parametro("ALIAS_MODELO", "champion")            # alias del registro: champion (producción) o challenger
PERMITIR_CHALLENGER = parametro("PERMITIR_CHALLENGER", True)    # si no hay @champion, usar @challenger
RUN_ID_MODELO = parametro("RUN_ID_MODELO", None)                # corrida del Avance 5 cuyo modelo usar si el registro no está disponible
ORIGEN_PRONOSTICO = parametro("ORIGEN_PRONOSTICO", None)        # None = último día con datos (operativo); una fecha = retrospectivo
USAR_CALENDARIO_PROMOS = parametro("USAR_CALENDARIO_PROMOS", True)
MODO_PUBLICO = parametro("MODO_PUBLICO", True)                  # True: no se imprimen importes
MIN_DIAS_CON_VENTA = 5                                          # mismas filas elegibles que en el entrenamiento (Avance 4 y 5)
TABLA_BITACORA = tabla("mlops", "pronosticos_bitacora")
RUN_ID_PRONOSTICO = "pron_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dt%H%M%S")
from uuid import uuid4
RUN_ID_PRONOSTICO += "_" + uuid4().hex[:6]
print("Corrida de pronóstico:", RUN_ID_PRONOSTICO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Modelo

# COMMAND ----------

titulo("1. MODELO REGISTRADO")
MODELO, INFO_MODELO = cargar_modelo_operativo(ALIAS_MODELO, RUN_ID_MODELO, PERMITIR_CHALLENGER)
ENTRADAS = columnas_modelo(MODELO) or []
USA_F = all(c in ENTRADAS for c in VARIABLES_PROMO_FUTURA)
mostrar(pd.DataFrame([{k: v for k, v in INFO_MODELO.items() if k != "etiquetas"}]), "Modelo cargado")
print(f"Variables de entrada del modelo: {len(ENTRADAS)} · usa promociones planeadas (f_*): {USA_F}")
hallazgo(f"Modelo {INFO_MODELO.get('modelo')} tomado de {INFO_MODELO['origen']} (alias {INFO_MODELO.get('alias')}, versión {INFO_MODELO.get('version')}), "
         f"entrenado en la corrida {INFO_MODELO.get('run_id_av5')}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Filas de la Oro a pronosticar

# COMMAND ----------

titulo("2. FILAS A PRONOSTICAR")
ORO = leer_tabla(TABLA_ORO, "20_features_oro")
try:
    VERSION_ORO = int(spark.sql(f"DESCRIBE HISTORY {TABLA_ORO} LIMIT 1").collect()[0]["version"])
except Exception:
    VERSION_ORO = None
if ORIGEN_PRONOSTICO is None:
    fila = ORO.where(F.col("conjunto") == "pronostico").agg(F.max("origen").alias("o")).collect()[0]
    if fila["o"] is None:
        raise RuntimeError("La Oro no tiene conjunto 'pronostico'. Corre 20_features_oro con GENERAR_PRONOSTICO = True.")
    ORIGEN, CONJUNTO, FASE = fila["o"], "pronostico", "operativo"
else:
    ORIGEN = pd.Timestamp(ORIGEN_PRONOSTICO).date()
    disponibles = {r["conjunto"] for r in ORO.where(F.col("origen") == F.lit(ORIGEN)).select("conjunto").distinct().collect()}
    if not disponibles:
        raise RuntimeError(f"La Oro no tiene filas con origen {ORIGEN}; orígenes de evaluación: ver mlops.oro_resumen.")
    CONJUNTO = "pronostico" if "pronostico" in disponibles else "evaluacion"
    FASE = "operativo" if CONJUNTO == "pronostico" else "retrospectivo"
elegible = ((F.col("dias_con_venta_hist") >= MIN_DIAS_CON_VENTA) & F.col("ingenuo_estacional").isNotNull() & (F.col("s_dias_activos_7") >= 5))
FILAS = ORO.where((F.col("conjunto") == CONJUNTO) & (F.col("origen") == F.lit(ORIGEN)) & elegible)
if USAR_CALENDARIO_PROMOS and USA_F:
    PLAN_DIA, INFO_CAL = cargar_variables_promo_futura(spark)
    if PLAN_DIA is None:
        aviso(f"El modelo usa promociones planeadas pero el calendario no está disponible ({INFO_CAL.get('motivo')}): se pronostica como si no hubiera promociones planeadas.")
    FILAS = unir_promociones_planeadas(FILAS, PLAN_DIA)
    print("Calendario de promociones:", INFO_CAL)
elif USAR_CALENDARIO_PROMOS and not USA_F:
    aviso("El modelo registrado no se entrenó con las variables f_* del calendario: el plan de promociones no influye en este pronóstico "
          "(reentrena con USAR_CALENDARIO_PROMOS = True en Avance5.43).")
    INFO_CAL = {"activo": False}
else:
    INFO_CAL = {"activo": False}
    FILAS = unir_promociones_planeadas(FILAS, None) if USA_F else FILAS
n = FILAS.agg(F.count("*").alias("filas"), F.countDistinct("tienda", "categoria_id").alias("series"), F.countDistinct("h").alias("h")).collect()[0]
print(f"Origen {ORIGEN} ({FASE}, conjunto {CONJUNTO}) · {n['filas']:,} filas · {n['series']:,} series · {n['h']} horizontes · Oro versión {VERSION_ORO}")
if n["filas"] == 0:
    raise RuntimeError("No hay filas elegibles para pronosticar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Puntuar y escribir `gold.pronostico_14d`

# COMMAND ----------

titulo("3. PRONÓSTICO")
with Cronometro() as reloj:
    try:
        PRON = puntuar_oro(FILAS, MODELO, por="h", extra=["f_promos_planeadas", "f_descuento_planeado"] if USA_F else [])
    except ValueError as e:
        if "calibraci" in str(e):
            raise RuntimeError(f"El modelo no puede pronosticar el origen {ORIGEN}: sus intervalos se calibraron con datos posteriores y la "
                               "protección anti-fuga lo impide. El acierto en producción se mide cuando llega la venta real de un origen "
                               "pronosticado hacia adelante (23_monitoreo); la referencia mientras tanto es la fase de prueba del Avance 5.") from e
        raise
print(f"{len(PRON):,} filas pronosticadas en {reloj.segundos / 60:.1f} min")
extra = {"modelo": INFO_MODELO.get("modelo") or "modelo_final", "run_id": INFO_MODELO.get("run_id_av5") or "",
         "version_oro": str(VERSION_ORO), "version_modelo": str(INFO_MODELO.get("version")), "alias_modelo": INFO_MODELO.get("alias"),
         "fase": FASE, "run_id_pronostico": RUN_ID_PRONOSTICO, "version_plan": str(INFO_CAL.get("version_plan", ""))}
DESTINO = escribir_pronostico(PRON, ORIGEN, extra)
hallazgo(f"Pronóstico {FASE} del origen {ORIGEN} escrito en {DESTINO}: {len(PRON):,} filas, {PRON[['tienda', 'categoria_id']].drop_duplicates().shape[0]:,} series, "
         f"{PRON['fecha_objetivo'].min()} → {PRON['fecha_objetivo'].max()}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Revisión (sin importes) y bitácora

# COMMAND ----------

titulo("4. REVISIÓN")
rev = pd.DataFrame([{
    "filas": len(PRON), "series": PRON[["tienda", "categoria_id"]].drop_duplicates().shape[0],
    "p10_le_pron_le_p90": float(((PRON["p10"] <= PRON["prediccion_puntual"]) & (PRON["prediccion_puntual"] <= PRON["p90"])).mean()),
    "ancho_relativo_mediano": float(((PRON["p90"] - PRON["p10"]) / PRON["prediccion_puntual"].clip(lower=1.0)).median()),
    "pronostico_cero": float((PRON["prediccion_puntual"] <= 0).mean()),
    "series_con_promo_planeada": float(PRON.groupby(["tienda", "categoria_id"])["f_promos_planeadas"].max().gt(0).mean()) if "f_promos_planeadas" in PRON else 0.0,
}])
mostrar(rev, "Revisión del pronóstico", formatos={"filas": "int", "series": "int", "p10_le_pron_le_p90": "pct", "ancho_relativo_mediano": "dec2",
                                                 "pronostico_cero": "pct", "series_con_promo_planeada": "pct"})
por_region = PRON.groupby(["region", "tipo_categoria"]).agg(series=("tienda", lambda s: s.nunique()), filas=("h", "size"),
                                                           ancho_rel=("p90", "size")).reset_index()
por_region["ancho_rel"] = PRON.assign(a=(PRON["p90"] - PRON["p10"]) / PRON["prediccion_puntual"].clip(lower=1.0)).groupby(["region", "tipo_categoria"])["a"].median().to_numpy()
mostrar(por_region, "Series y ancho relativo del intervalo por región × tipo", formatos={"series": "int", "filas": "int", "ancho_rel": "dec2"})
if not MODO_PUBLICO:
    tot = PRON.groupby("fecha_objetivo")[["prediccion_puntual", "p10", "p90"]].sum().reset_index()
    mostrar(tot, "Totales diarios (pesos)", formatos={"prediccion_puntual": "dinero", "p10": "dinero", "p90": "dinero"})
bit = pd.DataFrame([{"run_id_pronostico": RUN_ID_PRONOSTICO, "origen": ORIGEN, "fase": FASE, "modelo": extra["modelo"], "alias_modelo": extra["alias_modelo"],
                     "version_modelo": extra["version_modelo"], "run_id_entrenamiento": extra["run_id"], "version_oro": str(VERSION_ORO),
                     "version_plan": extra["version_plan"], "filas": int(len(PRON)), "series": int(rev["series"].iloc[0]),
                     "desde": PRON["fecha_objetivo"].min(), "hasta": PRON["fecha_objetivo"].max(), "minutos": float(reloj.segundos / 60),
                     "fecha": pd.Timestamp.now()}])
guardar(bit, "mlops", "pronosticos_bitacora", "Corridas de pronostico operativo: origen, modelo, version y filas (22)", modo="append")

# COMMAND ----------

cerrar("22_pronostico_operativo", siguiente="23_monitoreo · 30_simulacion_escenarios_whatif · 31_desagregacion_predicciones_sku · 40_salidas_dashboard")
