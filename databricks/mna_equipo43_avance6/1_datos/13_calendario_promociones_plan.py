# Databricks notebook source
# MAGIC %md
# MAGIC # 13 · Calendario de promociones planeadas
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC Convierte el **archivo editable** del calendario de promociones (Excel o CSV en el volumen `outputs/calendario/`) en las
# MAGIC tablas que usan los modelos y el what-if. El equipo (o Bnext) sólo edita el archivo; este notebook lo valida y lo
# MAGIC publica.
# MAGIC
# MAGIC **Fuentes del plan.**
# MAGIC 1. `calendario_promociones_plan.xlsx` (hoja `promociones`): promociones **futuras** — hoy un *placeholder* inventado con
# MAGIC    `fuente = placeholder`, que se sustituye por el plan real de Bnext (`fuente = bnext`) editando el archivo. Si el
# MAGIC    archivo no existe, este notebook escribe la plantilla en el volumen para que se edite.
# MAGIC 2. La vigencia **observada** de las promociones pasadas (`silver.calendario_promociones` del 09) como aproximación del
# MAGIC    plan histórico: se supone que cada promoción se conocía `DIAS_ANTICIPACION_HISTORICO` días antes de empezar
# MAGIC    (`fuente = historico_observado`). Con eso el modelo aprende el efecto de una promoción anunciada.
# MAGIC
# MAGIC **Sin fuga.** Cada promoción lleva `fecha_publicacion`; al construir variables para un origen T sólo cuentan las
# MAGIC publicadas en o antes de T (`unir_promociones_planeadas`).
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | el archivo del volumen · `silver.calendario_promociones`, `silver.promociones`, `silver.catalogo_categorias`, `silver.catalogo_tiendas` |
# MAGIC | **Escribe** | `silver.calendario_promociones_plan` (promoción × categoría × alcance) · `silver.calendario_promociones_plan_dia` (tienda × categoría × día) · `mlops.calendario_plan_versiones` · `outputs/calendario/catalogo_categorias.csv` |
# MAGIC | **Parámetros** | `ARCHIVO_CALENDARIO`, `INCLUIR_HISTORICO_OBSERVADO`, `DIAS_ANTICIPACION_DEFECTO`, `DIAS_ANTICIPACION_HISTORICO`, `FALLAR_SI_INVALIDO` |
# MAGIC | **Tiempo** | 5–15 min |

# COMMAND ----------

# MAGIC %pip install "openpyxl>=3.1,<4"

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

# ----------------------------------------------------------------- parámetros de este notebook (también como parámetros de Job)
CARPETA_CALENDARIO = volumen("outputs", "calendario")
ARCHIVO_CALENDARIO = parametro("ARCHIVO_CALENDARIO", CARPETA_CALENDARIO + "/calendario_promociones_plan.xlsx")
# Excel (.xlsx, hoja 'promociones') o CSV con las columnas del contrato ESQUEMA_CALENDARIO_PROMOS.
INCLUIR_HISTORICO_OBSERVADO = parametro("INCLUIR_HISTORICO_OBSERVADO", True)
# True → las promociones pasadas observadas (09) entran como plan histórico; False → sólo el archivo.
DIAS_ANTICIPACION_DEFECTO = parametro("DIAS_ANTICIPACION_DEFECTO", 14)      # fecha_publicacion si el archivo no la trae
DIAS_ANTICIPACION_HISTORICO = parametro("DIAS_ANTICIPACION_HISTORICO", 14)  # cuántos días antes se "conocía" una promoción observada
FALLAR_SI_INVALIDO = parametro("FALLAR_SI_INVALIDO", False)                 # True en Jobs: detenerse si el archivo tiene errores
IMPUTAR_DESCUENTO = parametro("IMPUTAR_DESCUENTO", True)
# True → si una fila no trae descuento_planeado, se usa el descuento histórico mediano de la categoría (de las promociones
#        observadas); así una promoción de la que sólo se conocen las fechas entra al modelo con un descuento plausible.
TIPO_TODAS_DEFECTO = parametro("TIPO_TODAS_DEFECTO", "mercancia")
# A qué tipo de categoría se expande una fila con categoria = TODAS cuando no indica tipo_categoria ("" = todos los tipos).
TABLA_VERSIONES = tabla("mlops", "calendario_plan_versiones")
print("Archivo del plan:", ARCHIVO_CALENDARIO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leer el archivo editable (o escribir la plantilla si no existe)

# COMMAND ----------

import hashlib
import io

COLUMNAS_PLAN = list(ESQUEMA_CALENDARIO_PROMOS)
# Placeholder INVENTADO para arrancar (las mismas filas de calendario/calendario_promociones_plan.xlsx del repositorio).
PLANTILLA_PLACEHOLDER = [
    ("PH-2026-10-OTONO", "Arranque de otoño refrescos y cerveza", "2026-10-02", "2026-10-11", "", "REFRESCOS", "", "", "", "", 0.10, "descuento", "2026-09-20", "placeholder", "INVENTADA: cae dentro del horizonte del pronóstico de octubre"),
    ("PH-2026-10-OTONO", "Arranque de otoño refrescos y cerveza", "2026-10-02", "2026-10-11", "", "CERVEZA", "", "", "", "", 0.10, "descuento", "2026-09-20", "placeholder", "INVENTADA"),
    ("PH-2026-10-HALLOWEEN", "Halloween botanas y dulces", "2026-10-26", "2026-10-31", "", "BOTANAS", "", "", "", "", 0.10, "descuento", "2026-10-10", "placeholder", "INVENTADA: sustituir por el plan real"),
    ("PH-2026-10-HALLOWEEN", "Halloween botanas y dulces", "2026-10-26", "2026-10-31", "", "CACAHUATE Y SEMILLAS", "", "", "", "", 0.10, "descuento", "2026-10-10", "placeholder", "INVENTADA"),
    ("PH-2026-11-DIA-MUERTOS", "Día de Muertos pan y café", "2026-10-30", "2026-11-02", "", "PAN Y GALLETA", "", "", "", "", 0.15, "descuento", "2026-10-15", "placeholder", "INVENTADA"),
    ("PH-2026-11-DIA-MUERTOS", "Día de Muertos pan y café", "2026-10-30", "2026-11-02", "", "CAFES", "", "", "", "", 0.15, "descuento", "2026-10-15", "placeholder", "INVENTADA"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "CERVEZA", "", "", "", "", 0.20, "descuento", "2026-10-20", "placeholder", "INVENTADA; fechas del Buen Fin por confirmar"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "REFRESCOS", "", "", "", "", 0.20, "descuento", "2026-10-20", "placeholder", "INVENTADA"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "BOTANAS", "", "", "", "", 0.20, "descuento", "2026-10-20", "placeholder", "INVENTADA"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "AGUAS", "", "", "", "", 0.15, "descuento", "2026-10-20", "placeholder", "INVENTADA"),
    ("PH-2026-11-2X1-YOGURT", "2x1 yogurt noviembre", "2026-11-17", "2026-11-30", "", "YOGURT", "", "", "", "", 0.50, "lleva 2 paga 1", "2026-11-01", "placeholder", "INVENTADA; 2x1 ≈ 50 % de descuento"),
    ("PH-2026-12-POSADAS", "Posadas cerveza y botanas", "2026-12-12", "2026-12-24", "", "CERVEZA", "", "", "", "", 0.10, "descuento", "2026-11-25", "placeholder", "INVENTADA"),
    ("PH-2026-12-POSADAS", "Posadas cerveza y botanas", "2026-12-12", "2026-12-24", "", "BOTANAS", "", "", "", "", 0.10, "descuento", "2026-11-25", "placeholder", "INVENTADA"),
    ("PH-2026-12-POSADAS", "Posadas cerveza y botanas", "2026-12-12", "2026-12-24", "", "REFRESCOS", "", "", "", "", 0.10, "descuento", "2026-11-25", "placeholder", "INVENTADA"),
    ("PH-2026-12-NAVIDAD", "Navidad y Año Nuevo", "2026-12-20", "2026-12-31", "", "CERVEZA", "", "", "", "", 0.15, "descuento", "2026-12-01", "placeholder", "INVENTADA; se traslapa con posadas a propósito"),
    ("PH-2026-12-NAVIDAD", "Navidad y Año Nuevo", "2026-12-20", "2026-12-31", "", "CHICHARRONES", "", "", "", "", 0.15, "descuento", "2026-12-01", "placeholder", "INVENTADA"),
    ("PH-2026-11-REGIONAL-EJEMPLO", "Ejemplo de alcance regional", "2026-11-20", "2026-11-22", "", "PALOMITAS", "", "", "", "NORTE", 0.20, "descuento", "2026-11-05", "placeholder", "INVENTADA: ejemplo de alcance por región"),
    ("PH-2026-11-SOLO-FECHAS", "Promoción de la que sólo se conocen las fechas (ejemplo)", "2026-11-05", "2026-11-08", "TODAS", "", "mercancia", "", "", "", "", "descuento", "2026-10-25", "placeholder", "INVENTADA: ejemplo de fechas sin categoría ni descuento; se expande a toda la mercancía con descuento imputado"),
]


def escribir_plantilla(ruta):
    """Escribe la plantilla (Excel o CSV según la extensión) con el placeholder y las instrucciones."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    datos = pd.DataFrame(PLANTILLA_PLACEHOLDER, columns=COLUMNAS_PLAN)
    if ruta.lower().endswith(".csv"):
        datos.to_csv(ruta, index=False)
        return
    instrucciones = pd.DataFrame({"instrucciones": [
        "Cada fila es una promoción × categoría × alcance; la misma promo_id puede repetirse.",
        "Obligatorias: promo_id, inicio_planeado, fin_planeado (AAAA-MM-DD) y categoria_id o categoria_nombre.",
        "categoria_nombre se traduce con silver.catalogo_categorias (ver catalogo_categorias.csv en esta carpeta).",
        "Alcance: tienda, zona y region vacíos = nacional; llena sólo el nivel que aplique.",
        "descuento_planeado en fracción (0.15 = 15 %); 2x1 = 0.50. fecha_publicacion = desde cuándo se conoce el plan.",
        "fuente: bnext (plan real) o placeholder (inventada: sustituir). Guardar, subir al volumen y correr 13.",
    ]})
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        datos.to_excel(w, sheet_name="promociones", index=False)
        instrucciones.to_excel(w, sheet_name="instrucciones", index=False)


def leer_archivo(ruta):
    if ruta.lower().endswith(".csv"):
        return pd.read_csv(ruta, dtype=str, keep_default_na=False)
    return pd.read_excel(ruta, sheet_name="promociones", dtype=str, keep_default_na=False)


titulo("1. ARCHIVO DEL CALENDARIO")
if not os.path.exists(ARCHIVO_CALENDARIO):
    escribir_plantilla(ARCHIVO_CALENDARIO)
    aviso(f"No existía el archivo del calendario: se escribió la plantilla con promociones placeholder (inventadas) en {ARCHIVO_CALENDARIO}. "
          "Edítala con el plan real de Bnext y vuelve a correr este notebook.")
with open(ARCHIVO_CALENDARIO, "rb") as f:
    contenido = f.read()
VERSION_PLAN = hashlib.sha256(contenido).hexdigest()[:12]
PLAN_CRUDO = leer_archivo(ARCHIVO_CALENDARIO)
PLAN_CRUDO.columns = [str(c).strip().lower() for c in PLAN_CRUDO.columns]
for c in COLUMNAS_PLAN:
    if c not in PLAN_CRUDO.columns:
        PLAN_CRUDO[c] = ""
PLAN_CRUDO = PLAN_CRUDO[COLUMNAS_PLAN].map(lambda v: str(v).strip() if v is not None else "") if hasattr(PLAN_CRUDO, "map") else PLAN_CRUDO[COLUMNAS_PLAN].applymap(lambda v: str(v).strip())
PLAN_CRUDO = PLAN_CRUDO[(PLAN_CRUDO["promo_id"] != "") | (PLAN_CRUDO["inicio_planeado"] != "")]
print(f"Archivo: {os.path.basename(ARCHIVO_CALENDARIO)} · versión {VERSION_PLAN} · {len(PLAN_CRUDO)} filas · "
      f"fuentes: {dict(PLAN_CRUDO['fuente'].replace('', 'sin_fuente').value_counts())}")

# COMMAND ----------

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# CUANDO BNEXT ENTREGUE SU CATÁLOGO DE PROMOCIONES COMO TABLA O ARCHIVO (en lugar del Excel): descomentar y ajustar
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 1. Indicar dónde está el catálogo (tabla de Unity Catalog, parquet o CSV en el volumen):
# ORIGEN_CATALOGO_BNEXT = tabla("silver", "promociones_bnext")          # o volumen("outputs", "calendario", "catalogo_bnext.csv")
# 2. Mapear los nombres de columnas de Bnext a las del contrato (ESQUEMA_CALENDARIO_PROMOS); las que no existan se dejan "":
# MAPEO_COLUMNAS_BNEXT = {
#     "promo_id": "id_promocion",            # identificador de la promoción en Bnext
#     "nombre": "nombre_promocion",
#     "inicio_planeado": "fecha_inicio",
#     "fin_planeado": "fecha_fin",
#     "categoria_id": "id_categoria",        # o "categoria_nombre": "categoria" si viene el nombre; "TODAS" si aplica a todas
#     "tienda": "",                          # vacío = nacional; o "store_code" si viene por tienda
#     "zona": "", "region": "",
#     "descuento_planeado": "porcentaje_descuento",   # 0–1 o 0–100 (el notebook lo normaliza); vacío = se imputa
#     "tipo_mecanica": "mecanica",
#     "fecha_publicacion": "fecha_aprobacion",        # vacío = inicio − DIAS_ANTICIPACION_DEFECTO
#     "comentario": "",
# }
# 3. Leer, renombrar y usar en lugar del Excel (todo lo demás del notebook sigue igual):
# if ORIGEN_CATALOGO_BNEXT.count(".") == 2:
#     crudo = spark.table(ORIGEN_CATALOGO_BNEXT).toPandas()
# elif ORIGEN_CATALOGO_BNEXT.lower().endswith(".csv"):
#     crudo = pd.read_csv(ORIGEN_CATALOGO_BNEXT, dtype=str, keep_default_na=False)
# else:
#     crudo = spark.read.parquet(ORIGEN_CATALOGO_BNEXT).toPandas()
# PLAN_CRUDO = pd.DataFrame({c: (crudo[origen].astype(str) if origen in crudo.columns else "") for c, origen in MAPEO_COLUMNAS_BNEXT.items()})
# for c in COLUMNAS_PLAN:
#     if c not in PLAN_CRUDO.columns:
#         PLAN_CRUDO[c] = ""
# PLAN_CRUDO["fuente"] = "bnext"
# PLAN_CRUDO = PLAN_CRUDO[COLUMNAS_PLAN]
# VERSION_PLAN = hashlib.sha256(PLAN_CRUDO.to_csv(index=False).encode("utf-8")).hexdigest()[:12]
# print(f"Catálogo de Bnext: {len(PLAN_CRUDO)} filas · versión {VERSION_PLAN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validación y traducción de categorías y alcances
# MAGIC Se traducen `categoria_nombre` → `categoria_id` con el catálogo, se completan `fecha_publicacion` y `descuento_planeado`,
# MAGIC se revisan fechas, rangos y alcances (región, zona o tienda que existan). Los errores se listan sin datos sensibles;
# MAGIC las filas con error se omiten (o detienen el notebook si `FALLAR_SI_INVALIDO = True`).
# MAGIC
# MAGIC **Si sólo se conocen las fechas** de una promoción: `categoria_id = TODAS` (opcionalmente `tipo_categoria = mercancia`)
# MAGIC la expande a todas las categorías de ese tipo y, si no trae descuento, se le imputa el descuento histórico mediano de
# MAGIC cada categoría (`descuento_imputado = 1`). Es una aproximación declarada: cuando llegue el detalle por categoría, se
# MAGIC sustituyen esas filas.

# COMMAND ----------

titulo("2. VALIDACIÓN")
CAT = leer_tabla(tabla("silver", "catalogo_categorias"), "08_limpieza_plata").select(
    F.col("categoria_id").cast("string").alias("categoria_id"), F.upper(F.trim(F.col("nombre"))).alias("nombre"), "tipo_categoria").toPandas()
TIENDAS = leer_tabla(tabla("silver", "catalogo_tiendas"), "08_limpieza_plata").select(
    F.col("tienda").cast("string").alias("tienda"), F.col("zona").cast("string").alias("zona"),
    F.col("region").cast("string").alias("region")).toPandas()
try:      # catálogo de referencia junto al archivo, para que quien edite el Excel encuentre los identificadores
    CAT.sort_values("nombre").to_csv(CARPETA_CALENDARIO + "/catalogo_categorias.csv", index=False)
    print(f"Catálogo de categorías de referencia escrito en {CARPETA_CALENDARIO}/catalogo_categorias.csv ({len(CAT)} categorías)")
except Exception as e:
    print(f"(no se pudo escribir el catálogo de referencia: {type(e).__name__})")

nombre_a_id = dict(zip(CAT["nombre"], CAT["categoria_id"]))
ids_validos = set(CAT["categoria_id"])
plan = PLAN_CRUDO.copy()
plan["fila"] = range(2, len(plan) + 2)      # número de fila en el archivo (encabezado = 1)
# ---- TODAS: una promoción de la que sólo se conocen las fechas se expande a todas las categorías (del tipo indicado)
es_todas = (plan["categoria_id"].str.upper() == "TODAS") | ((plan["categoria_id"] == "") & (plan["categoria_nombre"].str.upper() == "TODAS"))
if es_todas.any():
    tipo_filtro = plan["tipo_categoria"].str.lower().where(plan["tipo_categoria"] != "", TIPO_TODAS_DEFECTO)
    expandidas = []
    for i in plan.index[es_todas]:
        cats = CAT[CAT["tipo_categoria"].str.lower() == tipo_filtro[i]] if tipo_filtro[i] else CAT
        for cid in cats["categoria_id"]:
            fila = plan.loc[i].copy()
            fila["categoria_id"], fila["categoria_nombre"] = cid, ""
            expandidas.append(fila)
    n_todas = int(es_todas.sum())
    plan = pd.concat([plan[~es_todas], pd.DataFrame(expandidas)], ignore_index=True)
    hallazgo(f"{n_todas} filas con categoría TODAS se expandieron a {len(expandidas)} filas promoción × categoría "
             f"(tipo por defecto: {TIPO_TODAS_DEFECTO or 'todos'}).")
plan["categoria_id"] = plan["categoria_id"].where(plan["categoria_id"] != "", plan["categoria_nombre"].str.upper().map(nombre_a_id).fillna(""))
plan["descuento_planeado"] = pd.to_numeric(plan["descuento_planeado"].replace("", np.nan), errors="coerce")
plan.loc[plan["descuento_planeado"] > 1, "descuento_planeado"] = plan.loc[plan["descuento_planeado"] > 1, "descuento_planeado"] / 100  # 15 → 0.15
plan["descuento_imputado"] = 0
if IMPUTAR_DESCUENTO and plan["descuento_planeado"].isna().any() and existe(tabla("silver", "calendario_promociones")):
    hist = (spark.table(tabla("silver", "calendario_promociones")).select("promo_id", F.col("categoria_id").cast("string").alias("categoria_id"))
            .join(spark.table(tabla("silver", "promociones")).select("promo_id", "tasa_media"), "promo_id").where(F.col("tasa_media") > 0)
            .groupBy("categoria_id").agg(F.expr("percentile_approx(tasa_media, 0.5)").alias("descuento_hist")).toPandas())
    desc_cat = dict(zip(hist["categoria_id"], hist["descuento_hist"]))
    desc_global = float(hist["descuento_hist"].median()) if len(hist) else 0.0
    faltante = plan["descuento_planeado"].isna()
    plan.loc[faltante, "descuento_planeado"] = plan.loc[faltante, "categoria_id"].map(desc_cat).fillna(desc_global)
    plan.loc[faltante, "descuento_imputado"] = 1
    hallazgo(f"{int(faltante.sum())} filas sin descuento: se imputó el descuento histórico mediano de su categoría "
             f"(global {desc_global:.0%}); quedan marcadas con descuento_imputado = 1.")
plan["descuento_planeado"] = plan["descuento_planeado"].fillna(0.0).clip(0, 1)
for c in ("inicio_planeado", "fin_planeado", "fecha_publicacion"):
    plan[c] = pd.to_datetime(plan[c].replace("", np.nan), errors="coerce")
plan["fecha_publicacion"] = plan["fecha_publicacion"].fillna(plan["inicio_planeado"] - pd.Timedelta(days=DIAS_ANTICIPACION_DEFECTO))
plan["fuente"] = plan["fuente"].replace("", "bnext").str.lower()
plan["tipo_mecanica"] = plan["tipo_mecanica"].replace("", "descuento")

errores = []
def marcar(mascara, motivo):
    for fila in plan.loc[mascara, "fila"]:
        errores.append({"fila": int(fila), "motivo": motivo})
marcar(plan["promo_id"] == "", "promo_id vacío")
marcar(plan["inicio_planeado"].isna() | plan["fin_planeado"].isna(), "fecha de inicio o fin vacía o inválida (usar AAAA-MM-DD)")
marcar(plan["fin_planeado"] < plan["inicio_planeado"], "fin anterior al inicio")
marcar((plan["categoria_id"] == "") | ~plan["categoria_id"].isin(ids_validos), "categoría no encontrada en silver.catalogo_categorias (revisar categoria_id o categoria_nombre)")
marcar((plan["region"] != "") & ~plan["region"].str.upper().isin(set(TIENDAS["region"].dropna().str.upper())), "región no existe en el catálogo de tiendas")
marcar((plan["zona"] != "") & ~plan["zona"].str.upper().isin(set(TIENDAS["zona"].dropna().str.upper())), "zona no existe en el catálogo de tiendas")
marcar((plan["tienda"] != "") & ~plan["tienda"].isin(set(TIENDAS["tienda"])), "tienda no existe en el catálogo de tiendas")
marcar((plan["fin_planeado"] - plan["inicio_planeado"]).dt.days > 120, "promoción de más de 120 días (revisar fechas)")
ERRORES = pd.DataFrame(errores)
if len(ERRORES):
    mostrar(ERRORES.groupby("motivo").agg(filas=("fila", lambda s: ", ".join(str(v) for v in sorted(s)[:15])), n=("fila", "size")).reset_index(),
            "Filas con error (se omiten)")
    if FALLAR_SI_INVALIDO:
        raise ValueError(f"El calendario tiene {ERRORES['fila'].nunique()} filas con error; corrígelas en el archivo.")
    aviso(f"{ERRORES['fila'].nunique()} filas del archivo tienen errores y se omiten (ver tabla); el resto del plan sí se publica.")
plan = plan[~plan["fila"].isin(set(ERRORES["fila"]) if len(ERRORES) else set())].copy()
# traslapes de la misma promoción × categoría × alcance
plan["alcance"] = np.where(plan["tienda"] != "", "tienda", np.where(plan["zona"] != "", "zona", np.where(plan["region"] != "", "region", "nacional")))
dup = plan.duplicated(subset=["promo_id", "categoria_id", "tienda", "zona", "region"], keep=False)
if dup.any():
    aviso(f"{int(dup.sum())} filas repiten promo_id × categoría × alcance: se conservan todas (cuentan como una sola promoción por día).")
resumen_plan = (plan.groupby("fuente").agg(promociones=("promo_id", "nunique"), filas=("promo_id", "size"),
                                            desde=("inicio_planeado", "min"), hasta=("fin_planeado", "max"),
                                            descuento_medio=("descuento_planeado", "mean")).reset_index())
mostrar(resumen_plan, "Plan válido por fuente", formatos={"descuento_medio": "pct"})
mostrar(plan.groupby("alcance").agg(filas=("promo_id", "size")).reset_index(), "Filas por tipo de alcance")
if (plan["fuente"] == "placeholder").any():
    aviso(f"{int((plan['fuente'] == 'placeholder').sum())} filas son placeholder (inventadas): sustituirlas por el plan real de Bnext en {os.path.basename(ARCHIVO_CALENDARIO)}.")
hallazgo(f"Calendario versión {VERSION_PLAN}: {plan['promo_id'].nunique()} promociones válidas del archivo en {plan['categoria_id'].nunique()} categorías "
         f"({plan['inicio_planeado'].min().date() if len(plan) else '—'} → {plan['fin_planeado'].max().date() if len(plan) else '—'}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tabla del plan (promoción × categoría × alcance) y plan histórico observado

# COMMAND ----------

titulo("3. TABLAS DEL PLAN")
plan_tabla = plan.assign(inicio_planeado=plan["inicio_planeado"].dt.date, fin_planeado=plan["fin_planeado"].dt.date,
                         fecha_publicacion=plan["fecha_publicacion"].dt.date, version_plan=VERSION_PLAN, fecha_carga=pd.Timestamp.now())[
    ["promo_id", "nombre", "inicio_planeado", "fin_planeado", "categoria_id", "tienda", "zona", "region", "alcance",
     "descuento_planeado", "descuento_imputado", "tipo_mecanica", "fecha_publicacion", "fuente", "comentario", "version_plan", "fecha_carga"]]
guardar(plan_tabla, "silver", "calendario_promociones_plan",
        "Plan de promociones: una fila por promocion x categoria x alcance, leido del archivo editable del volumen (13)", modo="overwrite")
comentar_columnas(TABLA_CALENDARIO_PLAN, {k: v for k, v in ESQUEMA_CALENDARIO_PROMOS.items()})
PLAN_SDF = spark.table(TABLA_CALENDARIO_PLAN).withColumn("tienda", F.coalesce(F.col("tienda"), F.lit(""))) \
    .withColumn("zona", F.coalesce(F.col("zona"), F.lit(""))).withColumn("region", F.coalesce(F.col("region"), F.lit("")))

# ---- expansión del archivo a tiendas según el alcance (nacional / región / zona / tienda)
tiendas_sdf = spark.createDataFrame(TIENDAS[["tienda", "zona", "region"]].fillna(""))
archivo_dia = (PLAN_SDF.alias("p").join(tiendas_sdf.alias("t"),
                                       ((F.col("p.tienda") == "") | (F.col("p.tienda") == F.col("t.tienda")))
                                       & ((F.col("p.zona") == "") | (F.upper(F.col("p.zona")) == F.upper(F.col("t.zona"))))
                                       & ((F.col("p.region") == "") | (F.upper(F.col("p.region")) == F.upper(F.col("t.region")))))
               .select(F.col("t.tienda").alias("tienda"), F.col("p.categoria_id").alias("categoria_id"), F.col("p.promo_id").alias("promo_id"),
                       F.col("p.inicio_planeado").alias("inicio"), F.col("p.fin_planeado").alias("fin"),
                       F.col("p.fecha_publicacion").alias("fecha_publicacion"), F.col("p.descuento_planeado").alias("descuento"),
                       F.col("p.fuente").alias("fuente")))

# ---- histórico observado (09): la vigencia real de cada promoción por tienda × categoría, "publicada" N días antes
if INCLUIR_HISTORICO_OBSERVADO and existe(tabla("silver", "calendario_promociones")):
    promos_arch = {str(v) for v in plan["promo_id"].unique()}
    obs = (spark.table(tabla("silver", "calendario_promociones"))
           .join(spark.table(tabla("silver", "promociones")).select("promo_id", F.col("tasa_media").alias("descuento")), "promo_id", "left")
           .where(~F.col("promo_id").cast("string").isin(list(promos_arch)) if promos_arch else F.lit(True))
           .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"),
                   F.col("promo_id").cast("string").alias("promo_id"), F.col("inicio").cast("date").alias("inicio"), F.col("fin").cast("date").alias("fin"),
                   F.date_sub(F.col("inicio").cast("date"), int(DIAS_ANTICIPACION_HISTORICO)).alias("fecha_publicacion"),
                   F.coalesce(F.col("descuento").cast("double"), F.lit(0.0)).alias("descuento"), F.lit("historico_observado").alias("fuente")))
    n_obs = obs.agg(F.countDistinct("promo_id").alias("p"), F.count("*").alias("f"), F.min("inicio").alias("d"), F.max("fin").alias("h")).collect()[0]
    hallazgo(f"Plan histórico observado: {n_obs['p']:,} promociones ({n_obs['d']} → {n_obs['h']}) con fecha de publicación supuesta "
             f"{DIAS_ANTICIPACION_HISTORICO} días antes de su inicio.")
    todo = archivo_dia.unionByName(obs)
else:
    if INCLUIR_HISTORICO_OBSERVADO:
        aviso("No existe silver.calendario_promociones (corre 09): el plan sólo tiene lo que trae el archivo.")
    todo = archivo_dia

# ---- tienda × categoría × día con las promociones que lo cubren (lo que une la Oro)
dias = (todo.withColumn("fecha", F.explode(F.sequence(F.col("inicio"), F.col("fin"))))
        .withColumn("restantes", F.datediff(F.col("fin"), F.col("fecha"))))
PLAN_DIA = (dias.groupBy("tienda", "categoria_id", "fecha")
            .agg(F.collect_list(F.struct(F.col("promo_id"), F.col("fecha_publicacion"), F.col("descuento"), F.col("restantes"), F.col("fuente"))).alias("promos"),
                 F.countDistinct("promo_id").alias("n_promos"), F.max("descuento").alias("descuento_max"),
                 F.max(F.when(F.col("fuente") != "historico_observado", 1).otherwise(0)).alias("del_archivo"))
            .withColumn("version_plan", F.lit(VERSION_PLAN)).withColumn("mes", F.date_format("fecha", "yyyy-MM")))
guardar(PLAN_DIA, "silver", "calendario_promociones_plan_dia",
        "Promociones planeadas por tienda x categoria x dia (archivo + historico observado); cada promo trae su fecha_publicacion", particion="mes")
try:
    spark.sql(f"ALTER TABLE {TABLA_CALENDARIO_PLAN_DIA} SET TBLPROPERTIES ('mna.version_plan' = '{VERSION_PLAN}', "
              f"'mna.dias_anticipacion_historico' = '{DIAS_ANTICIPACION_HISTORICO}')")
    spark.sql(f"OPTIMIZE {TABLA_CALENDARIO_PLAN_DIA} ZORDER BY (tienda, categoria_id)")
except Exception as e:
    print(f"(propiedades u OPTIMIZE no disponibles: {type(e).__name__})")
comentar_columnas(TABLA_CALENDARIO_PLAN_DIA, {
    "tienda": "Tienda a la que aplica.", "categoria_id": "Categoría canónica.", "fecha": "Día cubierto por al menos una promoción planeada.",
    "promos": "Promociones que cubren el día: promo_id, fecha_publicacion, descuento (0-1), dias restantes, fuente.",
    "n_promos": "Promociones distintas ese día.", "descuento_max": "Mayor descuento planeado ese día.",
    "del_archivo": "1 si alguna viene del archivo editable (plan de Bnext o placeholder); 0 si todas son historico observado.",
    "version_plan": "Huella del archivo con el que se construyo.", "mes": "Particion AAAA-MM."})
r = spark.table(TABLA_CALENDARIO_PLAN_DIA).agg(F.count("*").alias("filas"), F.countDistinct("tienda").alias("tiendas"),
                                                F.countDistinct("categoria_id").alias("categorias"), F.min("fecha").alias("desde"), F.max("fecha").alias("hasta"),
                                                F.sum("del_archivo").alias("dias_del_archivo")).collect()[0].asDict()
mostrar(pd.DataFrame([r]), "Tabla tienda × categoría × día", formatos={"filas": "int", "tiendas": "int", "categorias": "int", "dias_del_archivo": "int"})
hallazgo(f"{TABLA_CALENDARIO_PLAN_DIA}: {r['filas']:,} filas tienda × categoría × día ({r['desde']} → {r['hasta']}); "
         f"{r['dias_del_archivo']:,} provienen del archivo editable.")
cobertura = (archivo_dia.groupBy("promo_id", "fuente").agg(F.countDistinct("tienda").alias("tiendas"), F.countDistinct("categoria_id").alias("categorias"))
             .orderBy("promo_id").toPandas())
mostrar(cobertura, "Alcance de cada promoción del archivo (tiendas y categorías a las que aplica)", formatos={"tiendas": "int", "categorias": "int"})
sin_tiendas = cobertura[cobertura["tiendas"] == 0]
if len(sin_tiendas):
    aviso(f"Promociones del archivo sin tiendas en su alcance: {', '.join(sin_tiendas['promo_id'])} (revisar región/zona/tienda).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Bitácora de versiones del plan

# COMMAND ----------

titulo("4. BITÁCORA")
bitacora = pd.DataFrame([{"version_plan": VERSION_PLAN, "archivo": os.path.basename(ARCHIVO_CALENDARIO), "fecha_carga": pd.Timestamp.now(),
                          "promociones_archivo": int(plan["promo_id"].nunique()), "filas_archivo": int(len(plan)),
                          "filas_con_error": int(ERRORES["fila"].nunique()) if len(ERRORES) else 0,
                          "placeholder": int((plan["fuente"] == "placeholder").sum()), "historico_observado": bool(INCLUIR_HISTORICO_OBSERVADO),
                          "filas_plan_dia": int(r["filas"]), "desde": str(r["desde"]), "hasta": str(r["hasta"])}])
guardar(bitacora, "mlops", "calendario_plan_versiones", "Versiones del calendario de promociones planeadas cargadas (13)", modo="append")
mostrar(bitacora.T.reset_index().rename(columns={"index": "campo", 0: "valor"}), "Esta carga")

# COMMAND ----------

cerrar("13_calendario_promociones_plan", siguiente="20_features_oro (las variables f_* se unen al entrenar y pronosticar)")
