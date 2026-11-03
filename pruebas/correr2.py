"""Corre notebooks en formato fuente de Databricks con Spark local + Delta, simulando dbutils y los volúmenes."""
import os
import sys
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
BASE = sys.argv[1]
NOTEBOOKS = sys.argv[2:]

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

b = (SparkSession.builder.master("local[4]").appName("prueba")
     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
     .config("spark.sql.warehouse.dir", os.path.join(BASE, "warehouse"))
     .config("spark.driver.memory", "6g").config("spark.sql.shuffle.partitions", "8")
     .config("spark.ui.enabled", "false").config("spark.ui.showConsoleProgress", "false")
     .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
     .config("spark.sql.ansi.enabled", "true")
     .config("spark.driver.extraJavaOptions", "-Dderby.system.home=" + os.path.join(BASE, "derby")
             + " --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED -Dio.netty.tryReflectionSetAccessible=true")
     .config("spark.executor.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED -Dio.netty.tryReflectionSetAccessible=true"))
spark = configure_spark_with_delta_pip(b).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.sql("CREATE DATABASE IF NOT EXISTS maestria")
if os.environ.get("SIMULAR_SERVERLESS"):
    # serverless: sin cache/persist, sólo algunas configuraciones permitidas, sin datos del clúster
    from pyspark.sql import DataFrame
    from pyspark.sql.conf import RuntimeConfig
    def _sin_cache(*a, **k):
        raise Exception("[NOT_SUPPORTED_WITH_SERVERLESS] PERSIST TABLE is not supported on serverless compute.")
    DataFrame.persist = _sin_cache
    DataFrame.cache = _sin_cache
    _set_original, _get_original = RuntimeConfig.set, RuntimeConfig.get
    PERMITIDAS = {"spark.sql.session.timeZone", "spark.sql.shuffle.partitions", "spark.sql.ansi.enabled"}
    def _set(self, k, v):
        if k not in PERMITIDAS:
            raise Exception(f"[CONFIG_NOT_AVAILABLE] Configuration {k} is not available.")
        return _set_original(self, k, v)
    def _get(self, k, *a):
        if k.startswith("spark.databricks.clusterUsageTags"):
            raise Exception(f"[CONFIG_NOT_AVAILABLE] Configuration {k} is not available.")
        return _get_original(self, k, *a)
    RuntimeConfig.set, RuntimeConfig.get = _set, _get
    print("(simulando serverless)")
# El catálogo local vive en memoria: si el almacén ya tiene tablas Delta de una corrida anterior, se vuelven a registrar
_almacen = os.path.join(BASE, "warehouse", "maestria.db")
if os.path.isdir(_almacen):
    for _t in sorted(os.listdir(_almacen)):
        if os.path.isdir(os.path.join(_almacen, _t, "_delta_log")):
            spark.sql(f"CREATE TABLE IF NOT EXISTS maestria.{_t} USING DELTA LOCATION '{os.path.join(_almacen, _t)}'")
if not spark.catalog.tableExists("maestria.ventas_companero"):
    spark.sql("CREATE TABLE maestria.ventas_companero USING DELTA AS "
              "SELECT 1001 AS store_code, DATE'2026-03-01' AS fecha, 10.0 AS venta_total")


class FS:
    def ls(self, ruta):
        ruta = ruta.replace("dbfs:", "")
        if not os.path.exists(ruta):
            raise FileNotFoundError(ruta)
        salida = []
        for n in sorted(os.listdir(ruta)):
            p = os.path.join(ruta, n)
            mt = int(os.path.getmtime(p) * 1000)
            if os.path.isdir(p):
                salida.append(SimpleNamespace(path=p + "/", name=n + "/", size=0, modificationTime=mt))
            else:
                salida.append(SimpleNamespace(path=p, name=n, size=os.path.getsize(p), modificationTime=mt))
        return salida


def _put(ruta, texto, overwrite=True):
    ruta = ruta.replace("dbfs:", "")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)
    return True
FS.put = staticmethod(_put)


class Widgets:
    """Emula dbutils.widgets: los valores vienen de la variable de entorno PARAMETROS_NOTEBOOK (JSON), como un Job."""
    def __init__(self):
        import json
        self._valores = json.loads(os.environ.get("PARAMETROS_NOTEBOOK", "{}"))
        self._defectos = {}

    def text(self, nombre, defecto="", etiqueta=None):
        self._defectos[nombre] = defecto

    def get(self, nombre):
        if nombre in self._valores:
            return str(self._valores[nombre])
        if nombre in self._defectos:
            return self._defectos[nombre]
        raise ValueError(f"widget {nombre} no existe")


dbutils = SimpleNamespace(fs=FS(), widgets=Widgets())


def display(x):
    print(x)


for nb in NOTEBOOKS:
    fuente = open(nb, encoding="utf-8").read()
    fuente = fuente.replace('CATALOGO       = "modeladoprescriptivoventas"', 'CATALOGO       = "spark_catalog"')
    fuente = fuente.replace('RAIZ_VOLUMENES = "/Volumes"', f'RAIZ_VOLUMENES = "{BASE}/Volumes"')
    celdas = fuente.split("# COMMAND ----------")
    # como en Databricks: el espacio de nombres de la libreta es el módulo __main__ (así pickle encuentra sus clases)
    import types
    _modulo = types.ModuleType("__main__")
    sys.modules["__main__"] = _modulo
    ns = _modulo.__dict__
    dbutils.widgets = Widgets()
    ns.update({"spark": spark, "dbutils": dbutils, "display": display, "__name__": "__main__"})
    print(f"\n{'#' * 100}\n# {os.path.basename(nb)}  ({len(celdas)} celdas)\n{'#' * 100}")
    t0 = time.time()
    for i, celda in enumerate(celdas):
        cuerpo = celda.replace("# Databricks notebook source", "").strip()
        if not cuerpo or cuerpo.startswith("# MAGIC %md") or cuerpo.startswith("# MAGIC %pip"):
            continue
        try:
            exec(compile(cuerpo, f"{os.path.basename(nb)}[celda {i}]", "exec"), ns)
        except Exception:
            tb = traceback.format_exc()
            print("\n".join([l for l in tb.splitlines() if not l.strip().startswith("at ")][-60:]))
            print(f"\n!!! FALLÓ {os.path.basename(nb)} en la celda {i}")
            sys.exit(1)
    print(f"\n=== {os.path.basename(nb)} OK en {time.time() - t0:.0f} s")
