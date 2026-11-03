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

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# @@COLUMNAS

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
