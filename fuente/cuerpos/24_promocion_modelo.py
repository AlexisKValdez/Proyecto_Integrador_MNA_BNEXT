# Databricks notebook source
# MAGIC %md
# MAGIC # 24 · Promoción del modelo: de `@challenger` a `@champion`
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC Gobierno del registro de modelos en Unity Catalog (`demanda_categoria`). Cada corrida completa de `Avance5.43` que
# MAGIC pasa la compuerta técnica registra una versión nueva con alias `@challenger`. Este notebook decide, con evidencia, si
# MAGIC esa versión pasa a `@champion` (la que usa `22_pronostico_operativo` en producción):
# MAGIC
# MAGIC 1. **Compuerta técnica** de la corrida (`mlops.av5_compuerta`): WMAPE ≤ línea base, sesgo, Diebold–Mariano, cobertura.
# MAGIC 2. **Comparación con el campeón vigente**: el retador no puede ser peor en la prueba no vista (WMAPE) más allá de la tolerancia.
# MAGIC 3. **Aprobaciones de negocio** registradas en la tabla `mlops.aprobaciones` (quién, cuándo, qué): alcance por categoría,
# MAGIC    política de servicios y disponibilidad de atributos al momento de pronosticar. Son datos, no constantes en el código.
# MAGIC
# MAGIC Con `ACCION = "evaluar"` sólo informa; con `ACCION = "promover"` mueve el alias (y conserva al campeón anterior como
# MAGIC `@anterior`) y deja constancia en `mlops.av5_ejecuciones` y `mlops.registro_modelos`.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | registro UC `demanda_categoria` · `mlops.av5_compuerta` · `mlops.av5_comparativa_prueba` · `mlops.aprobaciones` |
# MAGIC | **Escribe** | alias en el registro · `mlops.aprobaciones` (si no existe) · `mlops.registro_modelos` · `mlops.av5_ejecuciones` |
# MAGIC | **Parámetros** | `ACCION` (evaluar / promover), `TOLERANCIA_WMAPE`, `FORZAR` |
# MAGIC | **Tiempo** | 1–3 min |

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook (también como parámetros de Job)
ACCION = parametro("ACCION", "evaluar")                 # "evaluar" (sólo informa) o "promover" (mueve el alias si procede)
TOLERANCIA_WMAPE = parametro("TOLERANCIA_WMAPE", 0.0)  # el retador puede ser peor que el campeón hasta por esto (0 = no)
FORZAR = parametro("FORZAR", False)                     # True: promueve aunque falte evidencia (queda registrado que fue forzado)
NOMBRE_MODELO_UC = f"{CATALOGO}.{ESQUEMA_DATOS}.demanda_categoria"
TABLA_APROBACIONES = tabla("mlops", "aprobaciones")
TABLA_REGISTRO = tabla("mlops", "registro_modelos")
TABLA_EJECUCIONES = tabla("mlops", "av5_ejecuciones")
ASPECTOS = {"alcance_categoria": "Bnext confirma que el grano tienda × categoría y las categorías incluidas son los correctos",
            "politica_servicios": "Bnext confirma si las categorías de servicio se pronostican o se excluyen",
            "atributos_point_in_time": "Bnext confirma que los atributos de tienda y los calendarios estarán disponibles al pronosticar"}
USUARIO = None
try:
    USUARIO = spark.sql("SELECT current_user()").collect()[0][0]
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Versiones en el registro

# COMMAND ----------

titulo("1. REGISTRO DE MODELOS")
REGISTRO_OK, VERSIONES, CLIENTE = False, pd.DataFrame(), None
try:
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_registry_uri("databricks-uc")
    CLIENTE = MlflowClient()
    filas = []
    for mv in CLIENTE.search_model_versions(f"name = '{NOMBRE_MODELO_UC}'"):
        etq = dict(mv.tags or {})
        filas.append({"version": int(mv.version), "aliases": ",".join(sorted(mv.aliases or [])), "modelo": etq.get("modelo"),
                      "run_id_av5": etq.get("run_id_av5"), "wmape_desarrollo": etq.get("wmape_desarrollo"), "wmape_prueba": etq.get("wmape_prueba"),
                      "creada": pd.to_datetime(mv.creation_timestamp, unit="ms"), "run_id_mlflow": mv.run_id, "estado": mv.status})
    VERSIONES = pd.DataFrame(filas).sort_values("version", ascending=False) if filas else pd.DataFrame()
    REGISTRO_OK = True
except Exception as e:
    aviso(f"No se pudo leer el registro {NOMBRE_MODELO_UC} ({type(e).__name__}: {str(e)[:120]}). Sin registro no hay promoción: "
          "revisar permisos (USE SCHEMA, EXECUTE/CREATE MODEL) o que Avance5.43 haya registrado el modelo.")
if REGISTRO_OK and VERSIONES.empty:
    aviso(f"El registro {NOMBRE_MODELO_UC} no tiene versiones: corre Avance5.43 en modo completo (la compuerta técnica registra @challenger).")
if len(VERSIONES):
    mostrar(VERSIONES, "Versiones registradas")


def version_con(alias):
    if VERSIONES.empty:
        return None
    m = VERSIONES[VERSIONES["aliases"].str.split(",").map(lambda a: alias in a)]
    return m.iloc[0] if len(m) else None


CHALLENGER, CHAMPION = version_con("challenger"), version_con("champion")
print("Retador :", None if CHALLENGER is None else f"versión {CHALLENGER['version']} · {CHALLENGER['modelo']} · corrida {CHALLENGER['run_id_av5']}")
print("Campeón :", None if CHAMPION is None else f"versión {CHAMPION['version']} · {CHAMPION['modelo']} · corrida {CHAMPION['run_id_av5']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evidencia: compuerta técnica, prueba no vista y aprobaciones de negocio

# COMMAND ----------

titulo("2. EVIDENCIA")
EVIDENCIA = {"compuerta_tecnica": None, "mejor_o_igual_que_campeon": None, "aprobaciones_negocio": None}
if CHALLENGER is not None:
    t = tabla("mlops", "av5_compuerta")
    if existe(t):
        comp = spark.table(t).where(F.col("run_id") == CHALLENGER["run_id_av5"]).orderBy(F.col("fecha_publicacion").desc()).limit(1).toPandas()
        if len(comp):
            EVIDENCIA["compuerta_tecnica"] = bool(comp["technical_pass"].iloc[0])
            det = json.loads(comp["detalle_json"].iloc[0]) if isinstance(comp["detalle_json"].iloc[0], str) else {}
            mostrar(pd.DataFrame([{"comprobacion": k, "cumple": v} for k, v in det.get("checks", {}).items()]), "Compuerta técnica del retador")
            print("Motivos de bloqueo registrados en la corrida:", det.get("blocking_reasons"))
    w_ch = pd.to_numeric(pd.Series([CHALLENGER["wmape_prueba"], CHALLENGER["wmape_desarrollo"]]), errors="coerce")
    if CHAMPION is None:
        EVIDENCIA["mejor_o_igual_que_campeon"] = True
        print("No hay campeón: el retador sólo necesita la compuerta técnica y las aprobaciones.")
    else:
        w_cp = pd.to_numeric(pd.Series([CHAMPION["wmape_prueba"], CHAMPION["wmape_desarrollo"]]), errors="coerce")
        ref_ch, ref_cp = w_ch.dropna().iloc[0] if w_ch.notna().any() else np.nan, w_cp.dropna().iloc[0] if w_cp.notna().any() else np.nan
        EVIDENCIA["mejor_o_igual_que_campeon"] = bool(np.isnan(ref_cp) or (not np.isnan(ref_ch) and ref_ch <= ref_cp + TOLERANCIA_WMAPE))
        print(f"WMAPE retador {ref_ch} vs campeón {ref_cp} (tolerancia {TOLERANCIA_WMAPE}) → {'cumple' if EVIDENCIA['mejor_o_igual_que_campeon'] else 'NO cumple'}")
# aprobaciones de negocio como datos
if not existe(TABLA_APROBACIONES):
    base = spark.createDataFrame([(k, v, False, None, None, "pendiente") for k, v in ASPECTOS.items()],
                                 "aspecto string, descripcion string, aprobado boolean, aprobado_por string, fecha timestamp, comentario string")
    guardar(base, "mlops", "aprobaciones", "Aprobaciones de negocio necesarias para promover un modelo a produccion (24)")
    print(f"Se creó {TABLA_APROBACIONES} con los tres aspectos pendientes. Para aprobar uno:\n"
          f"  UPDATE {TABLA_APROBACIONES} SET aprobado = true, aprobado_por = '<nombre>', fecha = current_timestamp(), comentario = '<evidencia>' WHERE aspecto = 'alcance_categoria';")
APROB = spark.table(TABLA_APROBACIONES).toPandas()
mostrar(APROB, "Aprobaciones de negocio")
EVIDENCIA["aprobaciones_negocio"] = bool(len(APROB) and APROB["aprobado"].fillna(False).astype(bool).all())
mostrar(pd.DataFrame([{"criterio": k, "cumple": v} for k, v in EVIDENCIA.items()]), "Resumen de la evidencia")
PUEDE_PROMOVER = CHALLENGER is not None and all(bool(v) for v in EVIDENCIA.values())
hallazgo("Promoción " + ("PROCEDE: " if PUEDE_PROMOVER else "NO procede todavía: ")
         + ", ".join(f"{k} = {'sí' if v else 'no' if v is not None else 'sin dato'}" for k, v in EVIDENCIA.items()) + ".")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Acción

# COMMAND ----------

titulo("3. ACCIÓN")
if ACCION == "promover" and CHALLENGER is not None and REGISTRO_OK and (PUEDE_PROMOVER or FORZAR):
    v = int(CHALLENGER["version"])
    try:
        if CHAMPION is not None and int(CHAMPION["version"]) != v:
            CLIENTE.set_registered_model_alias(NOMBRE_MODELO_UC, "anterior", int(CHAMPION["version"]))
        CLIENTE.set_registered_model_alias(NOMBRE_MODELO_UC, "champion", v)
        try:
            CLIENTE.delete_registered_model_alias(NOMBRE_MODELO_UC, "challenger")
        except Exception:
            pass
        CLIENTE.set_model_version_tag(NOMBRE_MODELO_UC, v, "promovido_por", str(USUARIO))
        CLIENTE.set_model_version_tag(NOMBRE_MODELO_UC, v, "promovido_en", str(pd.Timestamp.now()))
        CLIENTE.set_model_version_tag(NOMBRE_MODELO_UC, v, "promocion_forzada", str(bool(FORZAR and not PUEDE_PROMOVER)))
        registro = pd.DataFrame([{"fecha": pd.Timestamp.now(), "modelo_uc": NOMBRE_MODELO_UC, "version": v, "accion": "promover_a_champion",
                                  "version_anterior": None if CHAMPION is None else int(CHAMPION["version"]), "run_id_av5": CHALLENGER["run_id_av5"],
                                  "usuario": USUARIO, "forzada": bool(FORZAR and not PUEDE_PROMOVER), "evidencia": json.dumps(EVIDENCIA)}])
        guardar(registro, "mlops", "registro_modelos", "Promociones y cambios de alias del modelo en Unity Catalog (24)", modo="append")
        evento = pd.DataFrame([{"run_id": CHALLENGER["run_id_av5"], "estado": "PROMOVIDO", "fecha_utc": dt.datetime.utcnow(),
                                "info_json": json.dumps({"version": v, "usuario": USUARIO, "forzada": bool(FORZAR and not PUEDE_PROMOVER)})}])
        guardar(evento, "mlops", "av5_ejecuciones", None, modo="append")
        hallazgo(f"Versión {v} promovida a @champion de {NOMBRE_MODELO_UC}" + (" (forzada)" if FORZAR and not PUEDE_PROMOVER else "") + ".")
    except Exception as e:
        aviso(f"No se pudo mover el alias: {type(e).__name__}: {str(e)[:160]}")
elif ACCION == "promover":
    print("No se promueve: " + ("no hay retador" if CHALLENGER is None else "falta evidencia (usa FORZAR = True sólo con acuerdo del equipo)"))
else:
    print("ACCION = evaluar: no se cambia nada en el registro.")

# COMMAND ----------

cerrar("24_promocion_modelo", siguiente="22_pronostico_operativo (con ALIAS_MODELO = champion)")
