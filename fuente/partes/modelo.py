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
