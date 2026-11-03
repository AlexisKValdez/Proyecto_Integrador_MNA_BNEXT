
# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.11 Registro de la corrida: bitácora, puntos de control, MLflow y Unity Catalog
# MAGIC - Cada corrida tiene un identificador `av5_43_<fecha UTC>_<8 caracteres>`. Sus eventos (iniciada, bloque terminado,
# MAGIC   modelo elegido, finalizada o error) se agregan a `mlops.av5_ejecuciones`.
# MAGIC - Después de cada bloque se guarda un **punto de control** (pickle con las sumas, la bolsa de meta-entrenamiento y los
# MAGIC   hiperparámetros) en el volumen `outputs`. Si la corrida se cae, `RUN_ID_REANUDAR` la continúa sin repetir bloques.
# MAGIC - Las tablas de resultados se agregan con la columna `run_id` a tablas fijas `mlops.av5_*` (no se crea una tabla nueva
# MAGIC   por corrida). Ninguna tabla de resultados lleva tienda, categoría ni filas individuales, salvo el pronóstico operativo.
# MAGIC - MLflow guarda parámetros, métricas, tablas, gráficas y el modelo final empaquetado (`pyfunc`) que se registra en
# MAGIC   Unity Catalog como nueva versión de `demanda_categoria` con alias `@challenger`.

# COMMAND ----------

def verificar_entorno_formal(spark):
    """Versiones efectivas de las bibliotecas (no instala nada; las que falten se reportan y sus modelos se omiten)."""
    import importlib
    import platform
    observed, missing = {"python": platform.python_version()}, []
    for name in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "catboost", "optuna", "shap", "mlflow", "cloudpickle"):
        try:
            observed[name] = str(getattr(importlib.import_module(name), "__version__", "sin_version_publicada"))
        except Exception:
            missing.append(name)
            observed[name] = "NO DISPONIBLE"
    try:
        observed["databricks_runtime"] = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
    except Exception:
        observed["databricks_runtime"] = "serverless (no publica versión)"
    return observed, missing


def huella_codigo():
    """SHA-256 de las funciones principales de esta libreta: identifica qué versión del código produjo la corrida."""
    import inspect
    texto = ""
    for nombre in ("FuenteAvance5", "fit_model", "tune_boosting", "fit_homogeneous", "fit_meta", "predict_ensembles",
                   "fit_conformal", "fit_quantiles", "AcumuladorAvance5", "matriz_decision", "evaluate_gate", "ejecutar_modelado"):
        try:
            texto += inspect.getsource(globals()[nombre])
        except Exception:
            texto += nombre
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _operacion_json(value):
    """Convierte fechas y tipos de numpy para guardarlos en JSON."""
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


class RegistroFormal:
    """Contexto de una corrida formal: `with RegistroFormal(...) as REGISTRO_AV5: ...`.

    run_id          identificador de la corrida (nuevo, o el de RUN_ID_REANUDAR)
    mlflow          módulo de MLflow si está disponible (si no, se registra sólo en Delta)
    mlflow_run_id   identificador del run en MLflow
    ruta_control    carpeta del volumen donde vive el punto de control de esta corrida
    manifest        fuente, entorno, huella del código y decisiones; se registra al inicio y al final
    """

    def __init__(self, spark, fuente, entorno, huella, n_jobs, run_id=None):
        self.spark, self.fuente, self.entorno = spark, fuente, entorno
        self.reanudada = run_id is not None
        self.run_id = run_id or ("av5_43_" + datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S") + "_" + uuid4().hex[:8])
        self.ruta_control = f"{RUTA_PUNTOS_CONTROL}/{self.run_id}"
        self.mlflow, self.mlflow_run_id, self.experimento = None, None, None
        self.manifest = {"run_id": self.run_id, "codigo_sha256": huella, "entorno": entorno, "hilos": int(n_jobs),
                         "modo_rapido": MODO_RAPIDO, "fuente": fuente.manifiesto, "reanudada": self.reanudada}

    def __enter__(self):
        try:
            import mlflow
            mlflow.autolog(disable=True)          # sin esto, Databricks registraría un run por cada modelo entrenado
            self.experimento = EXPERIMENTO_MLFLOW
            if self.experimento is None:
                usuario = self.spark.sql("SELECT current_user()").collect()[0][0]
                self.experimento = f"/Users/{usuario}/mna_equipo43_avance5"
            try:
                mlflow.set_experiment(self.experimento)
            except Exception:
                self.experimento = "/Shared/mna_equipo43_avance5"
                mlflow.set_experiment(self.experimento)
            if mlflow.active_run() is not None:
                mlflow.end_run()
            previo = None
            if self.reanudada:
                encontrados = mlflow.search_runs(filter_string=f"tags.run_id_av5 = '{self.run_id}'", output_format="list")
                previo = encontrados[0].info.run_id if encontrados else None
            run = mlflow.start_run(run_id=previo) if previo else mlflow.start_run(run_name=self.run_id)
            self.mlflow, self.mlflow_run_id = mlflow, run.info.run_id
            mlflow.set_tags({"equipo": "43", "avance": "5", "run_id_av5": self.run_id,
                             "version_gold": str(self.fuente.version), "codigo_sha256": self.manifest["codigo_sha256"],
                             "modo": "rapido" if MODO_RAPIDO else "completo"})
            self.log_dict("manifiesto_inicial", self.manifest)
        except Exception as e:
            print(f"(MLflow no disponible: {type(e).__name__}: {str(e)[:120]}; se registra sólo en Delta)")
            self.mlflow = None
        self.manifest["mlflow_run_id"] = self.mlflow_run_id
        self.manifest["mlflow_experimento"] = self.experimento
        self.evento("REANUDADA" if self.reanudada else "INICIADA", {"version_gold": self.fuente.version,
                                                                     "mlflow_run_id": self.mlflow_run_id})
        print(f"Corrida {self.run_id} · MLflow {self.mlflow_run_id or '(no disponible)'} · punto de control en {self.ruta_control}")
        return self

    def evento(self, estado, info=None):
        """Agrega un evento a mlops.av5_ejecuciones (append; nunca sobrescribe)."""
        fila = pd.DataFrame([{"run_id": self.run_id, "estado": str(estado), "fecha_utc": datetime.now(timezone.utc).replace(tzinfo=None),
                              "info_json": json.dumps(info or {}, ensure_ascii=False, default=_operacion_json)}])
        try:
            limpio, esquema = _para_spark(fila)
            (self.spark.createDataFrame(limpio, schema=esquema).write.format("delta").mode("append")
             .option("mergeSchema", "true").saveAsTable(TABLA_EJECUCIONES))
        except Exception as e:
            print(f"(no se pudo escribir el evento {estado}: {type(e).__name__})")

    def log_dict(self, name, value):
        if self.mlflow is not None:
            try:
                self.mlflow.log_dict(json.loads(json.dumps(value, ensure_ascii=False, default=_operacion_json)), f"metadatos/{name}.json")
            except Exception:
                pass

    def log_table(self, name, frame):
        """Tabla agregada a MLflow como CSV (nunca con identificadores de tienda)."""
        if self.mlflow is None or frame is None or not len(frame):
            return
        if "tienda" in frame.columns:
            raise ValueError("Las evidencias de MLflow no admiten identificadores de tienda.")
        try:
            with tempfile.TemporaryDirectory(prefix="av5_tabla_") as carpeta:
                destino = Path(carpeta) / f"{name}.csv"
                frame.to_csv(destino, index=False)
                self.mlflow.log_artifact(str(destino), artifact_path="tablas")
        except Exception:
            pass

    def log_metrics(self, metricas):
        if self.mlflow is not None:
            try:
                self.mlflow.log_metrics({k: float(v) for k, v in metricas.items() if v is not None and np.isfinite(v)})
            except Exception:
                pass

    def guardar_punto_control(self, estado) -> bool:
        """Guarda el estado de la corrida (pickle) en el volumen: primero en un archivo temporal y luego lo renombra."""
        try:
            os.makedirs(self.ruta_control, exist_ok=True)
            final, temporal = f"{self.ruta_control}/estado.pkl", f"{self.ruta_control}/estado.tmp"
            with open(temporal, "wb") as f:
                pickle.dump(estado, f, protocol=4)
            try:
                os.replace(temporal, final)
            except OSError:
                with open(final, "wb") as f:
                    pickle.dump(estado, f, protocol=4)
            return True
        except Exception as e:
            print(f"(no se pudo guardar el punto de control: {type(e).__name__}: {str(e)[:100]})")
            return False

    def cargar_punto_control(self):
        ruta = f"{self.ruta_control}/estado.pkl"
        if not os.path.exists(ruta):
            return None
        with open(ruta, "rb") as f:
            return pickle.load(f)

    def publicar_tablas(self, tables: Mapping) -> dict:
        """Agrega cada tabla a mlops.av5_<nombre> con run_id y fecha. Es idempotente: si la corrida ya había publicado
        (p. ej. se reanudó), primero borra sus filas. Rechaza tablas con identificadores de tienda."""
        publicadas = {}
        for name, frame in tables.items():
            if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
                raise ValueError(f"Nombre de salida no válido: {name}")
            if "tienda" in frame.columns:
                raise ValueError(f"La salida {name} contiene identificadores de tienda.")
            destino = tabla("mlops", f"av5_{name}")
            datos = frame.copy()
            for c in datos.columns:
                if datos[c].dtype == object and datos[c].map(lambda v: isinstance(v, (dict, list, tuple))).any():
                    datos[c] = datos[c].map(lambda v: json.dumps(v, ensure_ascii=False, default=_operacion_json))
            datos.insert(0, "run_id", self.run_id)
            datos["fecha_publicacion"] = pd.Timestamp.now()
            limpio, esquema = _para_spark(datos)
            borrar_corrida(destino, self.run_id)
            (self.spark.createDataFrame(limpio, schema=esquema).write.format("delta").mode("append")
             .option("mergeSchema", "true").saveAsTable(destino))
            publicadas[name] = destino
            self.log_table(name, frame)
        self.manifest["tablas_publicadas"] = publicadas
        return publicadas

    def __exit__(self, kind, error, traceback):
        if kind is None:
            self.evento("FINALIZADA", {"tablas": list(self.manifest.get("tablas_publicadas", {}))})
            self.log_dict("manifiesto_final", self.manifest)
        else:
            self.evento("ERROR", {"tipo_error": kind.__name__, "mensaje": str(error)[:500]})
        if self.mlflow is not None:
            try:
                self.mlflow.end_run(status="FINISHED" if kind is None else "FAILED")
            except Exception:
                pass
        return False


def predecir_candidato(candidate: Mapping, frame: pd.DataFrame) -> pd.DataFrame:
    """Predicción del paquete final (pronóstico puntual + P10/P50/P90 + techo operativo) para filas con las variables de
    la Oro. Acepta categóricas como texto (las codifica con el vocabulario guardado) o ya codificadas."""
    frame = frame.copy()
    for c in ("origen", "fecha_objetivo"):
        frame[c] = pd.to_datetime(frame[c])
    frame["h"] = frame["h"].astype("int16")
    if not ((frame["fecha_objetivo"] - frame["origen"]).dt.days.to_numpy() == frame["h"].to_numpy()).all():
        raise ValueError("La fecha objetivo debe ser origen más horizonte.")
    for c in candidate["cat_features"]:
        if frame[c].dtype == object:
            frame[c] = frame[c].map(candidate["codigos"][c]).fillna(-1).astype("int32")
    if "h_bloque" not in frame:
        frame["h_bloque"] = np.where(frame["h"] <= 7, 1, 2)
    name, bases = candidate["modelo"], candidate["bases"]
    if name == "B0_Ingenuo":
        point = frame["ingenuo_estacional"].to_numpy(dtype=np.float64)
    elif name in bases:
        point = predict_bundle(bases[name], frame)
    elif name == "E1_Bagging":
        point = predict_bundle(candidate["homogeneos"][name], frame)
    elif name in candidate.get("homogeneos", {}):
        point = predict_bundle(candidate["homogeneos"][name], frame)
    else:
        meta = candidate["meta"]
        point = predict_ensembles({b: predict_bundle(bases[b], frame) for b in meta.bases}, meta, frame)[name]
    intervalo = candidate.get("intervalos")
    if intervalo is None:
        q = pd.DataFrame({"p10": point, "p50": point, "p90": point}, index=frame.index)
    elif intervalo["metodo"] == "Q_Conformal":
        q = predict_conformal(intervalo, frame, point)
    else:
        q = predict_quantiles(intervalo, frame, prediccion=point)
    salida = pd.DataFrame({"prediccion_puntual": point, "p10": q["p10"].to_numpy(), "p50": q["p50"].to_numpy(),
                           "p90": q["p90"].to_numpy()}, index=frame.index)
    salida["techo_operativo"] = salida[NIVEL_TECHO_OPERATIVO]
    return salida


def registrar_modelo_uc(registro, candidate, ejemplo, metricas) -> dict:
    """Empaqueta el modelo final como MLflow pyfunc, lo registra como nueva versión de NOMBRE_MODELO_UC y le pone el alias
    @challenger. El ejemplo de entrada es sintético (ceros), para no guardar datos de Bnext en el registro."""
    import mlflow
    import mlflow.pyfunc
    from mlflow.models import infer_signature

    with warnings.catch_warnings():
        # MLflow ≥ 2.20 avisa si predict no tiene anotaciones de tipo; aquí la firma se declara explícitamente abajo
        warnings.simplefilter("ignore", UserWarning)

        class ModeloFinalAvance5(mlflow.pyfunc.PythonModel):
            """Modelo final del Avance 5: recibe filas con las variables de la Oro y regresa pronóstico, P10, P50 y P90."""

            def __init__(self, paquete):
                self.paquete = paquete

            def predict(self, context, model_input, params=None):
                return predecir_candidato(self.paquete, model_input)

    salida = predecir_candidato(candidate, ejemplo)
    firma = infer_signature(ejemplo, salida)
    requisitos = [f"{lib}=={registro.entorno.get(clave)}" for lib, clave in
                  (("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("catboost", "catboost"), ("scikit-learn", "sklearn"),
                   ("pandas", "pandas"), ("numpy", "numpy"), ("cloudpickle", "cloudpickle"))
                  if registro.entorno.get(clave) not in (None, "NO DISPONIBLE", "sin_version_publicada")]
    modelo = ModeloFinalAvance5(candidate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            info = mlflow.pyfunc.log_model(name="modelo_final", python_model=modelo, signature=firma, pip_requirements=requisitos)
        except TypeError:
            info = mlflow.pyfunc.log_model(artifact_path="modelo_final", python_model=modelo, signature=firma, pip_requirements=requisitos)
    resultado = {"model_uri": info.model_uri, "registrado": False}
    try:
        mlflow.set_registry_uri("databricks-uc")
        version = mlflow.register_model(info.model_uri, NOMBRE_MODELO_UC)
        from mlflow import MlflowClient
        cliente = MlflowClient()
        cliente.set_registered_model_alias(NOMBRE_MODELO_UC, "challenger", version.version)
        for clave, valor in {"run_id_av5": registro.run_id, "modelo": candidate["modelo"], **metricas}.items():
            cliente.set_model_version_tag(NOMBRE_MODELO_UC, version.version, clave, str(valor))
        resultado.update(registrado=True, version=str(version.version), alias="challenger")
    except Exception as e:
        resultado["motivo"] = f"{type(e).__name__}: {str(e)[:200]}"
    return resultado
