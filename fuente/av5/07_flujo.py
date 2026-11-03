
# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.12 Orquestación: `ejecutar_modelado`
# MAGIC Recorre los orígenes en orden cronológico y guarda un punto de control al terminar cada uno:
# MAGIC
# MAGIC 1. **Ajuste (20-may).** Optuna para I1 e I4; después todas las bases con paro temprano en el bloque de ajuste: el
# MAGIC    número de árboles queda **congelado** para el resto de la corrida. Las predicciones del bloque entran a la bolsa.
# MAGIC 2. **Orígenes OOF** (si `n_origenes_oof` > 0): bases entrenadas antes del ajuste; sus predicciones también entran.
# MAGIC 3. **Cortes de desarrollo.** En cada corte: bases y homogéneos con objetivo ≤ origen; ensambles heterogéneos con la
# MAGIC    bolsa (bloques anteriores); predicción del universo evaluable por lotes de horizonte; sumas en los acumuladores;
# MAGIC    las filas de la muestra común de este corte entran a la bolsa para los cortes siguientes.
# MAGIC 4. **Selección.** Matriz de decisión con los cortes de desarrollo. El modelo elegido se registra **antes** de ver
# MAGIC    septiembre.
# MAGIC 5. **Prueba (datos no vistos).** Todo se entrena con datos al 31-ago y se congela; pronostica los dos bloques de
# MAGIC    septiembre sin reentrenar. Se evalúan también los intervalos (conformal calibrado con el bloque del 18 al 31 de
# MAGIC    agosto y LightGBM cuantílico).
# MAGIC 6. **Modelo final.** El elegido se reentrena con todos los datos, se recalibran sus intervalos con septiembre, se
# MAGIC    calcula la importancia de variables y se pronostican los 14 días siguientes al último dato.

# COMMAND ----------

def _flujo_train(pool: pd.DataFrame, origin) -> pd.DataFrame:
    """Filas de entrenamiento disponibles en `origin`: origen < origin y objetivo ≤ origin (nada del futuro)."""
    cutoff = pd.Timestamp(origin)
    result = pool.loc[(pool["origen"] < cutoff) & (pool["fecha_objetivo"] <= cutoff)]
    if result.empty or result["y"].sum() <= 0:
        raise ValueError(f"No existe entrenamiento elegible anterior a {cutoff.date()}.")
    return result


def _flujo_observed(fuente, origin) -> pd.DataFrame:
    """Bloque observado de un origen: la muestra común si es de evaluación; todas sus filas si es de entrenamiento (OOF)."""
    conjunto = fuente.conjunto_que_contiene_origen(origin)
    return fuente.cargar_bloque(origin, comun=conjunto == "evaluacion", conjunto=conjunto, fraccion_series=1.0)


def _flujo_wide(frame: pd.DataFrame, preds: Mapping, bloque: str) -> pd.DataFrame:
    """Filas para la bolsa de meta-entrenamiento: columnas fijas + una columna por modelo con su predicción."""
    filas = frame[[c for c in COLUMNAS_META_POOL if c != "h_bloque"]].copy()
    filas["h_bloque"] = np.where(frame["h"].to_numpy() <= 7, 1, 2).astype("int8")
    filas["bloque"] = bloque
    for name, values in preds.items():
        filas[name] = np.asarray(values, dtype=np.float32)
    return filas.reset_index(drop=True)


def _agregar_meta_pool(pool, nuevas):
    return nuevas if pool is None else pd.concat([pool, nuevas], ignore_index=True, sort=False)


def _flujo_costos(costs: pd.DataFrame, fase, origen, ledger: list):
    for row in costs.to_dict("records"):
        ledger.append({"fase": fase, "origen": str(pd.Timestamp(origen).date()), "segundos_prediccion": 0.0, **row})


def _theta_para(fuente, datos, origen):
    """Tabla Theta del origen para las series de `datos` (sólo si I6 está activo)."""
    if "I6_Theta" not in POINT_BASE_MODELS:
        return None
    series = pd.DataFrame({"tienda": datos["tienda"].astype(str).to_numpy(),
                           "categoria_id": fuente.decodificar("categoria_id", datos["categoria_id"].to_numpy())})
    return TablaTheta(theta_spark(series, origen), fuente.etiquetas["categoria_id"])


def _predecir_intervalos(intervalos, chunk, prediccion):
    salida = {}
    for metodo, bundle in (intervalos or {}).items():
        salida[metodo] = (predict_conformal(bundle, chunk, prediccion) if metodo == "Q_Conformal"
                          else predict_quantiles(bundle, chunk, prediccion=prediccion))
    return salida


def _evaluar_bloque(fuente, bloque, modelos, meta, acumuladores, estado, intervalos=None, modelo_final=None):
    """Predice un bloque (corte o prueba) por lotes de horizonte, agrega a los acumuladores y agrega la muestra común a la
    bolsa de meta-entrenamiento. Regresa los segundos de predicción por modelo."""
    segundos = {}
    lotes = [lote.tolist() for lote in np.array_split(np.arange(1, 15), fuente.config["lotes_horizonte"])]
    for lote in lotes:
        datos = fuente.cargar_bloque(bloque["origen"], horizontes=lote, conjunto="evaluacion")
        theta = _theta_para(fuente, datos, bloque["origen"])
        for h in lote:
            chunk = datos.loc[datos["h"].to_numpy() == h]
            if chunk.empty:
                continue
            chunk = chunk.assign(h_bloque=np.where(chunk["h"].to_numpy() <= 7, 1, 2).astype("int8"))
            preds, seg = predecir_todos(modelos["bases"], modelos["homogeneos"], meta, chunk, theta=theta)
            quantiles = _predecir_intervalos(intervalos, chunk, preds[modelo_final]) if intervalos else None
            acumuladores["ampliada"].agregar(chunk, preds, quantiles, corte=bloque["corte"])
            comun = chunk["en_muestra_eval"].to_numpy() == 1
            if comun.any():
                sub = chunk.loc[comun]
                preds_c = {m: v[comun] for m, v in preds.items()}
                q_c = {k: q.loc[comun] for k, q in quantiles.items()} if quantiles else None
                acumuladores["comun"].agregar(sub, preds_c, q_c, corte=bloque["corte"])
                estado["meta_pool"] = _agregar_meta_pool(estado["meta_pool"], _flujo_wide(sub, preds_c, bloque["corte"]))
            for m, s in seg.items():
                segundos[m] = segundos.get(m, 0.0) + s
        del datos, theta
        gc.collect()
    return segundos


def _costos_por_modelo(nombre_corte, costos_bases, costos_homog, segundos_meta, segundos_pred):
    """Tiempos por modelo en un corte. Un ensamble cuenta el entrenamiento de todo lo que necesita (sus bases más el
    meta-modelo); E1 cuenta también el de I1, su primer miembro."""
    entrenamiento = {"B0_Ingenuo": 0.0}
    for row in costos_bases.to_dict("records"):
        entrenamiento[row["modelo"]] = float(row["segundos_entrenamiento"])
    for row in costos_homog.to_dict("records"):
        extra = entrenamiento.get("I1_LightGBM_Tweedie", 0.0) if row["modelo"] == "E1_Bagging" else 0.0
        entrenamiento[row["modelo"]] = float(row["segundos_entrenamiento"]) + extra
    if "I6_Theta" in POINT_BASE_MODELS:
        entrenamiento["I6_Theta"] = 0.0
    bases_total = sum(entrenamiento.get(b, 0.0) for b in POINT_BASE_MODELS)
    for name in (*POINT_HETEROGENEOUS_ACTIVOS, *MODELOS_DIAGNOSTICO):
        entrenamiento[name] = bases_total + segundos_meta
    return pd.DataFrame([{"modelo": m, "corte": nombre_corte, "segundos_entrenamiento": entrenamiento.get(m, 0.0),
                          "segundos_prediccion": segundos_pred.get(m, 0.0)} for m in AGG_MODELS])


def _nuevo_estado(fuente):
    """Estado de la corrida (lo que se guarda en cada punto de control)."""
    etq = fuente.etiquetas
    return {"version_estado": 1, "completados": [], "parametros": {}, "meta_pool": None,
            "acumuladores": {"desarrollo": {"ampliada": AcumuladorAvance5("desarrollo_ampliada", AGG_MODELS, etq),
                                            "comun": AcumuladorAvance5("desarrollo_comun", AGG_MODELS, etq)},
                             "prueba": {"ampliada": AcumuladorAvance5("prueba_ampliada", AGG_MODELS, etq),
                                        "comun": AcumuladorAvance5("prueba_comun", AGG_MODELS, etq)}},
            "costos": [], "auditoria": [], "ajuste": {}, "pesos": [], "decision": None, "prueba": {}, "final": {},
            "modelos_activos": list(AGG_MODELS)}


def _auditar(estado, etapa, origen, train=None, bloque=None):
    """Evidencia de que no hay fuga: la fecha objetivo máxima del entrenamiento nunca pasa del origen."""
    fila = {"etapa": etapa, "origen": str(pd.Timestamp(origen).date())}
    if train is not None:
        fila.update(filas_train=int(len(train)), max_objetivo_train=str(train["fecha_objetivo"].max().date()),
                    sin_fuga=bool(train["fecha_objetivo"].max() <= pd.Timestamp(origen)))
        if not fila["sin_fuga"]:
            raise RuntimeError(f"Fuga temporal en {etapa}: hay objetivos posteriores al origen.")
    if bloque is not None:
        fila.update(min_objetivo_evaluado=str(bloque["desde"]), max_objetivo_evaluado=str(bloque["hasta"]))
    estado["auditoria"].append(fila)


def ejecutar_modelado(fuente, registro, n_jobs=None, estado=None) -> dict:
    """Ejecuta el protocolo completo (ajuste → OOF → cortes → selección → prueba → modelo final) y regresa el estado.
    Con `estado` (punto de control) continúa en el primer paso no completado."""
    global PARAMETROS_VIGENTES
    inicio_total = time.perf_counter()
    jobs = runtime_threads(n_jobs)
    spec = {"numeric": list(fuente.num_features), "categorical": list(fuente.cat_features)}
    estado = estado or _nuevo_estado(fuente)
    if estado["modelos_activos"] != list(AGG_MODELS):
        raise RuntimeError("El punto de control se creó con otros modelos activos; revisa INCLUIR o inicia una corrida nueva.")
    if estado["parametros"]:
        PARAMETROS_VIGENTES = deepcopy(estado["parametros"])
    pool = fuente.cargar_entrenamiento()
    registro.evento("entrenamiento_cargado", {"filas": len(pool), "hilos": jobs, "variables": len(spec["numeric"]) + len(spec["categorical"])})
    completados = estado["completados"]

    def cerrar_paso(paso, info=None):
        completados.append(paso)
        estado["parametros"] = deepcopy(PARAMETROS_VIGENTES)
        registro.evento("paso_completado", {"paso": paso, **(info or {})})
        registro.guardar_punto_control(estado)

    # ---------------------------------------------------------------- 1. ajuste de hiperparámetros
    A = fuente.ajuste_origen
    if "ajuste" not in completados:
        subtitulo(f"1. Ajuste · origen {A}")
        train_A = _flujo_train(pool, A)
        _auditar(estado, "ajuste", A, train_A)
        tune_eval = _flujo_observed(fuente, A)
        mejores, pruebas, resumen, importancias = tune_boosting(train_A, tune_eval, spec, n_jobs=jobs)
        PARAMETROS_VIGENTES.update(mejores)
        if "I5_LightGBM_Log1p" in POINT_BASE_MODELS:
            PARAMETROS_VIGENTES["I5_LightGBM_Log1p"] = deepcopy(S6_POINT_PARAMETERS["I5_LightGBM_Log1p"])
        bases_A, costos_A = fit_individuals(train_A, spec, n_jobs=jobs, validacion=tune_eval)
        for name, bundle in bases_A.items():          # número de árboles congelado por paro temprano en el ajuste
            if "mejor_iteracion" in bundle.extras:
                PARAMETROS_VIGENTES.setdefault(name, deepcopy(bundle.params))
                PARAMETROS_VIGENTES[name]["n_estimators"] = int(bundle.extras["mejor_iteracion"])
        theta = _theta_para(fuente, tune_eval, A)
        preds = {b: (theta.predict(tune_eval) if b == "I6_Theta" else predict_bundle(bases_A[b], tune_eval)) for b in POINT_BASE_MODELS}
        estado["meta_pool"] = _agregar_meta_pool(estado["meta_pool"], _flujo_wide(tune_eval, preds, "ajuste"))
        estado["ajuste"] = {"pruebas": pruebas, "resumen": resumen, "importancias": importancias,
                            "wmape_bases_ajuste": {b: motor_metrics(tune_eval["y"], p)["wmape"] for b, p in preds.items()}}
        _flujo_costos(costos_A, "bases_ajuste", A, estado["costos"])
        mostrar(resumen, "Ajuste fino (Optuna) en el bloque de ajuste",
                formatos={"pruebas": "int", "minutos": "dec1", "wmape_inicial": "dec4", "wmape_mejor": "dec4",
                          "mejora_relativa": "pct", "arboles_prueba": "int"})
        del train_A, tune_eval, bases_A
        gc.collect()
        cerrar_paso("ajuste", {"parametros": {k: v.get("n_estimators") for k, v in PARAMETROS_VIGENTES.items() if isinstance(v, dict)}})

    # ---------------------------------------------------------------- 2. orígenes OOF
    for o in fuente.oof_origenes:
        paso = f"oof_{o}"
        if paso in completados:
            continue
        subtitulo(f"2. Origen OOF {o}")
        train_o = _flujo_train(pool, o)
        _auditar(estado, "oof", o, train_o)
        observado = _flujo_observed(fuente, o)
        bases_o, costos_o = fit_individuals(train_o, spec, n_jobs=jobs)
        theta = _theta_para(fuente, observado, o)
        preds = {b: (theta.predict(observado) if b == "I6_Theta" else predict_bundle(bases_o[b], observado)) for b in POINT_BASE_MODELS}
        estado["meta_pool"] = _agregar_meta_pool(estado["meta_pool"], _flujo_wide(observado, preds, f"oof_{o}"))
        _flujo_costos(costos_o, "bases_oof", o, estado["costos"])
        del train_o, observado, bases_o
        gc.collect()
        cerrar_paso(paso)

    # ---------------------------------------------------------------- 3. cortes de desarrollo
    acum_dev = estado["acumuladores"]["desarrollo"]
    for bloque in fuente.cortes:
        if bloque["corte"] in completados:
            continue
        origen = bloque["origen"]
        subtitulo(f"3. {bloque['corte']} · origen {origen} · pronostica {bloque['desde']} a {bloque['hasta']}")
        with Cronometro() as reloj_corte:
            train = _flujo_train(pool, origen)
            _auditar(estado, bloque["corte"], origen, train, bloque)
            bases, costos_b = fit_individuals(train, spec, n_jobs=jobs)
            homogeneos, costos_h = fit_homogeneous(train, spec, bases, n_jobs=jobs)
            del train
            gc.collect()
            with Cronometro() as reloj_meta:
                meta, pesos = fit_meta(estado["meta_pool"], origen, n_jobs=jobs)
            estado["pesos"].append(pesos.assign(corte=bloque["corte"]))
            segundos_pred = _evaluar_bloque(fuente, bloque, {"bases": bases, "homogeneos": homogeneos}, meta, acum_dev, estado)
            costos = _costos_por_modelo(bloque["corte"], costos_b, costos_h, reloj_meta.segundos, segundos_pred)
            for acum in acum_dev.values():
                acum.registrar_costos(costos)
            _flujo_costos(costos_b, "bases_corte", origen, estado["costos"])
            _flujo_costos(costos_h, "homogeneos_corte", origen, estado["costos"])
            estado["costos"].append({"fase": "meta_corte", "origen": origen, "modelo": "meta_heterogeneos",
                                     "segundos_entrenamiento": reloj_meta.segundos, "segundos_prediccion": 0.0})
        parcial = acum_dev["ampliada"].tablas()["por_corte"]
        fila = parcial.loc[parcial.corte.eq(bloque["corte"])].set_index("modelo")["wmape"]
        print(f"   {bloque['corte']} en {reloj_corte.segundos / 60:.1f} min · WMAPE: " +
              " · ".join(f"{m.split('_')[0]} {fila.get(m, np.nan):.4f}" for m in AGG_MODELS if m in fila.index))
        del bases, homogeneos, meta
        gc.collect()
        cerrar_paso(bloque["corte"], {"minutos": reloj_corte.segundos / 60})

    # ---------------------------------------------------------------- 4. selección (antes de ver septiembre)
    if "seleccion" not in completados:
        subtitulo("4. Selección del modelo final con los cortes de desarrollo")
        tablas_dev = acum_dev["ampliada"].tablas()
        DEC, puntos, elegido, argumento = matriz_decision(tablas_dev)
        estado["decision"] = {"matriz": DEC, "puntos": puntos, "modelo_final": elegido, "argumento": argumento,
                              "fecha_utc": datetime.now(timezone.utc).isoformat()}
        print(f"   Modelo elegido: {elegido}. {argumento}")
        cerrar_paso("seleccion", {"modelo_final": elegido, "nota": "elegido antes de evaluar los bloques de prueba"})
    elegido = estado["decision"]["modelo_final"]

    # ---------------------------------------------------------------- 5. prueba en datos no vistos
    acum_prueba = estado["acumuladores"]["prueba"]
    if fuente.pruebas and "prueba" not in completados:
        P1 = fuente.pruebas[0]["origen"]
        subtitulo(f"5. Prueba no vista · modelos congelados con datos al {P1}")
        train = _flujo_train(pool, P1)
        _auditar(estado, "prueba_entrenamiento", P1, train)
        bases, costos_b = fit_individuals(train, spec, n_jobs=jobs)
        homogeneos, costos_h = fit_homogeneous(train, spec, bases, n_jobs=jobs)
        meta, pesos = fit_meta(estado["meta_pool"], P1, n_jobs=jobs)
        estado["pesos"].append(pesos.assign(corte="prueba"))
        # calibración: el bloque más reciente de la bolsa que ya terminó en P1 (normalmente corte_4, 18–31 de agosto)
        bolsa = estado["meta_pool"]
        origen_cal = pd.Timestamp(bolsa.loc[(bolsa["fecha_objetivo"] <= pd.Timestamp(P1)) & bolsa["bloque"].str.startswith("corte_"),
                                            "origen"].max()).date().isoformat()
        calibracion = bolsa.loc[bolsa["origen"] == pd.Timestamp(origen_cal)]
        intervalos = {"Q_Conformal": fit_conformal(calibracion, elegido, calibration_end=P1)}
        if INCLUIR_Q_LIGHTGBM:
            train_q = _flujo_train(pool, origen_cal)
            bloque_cal = fuente.cargar_bloque(origen_cal, comun=True, conjunto="evaluacion", fraccion_series=1.0)
            intervalos["Q_LightGBM_CQR"] = fit_quantiles(train_q, bloque_cal, spec, cutoff_train=origen_cal, calibration_end=P1, n_jobs=jobs)
            _auditar(estado, "cuantiles_entrenamiento", origen_cal, train_q)
            _flujo_costos(intervalos["Q_LightGBM_CQR"]["costs"], "cuantiles_prueba", origen_cal, estado["costos"])
            del train_q, bloque_cal
        del train
        gc.collect()
        for bloque in fuente.pruebas:
            _auditar(estado, bloque["corte"], P1, bloque=bloque)
            segundos_pred = _evaluar_bloque(fuente, bloque, {"bases": bases, "homogeneos": homogeneos}, meta, acum_prueba,
                                            estado, intervalos, elegido)
            costos = _costos_por_modelo(bloque["corte"], costos_b, costos_h, 0.0, segundos_pred)
            for acum in acum_prueba.values():
                acum.registrar_costos(costos)
        estado["prueba"] = {"origen_entrenamiento": P1, "origen_calibracion": origen_cal,
                            "intervalos_auditoria": {m: {k: v for k, v in b.items() if k in ("calibration_end", "cutoff_train", "n_calibracion", "audit")}
                                                     for m, b in intervalos.items()}}
        del bases, homogeneos, meta, intervalos
        gc.collect()
        cerrar_paso("prueba")
    elif not fuente.pruebas:
        aviso("La Oro no tiene bloques de prueba: no hay evaluación en datos no vistos (carga septiembre y reconstruye 02 → 20).")

    # ---------------------------------------------------------------- 6. modelo final, intervalos, importancia y pronóstico
    if REENTRENAR_FINAL and "final" not in completados:
        F_ = fuente.fecha_maxima_objetivo
        subtitulo(f"6. Modelo final: {elegido} con todos los datos hasta {F_}")
        train = _flujo_train(pool, F_)
        _auditar(estado, "final", F_, train)
        bases, homogeneos, costos_b = _entrenar_final(elegido, train, spec, jobs)
        meta = None
        if elegido in POINT_HETEROGENEOUS_ACTIVOS or elegido in MODELOS_DIAGNOSTICO:
            meta, pesos = fit_meta(estado["meta_pool"], F_, n_jobs=jobs)
            estado["pesos"].append(pesos.assign(corte="final"))
        metodo = elegir_metodo_intervalos(acum_prueba)
        ultimo = estado["meta_pool"]["origen"].max()
        recientes = estado["meta_pool"].loc[estado["meta_pool"]["origen"] == ultimo]
        if metodo == "Q_LightGBM_CQR":
            origen_cal = pd.Timestamp(ultimo).date().isoformat()
            bloque_cal = fuente.cargar_bloque(origen_cal, comun=True, conjunto="evaluacion", fraccion_series=1.0)
            intervalo = fit_quantiles(_flujo_train(pool, origen_cal), bloque_cal, spec, cutoff_train=origen_cal,
                                      calibration_end=F_, n_jobs=jobs)
        else:
            intervalo = fit_conformal(recientes, elegido, calibration_end=F_)
        candidate = {"modelo": elegido, "bases": bases, "homogeneos": homogeneos, "meta": meta, "intervalos": intervalo,
                     "metodo_intervalos": metodo, "cat_features": list(fuente.cat_features), "codigos": fuente.codigos,
                     "feature_spec": spec, "max_target_train": str(train["fecha_objetivo"].max().date()),
                     "snapshot": {"tabla": fuente.tabla, "version": fuente.version}, "run_id": registro.run_id}
        _flujo_costos(costos_b, "reentrenamiento_final", F_, estado["costos"]) if len(costos_b) else None
        del train
        gc.collect()
        estado["final"] = {"candidate": candidate, "metodo_intervalos": metodo,
                           "importancia": importancia_variables(candidate, pool, spec)}
        if fuente.origen_pronostico:
            datos = fuente.cargar_bloque(fuente.origen_pronostico, conjunto="pronostico", fraccion_series=1.0 if not MODO_RAPIDO else None)
            datos = datos.assign(h_bloque=np.where(datos["h"].to_numpy() <= 7, 1, 2).astype("int8"))
            if "I6_Theta" in (getattr(meta, "bases", ()) or ()) or elegido == "I6_Theta":
                candidate["bases"]["I6_Theta"] = ModelBundle("I6_Theta", None, _theta_para(fuente, datos, fuente.origen_pronostico),
                                                             np.inf, 1.0, 0, {})
                candidate["requiere_spark"] = True
            salida = predecir_candidato(candidate, datos)
            estado["final"]["pronostico"] = pd.concat([datos[["tienda", "origen", "fecha_objetivo", "h"]].reset_index(drop=True),
                                                       pd.DataFrame({"categoria_id": fuente.decodificar("categoria_id", datos["categoria_id"].to_numpy()),
                                                                     "tipo_categoria": fuente.decodificar("tipo_categoria", datos["tipo_categoria"].to_numpy()),
                                                                     "region": fuente.decodificar("region", datos["region"].to_numpy())}),
                                                       salida.reset_index(drop=True)], axis=1)
            del datos
        cerrar_paso("final")
    estado["segundos_modelado"] = estado.get("segundos_modelado", 0.0) + (time.perf_counter() - inicio_total)
    return estado


def _entrenar_final(elegido, train, spec, jobs):
    """Entrena sólo lo que necesita el modelo elegido: la base misma; I1 y sus miembros (E1); el homogéneo (E2, E7);
    o todas las bases tabulares (ensambles heterogéneos)."""
    bases, homogeneos, costos = {}, {}, []
    if elegido in POINT_HETEROGENEOUS_ACTIVOS or elegido in MODELOS_DIAGNOSTICO:
        necesarias = [b for b in POINT_BASE_MODELS if b != "I6_Theta"]
    elif elegido == "E1_Bagging":
        necesarias = ["I1_LightGBM_Tweedie"]
    elif elegido in POINT_BASE_MODELS and elegido != "I6_Theta":
        necesarias = [elegido]
    else:
        necesarias = []
    for name in necesarias:
        bases[name], costo = fit_model(name, train, spec, n_jobs=jobs)
        costos.append(costo)
    if elegido == "E1_Bagging":
        h, c = fit_homogeneous(train, spec, bases, n_jobs=jobs)
        homogeneos["E1_Bagging"] = h["E1_Bagging"]
        costos += c.to_dict("records")
    elif elegido in ("E2_AdaBoost", "E7_Bosque_Aleatorio"):
        homogeneos[elegido], costo = fit_model(elegido, train, spec, n_jobs=jobs)
        costos.append(costo)
    return bases, homogeneos, pd.DataFrame(costos)


def elegir_metodo_intervalos(acum_prueba) -> str:
    """Método de intervalos del modelo final: METODO_INTERVALOS, o en 'automatico' el de menor pinball total en la prueba
    entre los que dejan la cobertura global dentro de la banda (si ninguno, el más cercano al 80 %)."""
    if METODO_INTERVALOS != "automatico":
        return METODO_INTERVALOS
    try:
        q = acum_prueba["ampliada"].tablas().get("cuantiles")
        g = q.loc[q.ambito.eq("global")].copy()
        g["pinball_total"] = g["pinball_p10"] + g["pinball_p50"] + g["pinball_p90"]
        en_banda = g.loc[g.cumple_banda]
        elegido = (en_banda.sort_values("pinball_total") if len(en_banda)
                   else g.assign(_d=(g.cobertura - NOMINAL_COVERAGE).abs()).sort_values("_d")).iloc[0]["metodo"]
        return str(elegido)
    except Exception:
        return "Q_Conformal"


def importancia_variables(candidate, pool, spec, filas=20_000):
    """Importancia de variables del modelo de árboles principal del modelo final: media de |SHAP| en una muestra (o la
    importancia por ganancia si SHAP no está disponible). En un ensamble se usa la base con más peso en el stacking."""
    principal = candidate["modelo"]
    meta = candidate.get("meta")
    if meta is not None:
        pesos = (meta.pesos_stacking[1] + meta.pesos_stacking[2]) / 2
        arboles = [b for b in meta.bases if b in ("I1_LightGBM_Tweedie", "I4_XGBoost_Tweedie", "I5_LightGBM_Log1p", "I2_CatBoost_Tweedie")]
        principal = max(arboles, key=lambda b: pesos[meta.bases.index(b)]) if arboles else "I1_LightGBM_Tweedie"
    elif principal == "E1_Bagging":
        principal = "I1_LightGBM_Tweedie"
    bundle = candidate["bases"].get(principal) or candidate["homogeneos"].get(principal)
    if bundle is None or principal in ("I3_ElasticNet", "I6_Theta", "B0_Ingenuo", "E2_AdaBoost"):
        return pd.DataFrame()
    muestra = _motor_training_frame(pool, filas, seed=SEMILLA + 7)
    try:
        import shap
        X = bundle.encoder.transform_catboost(muestra) if principal == "I2_CatBoost_Tweedie" else bundle.encoder.transform(muestra)
        valores = shap.TreeExplainer(bundle.estimator).shap_values(X)
        importancia, metodo = np.abs(np.asarray(valores)).mean(axis=0), "media de |SHAP|"
    except Exception:
        importancia = np.asarray(getattr(bundle.estimator, "feature_importances_", np.ones(len(bundle.encoder.feature_names))), dtype=float)
        metodo = "importancia nativa (ganancia o divisiones)"
    tabla_imp = pd.DataFrame({"variable": bundle.encoder.feature_names, "importancia": importancia})
    tabla_imp["importancia"] = tabla_imp["importancia"] / max(tabla_imp["importancia"].sum(), 1e-12)
    tabla_imp["modelo"], tabla_imp["metodo"] = principal, metodo
    tabla_imp["grupo"] = tabla_imp["variable"].map(grupo_variable)
    tabla_imp["descripcion"] = tabla_imp["variable"].map(DESCRIPCION_VARIABLES).fillna("")
    return tabla_imp.sort_values("importancia", ascending=False).reset_index(drop=True)


def grupo_variable(nombre):
    """Grupo de una variable para resumir la importancia."""
    if nombre.startswith("f_"):
        return "promociones futuras"
    if nombre.startswith("s_") or nombre in ("dias_con_venta_hist", "dias_historia"):
        return "historia de la serie"
    if nombre.startswith(("t_", "zc_", "pc_")):
        return "contexto de tienda, zona y país"
    if nombre in ("dow", "fin_de_semana", "dia_mes", "semana_mes", "quincena", "es_evento", "tipo_evento", "dias_a_evento",
                  "dias_desde_evento", "en_mundial", "h", "h_bloque"):
        return "calendario y horizonte"
    return "atributos de tienda y categoría"


def compilar_resultados(estado, fuente, registro) -> dict:
    """Arma las tablas que se publican (todas agregadas, sin identificadores) a partir del estado de la corrida."""
    dev = estado["acumuladores"]["desarrollo"]["ampliada"].tablas()
    dev_c = estado["acumuladores"]["desarrollo"]["comun"].tablas()
    hay_prueba = bool(estado["acumuladores"]["prueba"]["ampliada"]._sumas)
    pru = estado["acumuladores"]["prueba"]["ampliada"].tablas() if hay_prueba else {}
    pru_c = estado["acumuladores"]["prueba"]["comun"].tablas() if hay_prueba else {}
    elegido = estado["decision"]["modelo_final"]
    T = {"comparativa": dev["comparativa"].assign(fase="desarrollo", evaluacion="ampliada"),
         "comparativa_comun": dev_c["comparativa"].assign(fase="desarrollo", evaluacion="comun")}
    for nombre in ("por_corte", "por_horizonte", "diario", "segmentos", "calibracion", "residuos"):
        partes = [dev.get(nombre, pd.DataFrame()).assign(fase="desarrollo")]
        if hay_prueba:
            partes.append(pru.get(nombre, pd.DataFrame()).assign(fase="prueba"))
        T[nombre] = pd.concat(partes, ignore_index=True)
    if hay_prueba:
        T["comparativa_prueba"] = pru["comparativa"].assign(fase="prueba", evaluacion="ampliada")
        T["comparativa_prueba_comun"] = pru_c["comparativa"].assign(fase="prueba", evaluacion="comun")
        T["cuantiles"] = pru.get("cuantiles", pd.DataFrame()).assign(fase="prueba")
    # Diebold–Mariano del elegido contra el ingenuo y contra el mejor individual, por bloque
    mejor_ind = dev["comparativa"].loc[dev["comparativa"].tipo.eq("individual")].iloc[0]["modelo"]
    dms = []
    for fase, tablas_f in (("desarrollo", dev), ("prueba", pru)):
        if not tablas_f:
            continue
        for ref in ("B0_Ingenuo", mejor_ind):
            if ref != elegido:
                dms.append(dm_por_corte(tablas_f["diario"], elegido, referencia=ref).assign(fase=fase))
    T["dm"] = pd.concat(dms, ignore_index=True) if dms else pd.DataFrame()
    dec = estado["decision"]
    T["decision"] = dec["matriz"].assign(modelo_final=elegido, argumento=dec["argumento"])
    T["decision_puntos"] = dec["puntos"]
    T["pesos"] = pd.concat(estado["pesos"], ignore_index=True) if estado["pesos"] else pd.DataFrame()
    T["ajuste"] = estado["ajuste"].get("pruebas", pd.DataFrame())
    T["ajuste_resumen"] = estado["ajuste"].get("resumen", pd.DataFrame())
    T["ajuste_importancia"] = estado["ajuste"].get("importancias", pd.DataFrame())
    T["costos"] = pd.DataFrame(estado["costos"])
    T["auditoria"] = pd.DataFrame(estado["auditoria"])
    T["parametros"] = pd.DataFrame([{"modelo": m, "parametros": json.dumps(p, default=_operacion_json)}
                                    for m, p in PARAMETROS_VIGENTES.items()])
    T["importancia"] = estado.get("final", {}).get("importancia", pd.DataFrame())
    pool = estado["meta_pool"]
    if pool is not None:
        filas = pool.loc[pool["bloque"].str.startswith("corte_")]
        bases = [b for b in POINT_BASE_MODELS if b in filas.columns]
        if len(filas) and len(bases) > 1:
            errores = pd.DataFrame({b: filas["y"].to_numpy() - filas[b].to_numpy() for b in bases}).corr()
            T["correlacion_errores"] = errores.reset_index().rename(columns={"index": "base"})
    return T
