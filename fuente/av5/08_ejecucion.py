
# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.13 Compuerta, salida operativa, monitoreo y lectura de resultados

# COMMAND ----------

def preparar_compuerta(T: Mapping, estado, fuente) -> dict:
    """Reúne la evidencia de la corrida (cortes, prueba, intervalos, contratos y negocio) y evalúa la compuerta."""
    elegido = estado["decision"]["modelo_final"]
    pc = T["por_corte"]
    dev = pc.loc[pc.fase.eq("desarrollo") & pc.modelo.eq(elegido), ["corte", "wmape", "wmape_ingenuo", "sesgo_pct"]]
    dm = T["dm"]
    if len(dm):
        dm_ing = dm.loc[dm.fase.eq("desarrollo") & dm.referencia.eq("B0_Ingenuo"), ["corte", "dm_pvalue"]]
        dev = dev.merge(dm_ing, on="corte", how="left")
    else:
        dev = dev.assign(dm_pvalue=np.nan)
    prueba = pc.loc[pc.fase.eq("prueba") & pc.modelo.eq(elegido), ["corte", "wmape", "wmape_ingenuo", "sesgo_pct"]]
    metodo = estado.get("final", {}).get("metodo_intervalos") or "Q_Conformal"
    q = T.get("cuantiles", pd.DataFrame())
    q = q.loc[q.metodo.eq(metodo)] if len(q) else q
    contratos = fuente.validar_contratos(duplicados=False) if fuente._contratos is None else fuente._contratos
    controles = [k for k in contratos if k != "filas_snapshot"]
    calidad = {"critical_total": len(controles), "critical_passed": sum(int(contratos[k] == 0) for k in controles),
               "snapshot_verified": True, **evidencia_calidad_upstream()}
    comp = T["comparativa"].set_index("modelo")
    alcance = {"wmape_global": float(comp.loc[elegido, "wmape_media_cortes"]),
               "wmape_mercancia": float(comp.loc[elegido, "wmape_mercancia"]) if pd.notna(comp.loc[elegido].get("wmape_mercancia")) else None}
    compuerta = evaluate_gate(dev, q, calidad, EVIDENCIA_NEGOCIO, test_metrics=prueba, scope_metrics=alcance)
    compuerta.update(modelo=elegido, metodo_intervalos=metodo, evidencia_calidad=calidad, alcance=alcance)
    return compuerta


def guardar_pronostico(pronostico: pd.DataFrame, fuente, registro, elegido):
    """Escribe el pronóstico operativo en TABLA_PRONOSTICO reemplazando sólo el origen pronosticado (las corridas de otros
    orígenes se conservan). Es la única salida con tienda y categoría: vive en el catálogo privado."""
    datos = pronostico.copy()
    datos["modelo"], datos["run_id"], datos["version_oro"] = elegido, registro.run_id, str(fuente.version)
    datos["fecha_generacion"] = pd.Timestamp.now()
    for c in ("origen", "fecha_objetivo"):
        datos[c] = pd.to_datetime(datos[c]).dt.date
    limpio, esquema = _para_spark(datos)
    sdf = spark.createDataFrame(limpio, schema=esquema)
    origen = str(pd.Timestamp(pronostico["origen"].iloc[0]).date())
    escritor = sdf.write.format("delta").mode("overwrite")
    if existe(TABLA_PRONOSTICO):
        escritor = escritor.option("replaceWhere", f"origen = DATE'{origen}'").option("mergeSchema", "true")
    else:
        escritor = escritor.option("overwriteSchema", "true")
    escritor.saveAsTable(TABLA_PRONOSTICO)
    return TABLA_PRONOSTICO


def monitoreo_desde_prueba(acum_ampliada, elegido, metodo, wmape_desarrollo) -> pd.DataFrame:
    """Tabla de monitoreo por semana × región × tipo con el modelo elegido en la prueba: WMAPE, sesgo, cobertura y
    alertas. Es la misma tabla que un Job semanal llenaría en operación."""
    base = pd.DataFrame([dict(zip(["modelo", *AGG_DIMENSIONS], k), **dict(zip(AGG_POINT_SUMS, v)))
                         for k, v in acum_ampliada._sumas.items() if k[0] == elegido])
    if base.empty:
        return pd.DataFrame()
    base["semana"] = pd.to_datetime(base["fecha_objetivo"]).dt.to_period("W-SUN").dt.start_time.dt.date
    m = _agg_point_metrics(_agg_rollup(base, ["semana", "region", "tipo_categoria"], AGG_POINT_SUMS))
    if acum_ampliada._cuantiles:
        qb = pd.DataFrame([dict(zip(["metodo", *AGG_DIMENSIONS], k), **dict(zip(AGG_QUANTILE_SUMS, v)))
                           for k, v in acum_ampliada._cuantiles.items() if k[0] == metodo])
        if len(qb):
            qb["semana"] = pd.to_datetime(qb["fecha_objetivo"]).dt.to_period("W-SUN").dt.start_time.dt.date
            q = _agg_quantile_metrics(_agg_rollup(qb, ["semana", "region", "tipo_categoria"], AGG_QUANTILE_SUMS))
            m = m.merge(q[["semana", "region", "tipo_categoria", "cobertura"]], on=["semana", "region", "tipo_categoria"], how="left")
    m["alerta_error"] = m["wmape"] > wmape_desarrollo + 0.05
    m["alerta_sesgo"] = m["sesgo_pct"].abs() > 5
    m["alerta_cobertura"] = ~m.get("cobertura", pd.Series(np.nan, index=m.index)).between(*BANDA_COBERTURA)
    m["modelo"], m["fase"] = elegido, "prueba"
    return m[["semana", "region", "tipo_categoria", "modelo", "fase", "n_filas", "ventas_reales", "wmape", "sesgo_pct",
              *(["cobertura"] if "cobertura" in m else []), "alerta_error", "alerta_sesgo", "alerta_cobertura"]]


NOMBRES_TABLAS = ("comparativa", "comparativa_comun", "comparativa_prueba", "comparativa_prueba_comun", "por_corte",
                  "por_horizonte", "diario", "segmentos", "calibracion", "residuos", "cuantiles", "dm", "decision",
                  "decision_puntos", "pesos", "ajuste", "ajuste_resumen", "ajuste_importancia", "costos", "auditoria",
                  "parametros", "importancia", "correlacion_errores", "compuerta", "resumen_corrida")


def cargar_corrida(run_id=None) -> dict:
    """Lee de Delta las tablas publicadas de una corrida (por defecto, la última que terminó)."""
    if run_id is None:
        if not existe(TABLA_EJECUCIONES):
            raise RuntimeError("No hay corridas registradas: ejecuta la sección 4.")
        fin = spark.table(TABLA_EJECUCIONES).where(F.col("estado") == "FINALIZADA").orderBy(F.desc("fecha_utc")).limit(1).collect()
        if not fin:
            raise RuntimeError("No hay corridas terminadas en mlops.av5_ejecuciones.")
        run_id = fin[0]["run_id"]
    tablas = {"run_id": run_id}
    for nombre in NOMBRES_TABLAS:
        destino = tabla("mlops", f"av5_{nombre}")
        if existe(destino):
            df = spark.table(destino).where(F.col("run_id") == run_id).drop("run_id", "fecha_publicacion").toPandas()
            if len(df):
                tablas[nombre] = df
    return tablas

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Entorno y snapshot de la fuente
# MAGIC Se comprueban las bibliotecas (los modelos cuya biblioteca falte se omiten, con aviso), se fijan los modelos activos,
# MAGIC se lee la tabla Oro en una versión fija y se validan sus contratos (fechas, objetivo, llaves, conjuntos, referencia,
# MAGIC infinitos y duplicados). Sólo se imprimen conteos y fechas, nunca filas.

# COMMAND ----------

titulo("3. ENTORNO Y SNAPSHOT DE LA FUENTE")
ENTORNO_AV5, FALTANTES_AV5 = verificar_entorno_formal(spark)
mostrar(pd.DataFrame([{"biblioteca": k, "version": v} for k, v in ENTORNO_AV5.items()]), "Bibliotecas efectivas")


def _disponible(lib):
    return ENTORNO_AV5.get(lib, "NO DISPONIBLE") != "NO DISPONIBLE"


if not _disponible("lightgbm"):
    raise RuntimeError("LightGBM es indispensable (I1, E1, E6, E7 y cuantiles). Revisa la celda %pip install lightgbm.")
POINT_BASE_MODELS = tuple(m for m in CANDIDATOS_BASE if INCLUIR.get(m, True)
                          and (m != "I2_CatBoost_Tweedie" or _disponible("catboost"))
                          and (m != "I4_XGBoost_Tweedie" or _disponible("xgboost")))
POINT_HOMOGENEOUS = tuple(m for m in CANDIDATOS_HOMOGENEOS if INCLUIR.get(m, True))
POINT_HETEROGENEOUS_ACTIVOS = tuple(m for m in POINT_HETEROGENEOUS if INCLUIR.get(m, True))
AGG_MODELS = ("B0_Ingenuo", *POINT_BASE_MODELS, *POINT_HOMOGENEOUS, *POINT_HETEROGENEOUS_ACTIVOS, *MODELOS_DIAGNOSTICO)
for lib, modelo in (("catboost", "I2_CatBoost_Tweedie"), ("xgboost", "I4_XGBoost_Tweedie")):
    if not _disponible(lib):
        aviso(f"{lib} no está instalado: {modelo} queda fuera de la comparativa.")
if not _disponible("optuna"):
    aviso("Optuna no está instalado: no habrá ajuste fino (se usan los hiperparámetros del Avance 4).")
mostrar(pd.DataFrame([{"modelo": m, "etiqueta": AGG_LABELS[m], "tipo": _agg_model_type(m), "estrategia": ESTRATEGIA[m]}
                      for m in AGG_MODELS]), f"Candidatos activos ({len(AGG_MODELS)})")

FUENTE_AV5 = FuenteAvance5(spark)
if EJECUTAR_ENTRENAMIENTO:
    with Cronometro() as reloj:
        CONTRATOS_AV5 = FUENTE_AV5.validar_contratos(duplicados=not MODO_RAPIDO)
    print(f"Contratos del snapshot verificados en {reloj.segundos / 60:.1f} min: "
          + ", ".join(f"{k} = {v}" for k, v in CONTRATOS_AV5.items() if k != "filas_snapshot"))
def _ventana(origen):
    """Días que se pronostican desde un origen: T + 1 … T + 14."""
    o = pd.Timestamp(origen)
    return {"desde": (o + pd.Timedelta(days=1)).date(), "hasta": (o + pd.Timedelta(days=CONFIG_FUENTE["horizonte"])).date()}


BLOQUES_AV5 = pd.DataFrame(
    [{"bloque": "ajuste", "fase": "ajuste", "origen": FUENTE_AV5.ajuste_origen, **_ventana(FUENTE_AV5.ajuste_origen)}]
    + [{"bloque": f"oof_{o}", "fase": "oof", "origen": o, **_ventana(o)} for o in FUENTE_AV5.oof_origenes]
    + [{"bloque": b["corte"], "fase": b["fase"], "origen": b["origen"], "desde": b["desde"], "hasta": b["hasta"]}
       for b in FUENTE_AV5.cortes + FUENTE_AV5.pruebas]
    + ([{"bloque": "pronostico", "fase": "operación", "origen": FUENTE_AV5.origen_pronostico,
         **_ventana(FUENTE_AV5.origen_pronostico)}] if FUENTE_AV5.origen_pronostico else []))
mostrar(BLOQUES_AV5, "Bloques de la corrida (desarrollo = elegir; prueba = datos no vistos; pronóstico = salida operativa)")
mostrar(pd.DataFrame([{"variable": v, "tipo": "categórica" if v in FUENTE_AV5.cat_features else "numérica",
                       "descripcion": DESCRIPCION_VARIABLES.get(v, "")} for v in FUENTE_AV5.features_table]),
        f"Variables del modelo ({len(FUENTE_AV5.features_table)})", max_filas=80)
hallazgo(f"Tabla Oro versión {FUENTE_AV5.version} ({FUENTE_AV5.modo_lectura}); {len(FUENTE_AV5.cortes)} cortes de desarrollo, "
         f"{len(FUENTE_AV5.pruebas)} bloques de prueba no vistos, {len(FUENTE_AV5.features_table)} variables; "
         f"s_promos_vigentes_t0 {'incluida (causal)' if FUENTE_AV5.promos_vigentes_causal else 'excluida (la Oro no la declara causal)'}; "
         f"calendario de promociones futuras {'activo' if FUENTE_AV5._promos_futuras is not None else 'apagado'}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ejecución formal
# MAGIC Se abre la corrida (bitácora en Delta y run de MLflow), se ejecuta el protocolo de la sección 2.12 con puntos de
# MAGIC control, se arman las tablas agregadas, se evalúa la compuerta, se guarda el pronóstico operativo y la tabla de
# MAGIC monitoreo, se registra el modelo en Unity Catalog (si corresponde) y se publican los resultados en `mlops.av5_*`.
# MAGIC Si la celda se interrumpe, copia el `run_id` impreso al inicio en `RUN_ID_REANUDAR` y vuelve a correr la libreta.

# COMMAND ----------

titulo("4. EJECUCIÓN FORMAL")
RUN_ID_ACTUAL, TABLAS_SALIDA_AV5 = None, None
if EJECUTAR_ENTRENAMIENTO:
    try:
        with RegistroFormal(spark, FUENTE_AV5, ENTORNO_AV5, huella_codigo(), HILOS_AV5, run_id=RUN_ID_REANUDAR) as REGISTRO_AV5:
            ESTADO_PREVIO = REGISTRO_AV5.cargar_punto_control() if RUN_ID_REANUDAR else None
            if RUN_ID_REANUDAR and ESTADO_PREVIO is None:
                aviso(f"No se encontró punto de control para {RUN_ID_REANUDAR}: la corrida empieza desde el principio con ese identificador.")
            elif ESTADO_PREVIO is not None:
                print(f"Se reanuda desde el punto de control: pasos completados {ESTADO_PREVIO['completados']}")
            ESTADO_AV5 = ejecutar_modelado(FUENTE_AV5, REGISTRO_AV5, HILOS_AV5, estado=ESTADO_PREVIO)
            TABLAS_SALIDA_AV5 = compilar_resultados(ESTADO_AV5, FUENTE_AV5, REGISTRO_AV5)
            MODELO_FINAL = ESTADO_AV5["decision"]["modelo_final"]
            COMPUERTA_AV5 = preparar_compuerta(TABLAS_SALIDA_AV5, ESTADO_AV5, FUENTE_AV5)
            TABLAS_SALIDA_AV5["compuerta"] = pd.DataFrame([{
                "modelo": MODELO_FINAL, "estado_desarrollo": COMPUERTA_AV5["desarrollo_status"],
                "technical_pass": COMPUERTA_AV5["technical_pass"], "production_blocked": COMPUERTA_AV5["production_blocked"],
                "registry_action": COMPUERTA_AV5["registry_action"],
                "detalle_json": json.dumps(COMPUERTA_AV5, ensure_ascii=False, default=_operacion_json)}])
            REGISTRO_AV5.log_dict("compuerta", COMPUERTA_AV5)
            # ---- salida operativa y monitoreo (no se escriben en MODO_RAPIDO: serían cifras de una prueba de humo)
            final = ESTADO_AV5.get("final", {})
            RESUMEN_OPERATIVO = {}
            if "pronostico" in final and len(final["pronostico"]):
                pron = final["pronostico"]
                RESUMEN_OPERATIVO = {"origen_pronostico": str(pd.Timestamp(pron["origen"].iloc[0]).date()),
                                     "dias": f"{pron['fecha_objetivo'].min().date()} a {pron['fecha_objetivo'].max().date()}",
                                     "series_pronostico": int(pron[["tienda", "categoria_id"]].drop_duplicates().shape[0]),
                                     "filas_pronostico": int(len(pron)), "metodo_intervalos": final.get("metodo_intervalos"),
                                     "techo_operativo": NIVEL_TECHO_OPERATIVO}
                if not MODO_RAPIDO:
                    RESUMEN_OPERATIVO["tabla"] = guardar_pronostico(pron, FUENTE_AV5, REGISTRO_AV5, MODELO_FINAL)
            acum_p = ESTADO_AV5["acumuladores"]["prueba"]["ampliada"]
            wm_dev = float(TABLAS_SALIDA_AV5["comparativa"].set_index("modelo").loc[MODELO_FINAL, "wmape_media_cortes"])
            MONITOREO_AV5 = monitoreo_desde_prueba(acum_p, MODELO_FINAL, COMPUERTA_AV5["metodo_intervalos"], wm_dev) if acum_p._sumas else pd.DataFrame()
            if len(MONITOREO_AV5) and not MODO_RAPIDO:
                limpio, esquema = _para_spark(MONITOREO_AV5.assign(run_id=REGISTRO_AV5.run_id, fecha_calculo=pd.Timestamp.now()))
                borrar_corrida(TABLA_MONITOREO, REGISTRO_AV5.run_id)
                (spark.createDataFrame(limpio, schema=esquema).write.format("delta").mode("append")
                 .option("mergeSchema", "true").saveAsTable(TABLA_MONITOREO))
            # ---- registro en Unity Catalog
            REGISTRO_UC_AV5 = {"registrado": False, "motivo": "no aplica"}
            registrar = (not MODO_RAPIDO and COMPUERTA_AV5["technical_pass"]) if REGISTRAR_EN_UC is None else bool(REGISTRAR_EN_UC)
            candidato = final.get("candidate")
            if registrar and candidato is not None and REGISTRO_AV5.mlflow is not None and not candidato.get("requiere_spark"):
                inicio_ej = pd.Timestamp(FUENTE_AV5.fecha_maxima_objetivo) + pd.Timedelta(days=1)
                # firma del modelo: numéricas double (cualquier float cabe sin pérdida), categóricas como TEXTO (el modelo las
                # codifica con su vocabulario: así 22_pronostico_operativo le pasa la Oro tal cual) y fechas de origen/objetivo
                ejemplo = pd.DataFrame({**{c: np.zeros(2, dtype=np.float64) for c in FUENTE_AV5.num_features},
                                        **{c: [str(next(iter(FUENTE_AV5.codigos[c])) if FUENTE_AV5.codigos[c] else "0")] * 2
                                           for c in FUENTE_AV5.cat_features},
                                        "ingenuo_estacional": np.zeros(2), "origen": [inicio_ej] * 2,
                                        "fecha_objetivo": [inicio_ej + pd.Timedelta(days=1)] * 2})
                ejemplo["h"] = np.array([1, 1], dtype=np.int64)
                fila_p = TABLAS_SALIDA_AV5.get("comparativa_prueba", pd.DataFrame()).set_index("modelo") if "comparativa_prueba" in TABLAS_SALIDA_AV5 else None
                metricas_uc = {"wmape_desarrollo": round(wm_dev, 4)}
                if fila_p is not None and MODELO_FINAL in fila_p.index:
                    metricas_uc["wmape_prueba"] = round(float(fila_p.loc[MODELO_FINAL, "wmape_media_cortes"]), 4)
                try:
                    REGISTRO_UC_AV5 = registrar_modelo_uc(REGISTRO_AV5, candidato, ejemplo, metricas_uc)
                except Exception as e:
                    REGISTRO_UC_AV5 = {"registrado": False, "motivo": f"{type(e).__name__}: {str(e)[:200]}"}
            elif not registrar:
                REGISTRO_UC_AV5["motivo"] = ("modo rápido" if MODO_RAPIDO else "la compuerta técnica no pasó") if REGISTRAR_EN_UC is None else "REGISTRAR_EN_UC = False"
            # ---- métricas principales en MLflow y resumen de la corrida
            comp = TABLAS_SALIDA_AV5["comparativa"].set_index("modelo")
            REGISTRO_AV5.log_metrics({"wmape_desarrollo_final": comp.loc[MODELO_FINAL, "wmape_media_cortes"],
                                      "wmape_desarrollo_ingenuo": comp.loc["B0_Ingenuo", "wmape_media_cortes"],
                                      "sesgo_abs_medio_final": comp.loc[MODELO_FINAL, "sesgo_abs_medio_pct"]})
            TABLAS_SALIDA_AV5["resumen_corrida"] = pd.DataFrame([{
                "modelo_final": MODELO_FINAL, "version_oro": str(FUENTE_AV5.version), "modo": "rapido" if MODO_RAPIDO else "completo",
                "cortes": len(FUENTE_AV5.cortes), "bloques_prueba": len(FUENTE_AV5.pruebas),
                "minutos_modelado": ESTADO_AV5.get("segundos_modelado", 0.0) / 60, "mlflow_run_id": REGISTRO_AV5.mlflow_run_id,
                "registro_uc": json.dumps(REGISTRO_UC_AV5, ensure_ascii=False, default=_operacion_json),
                "salida_operativa": json.dumps(RESUMEN_OPERATIVO, ensure_ascii=False, default=_operacion_json),
                "argumento_decision": ESTADO_AV5["decision"]["argumento"]}])
            SALIDAS_AV5 = REGISTRO_AV5.publicar_tablas({k: v for k, v in TABLAS_SALIDA_AV5.items() if k in NOMBRES_TABLAS})
            RUN_ID_ACTUAL = REGISTRO_AV5.run_id
            print(f"Corrida {RUN_ID_ACTUAL} terminada: {len(SALIDAS_AV5)} tablas publicadas en mlops.av5_* · "
                  f"modelo final {MODELO_FINAL} · Unity Catalog: {REGISTRO_UC_AV5}")
    finally:
        FUENTE_AV5.liberar_memoria()
else:
    print("EJECUTAR_ENTRENAMIENTO = False: se muestran resultados ya guardados.")
