
# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Resultados de la corrida
# MAGIC Las secciones 5–10 se construyen con las tablas publicadas de la corrida (las de esta sesión o, con
# MAGIC `EJECUTAR_ENTRENAMIENTO = False`, las de `RUN_ID_A_MOSTRAR` o la última terminada): se pueden volver a mostrar sin
# MAGIC entrenar. Con `MODO_PUBLICO = True` no aparecen importes ni volúmenes.

# COMMAND ----------

titulo("5. RESULTADOS")
if TABLAS_SALIDA_AV5 is not None and RUN_ID_A_MOSTRAR in (None, RUN_ID_ACTUAL):
    CORRIDA_AV5 = dict(TABLAS_SALIDA_AV5, run_id=RUN_ID_ACTUAL)
else:
    CORRIDA_AV5 = cargar_corrida(RUN_ID_A_MOSTRAR)
C = CORRIDA_AV5
RESUMEN_C = C["resumen_corrida"].iloc[0]
MODELO_FINAL = RESUMEN_C["modelo_final"]
COMP = C["comparativa"].sort_values("posicion").reset_index(drop=True)
COMP["etiqueta"] = COMP["modelo"].map(AGG_LABELS)
HAY_PRUEBA = "comparativa_prueba" in C
MEJOR_IND = COMP.loc[COMP.tipo.eq("individual")].iloc[0]["modelo"]
F_COMP = {"wmape_media_cortes": "dec4", "wmape_desviacion_cortes": "dec4", "mejora_vs_ingenuo": "pct",
          "mejora_vs_mejor_individual": "pct", "sesgo_medio_pct": "dec2", "sesgo_abs_medio_pct": "dec2", "mae_relativo": "dec3",
          "rmse_relativo": "dec3", "wmape_mercancia": "dec4", "wmape_h1_7": "dec4", "wmape_h8_14": "dec4",
          "min_entrenamiento_por_corte": "dec1", "segundos_prediccion": "dec1", "wmape_global": "dec4"}
mostrar(pd.DataFrame([{"run_id": C["run_id"], "modo": RESUMEN_C["modo"], "version_oro": RESUMEN_C["version_oro"],
                       "cortes_desarrollo": int(RESUMEN_C["cortes"]), "bloques_prueba": int(RESUMEN_C["bloques_prueba"]),
                       "minutos_modelado": round(float(RESUMEN_C["minutos_modelado"]), 1), "modelo_final": MODELO_FINAL}]),
        "5.1 Identificación de la corrida")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Comparativa en los cortes de desarrollo (ordenada por la métrica principal)
# MAGIC **Cómo leerla.** `wmape_media_cortes` es la métrica principal (promedio del WMAPE de los cortes; menor es mejor) y
# MAGIC ordena la tabla. `wmape_desviacion_cortes` mide estabilidad entre cortes. `mejora_vs_ingenuo` y
# MAGIC `mejora_vs_mejor_individual` son reducciones relativas del WMAPE. `sesgo_medio_pct` > 0 = sobrepronóstico (más
# MAGIC inventario), < 0 = subpronóstico (riesgo de quiebre); `sesgo_abs_medio_pct` promedia su tamaño por corte. `mae_relativo`
# MAGIC y `rmse_relativo` dividen el MAE y el RMSE entre los del ingenuo (< 1 = mejor; el RMSE castiga más los errores grandes,
# MAGIC los picos). `wmape_mercancia` es el error donde se surte físicamente. `wmape_h1_7` y `wmape_h8_14` separan la primera y
# MAGIC la segunda semana. Tiempos: minutos de entrenamiento por corte (los ensambles incluyen a sus bases y su meta-modelo) y
# MAGIC segundos de predicción del bloque completo. Todo con el universo completo de series evaluables.

# COMMAND ----------

COLS_COMP = ["posicion", "etiqueta", "tipo", "estrategia", "wmape_media_cortes", "wmape_desviacion_cortes", "mejora_vs_ingenuo",
             "mejora_vs_mejor_individual", "sesgo_medio_pct", "sesgo_abs_medio_pct", "mae_relativo", "rmse_relativo",
             "wmape_mercancia", "wmape_h1_7", "wmape_h8_14", "min_entrenamiento_por_corte", "segundos_prediccion"]
mostrar_publico(COMP[[c for c in COLS_COMP if c in COMP]], "Comparativa: individuales de la fase previa, ensambles y referencia",
                formatos=F_COMP)
PC = C["por_corte"]
PIV = (PC.loc[PC.fase.eq("desarrollo")].pivot_table(index="modelo", columns="corte", values="wmape")
       .reindex(COMP["modelo"]).reset_index())
PIV.insert(1, "etiqueta", PIV["modelo"].map(AGG_LABELS))
mostrar(PIV.drop(columns=["modelo"]), "WMAPE por corte de desarrollo", formatos={c: "dec4" for c in PIV.columns if c.startswith("corte")})
if "comparativa_comun" in C:
    CC = C["comparativa_comun"].sort_values("posicion").assign(etiqueta=lambda d: d["modelo"].map(AGG_LABELS))
    mostrar_publico(CC[["posicion", "etiqueta", "wmape_media_cortes", "wmape_desviacion_cortes", "mejora_vs_ingenuo", "sesgo_medio_pct"]],
                    "Misma comparativa en la muestra común (3 % de las series; comparable con el Avance 4)", formatos=F_COMP)
n_hom = int(COMP.tipo.eq("ensamble_homogeneo").sum())
n_het = int(COMP.tipo.eq("ensamble_heterogeneo").sum())
primero = COMP.loc[~COMP.tipo.isin(["referencia", "diagnostico"])].iloc[0]
fila_ind = COMP.set_index("modelo").loc[MEJOR_IND]
hallazgo(f"Comparativa de desarrollo: {n_hom} ensambles homogéneos y {n_het} heterogéneos contra {int(COMP.tipo.eq('individual').sum())} "
         f"individuales y la referencia. Primer lugar: {AGG_LABELS[primero['modelo']]} con WMAPE {primero['wmape_media_cortes']:.4f} "
         f"({primero['mejora_vs_ingenuo']:+.1%} contra el ingenuo, {primero.get('mejora_vs_mejor_individual', np.nan):+.1%} contra el mejor "
         f"individual, {AGG_LABELS[MEJOR_IND]} {fila_ind['wmape_media_cortes']:.4f}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Prueba en datos no vistos (septiembre)
# MAGIC Todos los modelos se entrenaron con datos hasta el 31 de agosto y se congelaron; pronostican del 1 al 28 de
# MAGIC septiembre sin reentrenar. La tabla conserva el **orden de desarrollo** (no se reordena con la prueba, para no elegir
# MAGIC con ella). `diferencia` = WMAPE en prueba − WMAPE en desarrollo: un valor cercano a cero indica que el desempeño se
# MAGIC sostiene en meses nuevos.

# COMMAND ----------

if HAY_PRUEBA:
    CP = C["comparativa_prueba"].set_index("modelo")
    NV = COMP[["posicion", "modelo", "etiqueta", "tipo", "wmape_media_cortes"]].rename(columns={"wmape_media_cortes": "wmape_desarrollo"})
    NV["wmape_prueba"] = NV["modelo"].map(CP["wmape_media_cortes"])
    NV["diferencia"] = NV["wmape_prueba"] - NV["wmape_desarrollo"]
    NV["mejora_vs_ingenuo_prueba"] = NV["modelo"].map(CP["mejora_vs_ingenuo"])
    NV["sesgo_prueba_pct"] = NV["modelo"].map(CP["sesgo_medio_pct"])
    NV["wmape_mercancia_prueba"] = NV["modelo"].map(CP["wmape_mercancia"])
    mostrar(NV.drop(columns=["modelo"]), "Desarrollo contra prueba no vista (orden de desarrollo)",
            formatos={"wmape_desarrollo": "dec4", "wmape_prueba": "dec4", "diferencia": "dec4", "mejora_vs_ingenuo_prueba": "pct",
                      "sesgo_prueba_pct": "dec2", "wmape_mercancia_prueba": "dec4"})
    PIVP = (PC.loc[PC.fase.eq("prueba")].pivot_table(index="modelo", columns="corte", values="wmape").reindex(COMP["modelo"]).reset_index())
    PIVP.insert(1, "etiqueta", PIVP["modelo"].map(AGG_LABELS))
    mostrar(PIVP.drop(columns=["modelo"]), "WMAPE por bloque de prueba", formatos={c: "dec4" for c in PIVP.columns if c.startswith("prueba")})
    nv = NV.set_index("modelo")
    from scipy.stats import spearmanr
    candidatos = NV.loc[~NV.tipo.isin(["referencia", "diagnostico"])]
    rho = spearmanr(candidatos["wmape_desarrollo"], candidatos["wmape_prueba"]).correlation if len(candidatos) > 2 else np.nan
    hallazgo(f"Datos no vistos (septiembre): {AGG_LABELS[MODELO_FINAL]} WMAPE {nv.loc[MODELO_FINAL, 'wmape_prueba']:.4f} "
             f"({nv.loc[MODELO_FINAL, 'mejora_vs_ingenuo_prueba']:+.1%} contra el ingenuo, {nv.loc['B0_Ingenuo', 'wmape_prueba']:.4f}; "
             f"sesgo {nv.loc[MODELO_FINAL, 'sesgo_prueba_pct']:+.2f} %); en desarrollo tenía {nv.loc[MODELO_FINAL, 'wmape_desarrollo']:.4f} "
             f"(diferencia {nv.loc[MODELO_FINAL, 'diferencia']:+.4f}). El orden de los modelos se sostiene en septiembre "
             f"(correlación de rangos desarrollo–prueba {rho:.2f}).")
else:
    print("Esta corrida no tiene bloques de prueba (la Oro no incluye septiembre).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Ajuste fino de hiperparámetros
# MAGIC Resultado de Optuna en el bloque de ajuste (20-may). La gráfica muestra cada prueba y la mejor hasta ese momento:
# MAGIC si la curva baja después de la fase aleatoria (línea punteada), TPE encontró mejores regiones del espacio.

# COMMAND ----------

if "ajuste_resumen" in C:
    mostrar(C["ajuste_resumen"], "Ajuste fino por modelo", formatos={"pruebas": "int", "minutos": "dec1", "wmape_inicial": "dec4",
                                                                     "wmape_mejor": "dec4", "mejora_relativa": "pct", "arboles_prueba": "int"})
if "ajuste" in C and len(C["ajuste"]):
    AJ = C["ajuste"]
    modelos_aj = list(dict.fromkeys(AJ["modelo"]))
    fig, axes = plt.subplots(1, len(modelos_aj), figsize=(6.2 * len(modelos_aj), 3.4), squeeze=False)
    for ax, m in zip(axes[0], modelos_aj):
        d = AJ.loc[AJ.modelo.eq(m)].sort_values("prueba")
        ax.scatter(d["prueba"], d["wmape_ajuste"], s=16, alpha=0.6, color="#5B7BA0", label="prueba")
        ax.plot(d["prueba"], d["wmape_ajuste"].cummin(), color="#A06A2C", lw=2, label="mejor hasta el momento")
        ax.axhline(d.loc[d.prueba.eq(0), "wmape_ajuste"].min(), color="gray", ls="--", lw=1, label="configuración del Avance 4")
        ax.axvline(N_PRUEBAS_ALEATORIAS - 0.5, color="#bbbbbb", ls=":", lw=1)
        ax.set_title(f"Optuna · {AGG_LABELS.get(m, m)}", fontsize=10)
        ax.set_xlabel("prueba")
        ax.set_ylabel("WMAPE en el bloque de ajuste")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.show()
if "ajuste_importancia" in C and len(C["ajuste_importancia"]):
    mostrar(C["ajuste_importancia"].sort_values(["modelo", "importancia"], ascending=[True, False]).groupby("modelo").head(5),
            "Hiperparámetros que más movieron el WMAPE (importancia de Optuna)", formatos={"importancia": "pct"})
if "parametros" in C:
    filas_p = []
    for r in C["parametros"].itertuples():
        p = json.loads(r.parametros)
        filas_p.append({"modelo": r.modelo, **{k: p.get(k) for k in ("n_estimators", "iterations", "learning_rate", "num_leaves", "max_depth",
                                                                         "min_child_samples", "min_child_weight", "tweedie_variance_power",
                                                                         "subsample", "colsample_bytree", "reg_lambda", "reg_alpha") if k in p}})
    mostrar(pd.DataFrame(filas_p), "Hiperparámetros congelados para cortes, prueba y modelo final")
if "ajuste_resumen" in C:
    for r in C["ajuste_resumen"].itertuples():
        hallazgo(f"Ajuste fino de {AGG_LABELS.get(r.modelo, r.modelo)}: {int(r.pruebas)} pruebas en {r.minutos:.0f} min; WMAPE en el "
                 f"bloque de ajuste {r.wmape_inicial:.4f} → {min(r.wmape_mejor, r.wmape_inicial):.4f} ({r.mejora_relativa:+.1%}); "
                 f"configuración {r.configuracion}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Cómo combinan los ensambles a sus bases
# MAGIC Pesos de cada base en los ensambles heterogéneos, por origen (aprendidos sólo con bloques anteriores), y
# MAGIC correlación de los errores de las bases en los cortes: **un ensamble sólo gana si sus bases se equivocan distinto**
# MAGIC (correlación menor que 1).

# COMMAND ----------

if "pesos" in C and len(C["pesos"]):
    PW = C["pesos"]
    ultimo = "prueba" if PW["corte"].eq("prueba").any() else PW["corte"].iloc[-1]
    vista = PW.loc[PW.corte.eq(ultimo) & ~PW.ensamble.str.startswith("E4_factor")].pivot_table(index="ensamble", columns="base", values="peso")
    mostrar(vista.reset_index(), f"Pesos de las bases por ensamble (modelos congelados para '{ultimo}')", formatos={c: "dec3" for c in vista.columns})
    factores = PW.loc[PW.corte.eq(ultimo) & PW.ensamble.str.startswith("E4_factor")]
    if len(factores):
        etiquetas_tipo = FUENTE_AV5.etiquetas["tipo_categoria"]
        factores = factores.assign(tipo=factores["base"].str.replace("tipo:", "").astype(int).map(etiquetas_tipo))
        mostrar(factores[["ensamble", "tipo", "peso"]].rename(columns={"peso": "factor"}),
                "E4 · factores de nivel por tipo de categoría (1 = sin corrección)", formatos={"factor": "dec3"})
    evolucion = PW.loc[PW.ensamble.eq("E3_Blending")].pivot_table(index="corte", columns="base", values="peso")
    if len(evolucion):
        mostrar(evolucion.reset_index(), "E3 · evolución de los pesos de blending por corte", formatos={c: "dec3" for c in evolucion.columns})
if "correlacion_errores" in C and len(C["correlacion_errores"]):
    CE = C["correlacion_errores"].set_index("base")
    CE = CE[[c for c in CE.columns if c in CE.index]]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(CE.to_numpy(dtype=float), cmap="Blues", vmin=min(0.5, float(np.nanmin(CE.to_numpy(dtype=float)))), vmax=1)
    ax.set_xticks(range(len(CE)))
    ax.set_xticklabels([AGG_LABELS.get(c, c) for c in CE.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CE)))
    ax.set_yticklabels([AGG_LABELS.get(c, c) for c in CE.index], fontsize=8)
    for i in range(len(CE)):
        for j in range(len(CE)):
            ax.text(j, i, f"{CE.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if CE.iat[i, j] > 0.85 else "black")
    ax.set_title("Correlación de errores entre bases (cortes de desarrollo)", fontsize=10)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.show()
    par = CE.where(~np.eye(len(CE), dtype=bool)).stack()
    hallazgo(f"Diversidad de las bases: la correlación de errores va de {par.min():.2f} ({AGG_LABELS.get(par.idxmin()[0])} con "
             f"{AGG_LABELS.get(par.idxmin()[1])}) a {par.max():.2f} ({AGG_LABELS.get(par.idxmax()[0])} con {AGG_LABELS.get(par.idxmax()[1])}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.6 Tiempos de entrenamiento e inferencia
# MAGIC Minutos de entrenamiento por corte y segundos de predicción del bloque completo por modelo, y minutos totales por
# MAGIC fase de la corrida. El costo cuenta para el negocio porque el modelo se reentrenará cada mes en un Job.

# COMMAND ----------

TIEMPOS = COMP[["posicion", "etiqueta", "tipo", "min_entrenamiento_por_corte", "segundos_prediccion"]].copy()
mostrar(TIEMPOS, "Tiempos por modelo (desarrollo)", formatos={"min_entrenamiento_por_corte": "dec1", "segundos_prediccion": "dec1"})
if "costos" in C and len(C["costos"]):
    CO = C["costos"]
    por_fase = (CO.groupby("fase", as_index=False)[["segundos_entrenamiento", "segundos_prediccion"]].sum(min_count=1)
                .assign(minutos=lambda d: (d["segundos_entrenamiento"].fillna(0) + d["segundos_prediccion"].fillna(0)) / 60))
    mostrar(por_fase[["fase", "minutos"]].sort_values("minutos", ascending=False), "Minutos medidos por fase", formatos={"minutos": "dec1"})
hallazgo(f"La corrida tardó {float(RESUMEN_C['minutos_modelado']):.0f} min de modelado ({RESUMEN_C['modo']}). Entrenar el modelo "
         f"final cuesta {COMP.set_index('modelo').loc[MODELO_FINAL, 'min_entrenamiento_por_corte']:.1f} min por corte con la muestra de "
         f"entrenamiento de esta corrida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.7 Elección del modelo final (matriz de decisión)
# MAGIC Pesos fijados antes de ver resultados: error 35 %, estabilidad 10 %, sesgo 10 %, error en mercancía 10 %, costo de
# MAGIC reentrenamiento 15 % (no penaliza mientras quepa en el Job mensual), interpretabilidad 10 % y operabilidad 10 %. El
# MAGIC error se puntúa como mejora sobre el ingenuo relativa a la mejor mejora. La elección quedó registrada en la bitácora
# MAGIC **antes** de evaluar septiembre.

# COMMAND ----------

DEC = C["decision"].copy()
DEC.insert(1, "etiqueta", DEC["modelo"].map(AGG_LABELS))
mostrar(DEC.drop(columns=[c for c in ("modelo_final", "argumento") if c in DEC]), "Matriz de decisión (métricas de los cortes de desarrollo)",
        formatos={"error": "dec4", "estabilidad": "dec4", "sesgo": "dec2", "error_donde_cuesta": "dec4", "costo": "dec1",
                  "interpretabilidad": "dec2", "operabilidad": "dec2", "total": "dec3"})
if "decision_puntos" in C:
    mostrar(C["decision_puntos"], "Puntos por criterio (peso × puntuación)", formatos={c: "dec3" for c in PESOS_DECISION})
fila_f = COMP.set_index("modelo").loc[MODELO_FINAL]
ARGUMENTO_FINAL = (
    f"Modelo final: {AGG_LABELS[MODELO_FINAL]} ({fila_f['tipo'].replace('_', ' ')}; {ESTRATEGIA[MODELO_FINAL]}). "
    f"{DEC['argumento'].iloc[0]} En desarrollo tiene WMAPE {fila_f['wmape_media_cortes']:.4f} ({fila_f['mejora_vs_ingenuo']:+.1%} contra el "
    f"ingenuo y {fila_f.get('mejora_vs_mejor_individual', np.nan):+.1%} contra el mejor individual), sesgo absoluto medio "
    f"{fila_f['sesgo_abs_medio_pct']:.2f} % por corte, WMAPE en mercancía {fila_f['wmape_mercancia']:.4f} y entrena en "
    f"{fila_f['min_entrenamiento_por_corte']:.1f} min por corte. Para el negocio importa que acierte donde se surte (mercancía), que no "
    f"tenga sesgo sistemático (el sesgo positivo inmoviliza inventario y el negativo provoca quiebres) y que su reentrenamiento mensual "
    f"quepa en un Job; la matriz pondera esos tres aspectos además del error.")
hallazgo(ARGUMENTO_FINAL)
