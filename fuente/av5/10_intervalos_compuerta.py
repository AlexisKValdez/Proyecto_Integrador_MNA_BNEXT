
# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Intervalos, diagnóstico estadístico y compuerta de aceptación
# MAGIC ### 6.1 Intervalos P10–P90 en septiembre
# MAGIC `cobertura` = fracción de días reales dentro de [P10, P90] (objetivo 80 %, banda 75–85 %). `ancho_relativo` = ancho
# MAGIC medio del intervalo entre la venta media (más angosto = más útil, si cubre). Las pérdidas pinball relativas miden la
# MAGIC calidad de cada cuantil (menor es mejor). `tasa_cruces_final` debe ser 0 (P10 ≤ P50 ≤ P90).

# COMMAND ----------

titulo("6. INTERVALOS, DIAGNÓSTICO ESTADÍSTICO Y COMPUERTA")
COMPUERTA_C = json.loads(C["compuerta"]["detalle_json"].iloc[0]) if "compuerta" in C else {}
METODO_FINAL = COMPUERTA_C.get("metodo_intervalos", "Q_Conformal")
if "cuantiles" in C and len(C["cuantiles"]):
    Q = C["cuantiles"]
    GQ = Q.loc[Q.ambito.eq("global"), ["metodo", "cobertura", "cobertura_base", "ancho_relativo", "pinball_p10_rel", "pinball_p50_rel",
                                       "pinball_p90_rel", "tasa_cruces_final", "cumple_banda"]]
    mostrar(GQ, "Intervalos en la prueba no vista (global por método)",
            formatos={"cobertura": "pct", "cobertura_base": "pct", "ancho_relativo": "dec3", "pinball_p10_rel": "dec4",
                      "pinball_p50_rel": "dec4", "pinball_p90_rel": "dec4", "tasa_cruces_final": "pct2"})
    resumen_ambitos = []
    for metodo in GQ["metodo"]:
        for ambito in ("h", "tipo_categoria", "region", "corte×h×tipo", "tipo×region"):
            sub = Q.loc[Q.metodo.eq(metodo) & Q.ambito.eq(ambito)]
            if len(sub):
                resumen_ambitos.append({"metodo": metodo, "ambito": ambito, "grupos": len(sub), "en_banda": int(sub["cumple_banda"].sum()),
                                        "cobertura_min": sub["cobertura"].min(), "cobertura_max": sub["cobertura"].max()})
    mostrar(pd.DataFrame(resumen_ambitos), "6.2 Cobertura por ámbito (grupos dentro de la banda 75–85 %)",
            formatos={"cobertura_min": "pct", "cobertura_max": "pct"})
    fuera = Q.loc[Q.metodo.eq(METODO_FINAL) & Q.ambito.isin(["tipo_categoria", "region", "tipo×region"]) & ~Q.cumple_banda]
    if len(fuera):
        mostrar(fuera[["ambito", "tipo_categoria", "region", "cobertura", "ancho_relativo"]].head(20),
                f"Grupos fuera de la banda con {METODO_FINAL}", formatos={"cobertura": "pct", "ancho_relativo": "dec3"})
    g = GQ.set_index("metodo")
    hallazgo(f"Intervalos en septiembre: " + "; ".join(f"{m} cubre {r.cobertura:.1%} con ancho relativo {r.ancho_relativo:.2f}"
                                                         for m, r in g.iterrows())
             + f". El modelo final usa {METODO_FINAL} ({'método automático: menor pinball dentro de la banda' if METODO_INTERVALOS == 'automatico' else 'fijado en la configuración'}).")
else:
    print("Sin evaluación de intervalos (no hay bloques de prueba).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Sesgo del modelo final por bloque y 6.4 significancia (Diebold–Mariano)
# MAGIC Sesgo % = (Σ pronóstico − Σ real) / Σ real × 100: negativo = subpronóstico (riesgo de quiebre), positivo =
# MAGIC sobrepronóstico (inventario de más). Criterio: |sesgo| ≤ 3 % en cada corte de desarrollo y ≤ 5 % en la prueba.
# MAGIC Diebold–Mariano compara la pérdida diaria del modelo final contra el ingenuo y contra el mejor individual (estadístico
# MAGIC positivo = el modelo final se equivoca menos); con 14 días por bloque es una evidencia descriptiva.

# COMMAND ----------

SB = C["por_corte"].loc[C["por_corte"].modelo.eq(MODELO_FINAL), ["fase", "corte", "wmape", "wmape_ingenuo", "mejora_vs_ingenuo", "sesgo_pct"]].copy()
SB["limite_sesgo"] = np.where(SB["fase"].eq("prueba"), UMBRALES_COMPUERTA["sesgo_prueba_max_pct"], UMBRALES_COMPUERTA["sesgo_max_pct"])
SB["cumple_sesgo"] = SB["sesgo_pct"].abs() <= SB["limite_sesgo"]
mostrar(SB, f"Sesgo y error de {AGG_LABELS[MODELO_FINAL]} por bloque",
        formatos={"wmape": "dec4", "wmape_ingenuo": "dec4", "mejora_vs_ingenuo": "pct", "sesgo_pct": "dec2", "limite_sesgo": "dec1"})
if "dm" in C and len(C["dm"]):
    DM = C["dm"].copy()
    DM["referencia"] = DM["referencia"].map(AGG_LABELS)
    mostrar(DM[["fase", "corte", "referencia", "dm_estadistico", "dm_pvalue", "mejor_que_referencia", "n_dias"]],
            f"Diebold–Mariano: {AGG_LABELS[MODELO_FINAL]} contra cada referencia", formatos={"dm_estadistico": "dec3", "dm_pvalue": "dec4"})
    dmd = DM.loc[DM.fase.eq("desarrollo") & DM.referencia.eq(AGG_LABELS["B0_Ingenuo"])]
    hallazgo(f"Sesgo de {AGG_LABELS[MODELO_FINAL]}: de {SB['sesgo_pct'].min():+.2f} % a {SB['sesgo_pct'].max():+.2f} % por bloque; "
             f"{int(SB['cumple_sesgo'].sum())} de {len(SB)} bloques dentro del límite. Diebold–Mariano contra el ingenuo significativo "
             f"(p < 0.05) en {int((dmd['dm_pvalue'] < 0.05).sum())} de {len(dmd)} cortes de desarrollo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.5 Compuerta de aceptación
# MAGIC Las comprobaciones **técnicas** deciden si el modelo se registra como `@challenger` en Unity Catalog. Pasar a
# MAGIC producción (`@champion`) exige además las **aprobaciones de negocio** (alcance, política de servicios y disponibilidad
# MAGIC de atributos a la fecha del pronóstico) y las reglas críticas de calidad del pipeline: mientras Bnext no las confirme,
# MAGIC la compuerta bloquea producción por diseño, no por un error del modelo.

# COMMAND ----------

if COMPUERTA_C:
    mostrar(pd.DataFrame([{"comprobacion": k.replace("_", " "), "resultado": "CUMPLE" if v else "NO CUMPLE",
                           "bloquea_challenger": k in COMPUERTA_C.get("checks_tecnicos_bloqueantes", [])}
                          for k, v in COMPUERTA_C["checks"].items()]), "Comprobaciones de la compuerta")
    mostrar(pd.DataFrame({"motivo de bloqueo de producción": COMPUERTA_C["blocking_reasons"] or ["ninguno"]}), "Motivos registrados")
    registro_uc = json.loads(RESUMEN_C["registro_uc"]) if isinstance(RESUMEN_C.get("registro_uc"), str) else {}
    hallazgo(f"Compuerta: {COMPUERTA_C['desarrollo_status'].replace('_', ' ')}; acción de registro: {COMPUERTA_C['registry_action']}; "
             f"producción {'bloqueada' if COMPUERTA_C['production_blocked'] else 'habilitada'} "
             f"({len(COMPUERTA_C['blocking_reasons'])} {'motivo' if len(COMPUERTA_C['blocking_reasons']) == 1 else 'motivos'}). Unity Catalog: "
             + (f"versión {registro_uc.get('version')} de {NOMBRE_MODELO_UC} con alias @challenger." if registro_uc.get("registrado")
                else f"sin registro ({registro_uc.get('motivo', 'no aplica')})."))
