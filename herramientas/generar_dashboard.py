"""Genera el dashboard AI/BI de Databricks (formato .lvdash.json) y el archivo de consultas SQL equivalente.

Uso: python generar_dashboard.py <carpeta_destino> [catalogo.esquema]
"""
import json
import sys
from pathlib import Path

ESQ = sys.argv[2] if len(sys.argv) > 2 else "modeladoprescriptivoventas.maestria"
T = lambda n: f"{ESQ}.{n}"

DATASETS = {
    "ds_resumen": ("Resumen ejecutivo", f"""SELECT nivel, region, tipo_categoria, zona, series, tiendas, venta_pronostico_14d, p10_14d, p90_14d,
       venta_real_14d_previos, venta_real_14d_anteriores, variacion_vs_14d_previos, variacion_14d_previos_vs_anteriores,
       ancho_relativo_banda, pct_filas_con_promo_planeada, origen, desde, hasta
FROM {T('gold_dash_resumen_ejecutivo')}"""),
    "ds_serie": ("Serie de tiempo", f"""SELECT nivel, tienda, categoria_id, categoria_nombre, region, zona, tipo_categoria, fecha, tipo_dato,
       real, pronostico, p10, p90, COALESCE(real, pronostico) AS venta, tiendas, pct_series_con_promo
FROM {T('gold_dash_serie_tiempo')}"""),
    "ds_precision": ("Precisión del modelo", f"""SELECT fase, semana, region, tipo_categoria, modelo, n_filas, wmape, sesgo_pct, cobertura,
       alerta_error, alerta_sesgo, alerta_cobertura
FROM {T('gold_dash_precision')}"""),
    "ds_modelos": ("Comparativa de modelos", f"""SELECT fase, modelo, tipo, estrategia, wmape_media_cortes, mejora_vs_ingenuo, sesgo_abs_medio_pct, wmape_mercancia
FROM {T('gold_dash_precision_modelos')}"""),
    "ds_surtido": ("Surtido por producto", f"""SELECT s.tienda, CAST(s.categoria_id AS STRING) AS categoria_id, c.nombre AS categoria_nombre, s.tipo_categoria, s.region,
       s.producto_id, s.rank_sku, s.semana_horizonte, s.desde, s.hasta, s.share_venta, s.precio_medio,
       s.venta_p50_semana, s.unidades_p50_semana, s.techo_unidades_p90_semana
FROM {T('gold_surtido_sku_semana')} s
LEFT JOIN {T('silver_catalogo_categorias')} c ON CAST(s.categoria_id AS STRING) = CAST(c.categoria_id AS STRING)"""),
    "ds_promos": ("Promociones planeadas", f"""SELECT promo_id, nombre, CAST(categoria_id AS STRING) AS categoria_id, categoria_nombre, inicio_planeado, fin_planeado, alcance,
       region, zona, descuento_planeado, tipo_mecanica, fuente, aporte_plan_pct, dias_para_inicio, en_horizonte
FROM {T('gold_dash_promociones')}"""),
    "ds_whatif": ("Escenarios what-if", f"""SELECT id_escenario, descripcion_escenario, descuento_escenario, region, tipo_categoria, CAST(categoria_id AS STRING) AS categoria_id,
       series, venta_base, venta_escenario, uplift_pesos, uplift_pct, modo_simulacion, origen
FROM {T('gold_escenarios_whatif_resumen')}"""),
    "ds_alertas": ("Alertas", f"""SELECT fuente, tipo, severidad, mensaje, tienda, categoria_nombre, region, zona, tipo_categoria, cambio_7d, incertidumbre,
       dias_inactiva_7d, origen, fecha
FROM {T('gold_dash_alertas')}"""),
    "ds_estado": ("Estado del pipeline", f"""SELECT indicador, valor, detalle, estado, fecha_calculo
FROM {T('gold_dash_estado_pipeline')}"""),
}


def campo(nombre, expresion=None):
    return {"name": nombre, "expression": expresion or f"`{nombre}`"}


def widget(nombre, dataset, campos, spec, x, y, w, h, disaggregated=False):
    return {"widget": {"name": nombre, "queries": [{"name": "main_query", "query": {"datasetName": dataset, "fields": campos, "disaggregated": disaggregated}}],
                       "spec": spec}, "position": {"x": x, "y": y, "width": w, "height": h}}


def texto(nombre, md, x, y, w, h):
    return {"widget": {"name": nombre, "textbox_spec": md}, "position": {"x": x, "y": y, "width": w, "height": h}}


def filtro(nombre, dataset, columna, etiqueta, x, y, w=2, h=1, multi=True):
    qn = f"{dataset}_{columna}"
    return {"widget": {"name": nombre,
                       "queries": [{"name": qn, "query": {"datasetName": dataset, "fields": [campo(columna), campo(f"{columna}_associativity", "COUNT_IF(`associative_filter_predicate_group`)")],
                                                          "disaggregated": False}}],
                       "spec": {"version": 2, "widgetType": "filter-multi-select" if multi else "filter-single-select",
                                "encodings": {"fields": [{"fieldName": columna, "displayName": etiqueta, "queryName": qn}]}}},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def contador(nombre, dataset, columna, etiqueta, x, y, w=2, h=3, agg="SUM", formato=None):
    f = campo(f"{agg.lower()}({columna})", f"{agg}(`{columna}`)")
    spec = {"version": 2, "widgetType": "counter", "encodings": {"value": {"fieldName": f["name"], "displayName": etiqueta}}}
    if formato:
        spec["encodings"]["value"]["format"] = formato
    return widget(nombre, dataset, [f], spec, x, y, w, h)


def barras(nombre, dataset, xcol, ycol, etiqueta_x, etiqueta_y, x, y, w, h, color=None, agg="SUM", formato_y=None):
    campos = [campo(xcol), campo(f"{agg.lower()}({ycol})", f"{agg}(`{ycol}`)")] + ([campo(color)] if color else [])
    enc = {"x": {"fieldName": xcol, "scale": {"type": "categorical"}, "displayName": etiqueta_x},
           "y": {"fieldName": f"{agg.lower()}({ycol})", "scale": {"type": "quantitative"}, "displayName": etiqueta_y}}
    if formato_y:
        enc["y"]["format"] = formato_y
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}
    return widget(nombre, dataset, campos, {"version": 3, "widgetType": "bar", "encodings": enc, "frame": {"showTitle": True, "title": etiqueta_y}}, x, y, w, h)


def lineas(nombre, dataset, xcol, ycol, etiqueta_y, x, y, w, h, color=None, agg="SUM"):
    campos = [campo(xcol), campo(f"{agg.lower()}({ycol})", f"{agg}(`{ycol}`)")] + ([campo(color)] if color else [])
    enc = {"x": {"fieldName": xcol, "scale": {"type": "temporal"}, "displayName": xcol},
           "y": {"fieldName": f"{agg.lower()}({ycol})", "scale": {"type": "quantitative"}, "displayName": etiqueta_y}}
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}
    return widget(nombre, dataset, campos, {"version": 3, "widgetType": "line", "encodings": enc, "frame": {"showTitle": True, "title": etiqueta_y}}, x, y, w, h)


def tabla(nombre, dataset, columnas, titulo, x, y, w, h):
    campos = [campo(c) for c in columnas]
    enc = {"columns": [{"fieldName": c, "displayName": c.replace("_", " "), "order": i, "visible": True} for i, c in enumerate(columnas)]}
    return widget(nombre, dataset, campos, {"version": 1, "widgetType": "table", "encodings": enc, "frame": {"showTitle": True, "title": titulo}}, x, y, w, h, disaggregated=True)


PCT = {"type": "number-percent", "decimalPlaces": {"type": "exact", "places": 1}}
NUM = {"type": "number-plain", "abbreviation": "compact", "decimalPlaces": {"type": "exact", "places": 0}}

paginas = []
# ---- 1. Resumen
L = [texto("t_resumen", "## Demanda esperada en los próximos 14 días\nComparada con los 14 días reales anteriores. Filtra por región o tipo de categoría; los importes son los del catálogo privado.", 0, 0, 6, 1),
     filtro("f_res_region", "ds_resumen", "region", "Región", 0, 1), filtro("f_res_tipo", "ds_resumen", "tipo_categoria", "Tipo de categoría", 2, 1),
     filtro("f_res_nivel", "ds_resumen", "nivel", "Nivel", 4, 1, multi=False),
     contador("c_venta", "ds_resumen", "venta_pronostico_14d", "Venta esperada 14 días", 0, 2, formato=NUM),
     contador("c_prev", "ds_resumen", "venta_real_14d_previos", "Venta real 14 días previos", 2, 2, formato=NUM),
     contador("c_series", "ds_resumen", "series", "Series pronosticadas", 4, 2, formato=NUM),
     barras("b_region", "ds_resumen", "region", "venta_pronostico_14d", "Región", "Venta esperada 14 días por región", 0, 5, 3, 6),
     barras("b_tipo", "ds_resumen", "tipo_categoria", "venta_pronostico_14d", "Tipo", "Venta esperada 14 días por tipo de categoría", 3, 5, 3, 6),
     tabla("tb_resumen", "ds_resumen", ["nivel", "region", "tipo_categoria", "series", "venta_pronostico_14d", "venta_real_14d_previos", "variacion_vs_14d_previos",
                                        "ancho_relativo_banda", "pct_filas_con_promo_planeada"], "Detalle por nivel", 0, 11, 6, 7)]
paginas.append({"name": "p_resumen", "displayName": "Resumen", "layout": L})
# ---- 2. Pronóstico
L = [texto("t_pron", "## Historia reciente y pronóstico con banda P10–P90\nElige el nivel (nacional, región, zona, tipo, categoría, tienda o serie) y filtra. La banda agregada es la suma de P10 y P90 (conservadora).", 0, 0, 6, 1),
     filtro("f_pr_nivel", "ds_serie", "nivel", "Nivel", 0, 1, multi=False), filtro("f_pr_region", "ds_serie", "region", "Región", 2, 1),
     filtro("f_pr_zona", "ds_serie", "zona", "Zona", 4, 1),
     filtro("f_pr_tipo", "ds_serie", "tipo_categoria", "Tipo", 0, 2), filtro("f_pr_cat", "ds_serie", "categoria_nombre", "Categoría", 2, 2),
     filtro("f_pr_tienda", "ds_serie", "tienda", "Tienda", 4, 2),
     lineas("l_venta", "ds_serie", "fecha", "venta", "Venta real y pronóstico por día", 0, 3, 6, 7, color="tipo_dato"),
     lineas("l_p10", "ds_serie", "fecha", "p10", "Banda inferior P10", 0, 10, 3, 5), lineas("l_p90", "ds_serie", "fecha", "p90", "Banda superior P90", 3, 10, 3, 5),
     lineas("l_promo", "ds_serie", "fecha", "pct_series_con_promo", "Series con promoción (observada antes del origen, planeada después)", 0, 15, 6, 4, color="tipo_dato", agg="AVG")]
paginas.append({"name": "p_pronostico", "displayName": "Pronóstico", "layout": L})
# ---- 3. Precisión
L = [texto("t_prec", "## ¿Qué tan bien acierta el modelo?\nWMAPE (error relativo), sesgo y cobertura del intervalo por semana, región y tipo; fase `prueba` = septiembre no visto, `produccion_*` = pronósticos ya contrastados con la venta real.", 0, 0, 6, 1),
     filtro("f_pc_fase", "ds_precision", "fase", "Fase", 0, 1), filtro("f_pc_region", "ds_precision", "region", "Región", 2, 1), filtro("f_pc_tipo", "ds_precision", "tipo_categoria", "Tipo", 4, 1),
     contador("c_wmape", "ds_precision", "wmape", "WMAPE medio", 0, 2, agg="AVG", formato={"type": "number-plain", "decimalPlaces": {"type": "exact", "places": 3}}),
     contador("c_sesgo", "ds_precision", "sesgo_pct", "Sesgo medio (%)", 2, 2, agg="AVG", formato={"type": "number-plain", "decimalPlaces": {"type": "exact", "places": 1}}),
     contador("c_cob", "ds_precision", "cobertura", "Cobertura P10–P90", 4, 2, agg="AVG", formato=PCT),
     lineas("l_wmape", "ds_precision", "semana", "wmape", "WMAPE por semana", 0, 5, 6, 6, color="region", agg="AVG"),
     barras("b_cob", "ds_precision", "tipo_categoria", "cobertura", "Tipo", "Cobertura del intervalo por tipo", 0, 11, 3, 5, agg="AVG", formato_y=PCT),
     tabla("tb_modelos", "ds_modelos", ["fase", "modelo", "tipo", "wmape_media_cortes", "mejora_vs_ingenuo", "sesgo_abs_medio_pct", "wmape_mercancia"], "Comparativa de modelos (última corrida)", 3, 11, 3, 5)]
paginas.append({"name": "p_precision", "displayName": "Precisión", "layout": L})
# ---- 4. Surtido
L = [texto("t_surt", "## Surtido por producto (semana 1 = días 1–7, semana 2 = días 8–14)\nUnidades esperadas (P50) y techo que cubre la demanda al P90 por tienda × categoría × producto. `OTROS` agrupa la cola; no descuenta inventario.", 0, 0, 6, 1),
     filtro("f_su_tienda", "ds_surtido", "tienda", "Tienda", 0, 1), filtro("f_su_cat", "ds_surtido", "categoria_nombre", "Categoría", 2, 1),
     filtro("f_su_sem", "ds_surtido", "semana_horizonte", "Semana", 4, 1, multi=False),
     barras("b_sku", "ds_surtido", "producto_id", "techo_unidades_p90_semana", "Producto", "Techo de unidades P90 por producto", 0, 2, 6, 6),
     tabla("tb_sku", "ds_surtido", ["tienda", "categoria_nombre", "producto_id", "rank_sku", "semana_horizonte", "share_venta", "precio_medio", "unidades_p50_semana", "techo_unidades_p90_semana"],
           "Detalle por producto", 0, 8, 6, 8)]
paginas.append({"name": "p_surtido", "displayName": "Surtido", "layout": L})
# ---- 5. Promociones
L = [texto("t_promo", "## Promociones planeadas y escenarios\n`aporte_plan_pct` = cuánto suma el plan vigente frente a no tener promociones (según el modelo). La malla S1/S2/S3 simula promoción en todo el horizonte con 10/20/30 % de descuento; `placeholder` = promociones inventadas que hay que sustituir.", 0, 0, 6, 1),
     filtro("f_pm_fuente", "ds_promos", "fuente", "Fuente", 0, 1), filtro("f_pm_cat", "ds_promos", "categoria_nombre", "Categoría", 2, 1),
     tabla("tb_promos", "ds_promos", ["promo_id", "nombre", "categoria_nombre", "inicio_planeado", "fin_planeado", "alcance", "descuento_planeado", "fuente", "aporte_plan_pct", "dias_para_inicio"],
           "Promociones próximas y vigentes", 0, 2, 6, 6),
     filtro("f_wi_tipo", "ds_whatif", "tipo_categoria", "Tipo", 0, 8), filtro("f_wi_region", "ds_whatif", "region", "Región", 2, 8),
     barras("b_whatif", "ds_whatif", "id_escenario", "uplift_pesos", "Escenario", "Uplift en pesos por escenario (vs plan vigente)", 0, 9, 3, 6),
     barras("b_whatif_pct", "ds_whatif", "id_escenario", "uplift_pct", "Escenario", "Uplift medio (%) por escenario", 3, 9, 3, 6, agg="AVG")]
paginas.append({"name": "p_promociones", "displayName": "Promociones", "layout": L})
# ---- 6. Alertas
L = [texto("t_al", "## Alertas\nSeries con caída o subida fuerte frente a la semana previa, alta incertidumbre, tiendas inactivas y alertas del monitoreo del modelo.", 0, 0, 6, 1),
     filtro("f_al_sev", "ds_alertas", "severidad", "Severidad", 0, 1), filtro("f_al_tipo", "ds_alertas", "tipo", "Tipo", 2, 1), filtro("f_al_region", "ds_alertas", "region", "Región", 4, 1),
     barras("b_alertas", "ds_alertas", "tipo", "cambio_7d", "Tipo", "Alertas por tipo (conteo)", 0, 2, 6, 4, agg="COUNT"),
     tabla("tb_alertas", "ds_alertas", ["severidad", "tipo", "mensaje", "tienda", "categoria_nombre", "region", "zona", "cambio_7d", "incertidumbre", "fecha"], "Detalle", 0, 6, 6, 8)]
paginas.append({"name": "p_alertas", "displayName": "Alertas", "layout": L})
# ---- 7. Estado
L = [texto("t_est", "## Estado del pipeline y del modelo\nÚltima venta cargada, origen del pronóstico, compuerta, reglas de calidad y última corrida de cada notebook.", 0, 0, 6, 1),
     tabla("tb_estado", "ds_estado", ["indicador", "valor", "detalle", "estado", "fecha_calculo"], "Indicadores", 0, 1, 6, 10)]
paginas.append({"name": "p_estado", "displayName": "Estado", "layout": L})

dash = {"datasets": [{"name": k, "displayName": v[0], "queryLines": [l + "\n" for l in v[1].splitlines()]} for k, v in DATASETS.items()],
        "pages": paginas}
destino = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
destino.mkdir(parents=True, exist_ok=True)
json.dump(dash, open(destino / "mna_equipo43_demanda.lvdash.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(destino / "consultas.sql", "w", encoding="utf-8") as f:
    f.write("-- Consultas de los conjuntos de datos del tablero (una por vista). Esquema: " + ESQ + "\n\n")
    for k, (nombre, sql) in DATASETS.items():
        f.write(f"-- {k} · {nombre}\n{sql};\n\n")
print(f"dashboard: {len(dash['datasets'])} conjuntos de datos, {len(paginas)} páginas, {sum(len(p['layout']) for p in paginas)} widgets → {destino}")
