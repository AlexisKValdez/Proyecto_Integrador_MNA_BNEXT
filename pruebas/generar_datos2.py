"""Genera un conjunto sintético que imita los parquet de Bnext, con los defectos conocidos:
conflictos de tipo entre archivos, columnas nulas, nombres con espacios, duplicados exactos y
versiones, descuentos con error de escala, tiendas que abren/cierran, un apagón y promociones."""
import datetime as dt
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = sys.argv[1]
VOL = os.path.join(BASE, "Volumes", "spark_catalog", "maestria")
shutil.rmtree(os.path.join(BASE, "Volumes"), ignore_errors=True)
rng = np.random.default_rng(7)
MESES = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
TIENDAS = list(range(1001, 1031))
CATS = {101: "CERVEZA", 102: "BOTANAS", 5: "REFRESCOS", 548: "YOGURT", 715: "YOGURTH", 617: None, 758: "REFRESCO",
        10004: "SERVICIOS ARCADIAL", 677: "GENERICO"}
MARCAS = {i: f"MARCA_{i}" for i in range(1, 16)}
PRODUCTOS = []
for p in range(60):
    c = list(CATS)[p % len(CATS)]
    PRODUCTOS.append((50000 + p, c, 1 + p % 15, float(rng.integers(10, 80))))
apertura = {t: dt.date(2026, 6, 10) for t in TIENDAS[:3]}
cierre = {t: dt.date(2026, 6, 20) for t in TIENDAS[3:5]}
apagon = dt.date(2026, 4, 20)      # fuera de las ventanas de los cortes (en los datos reales los 4 cortes están completos)
# campañas promocionales por categoría: ventanas de 10–18 días con pausas de 2–3 semanas, descuento fijo por campaña y
# más unidades por línea mientras dura (así hay días con y sin promoción y un efecto que el modelo puede aprender)
CAMPANAS = []
_pid = 2000
for _i, _cat in enumerate(CATS):
    _ini = dt.date(2026, 3, 1) + dt.timedelta(days=5 * _i)
    while _ini < dt.date(2026, 11, 1):
        _dur = 10 + (_i % 3) * 4
        CAMPANAS.append((_pid, _cat, _ini, _ini + dt.timedelta(days=_dur - 1), [0.1, 0.2, 0.3][_pid % 3]))
        _pid += 1
        _ini = _ini + dt.timedelta(days=_dur + 14 + (_i % 2) * 7)
CAMPANA_POR_CAT = {}
for _c in CAMPANAS:
    CAMPANA_POR_CAT.setdefault(_c[1], []).append(_c)


def campana(cat, fecha):
    for c in CAMPANA_POR_CAT.get(cat, ()):
        if c[2] <= fecha <= c[3]:
            return c
    return None
linea_id = 10_000_000
filas_por_dia = {}
for mes in MESES:
    y, m = map(int, mes.split("-"))
    ndias = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).day
    for d in range(1, ndias + 1):
        fecha = dt.date(y, m, d)
        filas = []
        for t in TIENDAS:
            if t in apertura and fecha < apertura[t]:
                continue
            if t in cierre and fecha > cierre[t]:
                continue
            if fecha == apagon and t % 2 == 0:
                continue
            if fecha == dt.date(2026, 6, 7) and t % 10 == 0:     # día atípico: unas cuantas tiendas no venden
                continue
            factor = 1.4 if fecha.weekday() == 5 else 1.0
            factor *= 1.6 if fecha in (dt.date(2026, 6, 11), dt.date(2026, 6, 18), dt.date(2026, 9, 15)) else 1.0
            ntickets = rng.poisson(12 * factor)
            for k in range(ntickets):
                orden = int(f"{t}{d:02d}{m:02d}{k:03d}")
                hora = dt.datetime(y, m, d, int(rng.integers(7, 23)), int(rng.integers(0, 60)), int(rng.integers(0, 60)))
                for _ in range(int(rng.integers(1, 4))):
                    prod, cat, marca, precio = PRODUCTOS[int(rng.integers(0, len(PRODUCTOS)))]
                    camp = campana(cat, fecha)
                    qty = int(rng.integers(1, 4)) + (int(rng.integers(1, 3)) if camp is not None and rng.random() < 0.5 else 0)
                    sin = qty * precio
                    promo = camp is not None and rng.random() < 0.6
                    tasa = camp[4] if promo else 0.0
                    con = round(sin * (1 - tasa), 2)
                    pct = tasa * 100 * (100 if rng.random() < 0.3 else 1)   # error de escala en 30 %
                    linea_id += 1
                    filas.append({
                        "order_line_id": linea_id, "order_id": orden, "store_code": t, "category_id": cat,
                        "category": CATS[cat], "brand_id": marca, "brand": MARCAS[marca], "product_code": prod,
                        "order_line_qty_ordered": qty, "order_line_unit_price": precio,
                        "order_line_unit_cost": precio * 0.7 if rng.random() > 0.001 else 7501234567890.0,
                        "order_line_total_without_discount": sin, "order_line_total_with_discount": con,
                        "order_line_discount_percent": pct, "order_line_sumasix_discount": 0.0,
                        "$ Pago con puntos": 0.0,
                        "order_line_promo_id": float(camp[0]) if promo else None,
                        "order_line_promo_name": f"CAMPANA {cat} {camp[0]}" if promo else None,
                        "finish_order": hora.strftime("%Y-%m-%d %H:%M:%S"),
                        "sync_date": (hora + dt.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                        "order_status_id": 3 if (m == 7 and rng.random() < 0.01) else 1,
                        "store_type": "BARRIO" if t % 3 else "TRANSITO", "zone_id": t % 4,
                        "region": None if t % 3 == 0 else ("NORTE" if t % 4 in (0, 1) else "SUR"),
                        "jefatura_six": f"J{t % 5}" if t % 2 else None, "store_status_id": 6 if t % 7 else 8,
                        "order_line_total_taxes": round(con * 0.16 / 1.16, 2), "order_line_subtotal": round(sin / 1.16, 2) if rng.random() > 0.02 else 0.0,
                    })
        filas_por_dia[fecha] = filas

os.makedirs(VOL, exist_ok=True)
for mes in MESES:
    carpeta = os.path.join(VOL, mes)
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "_SUCCESS"), "w") as f:
        f.write("")
    for fecha, filas in filas_por_dia.items():
        if fecha.strftime("%Y-%m") != mes:
            continue
        if mes == "2026-07" and fecha.day > 15:          # la entrega real de julio trae sólo 1–15
            continue
        pdf = pd.DataFrame(filas)
        if fecha.day == 3:                        # duplicados exactos (reproceso)
            pdf = pd.concat([pdf, pdf.head(20)])
        pdf["date_orders"] = pdf["finish_order"].str[:10]
        if fecha.day == 1 and fecha.month in (4, 7):   # líneas de captura con importes de cientos de millones
            pdf.loc[pdf.index[:3], "order_line_total_with_discount"] = 3.0e8
            pdf.loc[pdf.index[:3], "order_line_total_without_discount"] = 3.0e8
        if fecha.day == 6:                        # líneas sin importe con descuento (venta nula)
            pdf.loc[pdf.index[:2], "order_line_total_with_discount"] = np.nan
        if fecha.day == 5:                        # fechas centinela que no caben en una fecha de Python
            pdf.loc[pdf.index[:5], "date_orders"] = "0001-01-01"
            pdf.loc[pdf.index[5:8], "sync_date"] = "+50000-01-01 00:00:00"
        if fecha.day == 4:                        # versiones distintas (corrección)
            v = pdf.head(10).copy()
            v["order_line_total_with_discount"] *= 0.5
            v["sync_date"] = (fecha + dt.timedelta(days=1)).strftime("%Y-%m-%d 10:00:00")
            pdf = pd.concat([pdf, v])
        tabla = pa.Table.from_pandas(pdf, preserve_index=False)
        # conflictos de tipo según el mes / día
        if mes == "2026-06":
            tabla = tabla.set_column(tabla.schema.get_field_index("order_line_id"), "order_line_id",
                                     pa.array(pdf["order_line_id"].astype("int32")))
        if fecha.day % 3 == 0:
            tabla = tabla.append_column("date_ordered", pa.array([None] * len(pdf), pa.null()))
        elif fecha.day % 3 == 1:
            tabla = tabla.append_column("date_ordered", pa.array([20260601.0] * len(pdf), pa.float64()))
        else:
            tabla = tabla.append_column("date_ordered", pa.array(["2026-06-01"] * len(pdf), pa.string()))
        if mes == "2026-07":
            tabla = tabla.append_column("fecha_archivo", pa.array([dt.datetime(fecha.year, fecha.month, fecha.day)] * len(pdf),
                                                                    pa.timestamp("ns")))
        if mes == "2026-07" and fecha.day in (10, 11):   # nombres tipo Spark (no indican el día)
            pq.write_table(tabla.slice(0, len(pdf) // 2), os.path.join(carpeta, f"part-0000{fecha.day}-a.snappy.parquet"),
                           version="2.6")
            pq.write_table(tabla.slice(len(pdf) // 2), os.path.join(carpeta, f"part-0000{fecha.day}-b.snappy.parquet"),
                           version="2.6")
        else:
            pq.write_table(tabla, os.path.join(carpeta, f"{fecha.day:02d}.parquet"), version="2.6")

# ------------------------------------------------------------------ externas y artefactos heredados
proc = os.path.join(VOL, "processed")
outs = os.path.join(VOL, "outputs")
os.makedirs(proc, exist_ok=True)
os.makedirs(outs, exist_ok=True)
pd.DataFrame({"id_tienda": TIENDAS[:-4] + [9999], "formato": "X", "LAT_DECIMAL": 25.6, "LON_DECIMAL": -100.3,
              "cp": "64000"}).to_csv(os.path.join(proc, "stores.csv"), index=False)
with open(os.path.join(proc, "CP.txt"), "w", encoding="latin-1") as f:
    f.write("El Catálogo Nacional de Códigos Postales es elaborado por Correos de México\n")
    f.write("d_codigo|d_asenta|d_mnpio|d_estado\n")
    f.write("64000|Centro|Monterrey|Nuevo León\n64010|Obispado|Monterrey|Nuevo León\n")
pd.DataFrame({"fecha": pd.date_range("2026-03-01", "2026-05-31").strftime("%Y-%m-%d"), "municipio": "Monterrey",
              "t_max": 30.0, "lluvia_mm": 0.0}).to_csv(os.path.join(proc, "historial_clima_enriquecido_mayo.csv"), index=False)
pd.DataFrame({"fecha": ["2026-03-07", "2026-04-11"], "partido": ["A vs B", "C vs D"]}).to_csv(
    os.path.join(proc, "calendario_ligamx_2026.csv"), index=False)

# ABT heredado (mar-may): agregado tienda-categoría-día SIN deduplicar y con relleno de ceros global (como el equipo anterior)
todas = pd.DataFrame([f for fecha, fs in filas_por_dia.items() if fecha < dt.date(2026, 6, 1) for f in fs])
todas["fecha"] = pd.to_datetime(todas["finish_order"]).dt.normalize()
dups = todas[todas["order_line_id"] % 97 == 0]                       # el ABT heredado arrastra duplicados
todas = pd.concat([todas, dups])
agg = (todas.groupby(["store_code", "category_id", "fecha"])
       .agg(venta_total=("order_line_total_with_discount", "sum"), unidades=("order_line_qty_ordered", "sum"),
            num_tickets=("order_id", "nunique")).reset_index())
series = agg[["store_code", "category_id"]].drop_duplicates()
dias = pd.DataFrame({"fecha": pd.date_range("2026-03-01", "2026-05-31")})
abt = series.merge(dias, how="cross").merge(agg, on=["store_code", "category_id", "fecha"], how="left")
abt[["venta_total", "unidades", "num_tickets"]] = abt[["venta_total", "unidades", "num_tickets"]].fillna(0)
abt = abt.sort_values(["store_code", "category_id", "fecha"])
g = abt.groupby(["store_code", "category_id"])["venta_total"]
abt["venta_lag_7"] = g.shift(7)
abt["venta_media_movil_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
abt["intensidad_promo"] = 0.2
abt["dia_semana"] = abt["fecha"].dt.dayofweek + 1
abt["municipio"] = "Monterrey"
abt["store_type"] = "BARRIO"
abt.to_parquet(os.path.join(proc, "dataset_modelado_unificado_fe_padded_mayo.parquet"), index=False)

# consolidados del equipo anterior (mar-may): crudo y "limpio"
crudo = pd.DataFrame([f for fecha, fs in filas_por_dia.items() if fecha < dt.date(2026, 6, 1) for f in fs])
crudo[["order_line_id", "order_id", "store_code", "category_id", "brand_id", "order_line_qty_ordered",
       "order_line_total_with_discount", "order_line_total_without_discount", "finish_order", "sync_date"]].to_parquet(
    os.path.join(proc, "ventas_consolidado_2026-08-21.parquet"), index=False)
limpio = crudo.drop_duplicates("order_line_id", keep="last")
limpio = limpio[limpio["order_line_unit_cost"] <= 5000]
limpio[["order_line_id", "store_code", "category_id", "order_line_total_with_discount", "finish_order"]].to_parquet(
    os.path.join(proc, "ventas_consolidado_limpio_final_mayo.parquet"), index=False)
# como el real: predicciones de 13 a 15 de julio, columnas y_real y pred_lightgbm_optuna
jul = pd.DataFrame([f for fecha, fs in filas_por_dia.items() if dt.date(2026, 7, 13) <= fecha <= dt.date(2026, 7, 15) for f in fs])
jul["fecha"] = pd.to_datetime(jul["finish_order"]).dt.date
preds = (jul.groupby(["store_code", "category_id", "fecha"])["order_line_total_with_discount"].sum()
         .reset_index().rename(columns={"order_line_total_with_discount": "y_real"}))
preds["pred_lightgbm_optuna"] = preds["y_real"] * 0.8 + 3
preds.to_parquet(os.path.join(outs, "preds_lightgbm_refinado_optuna.parquet"), index=False)
pd.DataFrame({"columna": ["a", "b"], "pct_nulos": [0.1, 0.0],
              "definicion_negocio": ["monto, con descuento, por línea", "sin definir"]}).to_csv(
    os.path.join(outs, "auditoria_columnas_final.csv"), index=False)
pd.DataFrame({"store_code": [1001, 1002], "fecha": ["2026-03-02", "2026-03-03"], "z": [5.1, 4.2]}).to_csv(
    os.path.join(outs, "anomalias_deteccion_final.csv"), index=False)
with open(os.path.join(outs, "best_params_refinamiento.json"), "w") as f:
    json.dump({"best_params": {"num_leaves": 90}, "best_value": 0.4987}, f)
with open(os.path.join(outs, "best_params_busqueda.json"), "w") as f:
    json.dump({"best_params": {"num_leaves": 102, "min_data_in_leaf": 510, "tweedie_variance_power": 1.24},
               "best_value": 0.5002}, f)
with open(os.path.join(outs, "lgbm_champion_v1.txt"), "w") as f:
    f.write("tree\nversion=v4\nnum_class=1\nobjective=tweedie tweedie_variance_power:1.24\n"
            "feature_names=venta_lag_1 venta_lag_7 venta_media_movil_7 intensidad_promo dia_semana\n")
    for i in range(120):
        f.write(f"\nTree={i}\nnum_leaves=31\n")
with open(os.path.join(outs, "inference_metadata_v1.json"), "w") as f:
    json.dump({"modelo": "lgbm_champion_v1", "fecha_corte": "2026-05-31", "features": ["venta_lag_1", "venta_lag_7"]}, f)
pd.DataFrame({"store_code": [1001], "category_id": [101], "fecha": [pd.Timestamp("2026-05-31")], "venta_lag_1": [10.0]}).to_parquet(
    os.path.join(outs, "last_state_lags_v1.parquet"), index=False)
print("líneas generadas:", sum(len(v) for v in filas_por_dia.values()))
