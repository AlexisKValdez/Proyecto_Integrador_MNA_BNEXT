"""Genera docs/DICCIONARIO_DATOS.md a partir del diccionario de variables de la Oro (partes/diccionario.py), del contrato
del calendario (partes/promociones.py) y de las tablas de salida (partes/modelo.py)."""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if not (BASE / "partes").exists() and (BASE / "fuente" / "partes").exists():   # en el repositorio la fuente vive en fuente/
    BASE = BASE / "fuente"
ns = {"tabla": lambda capa, n: f"{capa}.{n}", "CATALOGO": "modeladoprescriptivoventas", "ESQUEMA_DATOS": "maestria"}
exec(open(BASE / "partes" / "diccionario.py", encoding="utf-8").read(), ns)
src = open(BASE / "partes" / "promociones.py", encoding="utf-8").read()
exec(src[src.index("ESQUEMA_CALENDARIO_PROMOS = {"):src.index("def validar_calendario_promociones")], ns)
src = open(BASE / "partes" / "modelo.py", encoding="utf-8").read()
exec(src[src.index("DESCRIPCION_PRONOSTICO = {"):src.index("def cargar_modelo_operativo")], ns)

TABLAS = [
    ("bronze.ventas_lineas", "02", "Líneas de venta tal como llegan (partición anio_mes) con archivo de origen y fecha de carga."),
    ("silver.ventas_lineas", "08", "Líneas limpias, únicas y homologadas (partición anio_mes)."),
    ("silver.catalogo_categorias / catalogo_marcas / catalogo_tiendas", "08", "Dimensiones homologadas; tiendas con zona, región (imputada por zona si falta), jefatura, estatus y ventana activa."),
    ("silver.panel_categoria_dia", "09", "Una fila por tienda × categoría × día: venta, unidades, tickets, descuento, promociones vigentes, banderas de tienda inactiva, día sin datos y atípico."),
    ("silver.promociones / silver.calendario_promociones", "09", "Promociones observadas: vigencia (inicio, fin, censura) y sus tiendas × categorías."),
    ("silver.calendario_promociones_plan", "13", "Plan de promociones: una fila por promoción × categoría × alcance, con fecha_publicacion y fuente (bnext / placeholder / historico_observado)."),
    ("silver.calendario_promociones_plan_dia", "13", "Tienda × categoría × día con las promociones planeadas que lo cubren (arreglo de promo_id, fecha_publicacion, descuento, restantes, fuente)."),
    ("gold.features_origen", "20", "Tabla Oro: serie × origen × horizonte con objetivo y variables (sólo días ≤ origen); conjuntos entrenamiento / evaluacion / pronostico."),
    ("gold.pronostico_14d", "22 / Avance5.43", "Pronóstico operativo por tienda × categoría × día con P10/P50/P90 y techo; reemplaza sólo su origen."),
    ("gold.escenarios_whatif / gold.escenarios_whatif_resumen", "30", "Escenarios de promoción (plan, sin promo, 10/20/30 %) por serie × día y agregados por región × tipo × categoría."),
    ("gold.uplift_promocional_historico", "30", "Venta con promoción / sin promoción por categoría y profundidad (misma tienda, mes y día de semana); referencia descriptiva."),
    ("gold.predicciones_sku / gold.surtido_sku_semana", "31", "Pronóstico repartido a producto por participación de 28 días (fila OTROS) y vista semanal con unidades P50 y techo P90."),
    ("gold.dash_*", "40", "Tablas del tablero: resumen ejecutivo, serie de tiempo por nivel, precisión, promociones, alertas, estado del pipeline."),
    ("mlops.hallazgos / mlops.corridas", "todos", "Resumen de cada corrida de cada notebook y sus tiempos."),
    ("mlops.validaciones", "11", "Resultado de cada regla de calidad (PASA / FALLA, crítica o aviso) por corrida."),
    ("mlops.av5_*", "Avance5.43", "Resultados del Avance 5 por run_id: comparativas, por corte, por horizonte, diario, segmentos, calibración, residuos, cuantiles, DM, decisión, pesos, ajuste, costos, auditoría, parámetros, importancia, compuerta, resumen, interpretaciones."),
    ("mlops.av5_ejecuciones", "Avance5.43 / 24", "Bitácora de eventos de cada corrida (INICIADA, paso_completado, FINALIZADA, ERROR, REANUDADA, PROMOVIDO)."),
    ("mlops.monitoreo_modelo / mlops.monitoreo_horizonte", "Avance5.43 / 23", "WMAPE, sesgo y cobertura por semana × región × tipo (fase prueba y produccion) y por horizonte."),
    ("mlops.deriva_variables", "23", "PSI de las variables principales entre el entrenamiento y el conjunto que se pronostica."),
    ("mlops.alertas", "23", "Alertas con tipo, severidad, valor, umbral y si fueron atendidas."),
    ("mlops.aprobaciones", "24", "Aprobaciones de negocio (alcance, servicios, atributos) con responsable y fecha."),
    ("mlops.registro_modelos", "24", "Promociones y cambios de alias del modelo en Unity Catalog."),
    ("mlops.pronosticos_bitacora", "22", "Cada corrida de pronóstico: origen, fase, modelo, versión, filas."),
    ("mlops.calendario_plan_versiones", "13", "Versiones cargadas del archivo del calendario."),
]

out = ["# Diccionario de datos — Equipo 43", "",
       "Nombre real de cada tabla: `modeladoprescriptivoventas.maestria.<capa>_<nombre>` (por ejemplo `gold.pronostico_14d` → `gold_pronostico_14d`).",
       "Los comentarios de columna también están escritos en las tablas (Catalog Explorer).", "", "## Tablas", "",
       "| Tabla | La escribe | Contenido |", "| --- | --- | --- |"]
out += [f"| `{t}` | {q} | {d} |" for t, q, d in TABLAS]
out += ["", "## Variables de la tabla Oro (`gold.features_origen`)", "", "### Metadatos", "", "| Columna | Significado |", "| --- | --- |"]
out += [f"| `{k}` | {v} |" for k, v in ns["DESCRIPCION_METADATOS"].items()]
out += ["", "### Variables del modelo", "", "| Variable | Significado |", "| --- | --- |"]
out += [f"| `{k}` | {v} |" for k, v in ns["DESCRIPCION_VARIABLES"].items()]
out += ["", "## Contrato del calendario de promociones (archivo editable)", "", "| Columna | Tipo y significado |", "| --- | --- |"]
out += [f"| `{k}` | {v} |" for k, v in ns["ESQUEMA_CALENDARIO_PROMOS"].items()]
out += ["", "## Pronóstico operativo (`gold.pronostico_14d`)", "", "| Columna | Significado |", "| --- | --- |"]
out += [f"| `{k}` | {v} |" for k, v in ns["DESCRIPCION_PRONOSTICO"].items()]
destino = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "DICCIONARIO_DATOS.md"
destino.write_text("\n".join(out) + "\n", encoding="utf-8")
print(destino, len(out), "líneas")
