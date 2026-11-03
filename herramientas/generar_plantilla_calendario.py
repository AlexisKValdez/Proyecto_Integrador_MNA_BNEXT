"""Genera la plantilla editable del calendario de promociones planeadas (Excel + CSV) con un placeholder inventado.

Uso: python generar_plantilla_calendario.py <carpeta_destino>
El equipo sustituye las filas 'placeholder' por el plan real de Bnext (fuente = 'bnext') y vuelve a subir el archivo al
volumen outputs/calendario/; 13_calendario_promociones_plan lo valida y lo convierte en tablas.
"""
import csv
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

COLUMNAS = ["promo_id", "nombre", "inicio_planeado", "fin_planeado", "categoria_id", "categoria_nombre", "tienda", "zona",
            "region", "descuento_planeado", "tipo_mecanica", "fecha_publicacion", "fuente", "comentario"]
# Placeholder INVENTADO (sustituir): promociones nacionales por categoría para oct–dic 2026. Los nombres de categoría se
# traducen a categoria_id con silver.catalogo_categorias; los que no existan se reportan y se omiten.
FILAS = [
    ("PH-2026-10-OTONO", "Arranque de otoño refrescos y cerveza", "2026-10-02", "2026-10-11", "", "REFRESCOS", "", "", "", 0.10, "descuento", "2026-09-20", "placeholder", "INVENTADA: cae dentro del horizonte del pronóstico de octubre"),
    ("PH-2026-10-OTONO", "Arranque de otoño refrescos y cerveza", "2026-10-02", "2026-10-11", "", "CERVEZA", "", "", "", 0.10, "descuento", "2026-09-20", "placeholder", "INVENTADA"),
    ("PH-2026-10-HALLOWEEN", "Halloween botanas y dulces", "2026-10-26", "2026-10-31", "", "BOTANAS", "", "", "", 0.10, "descuento", "2026-10-10", "placeholder", "INVENTADA: sustituir por el plan real"),
    ("PH-2026-10-HALLOWEEN", "Halloween botanas y dulces", "2026-10-26", "2026-10-31", "", "CACAHUATE Y SEMILLAS", "", "", "", 0.10, "descuento", "2026-10-10", "placeholder", "INVENTADA"),
    ("PH-2026-11-DIA-MUERTOS", "Día de Muertos pan y café", "2026-10-30", "2026-11-02", "", "PAN Y GALLETA", "", "", "", 0.15, "descuento", "2026-10-15", "placeholder", "INVENTADA"),
    ("PH-2026-11-DIA-MUERTOS", "Día de Muertos pan y café", "2026-10-30", "2026-11-02", "", "CAFES", "", "", "", 0.15, "descuento", "2026-10-15", "placeholder", "INVENTADA"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "CERVEZA", "", "", "", 0.20, "descuento", "2026-10-20", "placeholder", "INVENTADA; fechas del Buen Fin por confirmar"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "REFRESCOS", "", "", "", 0.20, "descuento", "2026-10-20", "placeholder", "INVENTADA"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "BOTANAS", "", "", "", 0.20, "descuento", "2026-10-20", "placeholder", "INVENTADA"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "AGUAS", "", "", "", 0.15, "descuento", "2026-10-20", "placeholder", "INVENTADA"),
    ("PH-2026-11-2X1-YOGURT", "2x1 yogurt noviembre", "2026-11-17", "2026-11-30", "", "YOGURT", "", "", "", 0.50, "lleva 2 paga 1", "2026-11-01", "placeholder", "INVENTADA; 2x1 ≈ 50 % de descuento"),
    ("PH-2026-12-POSADAS", "Posadas cerveza y botanas", "2026-12-12", "2026-12-24", "", "CERVEZA", "", "", "", 0.10, "descuento", "2026-11-25", "placeholder", "INVENTADA"),
    ("PH-2026-12-POSADAS", "Posadas cerveza y botanas", "2026-12-12", "2026-12-24", "", "BOTANAS", "", "", "", 0.10, "descuento", "2026-11-25", "placeholder", "INVENTADA"),
    ("PH-2026-12-POSADAS", "Posadas cerveza y botanas", "2026-12-12", "2026-12-24", "", "REFRESCOS", "", "", "", 0.10, "descuento", "2026-11-25", "placeholder", "INVENTADA"),
    ("PH-2026-12-NAVIDAD", "Navidad y Año Nuevo", "2026-12-20", "2026-12-31", "", "CERVEZA", "", "", "", 0.15, "descuento", "2026-12-01", "placeholder", "INVENTADA; se traslapa con posadas a propósito"),
    ("PH-2026-12-NAVIDAD", "Navidad y Año Nuevo", "2026-12-20", "2026-12-31", "", "CHICHARRONES", "", "", "", 0.15, "descuento", "2026-12-01", "placeholder", "INVENTADA"),
    ("PH-2026-11-REGIONAL-EJEMPLO", "Ejemplo de alcance regional", "2026-11-20", "2026-11-22", "", "PALOMITAS", "", "", "NORTE", 0.20, "descuento", "2026-11-05", "placeholder", "INVENTADA: muestra cómo acotar por región (si la región no existe, no aplica a ninguna tienda)"),
]
INSTRUCCIONES = [
    "CALENDARIO DE PROMOCIONES PLANEADAS · Equipo 43",
    "",
    "1. Cada fila es una promoción × categoría × alcance. La misma promo_id puede repetirse en varias categorías o alcances.",
    "2. Obligatorias: promo_id, inicio_planeado, fin_planeado (AAAA-MM-DD) y categoria_id o categoria_nombre.",
    "   categoria_nombre se traduce a categoria_id con silver.catalogo_categorias (mismo texto, mayúsculas); el notebook 13",
    "   deja el catálogo en outputs/calendario/catalogo_categorias.csv para consultarlo.",
    "3. Alcance: tienda, zona y region vacíos = nacional. Llena sólo el nivel que aplique (p. ej. region = NORTE).",
    "4. descuento_planeado en fracción (0.15 = 15 %). Para 2x1 escribe 0.50; para 3x2, 0.33. Vacío = se usa 0.",
    "5. fecha_publicacion = desde cuándo se conoce el plan. Vacío = inicio_planeado − 14 días. Importa para el histórico:",
    "   el modelo sólo usa en cada fecha de pronóstico las promociones ya publicadas (así no hay fuga).",
    "6. fuente: 'bnext' para el plan real; 'placeholder' marca filas inventadas que hay que sustituir.",
    "7. Al terminar: guardar, subir el archivo al volumen /Volumes/modeladoprescriptivoventas/maestria/outputs/calendario/",
    "   (Catalog → Volumes → outputs → calendario → Upload) y correr 13_calendario_promociones_plan.",
    "8. Las promociones pasadas no se capturan aquí: el notebook 13 las toma de la vigencia observada (09) automáticamente.",
]


def main(destino):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "promociones"
    ws.append(COLUMNAS)
    for f in FILAS:
        ws.append(list(f))
    for i, c in enumerate(COLUMNAS, 1):
        ws.cell(row=1, column=i).font = Font(bold=True, color="FFFFFF")
        ws.cell(row=1, column=i).fill = PatternFill("solid", fgColor="1F4E78")
        ws.column_dimensions[get_column_letter(i)].width = max(14, min(42, len(c) + 4))
    ws.column_dimensions["N"].width = 60
    ws.freeze_panes = "A2"
    wi = wb.create_sheet("instrucciones")
    for linea in INSTRUCCIONES:
        wi.append([linea])
    wi.column_dimensions["A"].width = 120
    wi["A1"].font = Font(bold=True, size=13)
    for fila in wi.iter_rows():
        for celda in fila:
            celda.alignment = Alignment(wrap_text=False)
    wc = wb.create_sheet("categorias_referencia")
    wc.append(["categoria_id", "nombre", "tipo_categoria", "(el notebook 13 deja el catálogo real en catalogo_categorias.csv del volumen)"])
    wb.save(destino / "calendario_promociones_plan.xlsx")
    with open(destino / "calendario_promociones_plan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNAS)
        w.writerows(FILAS)
    print(f"plantilla escrita en {destino} ({len(FILAS)} filas placeholder)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
