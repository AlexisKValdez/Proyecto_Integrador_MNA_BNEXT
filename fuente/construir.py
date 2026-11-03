"""Arma los notebooks autocontenidos: inserta las celdas comunes (configuración, apoyo, columnas, promociones, …) en cada
cuerpo. Carpetas: 1_datos (00–13), 2_modelos (20, 21, Avance5.43), 3_operacion (22, 23, 24, 30, 31, 40)."""
import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
NOMBRES = ("config", "apoyo", "columnas", "estadisticas", "graficas", "eventos", "promociones", "diccionario", "modelo")
PARTES = {f"# @@{k.upper()}": open(os.path.join(BASE, "partes", f"{k}.py"), encoding="utf-8").read().rstrip() + "\n"
          for k in NOMBRES}
salida = os.path.join(BASE, "notebooks", "mna_equipo43")
shutil.rmtree(os.path.join(BASE, "notebooks"), ignore_errors=True)
for carpeta in ("1_datos", "2_modelos", "3_operacion"):
    os.makedirs(os.path.join(salida, carpeta), exist_ok=True)


def carpeta_de(nombre):
    if nombre.startswith("Avance"):
        return "2_modelos"
    if not nombre[0].isdigit():
        return ""
    if nombre.startswith("2"):
        return "2_modelos" if nombre[:2] in ("20", "21") else "3_operacion"
    if nombre[0] in "34":
        return "3_operacion"
    return "1_datos"


for nombre in sorted(os.listdir(os.path.join(BASE, "cuerpos"))):
    if not nombre.endswith(".py"):
        continue
    s = open(os.path.join(BASE, "cuerpos", nombre), encoding="utf-8").read()
    for marca, contenido in PARTES.items():
        s = s.replace(marca + "\n", contenido)
    assert "# @@" not in s, f"marca sin reemplazar en {nombre}"
    compile(s.replace("# MAGIC", "#"), nombre, "exec")          # error de sintaxis = no se entrega
    destino = os.path.join(salida, carpeta_de(nombre), nombre)
    open(destino, "w", encoding="utf-8").write(s)
    print(f"{nombre:38s} {s.count('# COMMAND ----------') + 1:3d} celdas  {len(s) / 1024:6.1f} KB  → {carpeta_de(nombre) or 'raíz'}")
