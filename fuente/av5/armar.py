"""Une las partes del notebook del Avance 5 (av5/NN_*.py) en cuerpos/Avance5.43.py."""
import glob
import os

BASE = os.path.dirname(os.path.abspath(__file__))
partes = sorted(glob.glob(os.path.join(BASE, "[0-9][0-9]_*.py")))
texto = "".join(open(p, encoding="utf-8").read().rstrip() + "\n" for p in partes)
destino = os.path.join(BASE, "..", "cuerpos", "Avance5.43.py")
open(destino, "w", encoding="utf-8").write(texto)
print(f"{len(partes)} partes → {os.path.normpath(destino)} ({texto.count('# COMMAND ----------') + 1} celdas, {len(texto) / 1024:.0f} KB)")
