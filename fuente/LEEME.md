# Fuente de los notebooks

Los notebooks de `databricks/` se construyen desde aquí: `partes/` (celdas comunes: configuración, apoyo, columnas, promociones, diccionario, modelo), `cuerpos/` (cada notebook sin las celdas comunes) y `av5/` (las 13 partes de Avance5.43). `python3 av5/armar.py && python3 construir.py` regenera `notebooks/`. Editar aquí y reconstruir evita cambiar la misma celda en 25 notebooks.
