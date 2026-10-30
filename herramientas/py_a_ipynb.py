"""Convierte un notebook en formato fuente de Databricks (.py con '# COMMAND ----------') a Jupyter (.ipynb).

Uso: python py_a_ipynb.py <entrada.py> <salida.ipynb>
Las celdas '# MAGIC %md' pasan a celdas de texto; '# MAGIC %pip …' a celdas de código con el comando mágico.
"""
import json
import sys


def convertir(texto: str) -> dict:
    texto = texto.replace("# Databricks notebook source\n", "", 1)
    celdas = []
    for bloque in texto.split("\n# COMMAND ----------\n"):
        lineas = bloque.strip("\n").split("\n")
        if not any(l.strip() for l in lineas):
            continue
        if all(l.startswith("# MAGIC") for l in lineas if l.strip()):
            contenido = [l[8:] if l.startswith("# MAGIC ") else l[7:] for l in lineas]
            if contenido and contenido[0].strip() == "%md":
                celdas.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in contenido[1:]]})
                continue
            celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": [l + "\n" for l in contenido]})
            continue
        celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": [l + "\n" for l in lineas]})
    for i, c in enumerate(celdas):
        c["id"] = f"celda-{i:03d}"            # nbformat 4.5 exige un identificador único por celda
        if c["source"]:
            c["source"][-1] = c["source"][-1].rstrip("\n")
    return {"cells": celdas, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    entrada, salida = sys.argv[1], sys.argv[2]
    nb = convertir(open(entrada, encoding="utf-8").read())
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"{salida}: {len(nb['cells'])} celdas")
