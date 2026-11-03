"""Pruebas de las funciones puras de la configuración común y del diccionario (pytest, sin Spark)."""
import datetime as dt
import os
import re
from types import SimpleNamespace

import pytest

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FUENTE = os.path.join(RAIZ, "fuente")


def cargar(nombre, extra=None):
    """Ejecuta una parte común en un espacio de nombres mínimo (spark y dbutils simulados)."""
    ns = {"spark": SimpleNamespace(sql=lambda q: SimpleNamespace(collect=lambda: [])), "dbutils": SimpleNamespace(), "print": lambda *a, **k: None}
    ns.update(extra or {})
    codigo = open(os.path.join(FUENTE, "partes", f"{nombre}.py"), encoding="utf-8").read()
    exec(codigo, ns)
    return ns


@pytest.fixture(scope="module")
def config():
    return cargar("config")


def test_cortes_fijos_completos(config):
    dias = [dt.date(2026, 3, 1) + dt.timedelta(days=k) for k in range(214)]          # mar–sep completos
    cortes = config["cortes_walk_forward"](dias)
    assert [c["corte"] for c in cortes] == ["corte_1", "corte_2", "corte_3", "corte_4", "prueba_1", "prueba_2"]
    assert all((c["hasta"] - c["desde"]).days == 13 for c in cortes)
    assert cortes[0]["fase"] == "desarrollo" and cortes[-1]["fase"] == "prueba"


def test_corte_en_hueco_se_omite(config):
    dias = [dt.date(2026, 3, 1) + dt.timedelta(days=k) for k in range(214) if not (dt.date(2026, 7, 2) <= dt.date(2026, 3, 1) + dt.timedelta(days=k) <= dt.date(2026, 7, 10))]
    nombres = [c["corte"] for c in config["cortes_walk_forward"](dias)]
    assert "corte_3" not in nombres and "corte_4" in nombres


def test_sin_septiembre_no_hay_prueba(config):
    dias = [dt.date(2026, 3, 1) + dt.timedelta(days=k) for k in range(184)]          # hasta el 31 de agosto
    assert all(not c["corte"].startswith("prueba") for c in config["cortes_walk_forward"](dias))


@pytest.mark.parametrize("texto, defecto, esperado", [
    ("", True, True), ("False", True, False), ("si", False, True), ("", None, None), ("none", None, None),
    ("av5_43_x", None, "av5_43_x"), ("2026-09", ["2026-03"], ["2026-09"]), ("2026-08, 2026-09", [], ["2026-08", "2026-09"]),
    ('["a","b"]', [], ["a", "b"]), ("7", 5, 7), ("0.25", 0.1, 0.25), ("", [1, 2], [1, 2]),
])
def test_parametro_desde_texto(config, texto, defecto, esperado):
    assert config["_parametro_desde_texto"](texto, defecto) == esperado


def test_parametro_bool_invalido(config):
    with pytest.raises(ValueError):
        config["_parametro_desde_texto"]("tal vez", True)


def test_parametro_sin_widgets_regresa_defecto(config):
    assert config["parametro"]("MODO_RAPIDO", True) is True


def test_nombres_de_tablas(config):
    assert config["tabla"]("gold", "pronostico_14d") == "modeladoprescriptivoventas.maestria.gold_pronostico_14d"
    assert config["volumen"]("2026-09").endswith("/maestria/2026-09")


def test_diccionario_cubre_las_variables():
    ns = cargar("diccionario")
    fuente = open(os.path.join(FUENTE, "av5", "01_configuracion.py"), encoding="utf-8").read()
    numericas = re.findall(r'"([a-z_0-9]+)"', fuente[fuente.index("NUMERICAS = ["):fuente.index("]", fuente.index("NUMERICAS = ["))])
    faltan = [v for v in numericas if v not in ns["DESCRIPCION_VARIABLES"]]
    assert not faltan, faltan
    for v in ("f_promos_planeadas", "f_descuento_planeado", "f_dias_promo_restantes", "s_promos_vigentes_t0"):
        assert v in ns["DESCRIPCION_VARIABLES"]


def test_notebooks_compilan():
    base = os.path.join(RAIZ, "databricks", "mna_equipo43_avance6")
    n = 0
    for carpeta, _, archivos in os.walk(base):
        for a in archivos:
            if a.endswith(".py"):
                s = open(os.path.join(carpeta, a), encoding="utf-8").read()
                for i, celda in enumerate(s.split("# COMMAND ----------")):
                    cuerpo = celda.replace("# Databricks notebook source", "").strip()
                    if cuerpo and not cuerpo.startswith("# MAGIC"):
                        compile(cuerpo, f"{a}[{i}]", "exec")
                n += 1
    assert n >= 20
