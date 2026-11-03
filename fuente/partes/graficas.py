# ─────────────────────── gráficas (un solo estilo para todo el proyecto) ───────────────────────
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

AZUL, GRIS, NARANJA, TINTA, SUAVE = "#2a78d6", "#8a8f98", "#d9822b", "#1f2328", "#5b616b"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c9ccd1",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e8eb",
                     "grid.linewidth": 0.6, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
                     "axes.labelcolor": SUAVE, "xtick.color": SUAVE, "ytick.color": SUAVE, "font.size": 9,
                     "axes.titlelocation": "left", "legend.frameon": False})


def grafica_serie(fechas, valores, titulo, unidad, eventos=None, color=AZUL, alto=2.8):
    """Serie diaria en una línea delgada, un solo eje; las fechas de eventos se marcan con líneas punteadas."""
    fig, ax = plt.subplots(figsize=(11, alto))
    ax.plot(pd.to_datetime(fechas), valores, color=color, linewidth=1.6)
    lo, hi = pd.to_datetime(fechas).min(), pd.to_datetime(fechas).max()
    for e in (eventos or []):
        f = pd.Timestamp(e["fecha"])
        if lo <= f <= hi:
            ax.axvline(f, color=NARANJA if e.get("tipo") == "futbol" else GRIS, linewidth=0.8, linestyle=":")
    ax.set_title(titulo)
    ax.set_ylabel(unidad)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.set_xlim(lo, hi)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def grafica_barras(etiquetas, valores, titulo, unidad, resaltar=None, formato="{:.2f}", referencia=None, horizontal=False):
    """Barras de una sola serie; la barra resaltada va en azul y las demás en gris; valor escrito en cada barra."""
    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.28 * len(etiquetas)) if horizontal else 3))
    colores = [AZUL if (resaltar is None or e in resaltar) else "#b8c4d6" for e in etiquetas]
    if horizontal:
        ax.barh(list(etiquetas)[::-1], list(valores)[::-1], color=colores[::-1], height=0.7)
        for i, v in enumerate(list(valores)[::-1]):
            ax.text(v, i, " " + formato.format(v), va="center", color=TINTA, fontsize=8)
        ax.set_xlabel(unidad)
    else:
        ax.bar(list(etiquetas), list(valores), color=colores, width=0.7)
        for i, v in enumerate(valores):
            ax.text(i, v, formato.format(v), ha="center", va="bottom", color=TINTA, fontsize=8)
        ax.set_ylabel(unidad)
    if referencia is not None:
        (ax.axvline if horizontal else ax.axhline)(referencia, color=GRIS, linewidth=0.8, linestyle="--")
    ax.set_title(titulo)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
