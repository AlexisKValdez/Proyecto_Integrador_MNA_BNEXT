# ─────────────────────── estadísticas del EDA (concentración, intermitencia, segmentos) ───────────────────────
def _arr(x):
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


# ----------------------------------------------------------------------------- concentración

def gini(valores):
    """Índice de Gini de valores no negativos: 0 = todos iguales, cercano a 1 = uno concentra todo."""
    x = np.sort(_arr(valores))
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    n = x.size
    i = np.arange(1, n + 1)
    return float((2 * np.sum(i * x)) / (n * x.sum()) - (n + 1) / n)


def curva_lorenz(valores, puntos=101):
    """(fracción acumulada de entidades, fracción acumulada del valor), de menor a mayor."""
    x = np.sort(_arr(valores))
    if x.size == 0 or x.sum() == 0:
        return np.linspace(0, 1, puntos), np.linspace(0, 1, puntos)
    acumulado = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    fraccion = np.linspace(0, 1, acumulado.size)
    malla = np.linspace(0, 1, puntos)
    return malla, np.interp(malla, fraccion, acumulado)


def participacion_top(valores, fracciones=(0.01, 0.05, 0.10, 0.20, 0.50)):
    """Parte del total que concentra el ``f`` % de entidades más grandes, para cada ``f``."""
    x = np.sort(_arr(valores))[::-1]
    total = x.sum()
    salida = {}
    for f in fracciones:
        k = max(1, int(np.ceil(f * x.size))) if x.size else 0
        salida[f] = float(x[:k].sum() / total) if total else float("nan")
    return salida


def fraccion_para(valores, objetivo=0.80):
    """Fracción de entidades (de mayor a menor) necesaria para acumular ``objetivo`` del total."""
    x = np.sort(_arr(valores))[::-1]
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    acumulado = np.cumsum(x) / x.sum()
    k = int(np.searchsorted(acumulado, objetivo - 1e-12) + 1)
    return float(min(k, x.size) / x.size)


# ----------------------------------------------------------------------------- intermitencia y segmentos

ADI_CORTE = 1.32
CV2_CORTE = 0.49
CLASES_SB = ("suave", "errática", "intermitente", "grumosa")


def clasificar_sb(adi, cv2, adi_corte=ADI_CORTE, cv2_corte=CV2_CORTE):
    """Clasificación de Syntetos y Boylan (2005) de una serie de demanda.

    * ``adi``: intervalo medio entre días con venta (días activos / días con venta).
    * ``cv2``: coeficiente de variación al cuadrado del tamaño de la venta en los días con venta.

    suave (ADI < 1.32, CV² < 0.49) · errática (ADI < 1.32, CV² ≥ 0.49) ·
    intermitente (ADI ≥ 1.32, CV² < 0.49) · grumosa (ADI ≥ 1.32, CV² ≥ 0.49).
    """
    adi = np.asarray(adi, dtype=float)
    cv2 = np.asarray(cv2, dtype=float)
    clase = np.where(adi < adi_corte,
                     np.where(cv2 < cv2_corte, CLASES_SB[0], CLASES_SB[1]),
                     np.where(cv2 < cv2_corte, CLASES_SB[2], CLASES_SB[3]))
    return np.where(np.isnan(adi) | np.isnan(cv2), "sin_datos", clase)


def clase_abc(valores, cortes=(0.80, 0.95)):
    """A = el conjunto mínimo de entidades (de mayor a menor) que alcanza el 80 % del valor;
    B = las siguientes hasta alcanzar el 95 %; C = el resto."""
    x = np.asarray(valores, dtype=float)
    orden = np.argsort(-x, kind="stable")
    acumulado = np.cumsum(x[orden]) / x.sum() if x.sum() else np.zeros_like(x)
    anterior = np.concatenate([[0.0], acumulado[:-1]])
    etiquetas = np.where(anterior < cortes[0], "A", np.where(anterior < cortes[1], "B", "C"))
    salida = np.empty(x.size, dtype=object)
    salida[orden] = etiquetas
    return salida.astype(str)


def clase_xyz(cv, cortes=(0.5, 1.0)):
    """X = estable (CV < 0.5), Y = variable (0.5–1.0), Z = muy variable (> 1.0); CV de la venta diaria."""
    cv = np.asarray(cv, dtype=float)
    return np.where(np.isnan(cv), "sin_datos", np.where(cv < cortes[0], "X", np.where(cv <= cortes[1], "Y", "Z")))


# ----------------------------------------------------------------------------- series

def autocorrelacion(x, rezagos=(1, 7, 14, 28)):
    """Autocorrelación muestral de una serie en los rezagos pedidos (NaN si la serie es constante)."""
    a = np.asarray(x, dtype=float)
    a = a - np.nanmean(a)
    den = np.nansum(a * a)
    salida = {}
    for k in rezagos:
        if den == 0 or k >= a.size:
            salida[k] = float("nan")
        else:
            salida[k] = float(np.nansum(a[k:] * a[:-k]) / den)
    return salida


def indice_relativo(valores, grupos):
    """Media por grupo dividida entre la media de las medias de grupo (1.0 = día promedio)."""
    v = np.asarray(valores, dtype=float)
    g = np.asarray(grupos)
    medias = {k: float(np.nanmean(v[g == k])) for k in sorted(set(g.tolist()))}
    base = np.nanmean(list(medias.values()))
    return {k: (m / base if base else float("nan")) for k, m in medias.items()}
