"""Funções de visualização compartilhadas pelos notebooks (estilo idêntico ao do Módulo 02)."""
from __future__ import annotations

from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .dados import FIGURAS


def configurar_estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "image.interpolation": "nearest",
        "font.family": "DejaVu Sans",
    })


def salvar_figura(fig, nome: str) -> str:
    """Salva a figura em outputs/figures/<nome>.png e devolve o caminho."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / f"{nome}.png"
    fig.savefig(destino, bbox_inches="tight", facecolor="white")
    return str(destino)


def grade(imagens: Sequence[np.ndarray], titulos: Sequence[str] | None = None, colunas: int = 6,
          largura: float = 2.4, cmap: str | None = None, cores_titulo: Sequence[str] | None = None):
    """Várias imagens em grade, sem eixos."""
    n = len(imagens)
    linhas = int(np.ceil(n / colunas))
    fig, axs = plt.subplots(linhas, colunas, figsize=(largura * colunas, largura * linhas * 1.05), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for i, img in enumerate(imagens):
        ax = axs.ravel()[i]
        ax.imshow(img, cmap=(cmap or "gray") if img.ndim == 2 else None)
        if titulos is not None:
            ax.set_title(titulos[i], fontsize=8, color=cores_titulo[i] if cores_titulo else "black")
    fig.tight_layout()
    return fig


def historico(hist, ax=None, metrica: str = "auc", titulo: str | None = None):
    """Curvas de treino × validação de uma métrica (marca a troca de fase do transfer learning)."""
    ax = ax or plt.gca()
    ax.plot(hist.epoca, hist[metrica], "o-", ms=3, label="treino")
    ax.plot(hist.epoca, hist[f"val_{metrica}"], "o-", ms=3, label="validação")
    if "fase" in hist and hist.fase.nunique() > 1:
        ax.axvline(hist[hist.fase == hist.fase.iloc[-1]].epoca.min() - 0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("época")
    ax.set_ylabel(metrica)
    ax.legend(frameon=False)
    if titulo:
        ax.set_title(titulo)
    return ax


def curva_roc(y, p, ax=None, rotulo: str = "", ponto: tuple[float, float] | None = None, **kw):
    """Curva ROC (sensibilidade × 1 − especificidade); `ponto` = (sens, esp) do limiar escolhido."""
    ax = ax or plt.gca()
    fpr, tpr, _ = roc_curve(y, p)
    linha, = ax.plot(fpr, tpr, label=rotulo, **kw)
    if ponto is not None:
        ax.plot(1 - ponto[1], ponto[0], "o", color=linha.get_color(), ms=6)
    ax.plot([0, 1], [0, 1], color="lightgray", lw=1, ls="--")
    ax.set_xlabel("1 − especificidade")
    ax.set_ylabel("sensibilidade")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return ax


def sobrepor_mapa(rgb: np.ndarray, mapa: np.ndarray, alpha: float = 0.45, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Pinta um mapa de calor [0, 1] (qualquer resolução) sobre a imagem RGB uint8."""
    h, w = rgb.shape[:2]
    m = cv2.resize(mapa.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    cor = cv2.cvtColor(cv2.applyColorMap((m * 255).astype(np.uint8), cmap), cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * rgb + alpha * cor).clip(0, 255).astype(np.uint8)
