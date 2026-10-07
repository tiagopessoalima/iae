"""Métricas de desempenho diagnóstico para triagem binária.

Convenção: ``y`` = 1 quando a condição está presente; ``p`` = probabilidade prevista; ``limiar`` decide
"encaminhar ao oftalmologista" (p ≥ limiar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def matriz_confusao(y: np.ndarray, p: np.ndarray, limiar: float) -> dict[str, int]:
    y, pred = np.asarray(y).astype(bool), np.asarray(p) >= limiar
    return {"VP": int((pred & y).sum()), "FN": int((~pred & y).sum()),
            "FP": int((pred & ~y).sum()), "VN": int((~pred & ~y).sum())}


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def metricas_no_limiar(y: np.ndarray, p: np.ndarray, limiar: float) -> dict[str, float]:
    """Sensibilidade, especificidade, VPP, VPN, acurácia, F1 e taxa de encaminhamento num limiar."""
    m = matriz_confusao(y, p, limiar)
    vp, fn, fp, vn = m["VP"], m["FN"], m["FP"], m["VN"]
    n = vp + fn + fp + vn
    sens, esp = _div(vp, vp + fn), _div(vn, vn + fp)
    vpp, vpn = _div(vp, vp + fp), _div(vn, vn + fn)
    return {"limiar": limiar, "sensibilidade": sens, "especificidade": esp, "VPP": vpp, "VPN": vpn,
            "acuracia": _div(vp + vn, n), "F1": _div(2 * vp, 2 * vp + fp + fn),
            "encaminhados": _div(vp + fp, n), **m}


def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def limiar_para_sensibilidade(y: np.ndarray, p: np.ndarray, sensibilidade_alvo: float = 0.85) -> float:
    """O **maior** limiar que ainda atinge a sensibilidade alvo (logo, a maior especificidade possível com ela).

    Deve ser escolhido na **validação** e depois aplicado, sem mudança, no teste e na base externa.
    """
    fpr, tpr, lim = roc_curve(y, p)
    ok = np.where(tpr >= sensibilidade_alvo)[0]
    return float(min(lim[ok[0]], 1.0))


def limiar_youden(y: np.ndarray, p: np.ndarray) -> float:
    """Limiar que maximiza sensibilidade + especificidade − 1 (índice de Youden)."""
    fpr, tpr, lim = roc_curve(y, p)
    return float(min(lim[np.argmax(tpr - fpr)], 1.0))


def bootstrap(y: np.ndarray, p: np.ndarray, funcao, n: int = 1000, semente: int = 0,
              grupos: np.ndarray | None = None) -> tuple[float, float, float]:
    """(estimativa, IC 95 % inferior, superior) de `funcao(y, p)` por bootstrap percentil.

    Com `grupos` (ex.: paciente), reamostra grupos inteiros — os dois olhos de uma pessoa não são independentes.
    """
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(semente)
    est = funcao(y, p)
    if grupos is None:
        indices = (rng.integers(0, len(y), len(y)) for _ in range(n))
    else:
        grupos = np.asarray(grupos)
        unicos = np.unique(grupos)
        por_grupo = {g: np.where(grupos == g)[0] for g in unicos}
        indices = (np.concatenate([por_grupo[g] for g in rng.choice(unicos, len(unicos))]) for _ in range(n))
    vals = []
    for idx in indices:
        if 0 < y[idx].sum() < len(idx):
            vals.append(funcao(y[idx], p[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(est), float(lo), float(hi)


def ic_proporcao(k: int, n: int) -> tuple[float, float]:
    """IC 95 % de Wilson para uma proporção k/n (ex.: sensibilidade = VP / (VP + FN))."""
    if n == 0:
        return float("nan"), float("nan")
    z, ph = 1.959964, k / n
    centro = (ph + z * z / (2 * n)) / (1 + z * z / n)
    meia = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centro - meia), float(centro + meia)


def relatorio(y: np.ndarray, p: np.ndarray, limiar: float, grupos: np.ndarray | None = None,
              n_boot: int = 1000) -> dict[str, float | str]:
    """Resumo para tabela: n, prevalência, AUC [IC], sensibilidade [IC] e especificidade [IC] no limiar."""
    y = np.asarray(y).astype(int)
    m = metricas_no_limiar(y, p, limiar)
    a, a_lo, a_hi = bootstrap(y, p, auc, n=n_boot, grupos=grupos)
    s_lo, s_hi = ic_proporcao(m["VP"], m["VP"] + m["FN"])
    e_lo, e_hi = ic_proporcao(m["VN"], m["VN"] + m["FP"])
    return {"n": len(y), "positivos": int(y.sum()), "prevalencia": y.mean(),
            "AUC": a, "AUC_IC": f"{a_lo:.3f}–{a_hi:.3f}",
            "sensibilidade": m["sensibilidade"], "sens_IC": f"{s_lo:.2f}–{s_hi:.2f}",
            "especificidade": m["especificidade"], "esp_IC": f"{e_lo:.2f}–{e_hi:.2f}",
            "VPP": m["VPP"], "VPN": m["VPN"], "encaminhados": m["encaminhados"]}


def curva_calibracao(y: np.ndarray, p: np.ndarray, faixas: int = 10) -> pd.DataFrame:
    """Probabilidade média prevista × frequência observada, por faixa de probabilidade (quantis)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["faixa"] = pd.qcut(df.p.rank(method="first"), faixas, labels=False)
    return df.groupby("faixa").agg(p_media=("p", "mean"), observado=("y", "mean"), n=("y", "size")).reset_index()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Erro quadrático médio da probabilidade (0 = perfeito; 0,25 = sempre dizer 50 %)."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def vpp_por_prevalencia(sensibilidade: float, especificidade: float, prevalencias) -> np.ndarray:
    """VPP esperado (teorema de Bayes) para uma lista de prevalências — o mesmo teste rende VPP muito diferente
    numa campanha de rua e num ambulatório de diabéticos."""
    pr = np.asarray(prevalencias, dtype=float)
    return sensibilidade * pr / (sensibilidade * pr + (1 - especificidade) * (1 - pr))
