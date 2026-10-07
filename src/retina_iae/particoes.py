"""Definição das três tarefas de triagem e divisão dos dados em treino / validação / teste / externo.

Cada tarefa é **binária** (condição presente = 1) e tem uma fonte de desenvolvimento, dividida aqui, e fontes
externas, usadas só na validação final (notebook 06). A divisão é estratificada pelo alvo e, quando a fonte
identifica o paciente (ODIR), feita **por paciente**: os dois olhos da mesma pessoa caem na mesma partição.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAREFAS: dict[str, dict] = {
    "rd": {
        "nome": "Retinopatia diabética referível",
        "desenvolvimento": ["APTOS"],
        "externo": ["DDR", "ODIR"],
        "descricao": "RD moderada ou pior (grau ICDR ≥ 2) × sem RD ou RD leve (graus 0–1). No ODIR, que não informa "
                     "o grau, positivo = 'diabetes' (qualquer grau) e negativo = 'normal'.",
    },
    "glaucoma": {
        "nome": "Glaucoma",
        "desenvolvimento": ["ODIR"],
        "externo": ["G1020"],
        "descricao": "Glaucoma (diagnóstico declarado pela fonte) × normal. O G1020 usa diagnóstico clínico (exame completo), "
                     "nem sempre visível na retinografia: é a base externa mais difícil do módulo.",
    },
    "dmri": {
        "nome": "Degeneração macular relacionada à idade",
        "desenvolvimento": ["ODIR"],
        "externo": ["RFMiD"],
        "descricao": "DMRI × normal. No RFMiD, a DMRI pode coexistir com outros achados (anotados no manifesto).",
    },
}
PARTICOES = ("treino", "validacao", "teste", "externo")
FRACOES_PADRAO = (0.70, 0.15, 0.15)


def alvo(linha: pd.Series, tarefa: str) -> int | None:
    """Alvo binário de uma imagem para a tarefa (1 = condição presente), ou None se a imagem não participa dela."""
    fonte, cond = linha["fonte"], linha["condicao"]
    if tarefa == "rd":
        if fonte in ("APTOS", "DDR"):
            g = int(linha["grau"])
            return None if g > 4 else int(g >= 2)     # DDR grau 5 = ilegível: fica de fora da tarefa
        return {"rd": 1, "normal": 0}.get(cond)
    return {tarefa: 1, "normal": 0}.get(cond)


def dividir(df: pd.DataFrame, semente: int = 2026, fracoes: tuple[float, float, float] = FRACOES_PADRAO) -> pd.Series:
    """Divide as linhas de `df` (colunas `id`, `paciente`, `alvo`) em treino/validacao/teste.

    Estratifica pelo alvo do *grupo* (paciente; ou a própria imagem, se não houver paciente). Um paciente com a
    condição em um só olho conta como positivo, e os dois olhos vão juntos para a mesma partição.
    """
    grupos = df["paciente"].where(df["paciente"].astype(str) != "", df["id"])
    alvo_grupo = df.groupby(grupos)["alvo"].max()
    rng = np.random.default_rng(semente)
    destino: dict[str, str] = {}
    for _, ids in alvo_grupo.groupby(alvo_grupo):
        ids = rng.permutation(ids.index.to_numpy())
        n_tr = round(fracoes[0] * len(ids))
        n_va = round(fracoes[1] * len(ids))
        for k, g in enumerate(ids):
            destino[g] = "treino" if k < n_tr else ("validacao" if k < n_tr + n_va else "teste")
    return grupos.map(destino)


def gerar_particoes(manifesto: pd.DataFrame, semente: int = 2026) -> pd.DataFrame:
    """Tabela (id, tarefa, fonte, alvo, particao) para as três tarefas, a partir do manifesto de imagens."""
    blocos = []
    for tarefa, cfg in TAREFAS.items():
        for papel, fontes in (("desenvolvimento", cfg["desenvolvimento"]), ("externo", cfg["externo"])):
            sub = manifesto[manifesto["fonte"].isin(fontes)].copy()
            if sub.empty:
                continue
            sub["alvo"] = sub.apply(lambda r: alvo(r, tarefa), axis=1)
            sub = sub[sub["alvo"].notna()].copy()
            sub["alvo"] = sub["alvo"].astype(int)
            sub["tarefa"] = tarefa
            sub["particao"] = dividir(sub, semente) if papel == "desenvolvimento" else "externo"
            blocos.append(sub[["id", "tarefa", "fonte", "alvo", "particao", "paciente"]])
    if not blocos:
        return pd.DataFrame(columns=["id", "tarefa", "fonte", "alvo", "particao", "paciente"])
    return pd.concat(blocos, ignore_index=True)
