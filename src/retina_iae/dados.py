"""Manifesto, partições e cache de tensores de entrada.

O cache guarda, para cada imagem do manifesto, o tensor ``uint8`` 224×224×3 produzido por
``retina_pdi.pipeline.preparar_para_inferencia`` (Módulo 02) com a configuração congelada em ``CONFIG_PRE`` —
a mesma função que o Raspberry Pi usa na hora da triagem. Treinar com outra rotina de pré-processamento
quebraria a paridade treino × inferência discutida no notebook 06 do Módulo 02.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
MANIFESTO = DATA / "manifest.csv"
PARTICOES = DATA / "particoes.csv"
CACHE = DATA / "cache"
MODELOS = RAIZ / "modelos"            # .tflite finais + configuração (versionados)
SAIDAS = RAIZ / "outputs"
FIGURAS = SAIDAS / "figures"
KERAS = SAIDAS / "keras"              # modelos .keras treinados (não versionados)

CONDICOES = ("normal", "rd", "glaucoma", "dmri")
NOMES_CONDICAO = {"normal": "Normal", "rd": "Retinopatia diabética", "glaucoma": "Glaucoma", "dmri": "DMRI",
                  "ilegivel": "Sem qualidade (DDR grau 5)"}
TAREFAS = ("rd", "glaucoma", "dmri")

# Pré-processamento congelado: igual ao `config_preprocessamento.json` exportado no notebook 06 do Módulo 02,
# exceto pela escala: guardamos uint8 e cada modelo normaliza internamente (ver retina_iae.modelos).
CONFIG_PRE = {"tamanho": 224, "realce": "clahe", "normalizacao": "uint8", "recortar_fov": True}


# =========================================================================== manifesto e partições


def carregar_manifesto() -> pd.DataFrame:
    """Lê `data/manifest.csv`. Levanta um erro claro se os dados ainda não foram baixados."""
    if not MANIFESTO.exists():
        raise FileNotFoundError(f"{MANIFESTO} não existe. Rode `python scripts/download_data.py` na raiz do projeto.")
    return pd.read_csv(MANIFESTO, keep_default_na=False)


def carregar_particoes(tarefa: str | None = None) -> pd.DataFrame:
    """Lê `data/particoes.csv` (id, tarefa, fonte, alvo, particao, paciente), opcionalmente de uma tarefa."""
    if not PARTICOES.exists():
        raise FileNotFoundError(f"{PARTICOES} não existe. Rode `python scripts/download_data.py`.")
    p = pd.read_csv(PARTICOES, keep_default_na=False)
    return p if tarefa is None else p[p.tarefa == tarefa].reset_index(drop=True)


def caminho_absoluto(arquivo: str | Path) -> Path:
    p = Path(arquivo)
    return p if p.is_absolute() else RAIZ / p


def carregar_rgb(arquivo: str | Path) -> np.ndarray:
    """Imagem uint8 RGB (usa o leitor do Módulo 02, que aceita acentos no caminho)."""
    from retina_pdi import dados as d2
    return d2.carregar_rgb(caminho_absoluto(arquivo))


# =========================================================================== cache de tensores


def _nome_cache(cfg: dict) -> str:
    return f"tensores_{cfg['tamanho']}_{cfg['realce'] or 'sem_realce'}"


def _preparar_um(arquivo: str, cfg: dict) -> np.ndarray:
    from retina_pdi import pipeline as P
    rgb = carregar_rgb(arquivo)
    return P.preparar_para_inferencia(rgb, cfg["tamanho"], cfg["realce"], "uint8", recortar=cfg["recortar_fov"])[0]


def construir_cache(cfg: dict | None = None, n_jobs: int = -1, forcar: bool = False) -> Path:
    """Pré-processa todas as imagens do manifesto e grava `data/cache/<nome>.npy` + índice `.csv` + `.json`.

    ≈ 10 mil imagens → ≈ 1,6 GB (224×224×3 uint8); 1–3 min com todos os núcleos.
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    cfg = {**CONFIG_PRE, **(cfg or {})}
    CACHE.mkdir(parents=True, exist_ok=True)
    base = CACHE / _nome_cache(cfg)
    man = carregar_manifesto()
    if base.with_suffix(".npy").exists() and not forcar:
        idx = pd.read_csv(base.with_suffix(".csv"))
        if len(idx) == len(man) and (idx.id.values == man.id.values).all():
            return base.with_suffix(".npy")
    t = cfg["tamanho"]
    saida = np.lib.format.open_memmap(base.with_suffix(".npy.tmp"), mode="w+", dtype=np.uint8,
                                      shape=(len(man), t, t, 3))
    lotes = [man.arquivo.iloc[i:i + 256].tolist() for i in range(0, len(man), 256)]
    k = 0
    with Parallel(n_jobs=n_jobs) as par:
        for lote in tqdm(lotes, desc=f"cache {base.name}"):
            for x in par(delayed(_preparar_um)(a, cfg) for a in lote):
                saida[k] = x
                k += 1
    saida.flush()
    del saida
    base.with_suffix(".npy.tmp").replace(base.with_suffix(".npy"))
    man[["id"]].to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return base.with_suffix(".npy")


class Cache:
    """Acesso aos tensores pré-processados por id (memória mapeada: só o que é usado vai para a RAM)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = {**CONFIG_PRE, **(cfg or {})}
        base = CACHE / _nome_cache(self.cfg)
        if not base.with_suffix(".npy").exists():
            raise FileNotFoundError(f"{base}.npy não existe. Rode `python scripts/preparar_cache.py`.")
        self.x = np.load(base.with_suffix(".npy"), mmap_mode="r")
        self.pos = pd.Series(np.arange(len(self.x)), index=pd.read_csv(base.with_suffix(".csv")).id)

    def __getitem__(self, ids) -> np.ndarray:
        if isinstance(ids, str):
            return np.asarray(self.x[self.pos[ids]])
        pos = self.pos.loc[list(ids)].to_numpy()
        ordem = np.argsort(pos)                     # leitura sequencial no disco é bem mais rápida
        out = np.empty((len(pos),) + self.x.shape[1:], np.uint8)
        out[ordem] = self.x[pos[ordem]]
        return out


def conjunto(tarefa: str, particao: str | list[str], cache: Cache | None = None, fonte: str | None = None,
             n_max: int | None = None, semente: int = 0) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(X uint8 N×224×224×3, y 0/1, tabela) de uma tarefa e partição(ões). `n_max` sorteia uma amostra estratificada."""
    p = carregar_particoes(tarefa)
    p = p[p.particao.isin([particao] if isinstance(particao, str) else particao)]
    if fonte is not None:
        p = p[p.fonte == fonte]
    if n_max is not None and len(p) > n_max:
        frac = n_max / len(p)
        p = p.groupby("alvo", group_keys=False).apply(lambda g: g.sample(max(1, round(len(g) * frac)),
                                                                         random_state=semente))
    p = p.reset_index(drop=True)
    cache = cache or Cache()
    return cache[p.id], p.alvo.to_numpy(np.float32), p
