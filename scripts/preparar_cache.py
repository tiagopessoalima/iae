#!/usr/bin/env python
"""Pré-processa todas as imagens do manifesto com o pipeline do Módulo 02 e grava o cache de tensores.

    python scripts/preparar_cache.py                      # configuração padrão (224 px, CLAHE)
    python scripts/preparar_cache.py --realce nenhum      # outra configuração (para os exercícios)
    python scripts/preparar_cache.py --forcar             # refaz

Cada imagem passa por ``retina_pdi.pipeline.preparar_para_inferencia`` (recorte do FOV, padding quadrado,
redimensionamento, realce) e vira um tensor uint8 224×224×3 em ``data/cache/tensores_<tamanho>_<realce>.npy``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from retina_iae import dados as D  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tamanho", type=int, default=D.CONFIG_PRE["tamanho"])
    ap.add_argument("--realce", default=D.CONFIG_PRE["realce"], choices=["clahe", "ben_graham", "nenhum"])
    ap.add_argument("--forcar", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1, help="processos paralelos (-1 = todos os núcleos)")
    args = ap.parse_args()
    cfg = {"tamanho": args.tamanho, "realce": None if args.realce == "nenhum" else args.realce}
    t0 = time.time()
    caminho = D.construir_cache(cfg, n_jobs=args.jobs, forcar=args.forcar)
    print(f"{caminho.relative_to(RAIZ)}  ({caminho.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
