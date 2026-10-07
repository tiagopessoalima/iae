#!/usr/bin/env python
"""Triagem de uma ou mais retinografias com os três modelos de `modelos/` (sem TensorFlow).

    python scripts/triar.py amostras/*.jpg
    python scripts/triar.py captura.png --threads 4 --sem-qualidade
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from retina_iae import dados as D  # noqa: E402
from retina_iae.triagem import Triagem  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagens", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sem-qualidade", action="store_true", help="não bloqueia capturas reprovadas no controle de qualidade")
    args = ap.parse_args()
    tri = Triagem(threads=args.threads, exigir_qualidade=not args.sem_qualidade)
    for caminho in args.imagens:
        r = tri.avaliar(D.carregar_rgb(Path(caminho).resolve()))
        print(f"\n== {Path(caminho).name}  ({r.tempos_ms.get('total', 0):.0f} ms)")
        print(tri.laudo(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
