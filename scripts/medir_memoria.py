#!/usr/bin/env python
"""Pico de memória de um processo que só faz a triagem (LiteRT + retina_pdi, sem TensorFlow).

    python scripts/medir_memoria.py amostras/*.jpg

Imprime uma linha JSON com a memória após cada etapa e o pico. Rodar num processo novo é essencial:
dentro de um notebook, o TensorFlow já carregado (≈ 1 GB) mascararia a medida.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))


def main() -> int:
    from retina_iae.inferencia import memoria_pico_mb, memoria_processo_mb
    out = {"inicio_MB": memoria_processo_mb()}
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401
    from retina_pdi import fluxo  # noqa: F401
    out["bibliotecas_MB"] = memoria_processo_mb()
    from retina_iae import dados as D
    from retina_iae.triagem import Triagem
    tri = Triagem(threads=4, exigir_qualidade=False)
    out["modelos_MB"] = memoria_processo_mb()
    for img in sys.argv[1:]:
        tri.avaliar(D.carregar_rgb(Path(img).resolve()))
    out["apos_triagem_MB"] = memoria_processo_mb()
    out["pico_MB"] = memoria_pico_mb()
    out["imagens"] = len(sys.argv) - 1
    print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in out.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
