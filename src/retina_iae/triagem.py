"""Triagem completa no dispositivo: captura → qualidade e pré-processamento (Módulo 02) → três modelos → decisão.

Só depende de NumPy/OpenCV, do ``retina_pdi`` e do LiteRT — roda no Raspberry Pi 5 sem TensorFlow.

>>> t = Triagem()                         # lê modelos/config_triagem.json
>>> r = t.avaliar(rgb)                    # rgb: captura uint8 H×W×3
>>> print(t.laudo(r))
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .inferencia import ModeloTFLite

PASTA_MODELOS = Path(__file__).resolve().parents[2] / "modelos"
CONFIG = "config_triagem.json"
AVISO = ("Resultado de TRIAGEM automatizada, não é diagnóstico. A decisão clínica cabe ao oftalmologista "
         "(protótipo de pesquisa — Edital FACEPE nº 14/2026).")


@dataclass
class ResultadoTriagem:
    aprovada_qualidade: bool
    motivos_qualidade: list[str]
    probabilidades: dict[str, float] = field(default_factory=dict)
    encaminhar: dict[str, bool] = field(default_factory=dict)
    tempos_ms: dict[str, float] = field(default_factory=dict)

    @property
    def encaminhar_algum(self) -> bool:
        return any(self.encaminhar.values())


class Triagem:
    """Carrega os três modelos e os limiares de `config_triagem.json` (gerado no notebook 06)."""

    def __init__(self, pasta: str | Path = PASTA_MODELOS, config: str = CONFIG, threads: int = 4,
                 exigir_qualidade: bool = True):
        self.pasta = Path(pasta)
        self.cfg = json.loads((self.pasta / config).read_text(encoding="utf-8"))
        self.pre = self.cfg["preprocessamento"]
        self.exigir_qualidade = exigir_qualidade
        self.modelos = {t: ModeloTFLite(self.pasta / m["arquivo"], threads=threads)
                        for t, m in self.cfg["modelos"].items()}
        self.limiares = {t: float(m["limiar"]) for t, m in self.cfg["modelos"].items()}
        self.nomes = {t: m.get("nome", t) for t, m in self.cfg["modelos"].items()}

    def preparar(self, rgb: np.ndarray):
        """Controle de qualidade + tensor uint8 (1, 224, 224, 3) — `retina_pdi.fluxo.processar_captura`."""
        from retina_pdi import fluxo
        return fluxo.processar_captura(rgb, tamanho=self.pre["tamanho"], realce=self.pre["realce"],
                                       normalizacao="uint8", exigir_qualidade=self.exigir_qualidade)

    def avaliar(self, rgb: np.ndarray) -> ResultadoTriagem:
        try:
            fl = self.preparar(rgb)
        except ValueError as e:     # ex.: captura inteiramente escura — o Módulo 02 não encontra o campo de visão
            return ResultadoTriagem(False, [f"campo de visão não encontrado ({e})"])
        r = ResultadoTriagem(fl.aprovada, list(fl.qualidade.motivos), tempos_ms=dict(fl.tempos_ms))
        if fl.tensor is None:
            return r
        for tarefa, modelo in self.modelos.items():
            t0 = time.perf_counter()
            p = float(modelo.prever(fl.tensor)[0])
            r.tempos_ms[f"modelo_{tarefa}"] = (time.perf_counter() - t0) * 1e3
            r.probabilidades[tarefa] = p
            r.encaminhar[tarefa] = p >= self.limiares[tarefa]
        r.tempos_ms["total"] = sum(v for k, v in r.tempos_ms.items() if k != "total")
        return r

    def laudo(self, r: ResultadoTriagem) -> str:
        """Texto curto para o operador do retinógrafo."""
        if not r.probabilidades:
            return ("CAPTURA REPROVADA no controle de qualidade — repetir a imagem.\nMotivos: "
                    + "; ".join(r.motivos_qualidade or ["(não informado)"]))
        linhas = []
        for t, p in r.probabilidades.items():
            sinal = "ENCAMINHAR" if r.encaminhar[t] else "sem alteração detectada"
            linhas.append(f"  {self.nomes[t]:<42s} {sinal:<24s} (escore {p:.2f}; limiar {self.limiares[t]:.2f})")
        conclusao = ("→ Encaminhar ao oftalmologista." if r.encaminhar_algum
                     else "→ Nenhum achado acima dos limiares de triagem.")
        return "\n".join(linhas + [conclusao, AVISO])
