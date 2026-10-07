"""Inferência de modelos `.tflite` com o LiteRT — **sem TensorFlow**, o mesmo código no PC e no Raspberry Pi 5.

Ordem de preferência do runtime: ``ai_edge_litert`` (atual) → ``tflite_runtime`` (antigo) → ``tensorflow.lite``.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


def _classe_interpretador():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai_edge_litert"
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite_runtime"
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter, "tensorflow.lite"


def _resolver_sem_delegates():
    try:
        from ai_edge_litert.interpreter import OpResolverType
    except ImportError:
        try:
            from tflite_runtime.interpreter import OpResolverType
        except ImportError:
            from tensorflow.lite.experimental import OpResolverType
    return OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES


def runtime() -> str:
    """Nome do runtime que será usado."""
    return _classe_interpretador()[1]


class ModeloTFLite:
    """Um modelo `.tflite` binário (saída = P(condição)), pronto para receber o tensor uint8 do Módulo 02.

    >>> m = ModeloTFLite("modelos/rd_mobilenetv3_small_int8.tflite", threads=4)
    >>> m.prever(tensor_uint8)          # (N, 224, 224, 3) uint8  →  array de N probabilidades
    """

    def __init__(self, caminho: str | Path | bytes, threads: int | None = 4, xnnpack: bool = True):
        Interpreter, self.runtime = _classe_interpretador()
        if isinstance(caminho, (bytes, bytearray)):
            self.caminho, self._conteudo = None, bytes(caminho)
        else:
            # Lemos os bytes em vez de passar `model_path`: no Windows o LiteRT não abre caminhos com acento ("Módulo 4").
            self.caminho = Path(caminho)
            self._conteudo = self.caminho.read_bytes()
        self.tamanho_bytes = len(self._conteudo)
        fonte = {"model_content": self._conteudo}      # a referência em self._conteudo mantém o buffer vivo
        # XNNPACK é o acelerador de CPU padrão do LiteRT (x86 e ARM). Alguns grafos int8 — ex.: o squeeze-and-excitation
        # do MobileNetV3 — não são aceitos por ele; nesse caso usamos os kernels de referência otimizados (mais lentos).
        self.xnnpack = xnnpack
        try:
            if not xnnpack:
                raise RuntimeError("XNNPACK desativado")
            self.interp = Interpreter(**fonte, num_threads=threads)
            self.interp.allocate_tensors()
        except RuntimeError:
            self.xnnpack = False
            self.interp = Interpreter(**fonte, num_threads=threads,
                                      experimental_op_resolver_type=_resolver_sem_delegates())
            self.interp.allocate_tensors()
        self.entrada = self.interp.get_input_details()[0]
        self.saida = self.interp.get_output_details()[0]
        self.threads = threads

    # ------------------------------------------------------------------ informações
    @property
    def forma_entrada(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.entrada["shape"])

    @property
    def tipo_entrada(self) -> np.dtype:
        return np.dtype(self.entrada["dtype"])

    def descrever(self) -> dict:
        q_e, q_s = self.entrada["quantization"], self.saida["quantization"]
        return {"arquivo": self.caminho.name if self.caminho else "(memória)", "tamanho_MB": self.tamanho_bytes / 1e6,
                "entrada": f"{self.forma_entrada} {self.tipo_entrada.name}", "entrada_scale_zp": q_e,
                "saida": f"{tuple(int(v) for v in self.saida['shape'])} {np.dtype(self.saida['dtype']).name}",
                "saida_scale_zp": q_s,
                "runtime": self.runtime, "xnnpack": self.xnnpack, "threads": self.threads}

    # ------------------------------------------------------------------ inferência
    def preparar(self, x_uint8: np.ndarray) -> np.ndarray:
        """Converte o tensor uint8 [0, 255] para o tipo de entrada do modelo (q = round(x/scale + zero_point))."""
        tipo = self.tipo_entrada
        if tipo == np.float32:
            return x_uint8.astype(np.float32)
        scale, zp = self.entrada["quantization"]
        if tipo == np.uint8 and abs(scale - 1.0) < 1e-6 and zp == 0:
            return x_uint8.astype(np.uint8)
        info = np.iinfo(tipo)
        return np.clip(np.round(x_uint8.astype(np.float32) / scale + zp), info.min, info.max).astype(tipo)

    def _saida(self) -> np.ndarray:
        y = self.interp.get_tensor(self.saida["index"])
        if np.dtype(self.saida["dtype"]) != np.float32:
            scale, zp = self.saida["quantization"]
            y = (y.astype(np.float32) - zp) * scale
        return y.reshape(-1)

    def prever(self, x_uint8: np.ndarray) -> np.ndarray:
        """Probabilidades para um lote (N, H, W, 3) uint8 — uma imagem por vez (o modelo tem lote fixo 1)."""
        if x_uint8.ndim == 3:
            x_uint8 = x_uint8[None]
        if x_uint8.shape[1:] != self.forma_entrada[1:]:
            raise ValueError(f"entrada {x_uint8.shape[1:]} ≠ modelo {self.forma_entrada[1:]}")
        out = np.empty(len(x_uint8), np.float32)
        for i in range(len(x_uint8)):
            self.interp.set_tensor(self.entrada["index"], self.preparar(x_uint8[i:i + 1]))
            self.interp.invoke()
            out[i] = self._saida()[0]
        return out

    def medir_latencia(self, x_uint8: np.ndarray | None = None, repeticoes: int = 50, aquecimento: int = 5) -> dict:
        """Tempo de `invoke()` em ms (sem pré-processamento): média, mediana, p95 e mínimo."""
        if x_uint8 is None:
            x_uint8 = np.random.default_rng(0).integers(0, 256, self.forma_entrada, dtype=np.uint8)
        self.interp.set_tensor(self.entrada["index"], self.preparar(x_uint8[:1]))
        for _ in range(aquecimento):
            self.interp.invoke()
        t = np.empty(repeticoes)
        for i in range(repeticoes):
            t0 = time.perf_counter()
            self.interp.invoke()
            t[i] = (time.perf_counter() - t0) * 1e3
        return {"media_ms": float(t.mean()), "mediana_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
                "min_ms": float(t.min())}


def memoria_processo_mb() -> float:
    """Memória residente (RSS) do processo atual, em MB (requer psutil)."""
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def memoria_pico_mb() -> float:
    """Pico de memória residente do processo desde o início, em MB (Linux/Raspberry Pi e Windows)."""
    try:
        import resource                                   # Linux / macOS
        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return pico / 1e3 if os.uname().sysname == "Linux" else pico / 1e6
    except ImportError:                                   # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e6
