"""Conversão Keras → TensorFlow Lite (formato do LiteRT), com quatro níveis de quantização.

| modo        | pesos   | ativações | entrada  | uso típico                                                     |
|-------------|---------|-----------|----------|----------------------------------------------------------------|
| `float32`   | float32 | float32   | float32  | referência: deve reproduzir o Keras                            |
| `float16`   | float16 | float32   | float32  | metade do tamanho; GPUs móveis                                  |
| `dinamica`  | int8    | float32   | float32  | ¼ do tamanho, sem dados de calibração                           |
| `int8`      | int8    | int8      | uint8    | ¼ do tamanho e aritmética inteira (CPU ARM, NPUs, Coral, Hailo) |

No modo `int8` a entrada é **uint8 [0, 255]** — a imagem pré-processada entra direto, sem conversão.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import keras
import numpy as np
import tensorflow as tf

MODOS = ("float32", "float16", "dinamica", "int8")


def _exportar(modelo: keras.Model, pasta: Path) -> Path:
    """SavedModel com assinatura de lote fixo 1 (o conversor congela os pesos como constantes a partir dele)."""
    t = modelo.input_shape[1]
    modelo.export(str(pasta), format="tf_saved_model", verbose=False,
                  input_signature=[tf.TensorSpec([1, t, t, 3], tf.float32, name="imagem")])
    return pasta


def dados_representativos(x: np.ndarray, n: int = 200, semente: int = 0):
    """Gerador de calibração para a quantização int8: `n` imagens do **treino**, uma por vez, como float32 [0, 255].

    O conversor roda o modelo nelas e registra a faixa (mín., máx.) de cada ativação para escolher `scale`/`zero_point`.
    """
    idx = np.random.default_rng(semente).choice(len(x), size=min(n, len(x)), replace=False)

    def gerador():
        for i in idx:
            yield [x[i:i + 1].astype(np.float32)]

    return gerador


def converter(modelo: keras.Model, modo: str = "int8", representativos: Iterable | None = None) -> bytes:
    """Converte e devolve o conteúdo do `.tflite` (lote fixo = 1, como no dispositivo)."""
    if modo not in MODOS:
        raise ValueError(f"modo desconhecido: {modo!r}; use um de {MODOS}")
    with tempfile.TemporaryDirectory() as tmp:
        conv = tf.lite.TFLiteConverter.from_saved_model(str(_exportar(modelo, Path(tmp) / "sm")))
        _configurar(conv, modo, representativos)
        return conv.convert()


def _configurar(conv, modo: str, representativos) -> None:
    if modo != "float32":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if modo == "float16":
        conv.target_spec.supported_types = [tf.float16]
    elif modo == "int8":
        if representativos is None:
            raise ValueError("a quantização int8 precisa de `representativos` (veja dados_representativos)")
        conv.representative_dataset = representativos
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.float32


def salvar_tflite(conteudo: bytes, caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(conteudo)
    return caminho
