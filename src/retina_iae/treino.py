"""Treinamento: pipeline tf.data, aumento de dados, pesos de classe e transfer learning em duas fases."""
from __future__ import annotations

import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers

from . import dados as D
from .modelos import descongelar, obter_base

AUTOTUNE = tf.data.AUTOTUNE


# =========================================================================== dados


class AumentoFotometrico(layers.Layer):
    """Ganho de brilho global e por canal (multiplicativo: a moldura preta continua preta).

    O ganho por canal imita o balanço de branco de câmeras diferentes — a principal fonte de variação entre bases.
    """

    def __init__(self, brilho: float = 0.15, cor: float = 0.06, semente: int | None = None, **kw):
        super().__init__(**kw)
        self.brilho, self.cor = brilho, cor
        self.gerador = keras.random.SeedGenerator(semente)

    def call(self, x, training=None):
        if not training:
            return x
        n = keras.ops.shape(x)[0]
        g = keras.random.uniform((n, 1, 1, 1), 1 - self.brilho, 1 + self.brilho, seed=self.gerador)
        c = keras.random.uniform((n, 1, 1, 3), 1 - self.cor, 1 + self.cor, seed=self.gerador)
        return keras.ops.clip(x * g * c, 0.0, 255.0)


def camada_aumento(semente: int = 0, rotacao: float = 0.5, zoom: float = 0.1) -> keras.Sequential:
    """Aumento de dados para retinografias, aplicado só no treino.

    Giros e espelhamentos não mudam o diagnóstico (o fundo de olho não tem "lado de cima" para a doença);
    a rotação de até ±180° e o zoom preenchem com preto, como a moldura real.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=semente),
        layers.RandomRotation(rotacao, fill_mode="constant", fill_value=0.0, seed=semente),
        layers.RandomZoom((-zoom, zoom), fill_mode="constant", fill_value=0.0, seed=semente),
        AumentoFotometrico(semente=semente),
    ], name="aumento")


def criar_dataset(x: np.ndarray, y: np.ndarray | None = None, lote: int = 32, treino: bool = False,
                  aumento: keras.Layer | None = None, tamanho: int | None = None, semente: int = 0) -> tf.data.Dataset:
    """tf.data a partir de tensores uint8 (N, H, W, 3): embaralha (treino), forma lotes, converte para float32 [0, 255],
    redimensiona se `tamanho` ≠ H (redução por média de área) e aplica o aumento (treino)."""
    ds = tf.data.Dataset.from_tensor_slices(x if y is None else (x, y))
    if treino:
        ds = ds.shuffle(len(x), seed=semente, reshuffle_each_iteration=True)
    ds = ds.batch(lote)

    def _converter(a, *b):
        a = tf.cast(a, tf.float32)
        if tamanho is not None and tamanho != x.shape[1]:
            a = tf.image.resize(a, (tamanho, tamanho), method="area")
        if treino and aumento is not None:
            a = aumento(a, training=True)
        return (a, *b) if b else a

    return ds.map(_converter, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def pesos_de_classe(y: np.ndarray) -> dict[int, float]:
    """Pesos inversamente proporcionais à frequência: cada classe contribui igualmente para a perda."""
    n, p = len(y), float(np.sum(y))
    return {0: n / (2 * (n - p)), 1: n / (2 * p)}


# =========================================================================== treino


def compilar(modelo: keras.Model, lr: float) -> None:
    modelo.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                   metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acuracia")])


def treinar(modelo: keras.Model, treino: tuple[np.ndarray, np.ndarray], validacao: tuple[np.ndarray, np.ndarray],
            epocas: int = 10, lr: float = 1e-3, lote: int = 32, aumento: bool = True, balancear: bool = True,
            paciencia: int = 4, tamanho: int | None = None, semente: int = 0, verbose: int = 0,
            compilar_modelo: bool = True) -> pd.DataFrame:
    """Treina até `epocas`, com parada antecipada pela AUC de validação (restaura a melhor época).

    Devolve o histórico por época (perda, AUC, acurácia, taxa de aprendizado e tempo).
    """
    keras.utils.set_random_seed(semente)
    tamanho = tamanho or modelo.input_shape[1]
    if compilar_modelo:
        compilar(modelo, lr)
    ds_tr = criar_dataset(*treino, lote=lote, treino=True, aumento=camada_aumento(semente) if aumento else None,
                          tamanho=tamanho, semente=semente)
    ds_va = criar_dataset(*validacao, lote=64, tamanho=tamanho)
    tempos: list[float] = []

    class _Cronometro(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            tempos.append(time.perf_counter() - self.t0)

    cb = [keras.callbacks.EarlyStopping("val_auc", mode="max", patience=paciencia, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau("val_auc", mode="max", factor=0.3, patience=max(1, paciencia // 2),
                                            min_lr=1e-6), _Cronometro()]
    h = modelo.fit(ds_tr, validation_data=ds_va, epochs=epocas, verbose=verbose, callbacks=cb, shuffle=False,
                   class_weight=pesos_de_classe(treino[1]) if balancear else None)
    hist = pd.DataFrame(h.history)
    hist.insert(0, "epoca", np.arange(1, len(hist) + 1))
    hist["segundos"] = tempos[:len(hist)]
    return hist


def treinar_duas_fases(modelo: keras.Model, treino, validacao, epocas_cabeca: int = 4, epocas_ajuste: int = 12,
                       lr_cabeca: float = 1e-3, lr_ajuste: float = 1e-4, fracao_ajuste: float = 1.0,
                       **kw) -> pd.DataFrame:
    """Transfer learning clássico:

    1. **cabeça**: base congelada, só a camada final aprende (taxa alta, poucas épocas);
    2. **ajuste fino**: descongela a fração final `fracao_ajuste` da base e treina tudo com taxa 10× menor.
    """
    base = obter_base(modelo)
    if base is not None:
        base.trainable = False
    h1 = treinar(modelo, treino, validacao, epocas=epocas_cabeca, lr=lr_cabeca, **kw)
    h1.insert(0, "fase", "cabeça")
    if base is None or epocas_ajuste == 0:
        return h1
    descongelar(modelo, fracao_ajuste)
    h2 = treinar(modelo, treino, validacao, epocas=epocas_ajuste, lr=lr_ajuste, **kw)
    h2.insert(0, "fase", "ajuste fino")
    h2["epoca"] += len(h1)
    return pd.concat([h1, h2], ignore_index=True)


# =========================================================================== predição e persistência


def prever(modelo: keras.Model, x: np.ndarray, lote: int = 64) -> np.ndarray:
    """Probabilidades P(condição) para tensores uint8 (N, H, W, 3)."""
    ds = criar_dataset(x, lote=lote, tamanho=modelo.input_shape[1])
    return modelo.predict(ds, verbose=0).ravel()


def extrair_caracteristicas(base: keras.Model, x: np.ndarray, lote: int = 64) -> np.ndarray:
    """Vetor de características (média global do último mapa) de cada imagem — o "embedding" da base."""
    ent = keras.Input(base.input_shape[1:])
    extrator = keras.Model(ent, layers.GlobalAveragePooling2D()(base(ent, training=False)))
    return extrator.predict(criar_dataset(x, lote=lote, tamanho=base.input_shape[1]), verbose=0)


def caminho_keras(tarefa: str, arquitetura: str) -> Path:
    return D.KERAS / f"{tarefa}_{arquitetura}.keras"


def salvar(modelo: keras.Model, tarefa: str, arquitetura: str) -> Path:
    D.KERAS.mkdir(parents=True, exist_ok=True)
    caminho = caminho_keras(tarefa, arquitetura)
    modelo.save(caminho)
    return caminho


def carregar(tarefa: str, arquitetura: str) -> keras.Model:
    return keras.models.load_model(caminho_keras(tarefa, arquitetura),
                                   custom_objects={"AumentoFotometrico": AumentoFotometrico})
