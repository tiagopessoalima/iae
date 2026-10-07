"""Grad-CAM (Selvaraju et al., 2017): onde, no último mapa de características, está a evidência para a decisão.

Peso de cada canal = média espacial do gradiente do *logit* em relação ao mapa; o mapa de calor é a soma
ponderada dos canais, só com a parte positiva. A resolução é a do último mapa (7×7 para entrada 224) —
uma indicação grosseira de região, não uma segmentação de lesões.
"""
from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from .modelos import obter_base


def gradcam(modelo: keras.Model, x_uint8: np.ndarray, camada: str = "pool4") -> np.ndarray:
    """Mapa de calor [0, 1] (h, w) para uma imagem uint8 (H, W, 3).

    Classificadores com base pré-treinada usam a saída da base; a `cnn_simples` usa a camada `camada`.
    """
    x = tf.convert_to_tensor(x_uint8[None].astype(np.float32))
    densa = modelo.get_layer("prob")
    base = obter_base(modelo)
    if base is None:
        extrator = keras.Model(modelo.inputs, modelo.get_layer(camada).output)
    else:
        extrator = base
    with tf.GradientTape() as tape:
        mapa = extrator(x, training=False)
        tape.watch(mapa)
        logit = tf.matmul(tf.reduce_mean(mapa, axis=(1, 2)), densa.kernel) + densa.bias
    grad = tape.gradient(logit, mapa)
    pesos = tf.reduce_mean(grad, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(pesos * mapa, axis=-1))[0].numpy()
    return cam / cam.max() if cam.max() > 0 else cam
