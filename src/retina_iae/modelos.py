"""Arquiteturas usadas no módulo: uma CNN pequena (notebook 01), MobileNetV3-Small e EfficientNet-Lite0.

Convenção de entrada de **todos** os modelos: tensor ``float32`` (lote, H, W, 3) em RGB com valores em **[0, 255]**
— exatamente o tensor ``uint8`` de ``retina_pdi.pipeline.preparar_para_inferencia(..., normalizacao="uint8")``
convertido para float. Cada modelo faz a própria normalização na primeira camada; assim o mesmo tensor serve para
qualquer arquitetura e, depois da quantização int8, o ``.tflite`` aceita a imagem ``uint8`` diretamente.
"""
from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
from keras import layers

ARQUITETURAS = {
    "cnn_simples": "CNN de 4 blocos treinada do zero (notebook 01)",
    "mobilenetv3_small": "MobileNetV3-Small (Howard et al., 2019), pesos ImageNet do Keras",
    "efficientnet_lite0": "EfficientNet-Lite0 (Tan & Le, 2019; variante Lite do TensorFlow), pesos ImageNet",
}

# =========================================================================== CNN simples


def cnn_simples(entrada: int = 128, filtros: tuple[int, ...] = (16, 32, 64, 128), dropout: float = 0.3,
                nome: str = "cnn_simples") -> keras.Model:
    """CNN didática: blocos [Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2], média global e uma saída sigmoide."""
    ent = keras.Input((entrada, entrada, 3), name="imagem")
    x = layers.Rescaling(1 / 255, name="escala_0_1")(ent)
    for i, f in enumerate(filtros):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i + 1}")(x)
        x = layers.BatchNormalization(name=f"bn{i + 1}")(x)
        x = layers.ReLU(name=f"relu{i + 1}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    return keras.Model(ent, layers.Dense(1, activation="sigmoid", name="prob")(x), name=nome)


# =========================================================================== EfficientNet-Lite0

# (expansão, kernel, stride, canais de saída, repetições) — idêntico ao EfficientNet-B0, sem squeeze-and-excitation
LITE0_BLOCOS = [
    (1, 3, 1, 16, 1),
    (6, 3, 2, 24, 2),
    (6, 5, 2, 40, 2),
    (6, 3, 2, 80, 3),
    (6, 5, 1, 112, 3),
    (6, 5, 2, 192, 4),
    (6, 3, 1, 320, 1),
]
PESOS_LITE0 = ("timm/tf_efficientnet_lite0.in1k", "model.safetensors")


def _conv_bn(x, filtros, kernel, stride, nome_conv, nome_bn, ativacao=True, depthwise=False):
    if depthwise:
        x = layers.DepthwiseConv2D(kernel, strides=stride, padding="same", use_bias=False, name=nome_conv)(x)
    else:
        x = layers.Conv2D(filtros, kernel, strides=stride, padding="same", use_bias=False, name=nome_conv)(x)
    x = layers.BatchNormalization(epsilon=1e-3, name=nome_bn)(x)
    return layers.ReLU(6.0, name=nome_bn + "_relu6")(x) if ativacao else x


def efficientnet_lite0_base(entrada: int = 224, com_topo_imagenet: bool = False) -> keras.Model:
    """EfficientNet-Lite0, construída bloco a bloco (os nomes das camadas seguem os pesos do `timm`).

    Diferenças para o EfficientNet-B0, pensadas para quantização e aceleradores de borda:
    sem *squeeze-and-excitation*, ReLU6 no lugar de *swish* e tronco/cabeça de largura fixa.
    Entrada em [0, 255]; a primeira camada leva para [−1, 1], a escala usada no pré-treino.
    Saída: mapa de características (H/32, W/32, 1280) — ou as 1000 classes do ImageNet, com `com_topo_imagenet`.
    """
    ent = keras.Input((entrada, entrada, 3), name="imagem")
    x = layers.Rescaling(1 / 127.5, offset=-1.0, name="escala_menos1_1")(ent)
    x = _conv_bn(x, 32, 3, 2, "conv_stem", "bn1")
    c_in = 32
    for i, (exp, k, s, c_out, rep) in enumerate(LITE0_BLOCOS):
        for j in range(rep):
            p, stride, atalho = f"blocks_{i}_{j}_", (s if j == 0 else 1), x
            if exp == 1:   # bloco "depthwise separable": dw → projeção
                x = _conv_bn(x, None, k, stride, p + "conv_dw", p + "bn1", depthwise=True)
                x = _conv_bn(x, c_out, 1, 1, p + "conv_pw", p + "bn2", ativacao=False)
            else:          # MBConv / "inverted residual": expansão 1×1 → dw k×k → projeção 1×1 (linear)
                x = _conv_bn(x, c_in * exp, 1, 1, p + "conv_pw", p + "bn1")
                x = _conv_bn(x, None, k, stride, p + "conv_dw", p + "bn2", depthwise=True)
                x = _conv_bn(x, c_out, 1, 1, p + "conv_pwl", p + "bn3", ativacao=False)
            if stride == 1 and c_in == c_out:
                x = layers.Add(name=p + "soma")([atalho, x])
            c_in = c_out
    x = _conv_bn(x, 1280, 1, 1, "conv_head", "bn2")
    if com_topo_imagenet:
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(1000, activation="softmax", name="classifier")(x)
    return keras.Model(ent, x, name="efficientnet_lite0")


def carregar_pesos_lite0(modelo: keras.Model, caminho: str | Path | None = None) -> int:
    """Copia os pesos ImageNet do `timm` (formato safetensors, convertido do checkpoint oficial do TensorFlow).

    Conversões de layout PyTorch → Keras: Conv2D (out, in, kh, kw) → (kh, kw, in, out);
    DepthwiseConv2D (C, 1, kh, kw) → (kh, kw, C, 1); Dense (out, in) → (in, out). Devolve o nº de camadas carregadas.
    """
    from safetensors.numpy import load_file
    if caminho is None:
        from huggingface_hub import hf_hub_download
        caminho = hf_hub_download(*PESOS_LITE0)
    w = load_file(str(caminho))
    n = 0
    for camada in modelo.layers:
        nome = camada.name
        chave = ".".join(nome.split("_", 3)) if nome.startswith("blocks_") else nome
        if isinstance(camada, layers.DepthwiseConv2D):
            camada.set_weights([w[chave + ".weight"].transpose(2, 3, 0, 1)])
        elif isinstance(camada, layers.Conv2D):
            camada.set_weights([w[chave + ".weight"].transpose(2, 3, 1, 0)])
        elif isinstance(camada, layers.BatchNormalization):
            camada.set_weights([w[chave + s] for s in (".weight", ".bias", ".running_mean", ".running_var")])
        elif isinstance(camada, layers.Dense) and nome == "classifier":
            camada.set_weights([w["classifier.weight"].T, w["classifier.bias"]])
        else:
            continue
        n += 1
    return n


# =========================================================================== bases e classificadores


def criar_base(arquitetura: str, entrada: int = 224, pesos: str | None = "imagenet") -> keras.Model:
    """Extrator de características pré-treinado: imagem [0, 255] → mapa (H/32, W/32, C)."""
    if arquitetura == "mobilenetv3_small":
        return keras.applications.MobileNetV3Small(
            input_shape=(entrada, entrada, 3), include_top=False, weights=pesos, include_preprocessing=True)
    if arquitetura == "efficientnet_lite0":
        base = efficientnet_lite0_base(entrada)
        if pesos == "imagenet":
            carregar_pesos_lite0(base)
        return base
    raise ValueError(f"arquitetura desconhecida: {arquitetura!r}; use uma de {list(ARQUITETURAS)}")


def criar_classificador(arquitetura: str = "mobilenetv3_small", entrada: int = 224, pesos: str | None = "imagenet",
                        dropout: float = 0.2, nome: str | None = None) -> keras.Model:
    """Base pré-treinada + cabeça binária (média global → dropout → 1 neurônio sigmoide = P(condição)).

    A base é chamada com ``training=False``: mesmo quando descongelada no ajuste fino, as camadas de
    BatchNorm continuam usando as estatísticas do ImageNet (com lotes pequenos, reestimá-las desestabiliza o treino).
    """
    if arquitetura == "cnn_simples":
        return cnn_simples(entrada, nome=nome or "cnn_simples")
    base = criar_base(arquitetura, entrada, pesos)
    base.trainable = False
    ent = keras.Input((entrada, entrada, 3), name="imagem")
    x = base(ent, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    saida = layers.Dense(1, activation="sigmoid", name="prob")(x)
    return keras.Model(ent, saida, name=nome or arquitetura)


def obter_base(modelo: keras.Model) -> keras.Model | None:
    """A sub-rede base (extrator) de um classificador criado por `criar_classificador`, se houver."""
    return next((c for c in modelo.layers if isinstance(c, keras.Model)), None)


def descongelar(modelo: keras.Model, fracao: float = 1.0) -> int:
    """Libera para treino a fração final das camadas da base (BatchNorm continua congelada). Devolve quantas."""
    base = obter_base(modelo)
    if base is None:
        return 0
    base.trainable = True
    camadas = base.layers
    corte = int(round(len(camadas) * (1 - fracao)))
    n = 0
    for i, c in enumerate(camadas):
        c.trainable = i >= corte and not isinstance(c, layers.BatchNormalization)
        n += int(c.trainable and bool(c.weights))
    return n


# =========================================================================== custo computacional


def _camadas_folha(modelo: keras.Model):
    for c in modelo.layers:
        if isinstance(c, keras.Model):
            yield from _camadas_folha(c)
        else:
            yield c


def contar_macs(modelo: keras.Model) -> int:
    """Multiplicações-acumulações (MACs) de uma inferência, somando convoluções e camadas densas.

    1 MAC ≈ 2 FLOPs. Ignora ativações, BatchNorm e somas (custo pequeno comparado às convoluções).
    """
    total = 0
    for c in _camadas_folha(modelo):
        if isinstance(c, (layers.Conv2D, layers.DepthwiseConv2D, layers.Dense)):
            saida = c.output.shape
            kernel = c.weights[0].shape
            if isinstance(c, layers.Dense):
                total += int(np.prod(kernel))
            elif isinstance(c, layers.DepthwiseConv2D):
                total += int(saida[1] * saida[2] * kernel[0] * kernel[1] * kernel[2] * kernel[3])
            else:
                total += int(saida[1] * saida[2] * np.prod(kernel))
    return total


def resumo_custo(modelo: keras.Model) -> dict[str, float]:
    """Parâmetros (milhões), MACs (milhões) e tamanho em MB para float32 e int8."""
    p = modelo.count_params()
    return {"parametros_M": p / 1e6, "MACs_M": contar_macs(modelo) / 1e6,
            "float32_MB": p * 4 / 1e6, "int8_MB": p / 1e6}
