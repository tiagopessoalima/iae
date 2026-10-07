"""Testes do pacote retina_iae. Os que usam os dados baixados ou os modelos finais são pulados se eles não existirem."""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd
import pytest

from retina_iae import dados as D, metricas as Me, particoes as Pa

TEM_DADOS = D.MANIFESTO.exists() and D.PARTICOES.exists()
TEM_MODELOS = (D.MODELOS / "config_triagem.json").exists()
precisa_dados = pytest.mark.skipif(not TEM_DADOS, reason="rode scripts/download_data.py")


def retina_sintetica(h=400, w=560, raio=180, seed=0):
    """Disco alaranjado com textura suave sobre fundo preto + um 'vaso' escuro (igual ao teste do Módulo 02)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    disco = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < raio ** 2
    img = np.zeros((h, w, 3), np.uint8)
    base = np.array([190, 100, 40], np.float32) * (0.85 + 0.15 * rng.random((h, w, 1)))
    img[disco] = base[disco].astype(np.uint8)
    cv2.line(img, (w // 2 - 150, h // 2), (w // 2 + 150, h // 2 + 30), (120, 40, 20), 3)
    img[~disco] = 0
    return img


# =========================================================================== partições


def test_alvo_por_tarefa():
    linha = lambda f, c, g="": pd.Series({"fonte": f, "condicao": c, "grau": g})  # noqa: E731
    assert Pa.alvo(linha("APTOS", "rd", 1), "rd") == 0          # RD leve não é referível
    assert Pa.alvo(linha("APTOS", "rd", 2), "rd") == 1
    assert Pa.alvo(linha("DDR", "ilegivel", 5), "rd") is None   # sem qualidade: fora da tarefa
    assert Pa.alvo(linha("ODIR", "rd"), "rd") == 1
    assert Pa.alvo(linha("ODIR", "glaucoma"), "dmri") is None
    assert Pa.alvo(linha("ODIR", "normal"), "glaucoma") == 0


def test_dividir_mantem_paciente_junto_e_estratifica():
    rng = np.random.default_rng(1)
    n_pac = 400
    df = pd.DataFrame({"id": [f"i{k}" for k in range(2 * n_pac)],
                       "paciente": [f"p{k // 2}" for k in range(2 * n_pac)],
                       "alvo": rng.random(2 * n_pac) < 0.2})
    df["alvo"] = df.alvo.astype(int)
    part = Pa.dividir(df, semente=0)
    assert set(part) == {"treino", "validacao", "teste"}
    assert (df.assign(p=part).groupby("paciente").p.nunique() == 1).all()       # sem vazamento
    frac = part.value_counts(normalize=True)
    assert abs(frac["treino"] - 0.70) < 0.03 and abs(frac["teste"] - 0.15) < 0.03
    prev = df.assign(p=part).groupby(["paciente", "p"]).alvo.max().groupby("p").mean()
    assert prev.max() - prev.min() < 0.06                                        # prevalência parecida


@precisa_dados
def test_particoes_baixadas_sem_vazamento():
    p = D.carregar_particoes()
    dev = p[(p.particao != "externo") & (p.paciente != "")]
    assert (dev.groupby(["tarefa", "paciente"]).particao.nunique() == 1).all()
    for t, cfg in Pa.TAREFAS.items():
        sub = p[p.tarefa == t]
        assert set(sub[sub.particao == "externo"].fonte) <= set(cfg["externo"])
        assert set(sub[sub.particao != "externo"].fonte) <= set(cfg["desenvolvimento"])


# =========================================================================== métricas


def test_metricas_no_limiar():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.1, 0.3, 0.05])
    m = Me.metricas_no_limiar(y, p, 0.5)
    assert (m["VP"], m["FN"], m["FP"], m["VN"]) == (2, 1, 1, 3)
    assert m["sensibilidade"] == pytest.approx(2 / 3) and m["especificidade"] == pytest.approx(3 / 4)


def test_limiar_para_sensibilidade():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.3
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, len(y)), 0, 1)
    lim = Me.limiar_para_sensibilidade(y, p, 0.9)
    assert Me.metricas_no_limiar(y, p, lim)["sensibilidade"] >= 0.9
    assert Me.metricas_no_limiar(y, p, lim + 1e-3)["sensibilidade"] < 0.9 + 1e-3 + 0.01


def test_ic_wilson_contem_proporcao():
    lo, hi = Me.ic_proporcao(45, 50)
    assert lo < 0.9 < hi and 0 <= lo and hi <= 1


def test_vpp_cai_com_prevalencia():
    v = Me.vpp_por_prevalencia(0.9, 0.9, [0.5, 0.1, 0.01])
    assert v[0] > v[1] > v[2] and v[0] == pytest.approx(0.9)


# =========================================================================== modelos, conversão e inferência

tf = pytest.importorskip("tensorflow")
from retina_iae import conversao as C, inferencia as I, modelos as M  # noqa: E402


def test_lite0_arquitetura():
    base = M.efficientnet_lite0_base(224)
    assert base.output_shape == (None, 7, 7, 1280)
    assert abs(base.count_params() - 3_371_872) < 50_000           # ≈ 3,4 M sem a cabeça do ImageNet
    assert 350 < M.contar_macs(base) / 1e6 < 420                     # ≈ 0,39 GMACs


def test_conversao_float32_reproduz_keras_e_int8_aproxima():
    m = M.cnn_simples(32, filtros=(4, 8))
    x = np.random.default_rng(0).integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    pk = m.predict(x.astype(np.float32), verbose=0).ravel()
    f32 = I.ModeloTFLite(C.converter(m, "float32"))
    assert np.allclose(f32.prever(x), pk, atol=1e-5)
    i8 = I.ModeloTFLite(C.converter(m, "int8", C.dados_representativos(x, 8)))
    assert i8.tipo_entrada == np.uint8
    assert np.abs(i8.prever(x) - pk).max() < 0.05


def test_carrega_modelo_em_caminho_com_acento(tmp_path):
    """No Windows o LiteRT não abre `model_path` com acento (ex.: 'Módulo 4'); ModeloTFLite lê os bytes."""
    destino = C.salvar_tflite(C.converter(M.cnn_simples(32, filtros=(4,)), "float32"), tmp_path / "Módulo 4" / "m.tflite")
    assert I.ModeloTFLite(destino).forma_entrada == (1, 32, 32, 3)


def test_preparar_quantiza_entrada():
    m = M.cnn_simples(32, filtros=(4,))
    mod = I.ModeloTFLite(C.converter(m, "float32"))
    x = np.full((1, 32, 32, 3), 200, np.uint8)
    assert mod.preparar(x).dtype == np.float32 and mod.preparar(x).max() == 200.0
    mod.entrada = {**mod.entrada, "dtype": np.int8, "quantization": (1.0, -128)}
    assert mod.preparar(x).dtype == np.int8 and int(mod.preparar(x).max()) == 72


def test_triagem_ponta_a_ponta(tmp_path):
    from retina_iae.triagem import Triagem
    x = np.random.default_rng(0).integers(0, 256, (4, 224, 224, 3), dtype=np.uint8)
    cfg = {"preprocessamento": {"tamanho": 224, "realce": "clahe"}, "modelos": {}}
    for t in ("rd", "glaucoma", "dmri"):
        C.salvar_tflite(C.converter(M.cnn_simples(224, filtros=(4,)), "int8", C.dados_representativos(x, 4)),
                        tmp_path / f"{t}.tflite")
        cfg["modelos"][t] = {"arquivo": f"{t}.tflite", "limiar": 0.5, "nome": t}
    (tmp_path / "config_triagem.json").write_text(json.dumps(cfg), encoding="utf-8")
    tri = Triagem(tmp_path, exigir_qualidade=False)
    r = tri.avaliar(retina_sintetica())
    assert set(r.probabilidades) == {"rd", "glaucoma", "dmri"}
    assert all(0 <= p <= 1 for p in r.probabilidades.values())
    assert "TRIAGEM" in tri.laudo(r)
    preto = tri.avaliar(np.zeros((300, 300, 3), np.uint8))      # câmera tampada: reprova, não quebra
    assert not preto.aprovada_qualidade and not preto.probabilidades
    assert "REPROVADA" in tri.laudo(preto)


@pytest.mark.skipif(not TEM_MODELOS, reason="modelos finais ainda não exportados (notebook 06)")
def test_modelos_finais_carregam():
    from retina_iae.triagem import Triagem
    tri = Triagem()
    for m in tri.modelos.values():
        assert m.forma_entrada == (1, 224, 224, 3)


@precisa_dados
def test_cache_bate_com_pipeline_do_modulo2():
    """Paridade: o tensor do cache é exatamente o que o Raspberry Pi vai produzir para a mesma imagem."""
    from retina_pdi import pipeline as P
    try:
        cache = D.Cache()
    except FileNotFoundError:
        pytest.skip("rode scripts/preparar_cache.py")
    linha = D.carregar_manifesto().sample(1, random_state=0).iloc[0]
    x = P.preparar_para_inferencia(D.carregar_rgb(linha.arquivo), 224, "clahe", "uint8")[0]
    assert np.array_equal(cache[linha.id], x)
