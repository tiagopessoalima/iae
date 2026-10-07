#!/usr/bin/env python
"""Baixa as retinografias públicas usadas para treinar e validar os modelos do Módulo 04.

Um modelo binário por condição, cada um com uma fonte de **desenvolvimento** (treino/validação/teste)
e uma fonte **externa** (outra câmera, outro país, outro protocolo de rotulagem):

| Modelo   | Desenvolvimento                         | Validação externa                          |
|----------|-----------------------------------------|--------------------------------------------|
| RD       | APTOS 2019 (3 662, graus 0–4)           | DDR (550, graus 0–4 + "ilegível") · ODIR   |
| Glaucoma | ODIR (glaucoma × normal, por paciente)  | G1020 (1 020)                               |
| DMRI     | ODIR (DMRI × normal, por paciente)      | RFMiD (DMRI × normal)                       |

Todas as fontes estão no Hugging Face Hub, sem cadastro. Os Parquets (APTOS, ODIR, G1020) são baixados
inteiros (≈ 0,9 GB); do DDR e do RFMiD baixamos só os arquivos sorteados (≈ 0,8 GB), reduzidos para no
máximo 1024 px antes de gravar. No disco ficam ≈ 0,5 GB.

Saídas:
* ``data/raw/<fonte>/<condicao>/<id>.jpg``
* ``data/manifest.csv``  — uma linha por imagem (fonte, condição, grau, paciente, licença...)
* ``data/particoes.csv`` — uma linha por (imagem, tarefa): alvo 0/1 e partição treino/validacao/teste/externo

Uso:
    python scripts/download_data.py                  # tudo
    python scripts/download_data.py --only aptos     # só uma fonte (as partições são sempre refeitas)
    python scripts/download_data.py --force          # rebaixa
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from retina_iae.particoes import gerar_particoes  # noqa: E402

RAW = RAIZ / "data" / "raw"
CACHE = RAIZ / "data" / "cache" / "hf"
MANIFESTO = RAIZ / "data" / "manifest.csv"
PARTICOES = RAIZ / "data" / "particoes.csv"
LADO_MAX = 1024
SEMENTE = 2026

REF = {
    "APTOS": ("Asia Pacific Tele-Ophthalmology Society. APTOS 2019 Blindness Detection (Kaggle), 2019. "
              "Espelho: hf.co/datasets/bumbledeep/aptos"),
    "DDR": ("Li, T. et al. Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening. "
            "Information Sciences 501:511-522, 2019. Espelho: hf.co/datasets/ctmedtech/DDR-dataset"),
    "G1020": ("Bajwa, M. N. et al. G1020: A Benchmark Retinal Fundus Image Dataset for Computer-Aided Glaucoma "
              "Detection. IJCNN 2020. Espelho: hf.co/datasets/ai4ophth/G1020_dataset"),
    "ODIR": ("ODIR-2019, Peking University (Shanggong Medical Technology). "
             "Espelho: hf.co/datasets/bumbledeep/odir. Uso educacional/pesquisa."),
    "RFMiD": ("Pachade, S. et al. Retinal Fundus Multi-Disease Image Dataset (RFMiD). Data 6(2):14, 2021. "
              "Espelho: hf.co/datasets/ctmedtech/RFMID"),
}
LICENCA = {"APTOS": "termos da competição Kaggle (pesquisa, não comercial)", "DDR": "CC BY 4.0",
           "G1020": "uso acadêmico (ver página dos autores)", "ODIR": "uso educacional/pesquisa", "RFMiD": "CC BY 4.0"}

COLUNAS = ["id", "fonte", "condicao", "grau", "rotulo_original", "paciente", "arquivo", "largura", "altura",
           "largura_original", "altura_original", "licenca", "referencia", "observacao"]


# --------------------------------------------------------------------------- utilidades
def _tentar(fn, tentativas: int = 5, espera: float = 4.0):
    """Repete `fn` com backoff — conexões longas ao Hub às vezes caem."""
    for k in range(tentativas):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if k == tentativas - 1:
                raise
            print(f"   ! {type(e).__name__}: {e} — nova tentativa em {espera * (k + 1):.0f}s")
            time.sleep(espera * (k + 1))


def _hub(repo: str, arquivo: str) -> Path:
    """Baixa (com cache e retomada) um arquivo de um dataset do Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    return Path(_tentar(lambda: hf_hub_download(repo, arquivo, repo_type="dataset", cache_dir=CACHE)))


def _gravar(dados: bytes, destino: Path) -> tuple[int, int, int, int]:
    """Grava a imagem como JPEG com lado maior ≤ LADO_MAX.

    JPEGs que já cabem no limite são copiados byte a byte (sem recompressão). Devolve
    (largura, altura, largura_original, altura_original).
    """
    img = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"imagem ilegível: {destino.name}")
    h0, w0 = img.shape[:2]
    destino.parent.mkdir(parents=True, exist_ok=True)
    if max(h0, w0) <= LADO_MAX and dados[:2] == b"\xff\xd8":
        destino.write_bytes(dados)
        return w0, h0, w0, h0
    if max(h0, w0) > LADO_MAX:
        f = LADO_MAX / max(h0, w0)
        img = cv2.resize(img, (round(w0 * f), round(h0 * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    destino.write_bytes(buf.tobytes())  # (cv2.imwrite não aceita acentos no caminho no Windows)
    return img.shape[1], img.shape[0], w0, h0


def _rel(p: Path) -> str:
    return p.relative_to(RAIZ).as_posix()


def _registro(fonte, id_, condicao, arq, tam, grau="", rotulo="", paciente="", obs=""):
    w, h, w0, h0 = tam
    return dict(id=id_, fonte=fonte, condicao=condicao, grau=grau, rotulo_original=rotulo, paciente=paciente,
                arquivo=_rel(arq), largura=w, altura=h, largura_original=w0, altura_original=h0,
                licenca=LICENCA[fonte], referencia=REF[fonte], observacao=obs)


def _limpar(fonte: str, force: bool) -> Path:
    destino = RAW / fonte.lower()
    if force and destino.exists():
        shutil.rmtree(destino)
    return destino


def _ler_parquet(caminho: Path, colunas: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq
    return pq.read_table(caminho, columns=colunas).to_pandas()


def _nome_original(img: dict) -> str:
    return Path(img.get("path") or "").name


# --------------------------------------------------------------------------- APTOS 2019 (RD, desenvolvimento)
def baixar_aptos(force: bool) -> list[dict]:
    """APTOS 2019: 3 662 retinografias de triagem na Índia, graus 0–4 (escala ICDR). O espelho já está em 224×224."""
    destino = _limpar("APTOS", force)
    tab = _ler_parquet(_hub("bumbledeep/aptos", "data/train-00000-of-00001.parquet"), ["image", "label_code", "label"])
    regs = []
    for i, r in tqdm(tab.iterrows(), total=len(tab), desc="APTOS"):
        grau = int(r.label_code)
        cond = "normal" if grau == 0 else "rd"
        id_ = f"aptos_{i:04d}"
        arq = destino / cond / f"{id_}.jpg"
        regs.append(_registro("APTOS", id_, cond, arq, _gravar(r.image["bytes"], arq), grau=grau, rotulo=r.label,
                              obs=_nome_original(r.image)))
    print(f"APTOS: {len(regs)} imagens")
    return regs


# --------------------------------------------------------------------------- G1020 (glaucoma, externo)
def baixar_g1020(force: bool) -> list[dict]:
    """G1020: 1 020 imagens de 432 pacientes (Alemanha), diagnóstico clínico de glaucoma (exame completo,
    nem sempre visível na retinografia). Base externa do modelo de glaucoma."""
    destino = _limpar("G1020", force)
    tab = _ler_parquet(_hub("ai4ophth/G1020_dataset", "data/train-00000-of-00001.parquet"),
                       ["fundus image", "sparse text"])
    regs = []
    for i, r in tqdm(tab.iterrows(), total=len(tab), desc="G1020"):
        rot = str(r["sparse text"]).strip().lower()
        cond = {"normal": "normal", "glaucoma": "glaucoma"}[rot]
        id_ = f"g1020_{i:04d}"
        arq = destino / cond / f"{id_}.jpg"
        regs.append(_registro("G1020", id_, cond, arq, _gravar(r["fundus image"]["bytes"], arq), rotulo=rot,
                              obs=_nome_original(r["fundus image"])))
    print(f"G1020: {len(regs)} imagens")
    return regs


# --------------------------------------------------------------------------- ODIR-5K (glaucoma e DMRI dev.; RD externo)
def baixar_odir(force: bool) -> list[dict]:
    """ODIR-5K (512×512): olhos esquerdo e direito de 3 358 pacientes (China). Usamos as imagens rotuladas como
    normal, diabetes (RD de qualquer grau), glaucoma e DMRI; o paciente fica registrado para dividir sem vazamento."""
    destino = _limpar("ODIR", force)
    tab = _ler_parquet(_hub("bumbledeep/odir", "data/train-00000-of-00001.parquet"),
                       ["image", "patient_id", "label", "age", "sex"])
    mapa = {"normal": "normal", "diabetes": "rd", "glaucoma": "glaucoma", "age_related_macular_degeneration": "dmri"}
    tab = tab[tab.label.isin(mapa)]
    regs = []
    for i, r in tqdm(tab.iterrows(), total=len(tab), desc="ODIR"):
        cond = mapa[r.label]
        id_ = f"odir_{i:04d}"
        arq = destino / cond / f"{id_}.jpg"
        regs.append(_registro("ODIR", id_, cond, arq, _gravar(r.image["bytes"], arq), rotulo=r.label,
                              paciente=f"odir_p{int(r.patient_id)}",
                              obs=f"{_nome_original(r.image)}; idade {r.age}; sexo {r.sex}"))
    print(f"ODIR: {len(regs)} imagens")
    return regs


# --------------------------------------------------------------------------- DDR (RD, externo)
def baixar_ddr(force: bool) -> list[dict]:
    """DDR (China, 147 hospitais, várias câmeras): amostra estratificada da partição *valid* do DR_grading.

    Graus 0–4 na escala ICDR e **grau 5 = imagem sem qualidade para graduação** — usadas no notebook 06 para
    testar o controle de qualidade do Módulo 2 antes dos modelos.
    """
    destino = _limpar("DDR", force)
    repo, base = "ctmedtech/DDR-dataset", "DR_grading"
    rotulos = pd.read_csv(_hub(repo, f"{base}/valid.txt"), sep=" ", header=None, names=["arquivo", "grau"])
    cotas = {0: 200, 1: 60, 2: 140, 3: 40, 4: 60, 5: 50}
    amostra = pd.concat([rotulos[rotulos.grau == g].sample(n, random_state=SEMENTE) for g, n in cotas.items()])

    def _um(linha):
        grau = int(linha.grau)
        cond = "ilegivel" if grau == 5 else ("normal" if grau == 0 else "rd")
        id_ = f"ddr_{Path(linha.arquivo).stem}"
        arq = destino / cond / f"{id_}.jpg"
        tam = _gravar(_hub(repo, f"{base}/valid/{linha.arquivo}").read_bytes(), arq)
        return _registro("DDR", id_, cond, arq, tam, grau=grau, rotulo=f"grau {grau}", obs=linha.arquivo)

    with ThreadPoolExecutor(8) as ex:
        regs = list(tqdm(ex.map(_um, amostra.itertuples()), total=len(amostra), desc="DDR"))
    print(f"DDR: {len(regs)} imagens")
    return regs


# --------------------------------------------------------------------------- RFMiD (DMRI, externo)
def baixar_rfmid(force: bool) -> list[dict]:
    """RFMiD (Índia, 3 câmeras): todas as DMRI (ARMD) das partições de validação e teste + o dobro de normais."""
    destino = _limpar("RFMiD", force)
    repo = "ctmedtech/RFMID"
    from huggingface_hub import HfApi
    arquivos = _tentar(lambda: HfApi().list_repo_files(repo, repo_type="dataset"))
    planos = [("Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv", "validacao"),
              ("Test_Set/Test_Set/RFMiD_Testing_Labels.csv", "teste")]
    tarefas = []
    for csv, nome in planos:
        lab = pd.read_csv(_hub(repo, csv))
        pngs = {Path(a).name: a for a in arquivos if a.startswith(csv.split("/")[0]) and a.endswith(".png")}
        dmri = lab[lab.ARMD == 1]
        normais = lab[lab.Disease_Risk == 0].sample(2 * len(dmri), random_state=SEMENTE)
        for cond, sub in (("dmri", dmri), ("normal", normais)):
            for linha in sub.itertuples():
                outros = [c for c in lab.columns[2:] if c != "ARMD" and getattr(linha, c) == 1]
                tarefas.append((cond, f"rfmid_{nome}_{int(linha.ID):03d}", pngs[f"{int(linha.ID)}.png"],
                                "ARMD" + (" + " + ", ".join(outros) if outros else "") if cond == "dmri" else "normal"))

    def _um(t):
        cond, id_, caminho, rot = t
        arq = destino / cond / f"{id_}.jpg"
        tam = _gravar(_hub(repo, caminho).read_bytes(), arq)
        return _registro("RFMiD", id_, cond, arq, tam, rotulo=rot, obs=caminho.split("/")[-2] + "/" + Path(caminho).name)

    with ThreadPoolExecutor(8) as ex:
        regs = list(tqdm(ex.map(_um, tarefas), total=len(tarefas), desc="RFMiD"))
    print(f"RFMiD: {len(regs)} imagens")
    return regs


# --------------------------------------------------------------------------- manifesto e partições
FONTES = {"aptos": baixar_aptos, "g1020": baixar_g1020, "odir": baixar_odir, "ddr": baixar_ddr, "rfmid": baixar_rfmid}


def atualizar_manifesto(novos: list[dict], fontes_atualizadas: set[str]) -> pd.DataFrame:
    """Mescla os novos registros ao manifesto existente (preserva fontes não reprocessadas)."""
    antigo = pd.read_csv(MANIFESTO, keep_default_na=False) if MANIFESTO.exists() else pd.DataFrame(columns=COLUNAS)
    antigo = antigo[~antigo["fonte"].str.lower().isin(fontes_atualizadas)]
    df = pd.concat([antigo, pd.DataFrame(novos, columns=COLUNAS)], ignore_index=True)
    df = df.fillna("").sort_values(["fonte", "id"]).reset_index(drop=True)
    MANIFESTO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFESTO, index=False, encoding="utf-8")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(FONTES), help="baixa somente estas fontes")
    ap.add_argument("--force", action="store_true", help="apaga e rebaixa as fontes selecionadas")
    args = ap.parse_args()

    selecionadas = args.only or list(FONTES)
    novos: list[dict] = []
    for nome in selecionadas:
        print(f"\n=== {nome.upper()} ===")
        novos += FONTES[nome](args.force)

    df = atualizar_manifesto(novos, set(selecionadas))
    part = gerar_particoes(df, semente=SEMENTE)
    part.to_csv(PARTICOES, index=False, encoding="utf-8")
    print("\nManifesto:", _rel(MANIFESTO))
    print(df.groupby(["fonte", "condicao"]).size().unstack(fill_value=0).to_string())
    print("\nPartições:", _rel(PARTICOES))
    print(part.groupby(["tarefa", "particao", "alvo"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
