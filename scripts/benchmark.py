#!/usr/bin/env python
"""Mede latência e memória dos modelos `.tflite` e da triagem completa — no PC ou no Raspberry Pi 5.

    python scripts/benchmark.py                          # modelos/*.tflite, threads 1/2/4, imagens de amostras/
    python scripts/benchmark.py --threads 4 --repeticoes 100
    python scripts/benchmark.py --modelos "modelos/rd_*.tflite"

Só precisa de NumPy, OpenCV, retina_pdi, LiteRT e psutil (sem TensorFlow). Grava
``outputs/benchmark_<máquina>_<data>.csv`` e compara com as metas do edital:
≤ 3 s por condição (pré-processamento + modelo) e ≤ 512 MB de RAM por inferência.
"""
from __future__ import annotations

import argparse
import glob
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from retina_iae.inferencia import ModeloTFLite, memoria_processo_mb, runtime  # noqa: E402

META_S_POR_CONDICAO = 3.0
META_RAM_MB = 512


def info_sistema() -> dict:
    info = {"maquina": socket.gethostname(), "sistema": f"{platform.system()} {platform.release()}",
            "arquitetura": platform.machine(), "python": platform.python_version(), "runtime": runtime()}
    try:
        import psutil
        info["nucleos"] = psutil.cpu_count(logical=False)
        info["ram_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    modelo = Path("/proc/device-tree/model")
    if modelo.exists():                                   # Raspberry Pi
        info["placa"] = modelo.read_text(errors="ignore").strip("\x00 \n")
    info["temperatura_C"] = temperatura()
    return info


def temperatura() -> float | None:
    """Temperatura do SoC (Raspberry Pi / Linux). O Pi 5 reduz o clock acima de ~80 °C — anote junto da latência."""
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return round(int(p.read_text()) / 1000, 1)
    except (OSError, ValueError):
        return None


def medir_modelos(arquivos: list[Path], threads: list[int], repeticoes: int) -> pd.DataFrame:
    linhas = []
    for arq in arquivos:
        for th in threads:
            m0 = memoria_processo_mb()
            t0 = time.perf_counter()
            m = ModeloTFLite(arq, threads=th)
            carga_ms = (time.perf_counter() - t0) * 1e3
            lat = m.medir_latencia(repeticoes=repeticoes)
            linhas.append({"modelo": arq.name, "threads": th, "tamanho_MB": round(m.tamanho_bytes / 1e6, 2),
                           "entrada": m.tipo_entrada.name, "xnnpack": m.xnnpack, "carga_ms": round(carga_ms, 1),
                           **{k: round(v, 2) for k, v in lat.items()},
                           "RAM_modelo_MB": round(memoria_processo_mb() - m0, 1), "temperatura_C": temperatura()})
            del m
            print(f"  {arq.name:<42s} threads={th}  mediana {linhas[-1]['mediana_ms']:7.2f} ms  "
                  f"p95 {linhas[-1]['p95_ms']:7.2f} ms  xnnpack={linhas[-1]['xnnpack']}")
    return pd.DataFrame(linhas)


def medir_triagem(imagens: list[Path], threads: int) -> pd.DataFrame:
    from retina_iae import dados as D
    from retina_iae.triagem import Triagem
    tri = Triagem(threads=threads, exigir_qualidade=False)
    linhas = []
    for img in imagens:
        rgb = D.carregar_rgb(img)
        tri.avaliar(rgb)                                    # aquecimento
        r = tri.avaliar(rgb)
        linhas.append({"imagem": img.name, "resolucao": f"{rgb.shape[1]}x{rgb.shape[0]}",
                       "qualidade_ok": r.aprovada_qualidade, **{f"ms_{k}": round(v, 1) for k, v in r.tempos_ms.items()},
                       "RAM_processo_MB": round(memoria_processo_mb(), 1),
                       **{f"p_{k}": round(v, 3) for k, v in r.probabilidades.items()}})
        print(f"  {img.name:<34s} total {r.tempos_ms['total']:7.1f} ms   RAM {linhas[-1]['RAM_processo_MB']:.0f} MB")
    return pd.DataFrame(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", default=str(RAIZ / "modelos" / "*.tflite"))
    ap.add_argument("--imagens", default=str(RAIZ / "amostras" / "*.jpg"))
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--repeticoes", type=int, default=50)
    ap.add_argument("--sem-triagem", action="store_true", help="mede só os modelos")
    args = ap.parse_args()

    info = info_sistema()
    print("Sistema:", ", ".join(f"{k}={v}" for k, v in info.items()))
    arquivos = sorted(Path(p) for p in glob.glob(args.modelos))
    if not arquivos:
        print("nenhum .tflite encontrado em", args.modelos)
        return 1
    print(f"\n1) Latência de invoke() — {len(arquivos)} modelos × threads {args.threads}")
    mod = medir_modelos(arquivos, args.threads, args.repeticoes)

    saida = RAIZ / "outputs"
    saida.mkdir(exist_ok=True)
    carimbo = f"{info['maquina']}_{datetime.now():%Y%m%d_%H%M}"
    mod.assign(**{k: v for k, v in info.items() if k != "temperatura_C"}).to_csv(
        saida / f"benchmark_modelos_{carimbo}.csv", index=False)

    imagens = sorted(Path(p) for p in glob.glob(args.imagens))
    if args.sem_triagem or not imagens or not (RAIZ / "modelos" / "config_triagem.json").exists():
        return 0
    th = max(args.threads)
    print(f"\n2) Triagem completa (qualidade + pré-processamento + 3 modelos), threads={th}")
    tri = medir_triagem(imagens, th)
    tri.assign(**{k: v for k, v in info.items() if k != "temperatura_C"}).to_csv(
        saida / f"benchmark_triagem_{carimbo}.csv", index=False)

    modelos_ms = tri[[c for c in tri.columns if c.startswith("ms_modelo_")]]
    pre_ms = tri[[c for c in ("ms_fov", "ms_qualidade", "ms_preparacao") if c in tri]].sum(axis=1)
    pior_condicao_s = (pre_ms + modelos_ms.max(axis=1)).max() / 1e3
    ram = tri.RAM_processo_MB.max()
    print("\nMetas do edital:")
    print(f"  tempo por condição (pior caso)  {pior_condicao_s:6.2f} s   meta ≤ {META_S_POR_CONDICAO:.0f} s   "
          f"{'OK' if pior_condicao_s <= META_S_POR_CONDICAO else 'ACIMA'}")
    print(f"  RAM do processo (pico medido)   {ram:6.0f} MB  meta ≤ {META_RAM_MB} MB  "
          f"{'OK' if ram <= META_RAM_MB else 'ACIMA'}")
    print(f"\nCSVs em {saida.relative_to(RAIZ)}/  (anexe-os ao fórum do módulo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
