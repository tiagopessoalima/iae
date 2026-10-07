#!/usr/bin/env python
"""Executa (headless) um ou mais notebooks e grava as saídas nos próprios .ipynb.

    python scripts/executar_notebooks.py                 # todos os notebooks/*.ipynb, em ordem
    python scripts/executar_notebooks.py 03 04           # só os que começam com 03 e 04
    python scripts/executar_notebooks.py --limpar        # apaga as saídas (para versionar sem outputs)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

if sys.platform == "win32":  # evita o aviso do zmq/tornado com o loop padrão do Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "notebooks"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixos", nargs="*", help="prefixos numéricos dos notebooks (ex.: 03 04)")
    ap.add_argument("--limpar", action="store_true", help="apenas remove as saídas das células")
    ap.add_argument("--timeout", type=int, default=3600, help="tempo máximo por célula, em segundos")
    args = ap.parse_args()

    alvos = sorted(p for p in PASTA.glob("*.ipynb") if not args.prefixos or p.name.startswith(tuple(args.prefixos)))
    if not alvos:
        print("nenhum notebook encontrado")
        return 1
    falhas = 0
    for p in alvos:
        nb = nbformat.read(p, as_version=4)
        if args.limpar:
            for c in nb.cells:
                if c.cell_type == "code":
                    c.outputs, c.execution_count = [], None
            nbformat.write(nb, p)
            print(f"limpo   {p.name}")
            continue
        t0 = time.time()
        try:
            NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(PASTA)}}
                           ).execute()
            print(f"ok      {p.name}  ({time.time() - t0:.0f}s)")
        except CellExecutionError as e:
            falhas += 1
            print(f"ERRO    {p.name}\n{str(e)[-1500:]}")
        finally:
            nbformat.write(nb, p)  # grava mesmo com erro, para inspecionar a célula que falhou
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
