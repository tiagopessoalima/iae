#!/usr/bin/env bash
# Prepara o ambiente do Módulo 04 em Linux/macOS (PC de treino). Para o Raspberry Pi, use pi/setup_pi.sh.
#   ./scripts/setup.sh              # venv + dependências + pacote + dados + cache + testes
#   ./scripts/setup.sh --sem-dados  # pula o download das imagens (~1,7 GB) e o cache
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt
$PY -m pip install -e ".[dev]"
$PY -m ipykernel install --user --name retina-modulo04 --display-name "Python (Retina em Foco - M04)"

if [ "${1:-}" != "--sem-dados" ]; then
    $PY scripts/download_data.py
    $PY scripts/preparar_cache.py
fi
$PY -m pytest -q
echo
echo "Pronto. Abra o VS Code / JupyterLab e escolha o interpretador .venv (kernel 'Python 3')."
echo "  JupyterLab: .venv/bin/jupyter lab notebooks"
