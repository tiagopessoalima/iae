#!/usr/bin/env bash
# Prepara o Raspberry Pi 5 para a sessão presencial do Módulo 04 (inferência local).
#   git clone https://github.com/tiagopessoalima/iae.git && cd iae && ./pi/setup_pi.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! python3 -c "import venv" 2>/dev/null; then
    sudo apt-get update && sudo apt-get install -y python3-venv
fi
[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install -r pi/requirements-pi.txt
$PY -m pip install -e . --no-deps

echo
$PY -c "from retina_iae.inferencia import runtime; import platform; print('runtime:', runtime(), '| python', platform.python_version(), '|', platform.machine())"
cat /proc/device-tree/model 2>/dev/null && echo
echo "Pronto. Próximos passos (docs/ROTEIRO_PI.md):"
echo "  .venv/bin/python scripts/triar.py amostras/*.jpg"
echo "  .venv/bin/python scripts/benchmark.py"
