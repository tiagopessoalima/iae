# Roteiro da sessão presencial — Inferência local no Raspberry Pi 5

**Módulo 04 · IA Embarcada** · Lab Maker do IFPE Campus Palmares · 5 h · equipes de 3–4 pessoas

**Objetivo:** medir, no próprio dispositivo, a latência e o consumo de memória da triagem completa (RD, glaucoma e DMRI) e verificar as metas do edital:
**≤ 3 s por condição** e **≤ 512 MB de RAM por inferência**. Pré-requisito: notebooks 04 e 05 lidos (o 05 é o ensaio desta sessão no PC).

## Material por bancada
* Raspberry Pi 5 (4 ou 8 GB) com **dissipador ativo** (Active Cooler) e fonte oficial de 27 W
* cartão microSD (≥ 32 GB) com **Raspberry Pi OS 64-bit** (Bookworm ou Trixie) e acesso à internet
* teclado, mouse e monitor — ou acesso por SSH a partir de um notebook
* (bloco 5) o protótipo do retinógrafo do Módulo 03 com a câmera conectada

## 0:00–0:40 · Preparação
```bash
git clone https://github.com/tiagopessoalima/iae.git
cd iae
./pi/setup_pi.sh                     # cria .venv e instala LiteRT, OpenCV, NumPy e o retina_pdi (sem TensorFlow)
cat /proc/device-tree/model; free -h; vcgencmd measure_temp
```
Anote: modelo da placa, RAM, versão do sistema (`cat /etc/os-release`), temperatura em repouso. O `setup_pi.sh` imprime o runtime
(`ai_edge_litert`) e a versão do Python.

## 0:40–1:10 · Primeira triagem
```bash
.venv/bin/python scripts/triar.py amostras/*.jpg
```
Para cada uma das seis amostras: o laudo faz sentido com o rótulo da fonte (`amostras/LEIAME.md`)? Quais o controle de qualidade reprova?
Espere reprovações também de imagens **legíveis**: os limiares de qualidade do Módulo 02 são rígidos demais para o DDR (notebook 06, seção 6).
Rode de novo com `--sem-qualidade` e compare os escores — inclusive os da imagem ilegível.

## 1:10–2:10 · Latência
```bash
.venv/bin/python scripts/benchmark.py                      # modelos × threads 1, 2, 4 + triagem completa
```
O script grava `outputs/benchmark_modelos_<placa>_<data>.csv` e `outputs/benchmark_triagem_…csv`. Para os modelos de comparação (quatro formatos do modelo de RD):
```bash
.venv/bin/python scripts/benchmark.py --modelos "modelos/comparacao/*.tflite" --sem-triagem
```
Perguntas: qual formato é mais rápido **no Pi**? A ordem é a mesma do PC (notebook 05)? O MobileNetV3 int8 roda com XNNPACK (coluna `xnnpack`)?

## 2:10–2:40 · Memória
```bash
.venv/bin/python scripts/medir_memoria.py amostras/*.jpg
/usr/bin/time -v .venv/bin/python scripts/triar.py amostras/*.jpg 2>&1 | grep "Maximum resident"
```
Quanto da memória é do interpretador e das bibliotecas e quanto é dos modelos? Abra outro terminal com `htop` durante a triagem.

## 2:40–3:40 · Estresse térmico
Em um terminal, deixe a medição longa rodando; no outro, registre a temperatura a cada 10 s:
```bash
.venv/bin/python scripts/benchmark.py --threads 4 --repeticoes 2000 --sem-triagem
watch -n 10 vcgencmd measure_temp      # e: vcgencmd get_throttled
```
Repita **com e sem** a ventoinha (desconecte o cabo do Active Cooler com o Pi desligado). A latência muda quando o SoC passa de ~80 °C?
`vcgencmd get_throttled` diferente de `0x0` indica redução de clock.

## 3:40–4:30 · Captura real
Capture imagens com o retinógrafo (software do Módulo 03) em resolução máxima e rode:
```bash
.venv/bin/python scripts/triar.py captura_*.jpg
```
O controle de qualidade aprova? Se não, por quê (os motivos aparecem no laudo)? Quanto tempo leva a triagem de uma captura de 12 MP?
Teste a otimização do notebook 05: reduza a captura para 1024 px antes (`python -c` com `cv2.resize`) e compare.
**Lembrete:** os escores do retinógrafo **não** são válidos clinicamente — os modelos nunca viram essa câmera e o limiar não foi recalibrado nela (notebook 06).

## 4:30–5:00 · Consolidação
Preencha a tabela e publique-a no fórum do módulo com os CSVs de `outputs/`:

| Medida | PC (notebook 05) | Raspberry Pi 5 |
|---|---|---|
| latência int8 EfficientNet-Lite0, 4 threads (ms) | | |
| latência int8 MobileNetV3-Small, 4 threads (ms) | | |
| triagem completa, captura 12 MP (s) | | |
| triagem com redução prévia para 1024 px (s) | | |
| pico de memória (MB) | | |
| temperatura após 10 min, com / sem ventoinha (°C) | — | |
| metas do edital atendidas? | | |

## Problemas comuns
* `pip` muito lento ou sem wheel: atualize o pip (`.venv/bin/python -m pip install -U pip`); em Python 3.13 (Trixie) confira se há wheel do `ai-edge-litert`.
* `ModuleNotFoundError: retina_pdi`: rode de novo `./pi/setup_pi.sh` (precisa de acesso ao GitHub).
* Latências muito variáveis: feche o navegador, desligue o modo gráfico (`sudo systemctl isolate multi-user.target`) e repita.
