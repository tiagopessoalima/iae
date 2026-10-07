# Retina em Foco · Módulo 04 — IA Embarcada

Material prático do **Módulo 04** (20 h · 5 h presenciais · Semanas 3–8) do curso de formação técnica do projeto *Retina em Foco*
(Edital nº 14/2026 FACEPE — Compet Superior · IFPE Campus Palmares · Centro Médico Senador José Ermírio de Moraes).

Ementa: **01** aprendizado profundo e redes convolucionais · **02** transfer learning · **03** MobileNetV3-Small e EfficientNet-Lite0 ·
**04** quantização e TensorFlow Lite · **05** inferência em Raspberry Pi · **06** validação experimental dos modelos.
Cada item tem um notebook executável. Ao final, três modelos de triagem — **retinopatia diabética referível, glaucoma e DMRI** — quantizados em int8
e rodando localmente num Raspberry Pi 5, a partir do tensor produzido pelo pré-processamento do [Módulo 02](https://github.com/tiagopessoalima/pdib).

### Resultados (modelos int8 em `modelos/`, EfficientNet-Lite0, LiteRT)

| Modelo | Teste interno (mesma fonte) | Validação externa (outra câmera/país) | No PC · captura 12 MP |
|---|---|---|---|
| RD referível | APTOS: AUC 0,986 · sens. 0,89 · esp. 0,96 | DDR: AUC 0,933 · sens. 0,67 · esp. 0,98 — ODIR*: AUC 0,675 | ≈ 4 ms por modelo |
| Glaucoma | ODIR: AUC 0,911 · sens. 0,95 · esp. 0,67 | G1020: AUC 0,541 (não generaliza) | |
| DMRI | ODIR: AUC 0,897 · sens. 0,76 · esp. 0,85 | RFMiD: AUC 0,978 · sens. 0,93 · esp. 0,97 | |
| **Triagem completa** | | | **0,44 s** (meta ≤ 3 s) · **218 MB** (meta ≤ 512 MB) |

\* No ODIR, "diabetes" inclui RD leve, que é negativa para o modelo. Intervalos de confiança, limiares e limitações em
[`modelos/CARTAO_DO_MODELO.md`](modelos/CARTAO_DO_MODELO.md). Os números do Raspberry Pi 5 são coletados na sessão presencial.

Algumas lições que os notebooks medem, em vez de afirmar: o aumento de dados **não** ajudou a CNN pequena; uma linha de base que acertava 0,94 no APTOS
cai ao acaso (0,53) numa câmera nova; o **MobileNetV3-Small int8 não roda no XNNPACK** e fica dezenas de vezes mais lento que o próprio float32 (≈ 100 ms × 1,5 ms), enquanto o
EfficientNet-Lite0 int8 roda sem perda de AUC — foi por isso que ele foi escolhido para as três tarefas.

## Início rápido

**Windows (PowerShell)**
```powershell
.\scripts\setup.ps1            # cria .venv, instala dependências, baixa as imagens (~1,7 GB), gera o cache e roda os testes
.\.venv\Scripts\jupyter lab notebooks
```
**Linux / macOS**
```bash
./scripts/setup.sh
.venv/bin/jupyter lab notebooks
```
**Raspberry Pi 5** (só inferência, sem TensorFlow e sem baixar as bases)
```bash
./pi/setup_pi.sh
.venv/bin/python scripts/triar.py amostras/*.jpg
.venv/bin/python scripts/benchmark.py
```
No **VS Code**, abra a pasta, escolha o interpretador `.venv` e o kernel *Python 3*.

Passo a passo manual: `python -m venv .venv` → `pip install -r requirements.txt` → `pip install -e ".[dev]"` →
`python scripts/download_data.py` → `python scripts/preparar_cache.py`.
`requirements-lock.txt` guarda as versões exatas com que tudo foi testado (Python 3.12, TensorFlow 2.21, Keras 3.15, LiteRT 2.3, NumPy 2.5, OpenCV 4.14).

> **GPU:** o TensorFlow não usa GPU no Windows nativo. Não faz falta: os modelos são pequenos e cada treino leva de 2 a 15 minutos numa CPU de 8 núcleos.
> Com Linux/WSL2 e GPU NVIDIA, o mesmo código usa a GPU sem mudanças.

## Notebooks (`notebooks/`)

| # | Notebook | Você vai… |
|---|---|---|
| 00 | `00_Visao_Geral_e_Dados` | conhecer as 5 bases, as 3 tarefas, as partições (por paciente) e verificar a paridade com o pré-processamento do Módulo 02 |
| 01 | `01_Redes_Convolucionais` | provar que `Conv2D` = `cv2.filter2D`; montar e treinar uma CNN do zero; campo receptivo; aumento de dados; linha de base; mapas de ativação e Grad-CAM |
| 02 | `02_Transfer_Learning` | ver o que o ImageNet "acha" de uma retina; extrator congelado × ajuste fino em duas fases; curva de dados; o espaço de características separa **câmeras** |
| 03 | `03_MobileNetV3_e_EfficientNet_Lite` | convolução separável, MBConv, SE, hard-swish; custo × acurácia; treinar os **seis** modelos (3 tarefas × 2 arquiteturas) |
| 04 | `04_Quantizacao_e_TFLite` | quantização afim; float32/float16/dinâmica/int8; erro, AUC e latência; por que o MobileNetV3 int8 não roda no XNNPACK; **exportar `modelos/`** |
| 05 | `05_Inferencia_no_Raspberry_Pi` | LiteRT sem TensorFlow; latência × threads × formato; triagem de uma captura de 12 MP etapa por etapa; memória; roteiro do Lab Maker |
| 06 | `06_Validacao_Experimental` | limiar de triagem; IC; **validação externa**; RD por grau; recalibração na câmera nova; calibração e VPP; controle de qualidade; erros com Grad-CAM; **cartão do modelo** |

Os notebooks vêm **já executados** (saídas embutidas). Para reexecutar: `python scripts/executar_notebooks.py` (≈ 1h15 em CPU de 8 núcleos, 35 min só no 03; aceita prefixos, ex.: `04 05`).
O 03 treina os modelos usados pelo 04, e o 04 exporta os usados pelo 05 e 06 — mas `modelos/` já vem pronto, então dá para começar por qualquer um.
Cada notebook termina com **exercícios propostos** — base da entrega "Exercícios de treinamento e quantização — modelos de RD, glaucoma e DMRI".
A sessão presencial (5 h) segue [`docs/ROTEIRO_PI.md`](docs/ROTEIRO_PI.md).

## Estrutura

```
iae/
├── notebooks/            # 00–06 (executados)
├── src/retina_iae/       # pacote reutilizável
│   ├── dados.py          #   manifesto, partições, cache de tensores (pré-processamento do Módulo 02)
│   ├── particoes.py      #   definição das 3 tarefas e divisão estratificada por paciente
│   ├── modelos.py        #   CNN simples, MobileNetV3-Small, EfficientNet-Lite0 (implementado camada a camada), custo
│   ├── treino.py         #   tf.data, aumento de dados, transfer learning em duas fases
│   ├── conversao.py      #   Keras → .tflite (float32, float16, dinâmica, int8)            [TensorFlow]
│   ├── inferencia.py     #   LiteRT: ModeloTFLite, latência, memória                        [sem TensorFlow]
│   ├── triagem.py        #   captura → qualidade (M02) → 3 modelos → laudo                  [sem TensorFlow]
│   ├── metricas.py       #   sensibilidade, especificidade, AUC com IC, limiar, calibração, VPP
│   ├── explicacao.py     #   Grad-CAM
│   └── visualizacao.py
├── modelos/              # .tflite int8 finais + config_triagem.json + CARTAO_DO_MODELO.md (versionados)
│   └── comparacao/       #   modelo de RD nos 4 formatos × 2 arquiteturas (para medir no Pi)
├── amostras/             # 6 retinografias CC BY 4.0 para testar no Pi sem baixar as bases
├── scripts/              # download_data · preparar_cache · executar_notebooks · triar · benchmark · medir_memoria · setup.ps1/.sh
├── pi/                   # setup_pi.sh · requirements-pi.txt (LiteRT, sem TensorFlow)
├── tests/                # pytest
├── data/                 # manifest.csv e particoes.csv (versionados) · raw/ · cache/ (gerados)
├── docs/                 # DADOS.md (fontes, licenças, cuidados) · ROTEIRO_PI.md (sessão presencial)
└── requirements.txt · requirements-lock.txt · pyproject.toml
```

## Usando os modelos fora dos notebooks

```python
from retina_iae import dados
from retina_iae.triagem import Triagem

tri = Triagem(threads=4)                                   # lê modelos/config_triagem.json
r = tri.avaliar(dados.carregar_rgb("captura.jpg"))         # qualidade (M02) → tensor → 3 modelos int8
print(tri.laudo(r))                                        # encaminhar ou não, por condição, com escore e limiar
r.probabilidades, r.encaminhar, r.tempos_ms
```

O pré-processamento é o do Módulo 02 (`retina_pdi`, **fixado num commit** em `requirements.txt`): o mesmo tensor no treino e no dispositivo.

## Testes

```bash
.venv/Scripts/python -m pytest -q     # Windows   ·   .venv/bin/python -m pytest -q  (Linux)
```
Cobrem as regras das tarefas, a divisão sem vazamento de paciente, a arquitetura do EfficientNet-Lite0, a conversão float32/int8, a quantização da entrada,
as métricas, a triagem ponta a ponta (inclusive com captura toda preta) e a **paridade** entre o cache de treino e o `retina_pdi`.

## Aviso

Material didático de um **protótipo de pesquisa**. Os modelos fazem **triagem**, não diagnóstico; não foram validados clinicamente nem na câmera do projeto
e não devem ser usados em pacientes. Veja as limitações em [`modelos/CARTAO_DO_MODELO.md`](modelos/CARTAO_DO_MODELO.md) e [`docs/DADOS.md`](docs/DADOS.md).
