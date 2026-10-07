# Dados: fontes, licenças, tarefas e cuidados

Todas as imagens são **retinografias reais, públicas e anonimizadas**, baixadas por `python scripts/download_data.py`
(≈ 1,7 GB baixados do Hugging Face Hub, sem cadastro; ≈ 0,5 GB em disco). O script grava:

* `data/raw/<fonte>/<condicao>/<id>.jpg` — imagens com lado maior ≤ 1024 px (JPEGs que já cabem são copiados sem recompressão);
* `data/manifest.csv` — uma linha por imagem (versionado);
* `data/particoes.csv` — uma linha por (imagem, tarefa) com alvo e partição (versionado; semente 2026).

Depois, `python scripts/preparar_cache.py` gera `data/cache/tensores_224_clahe.npy` (1,6 GB) com o pré-processamento do Módulo 02.

| Fonte | Conteúdo usado | Resolução aqui | Licença / uso | Espelho |
|---|---|---|---|---|
| **APTOS 2019** (Aravind Eye Hospital, Índia) | 3 662 imagens, graus 0–4 (ICDR) | 224×224 (espelho já reduzido) | termos da competição Kaggle (pesquisa, não comercial) | `hf.co/datasets/bumbledeep/aptos` |
| **DDR** (147 hospitais, China) | 550 imagens sorteadas da partição *valid*: graus 0 (200), 1 (60), 2 (140), 3 (40), 4 (60), 5 = ilegível (50) | ≤ 1024 px | CC BY 4.0 | `hf.co/datasets/ctmedtech/DDR-dataset` |
| **G1020** (Alemanha) | 1 020 imagens: 724 normais, 296 glaucoma | 512 px | uso acadêmico (confira a página dos autores) | `hf.co/datasets/ai4ophth/G1020_dataset` |
| **ODIR-5K** (China) | 5 031 imagens de 2 914 pacientes: normal 2 873, diabetes 1 608, glaucoma 284, DMRI 266 | 512×512 | uso educacional/pesquisa | `hf.co/datasets/bumbledeep/odir` |
| **RFMiD** (Índia) | 207 imagens das partições de validação e teste: todas as 69 com DMRI (ARMD) + 138 normais sorteadas | ≤ 1024 px | CC BY 4.0 | `hf.co/datasets/ctmedtech/RFMID` |

Total: **10 470 imagens**.

## Tarefas e partições (`retina_iae.particoes`)

| Tarefa | Positivo × negativo | Desenvolvimento (treino/validação/teste 70/15/15) | Externo |
|---|---|---|---|
| `rd` | RD **referível** (grau ≥ 2) × graus 0–1 | APTOS | DDR (graus 0–4); ODIR (diabetes × normal) |
| `glaucoma` | glaucoma × normal | ODIR (divisão **por paciente**) | G1020 |
| `dmri` | DMRI × normal | ODIR (divisão **por paciente**) | RFMiD |

Divisão estratificada pelo alvo. No ODIR, os dois olhos de um paciente ficam sempre na mesma partição.
As imagens DDR de grau 5 não participam de nenhuma tarefa: servem para testar o controle de qualidade (notebook 06).

### Bancos do edital
O edital cita Messidor-2, APTOS 2019, DDR (RD), REFUGE (glaucoma) e ADAM (DMRI). Usamos **APTOS 2019** e **DDR**.
REFUGE e ADAM exigem cadastro (e o espelho aberto do REFUGE tem estrutura corrompida); em seu lugar, **G1020** (glaucoma) e **ODIR/RFMiD** (DMRI).
O Messidor-2 tem espelho aberto (`hf.co/datasets/sngsfydy/Messidor2`, 2,5 GB) e fica como exercício do notebook 06.

## Como citar
* APTOS — Asia Pacific Tele-Ophthalmology Society. *APTOS 2019 Blindness Detection* (Kaggle), 2019.
* DDR — Li, T. et al. *Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening.* Information Sciences 501:511–522, 2019.
* G1020 — Bajwa, M. N. et al. *G1020: A Benchmark Retinal Fundus Image Dataset for Computer-Aided Glaucoma Detection.* IJCNN, 2020.
* ODIR-5K — Peking University / Shanggong Medical Technology, *ODIR-2019 (Ocular Disease Intelligent Recognition)*.
* RFMiD — Pachade, S. et al. *Retinal Fundus Multi-Disease Image Dataset (RFMiD).* Data 6(2):14, 2021.
* Pesos pré-treinados — MobileNetV3-Small: Keras Applications (ImageNet). EfficientNet-Lite0: `timm/tf_efficientnet_lite0.in1k`
  (Apache 2.0), convertidos do checkpoint oficial do TensorFlow.

Os espelhos do Hugging Face são cópias de terceiros; suas etiquetas de licença **não substituem** as dos autores. Antes de publicar resultados,
confira a licença vigente em cada página original.

## Cuidados de interpretação
1. **Rótulos herdados, não revistos** por oftalmologista do projeto.
2. **"RD" difere entre bases.** APTOS e DDR têm grau; o ODIR só diz "diabetes" (qualquer grau, inclusive RD leve, que é *negativa* na nossa tarefa).
3. **G1020: diagnóstico clínico, difícil de ver na imagem.** Os rótulos vêm do exame oftalmológico completo; numa validação cruzada *dentro* do G1020,
   um extrator ImageNet + regressão logística fica em AUC ≈ 0,51. Por isso ele é a base **externa** do glaucoma (e o desenvolvimento usa o ODIR).
4. **RFMiD é multirrótulo**: parte das DMRI tem outros achados (coluna `rotulo_original`).
5. **APTOS em 224 px** no espelho aberto: lesões muito pequenas ficam no limite da resolução.
6. **Amostras pequenas** para glaucoma e DMRI → intervalos de confiança largos. Resultados são **ilustrativos**, não estimativas de desempenho clínico.
7. **Ética.** Somente bases públicas e anonimizadas; sem procedimentos com seres humanos (Edital nº 14/2026, §2.3.2).

## Esquema de `data/manifest.csv`
| Coluna | Significado |
|---|---|
| `id` | identificador único (`<fonte>_<n>`) |
| `fonte` | `APTOS`, `DDR`, `G1020`, `ODIR` ou `RFMiD` |
| `condicao` | `normal`, `rd`, `glaucoma`, `dmri` ou `ilegivel` |
| `grau` | grau de RD 0–4 (5 = ilegível no DDR); vazio quando a fonte não informa |
| `rotulo_original` | rótulo na base de origem |
| `paciente` | identificador do paciente (só ODIR) |
| `arquivo` | caminho relativo à raiz do projeto |
| `largura`, `altura` / `largura_original`, `altura_original` | tamanho gravado / tamanho na fonte |
| `licenca`, `referencia`, `observacao` | proveniência (no ODIR, `observacao` traz idade e sexo) |

Para acrescentar imagens do **retinógrafo do projeto**, use `fonte = "OIO"`, as mesmas colunas, inclua a fonte em `particoes.TAREFAS` (como externa)
e refaça o cache (`scripts/preparar_cache.py --forcar`).
