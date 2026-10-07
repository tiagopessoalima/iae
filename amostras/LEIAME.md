# Amostras para o Raspberry Pi

Seis retinografias para testar a triagem no dispositivo sem baixar as bases completas (`scripts/triar.py amostras/*.jpg`,
`scripts/benchmark.py`). Sorteadas com semente fixa (sem escolha manual) entre as imagens de bases com licença **CC BY 4.0**
e reduzidas para 1024 px de largura. Também fazem parte das partições **externas** — nenhuma foi usada no treino.

| Arquivo | Fonte | Imagem original | Rótulo da fonte |
|---|---|---|---|
| `ddr_normal_grau0.jpg` | DDR | `DR_grading/valid/20170330111147398.jpg` | grau 0 (sem RD) |
| `ddr_rd_grau2.jpg` | DDR | `DR_grading/valid/007-5818-300.jpg` | grau 2 (RD moderada) |
| `ddr_rd_grau4.jpg` | DDR | `DR_grading/valid/007-6867-400.jpg` | grau 4 (RD proliferativa) |
| `ddr_ilegivel_grau5.jpg` | DDR | `DR_grading/valid/007-8215-601.jpg` | grau 5 (sem qualidade para graduação) |
| `rfmid_dmri.jpg` | RFMiD | `Test_Set/Test/188.png` | ARMD (DMRI) |
| `rfmid_normal.jpg` | RFMiD | `Evaluation_Set/Validation/593.png` | normal |

**Créditos e licença (CC BY 4.0):**
* DDR — Li, T. et al. *Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening.* Information Sciences 501:511–522, 2019.
* RFMiD — Pachade, S. et al. *Retinal Fundus Multi-Disease Image Dataset (RFMiD).* Data 6(2):14, 2021.

Modificação: redimensionamento para 1024 px de largura e recompressão JPEG (qualidade 95).
