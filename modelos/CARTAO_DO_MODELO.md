# Cartão do modelo — Retina em Foco · triagem embarcada (v1.0)

Gerado por `notebooks/06_Validacao_Experimental.ipynb` em 07/10/2026. Projeto Retina em Foco — IFPE Campus Palmares, Edital FACEPE nº 14/2026.

## Uso pretendido
Triagem de retinografias coloridas em um retinógrafo portátil (Raspberry Pi 5), para **apoiar a decisão de encaminhamento** ao oftalmologista. **Não é diagnóstico.** Protótipo de pesquisa, sem validação clínica prospectiva nem registro na Anvisa; não usar em pacientes.

## Entrada e pré-processamento
Retinografia RGB → `retina_pdi.pipeline.preparar_para_inferencia` (commit `70d02c46d3`): recorte do FOV, quadrado, 224×224, CLAHE na luminância → tensor uint8. A captura deve passar no controle de qualidade do Módulo 02.

## Modelos
| Condição | Arquivo | Arquitetura | Limiar | Critério do limiar |
|---|---|---|---|---|
| Retinopatia diabética referível | `rd_efficientnet_lite0_int8.tflite` | efficientnet_lite0 (int8) | 0.7812 | maior limiar com sensibilidade >= 90% na validação |
| Glaucoma | `glaucoma_efficientnet_lite0_int8.tflite` | efficientnet_lite0 (int8) | 0.3320 | maior limiar com sensibilidade >= 90% na validação |
| DMRI | `dmri_efficientnet_lite0_int8.tflite` | efficientnet_lite0 (int8) | 0.0938 | maior limiar com sensibilidade >= 90% na validação |

## Desempenho (modelo int8, LiteRT)
n (positivos) · AUC (IC 95 %, bootstrap por paciente) · sensibilidade e especificidade no limiar (IC 95 % de Wilson).

### Retinopatia diabética referível
| Conjunto | n (pos.) | AUC | Sensibilidade | Especificidade |
|---|---|---|---|---|
| externo · DDR | 500 (240) | 0.933 (0.910–0.955) | 0.67 (0.61–0.73) | 0.98 (0.96–0.99) |
| externo · ODIR | 4481 (1608) | 0.675 (0.655–0.696) | 0.24 (0.22–0.27) | 0.97 (0.96–0.97) |
| teste · APTOS | 550 (223) | 0.986 (0.978–0.993) | 0.89 (0.84–0.92) | 0.96 (0.93–0.97) |

### Glaucoma
| Conjunto | n (pos.) | AUC | Sensibilidade | Especificidade |
|---|---|---|---|---|
| externo · G1020 | 1020 (296) | 0.541 (0.499–0.580) | 0.36 (0.31–0.42) | 0.70 (0.67–0.74) |
| teste · ODIR | 484 (42) | 0.911 (0.861–0.953) | 0.95 (0.84–0.99) | 0.67 (0.63–0.71) |

### DMRI
| Conjunto | n (pos.) | AUC | Sensibilidade | Especificidade |
|---|---|---|---|---|
| externo · RFMiD | 207 (69) | 0.978 (0.948–0.998) | 0.93 (0.84–0.97) | 0.97 (0.93–0.99) |
| teste · ODIR | 468 (41) | 0.897 (0.826–0.957) | 0.76 (0.61–0.86) | 0.85 (0.82–0.88) |

## Dados de treinamento
RD: APTOS 2019 (Índia) · Glaucoma e DMRI: ODIR-5K (China), divisão por paciente. Detalhes, licenças e divisão em `docs/DADOS.md` e `data/particoes.csv`.

## Limitações conhecidas
* Rótulos herdados das bases públicas, não revistos por oftalmologista do projeto.
* Desempenho cai em câmeras/populações diferentes das de treino (ver validação externa); **o limiar deve ser recalibrado no retinógrafo do projeto** com imagens locais rotuladas antes de qualquer uso.
* Escores não são probabilidades calibradas; não exibir como porcentagem.
* Cada modelo foi treinado contra imagens **normais**: a especificidade diante de **outras** doenças não foi medida. Ex.: retinas com RD podem receber escore alto de DMRI, e drusas ou cicatrizes podem disparar o modelo de RD. Encaminhamentos por uma condição devem ser lidos como 'retina alterada', não como o diagnóstico daquela condição.
* Glaucoma: desempenho muito menor na base externa G1020, cujo rótulo é diagnóstico clínico completo (ver validação externa).
* Amostras pequenas para glaucoma e DMRI: intervalos de confiança largos.
* Nenhuma avaliação por subgrupos (idade, sexo, etnia, comorbidades) com poder estatístico.
* Imagens sem qualidade produzem escores sem significado: o controle de qualidade é obrigatório.
