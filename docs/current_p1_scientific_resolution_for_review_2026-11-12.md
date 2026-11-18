# Decisão científica P1 — parecer de revisão, ainda sem aprovação (12/11/2026)

**Estado: PROPOSTA PARA DECISÃO DO MANTENEDOR.** Este é o ponto de encerramento da triagem exploratória de fontes, não uma nova auditoria de datasets e **não** um protocolo científico congelado. O registro verificável das escolhas é [`current_p1_scientific_resolution_for_review_2026-11-12.json`](current_p1_scientific_resolution_for_review_2026-11-12.json), associado ao [PR #258](https://github.com/fbreve/pypcc/pull/258).

## Parecer científico proposto

**Recomendo GO somente para formalizar um estudo exploratório de estabilidade de classificação**, sujeito à aprovação explícita dos itens S1–S6 abaixo. Ainda é **NO GO** para qualquer execução real, geração de máscaras de rótulos de datasets reais, alegação de validação confirmatória ou merge em produção.

| Item | Recomendação a decidir | Motivo e limite |
| --- | --- | --- |
| **S1 — Conjuntos e histórico** | Propor **Vertebral 3-class, Raisin, Wholesale** (IDs UCI 212, 850, 292) | São três domínios diferentes. Rice/Raisin não devem contar como dois domínios. A revisão abrange dez fontes experimentais públicas delimitadas, **não** todos os trabalhos, teses e resultados locais. O usuário deve declarar qualquer uso anterior conhecido sem a obrigação irrealista de provar ausência de todo ensaio privado. |
| **S2 — Alvos e origem** | Propor `Class` 3-class, `Class` binário, `Channel` binário | Para **Wholesale**, o alvo `Channel` é **tarefa derivada**, diferente do alvo `Region` atribuído nos metadados UCI; `Channel` e `Region` jamais entram como atributos. Atribuição conforme CC BY 4.0. Não presumir independência de pacientes, lotes de imagens e grupos de clientes sem seus IDs. |
| **S3 — Pré-processamento** | Verificar não finitos, não eliminar registros; **z-score populacional (`ddof=0`)** em Vertebral e Raisin; **`log1p` + z-score populacional** nos seis gastos do Wholesale | Evita seleção oportunista a partir de resultados do PCC; reconhece a assimetria dos gastos. Transformação não supervisionada em **todas as linhas de X** deve ser declarada como parte do estudo transdutivo. |
| **S4 — Grafo** | Preservar `build_knn_graph(X, k_nn=10, metric="minkowski", p=2, n_jobs=1)` da produção | Simetrização não direcionada; lista densa ordenada `neib_list` com padding `-1`. Não substituir por CSR. P2 deve conferir self-loops, simetria, empates, ordenação, caminho `build_graph`/`set_graph` e **hash dos vizinhos efetivamente usados**. |
| **S5 — Desenho e sementes** | Três conjuntos, oito máscaras por conjunto, três rotulados/classe, três backends, 50.000 iterações sem parada antecipada | **72 ajustes só depois de aprovação**, checkpoints 0, 1, 10, 100, 1.000, 5.000, 25.000, 50.000. Sementes de máscaras PCG64 separadas das sementes nativas derivadas com SHA-256. Mesmo grafo e máscara para os três backends; RNG nativo não precisa ser bit a bit idêntico entre eles. |
| **S6 — Análise** | **Exploratória descritiva** por dataset, máscara e backend | Primária: proporção de nós inicialmente sem rótulos cujo argmax muda de 5.000 para 50.000. Relatar as oito máscaras por conjunto/backend, média/mediana/mínimo/máximo e média **não ponderada entre três conjuntos**. Empates exatos documentados. Sem `p`-values, IC de 95%, equivalência, generalização populacional nem pseudorreplicação de nós. |

## O que significa autorizar P1

A decisão não deve afirmar que os candidatos são “nunca utilizados”, que 440 clientes são 440 organizações independentes, nem que os indivíduos clínicos não têm relações. Essas verificações **não foram realizadas**. A decisão aceitável, caso o mantenedor concorde, é **assumir e divulgar tais limitações para um estudo descritivo sobre o comportamento do algoritmo**.

**O mantenedor precisa registrar explicitamente** se conhece usos locais ou não publicados desses conjuntos (inclusive sob outros nomes), se concorda com `Channel` como tarefa binária derivada, se aceita a falta de identificadores de agrupamento e se aprova S1–S6 como um todo ou solicita alterações. O presente documento contém recomendações **pendentes**, sem aprovação por omissão.

Quando essas decisões forem tomadas, o próximo artefato será um **protocolo P1 imutável com SHA científico próprio** (não o checksum do documento de revisão). Em seguida virá um **P2 separado** para testar versões e binários reais, limites de CPU/RAM/disco, construção do grafo, hashes de máscaras e segurança das exportações. **Somente um GO explícito ao SHA P2** poderá autorizar o primeiro ajuste real do PCC.

## Registros de suporte

- [Protocolo P1 original ainda não aprovado](current_p1_prospective_design_draft_2026-11-12.json) e [decisões científicas técnicas](current_p1_scientific_decision_packet_2026-11-12.json).
- [Sementes e análise exploratória](current_p1_rng_and_exploratory_analysis_2026-11-12.md).
- [Três primeiros artigos verificados](current_p1_public_literature_exposure_2026-11-12.md) e [sete fontes adicionais](current_p1_additional_public_papers_2026-11-12.md).
- [PR #260 de qualidade dos arquivos originais](https://github.com/fbreve/pypcc/pull/260): verificação de hashes, classes, duplicatas, limites de quase duplicatas, Wholesale por região e gastos. Os resultados são metadados descritivos, não resultados do PCC.

**Invariantes atuais:** `dataset_registry.selected=[]`; `scientific_protocol_frozen=false`; P1 SHA `null`; P2 SHA `null`; zero máscaras reais/zero fits; `execution_authorized=false`. Permanecer como PR **DRAFT/UNMERGED**, sem mudanças em `master`.
