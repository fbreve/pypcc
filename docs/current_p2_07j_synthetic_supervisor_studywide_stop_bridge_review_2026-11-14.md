# PyPCC — P2-07J: supervisor artificial H → STOP global I (2026-11-14)

**Ramo de pesquisa empilhado sobre o PR #283. Nenhuma alteração em produção ou merge.** Todos os 72 casos científicos originais permanecem **NOT_RUN_NO_APPROVAL**. Não são executados UCI, PCC, grafos, máscaras, sementes científicas ou nenhum ajuste real.

## Escopo novo

No P2-07H (#282), o supervisor único controlou árvores artificiais de três processos usando a imagem OCI original, cgroup v2, limites de memória/tmpfs e identificação host PID + starttime. No P2-07I (#283), a política de STOP para os 72 IDs científicos foi simulada sem lançar processos.

O P2-07J conecta os dois: percorre os 72 IDs originais e chama o supervisor H.one() no ID previamente selecionado para injetar uma falha **em uma árvore artificial**. Após um resultado de falha, exige comprovante de encerramento e limpeza dos três processos/contêiner e registra o STOP global. Os IDs posteriores não são despachados nem administrativamente validados. Resultado inesperado, saída normal considerada falha, limpeza incompleta, registro truncado ou reordenado, repetição e falsa aprovação científica são bloqueados.

## Matriz fixa

| Falha do STOP I | Índice original (zero-based) | Evidência |
| --- | ---: | --- |
| WALL_DEADLINE | 0 | Árvore artificial H real, limite amostrado de wall |
| CPU_TIME_BUDGET | 35 | Árvore artificial H real, CPU acumulada de cgroup |
| CGROUP_MEMORY_OOM | 71 | Árvore artificial H real, OOM do kernel |
| TMPFS_SCRATCH_ENOSPC | 0 | Árvore artificial H real, ENOSPC de tmpfs |
| WORKER_ROOT_FAILED | 35 | Árvore artificial H real, saída 23 do PID raiz |
| DESCENDANT_CLEANUP_UNVERIFIED | 71 | Injeção administrativa antes do processo; evidência H de limpeza falsa também é rejeitada pelos testes adversariais |
| STUDY_WALL_DEADLINE | 0 | Injeção administrativa; **não** se simula realmente esperar 48 h |

Resultado esperado do teste integrado: **5 lançamentos artificiais**, duas injeções prévias, sete campanhas de 72 IDs (504 linhas exclusivamente administrativas), cinco diários H de cinco eventos (25 registros) e **zero execuções de PCC**. Os sete cenários não substituem os 22 testes administrativos anteriores do P2-07I e não cobrem, em runtime real, todas as combinações de posição e falha.

A trilha de 504 registros é escrita com fsync por registro, sem sobrescrita, com SHA256 de conteúdo. Cada linha registra estado científico NOT_RUN_NO_APPROVAL e PCC_worker_invocations=0; os cinco diários H exigem exatamente cinco eventos em ordem e prova de três identidades de processo encerradas. Hash não equivale a assinatura criptográfica.

## Proveniência exigida

- P1 científico congelado: f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166.
- P2-07H, [execução original #43547341278](https://github.com/fbreve/pypcc/actions/runs/43547341278), artefato fonte-sem-PCC 14094039842, JSON canônico SHA256 0f96c4c91afbd0840f50adc430f42017207b0da71e3a6126c012d50d804388db.
- OCI original reaproveitada da [execução #43531054368](https://github.com/fbreve/pypcc/actions/runs/43531054368). Gzip SHA256 008b189f5326036aa6de95cb189dd2de3984cd0d93641c8; config id sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55. Sem reconstrução, acesso a registries ou arquivamento de cópia.
- Proposta P2-07C (ainda NÃO aprovada): 1.800 s wall e 3.600 CPU-s por slot, 48 h para todo estudo. Estes valores não são medidos ou demonstrados para os conjuntos reais.

A CI deve executar 13 testes adversariais sem Docker em Linux e Windows, Python 3.10/3.12, além de um job Linux real de cgroup v2 com os cinco processos artificiais supervisionados. A conclusão PASS só poderá ser registrada após verificar o job e os artefatos preservados.

## Limitações e próximos gates

Isto não mede limites de memória/tempo para grafos UCI reais ou 50k iterações, não prova fechamento de toolchain/compiladores/host, não implementa um executor científico e não aprova o teto de 48 horas. A proteção real de processos em Windows e a persistência duradoura da OCI continuam pendentes.

**STOP obrigatório**: P2_01_final_execution_toolchain_lock_pass=false; P2_07_full_resources_gate_pass=false; P2_execution_preflight_sha256=null; P2_execution_authorized=false; real_PCC_fits_executed=0. Um manifesto P2-08 separado, verificável e expressamente autorizado será necessário antes de qualquer ajuste real.
