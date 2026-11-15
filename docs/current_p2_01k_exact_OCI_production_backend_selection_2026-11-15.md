# PyPCC — P2-01K: seleção real dos três backends na OCI original (15/11/2026)

**PR DRAFT baseado no P2-07O #289.** Nenhum algoritmo de produção, binário nativo, imagem OCI, grafo, máscara ou protocolo científico P1 é alterado. Os 72 ajustes reais continuam `NOT_RUN_NO_APPROVAL`.

## Problema encontrado na auditoria

A imagem OCI binária arquivada já inclui `/opt/p2-native/pcc_step*.so` e `pcc_step_live_research*.so` preservados com os hashes P2-01F. Porém, as fases anteriores de preparação do grafo e compilação Numba iniciavam seus contêineres com `PYTHONPATH=/workspace`, ocultando `/opt/p2-native` da resolução normal de imports (o Dockerfile original configura `PYTHONPATH=/opt/p2-source-helpers`).

O código original `pcc.py` permite **fallback silencioso**: quando `impl="cython"` é solicitado mas `_HAS_CYTHON=False`, a função `_get_backend_fn()` tenta Numba antes de NumPy. Isso pode invalidar uma execução científica caso o runner julgue que usar `impl="cython"` seja prova de que o kernel Cython realmente foi selecionado.

A **presença do binário na OCI** e o **teste import-only por caminho explícito no P2-01H** não provam a resolução automática pelo `pcc.py` em um futuro runner.

## Comparação operacional import-only

Reutiliza a imagem existente, com gzip SHA256 `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8` e config ID `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`. Dois contêineres **separados**, sem rede, não privilegiados, rootfs somente leitura, código original em bind readonly, 2 GiB de memória *proposta*, sem invocação dos kernels:

1. **Baseline:** `PYTHONPATH=/workspace`, idêntico ao override de P2-07M/O. Verifica se o arquivo Cython não é encontrado pelo import comum e se a solicitação de Cython resolve a função Numba, enquanto os imports NumPy/Numba continuam corretos.
2. **Configuração corrigida, sem mudar imagem:** `PYTHONPATH=/workspace:/opt/p2-native:/opt/p2-source-helpers`. Confere os `__file__` e SHA256 de ambos os módulos Cython congelados e verifica a **identidade de objeto da função** retornada por `ParticleCompetitionAndCooperation(impl=...)._get_backend_fn()`: NumPy deve ser `pcc_numpy.pcc_propagate_numpy`, Numba deve ser `pcc_numba.pcc_propagate_numba`, Cython deve ser exatamente `pcc_step.pcc_propagate` (sem fallback). O observador `pcc_step_live_research` deve importar do segundo ELF com seu hash exato.

O teste não monta matrizes de dados, não chama `build_graph` ou `set_graph`, não chama `fit_predict`, não executa `CPUDispatcher`, não compila assinatura de Numba e não invoca o observador Cython. Não gera seeds, máscaras ou checkpoints. Somente metadados de importação e hashes binários entram no relatório.

## Resultado e limitação

**Antes da execução do CI, a presença e o efeito real do fallback são hipótese verificável, não resultado confirmado.** O gate técnico exige 12 testes de contrato de fonte × 4 ambientes Linux/Windows Py3.10/3.12, mais os dois contêineres dentro da OCI original e regressão de produção independente. Qualquer divergência é NO-GO, sem afirmar import bem-sucedido.

Mesmo com todos os `PASS`, a prova se limita a resolver os *entrypoints* originais: **não valida observação de marcos reais, custo de 50.000 iterações, fidelidade de snapshots/RNG, cgroup no ajuste completo, nem fechamento completo do toolchain compilador/SDK/OS**, e não autoriza experimentos.

A OCI ainda está arquivada de maneira temporária no GitHub Actions; a disponibilidade em armazenamento durável externo (como NAS/local) não foi confirmada. É necessário preservar o arquivo com o SHA acima antes do vencimento de retenção, sem expor a imagem em publicação pública indevida.

**Estado científico exigido:** `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`. P2-08 ainda depende de um manifesto exato, host e política aprovados explicitamente.

**Após resultado de CI:** integrar a exigência de identidade real do backend ao futuro *execution worker* e não aceitar fallback de `impl` em nenhum dos 72 casos; nenhum ajuste real antes da autorização.
