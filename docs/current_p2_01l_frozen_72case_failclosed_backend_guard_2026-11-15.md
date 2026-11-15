# PyPCC — P2-01L: guardião reutilizável da identidade exata de backend, antes de cada caso P1 (15/11/2026)

**PR experimental DRAFT sobre P2-01K #290; não altera `master` ou kernels originais.** Protocolo científico exploratório P1 SHA256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. Os 72 ajustes permanecem `NOT_RUN_NO_APPROVAL`, sem autorização P2-08.

## Problema corrigido com verificação executável

O P2-01K comprovou que `impl="cython"` no `pcc.py` original pode retornar **Numba**, dependendo de `PYTHONPATH`. A disponibilidade do arquivo Cython na OCI e o valor `model.impl` não são evidências suficientes da implementação selecionada.

O novo módulo `experiments/p2_01l_frozen_72case_backend_guard_no_fit.py` disponibiliza `assert_case_backend_exact(protocol, uci_id, replicate, backend)` como **pré-requisito reutilizável** para um futuro executor científico. Antes de qualquer ajuste, ele:

1. Confere se o caso pertence exatamente à lista congelada das 72 posições originais (3 fontes × 8 máscaras × 3 backends), com 50.000 iterações propostas, `early_stop=False` e sem reordenação; recusa aliases como `auto`.
2. Reconfere os Git blobs dos fontes originais `pcc.py`, `pcc_numpy.py`, `pcc_numba.py`, `pcc_graph.py` e os caminhos físicos dos módulos importados.
3. Instancia apenas um `ParticleCompetitionAndCooperation(impl=...,n_jobs=1,update_mode="sequential")` ainda **sem grafo e sem partículas**; chama apenas `_get_backend_fn()` para selecionar o ponteiro de função, sem invocá-lo.
4. Exige **identidade de objeto** com a função original exata de NumPy, Numba ou Cython. No Cython, recusa `_HAS_CYTHON=False`, mesmo que haja fallback Numba; valida arquivo nativo original e seu SHA256 binário `cec70629e10b1f6e9f53274d571a52ad2177d31b89d2c189840f3abda2944a8a` e o módulo original do observador Cython `1ea9f1c151df795071b3fe449df2a3e35ab72bb79156b073df9f8297c66d899f`.
5. Rejeita discrepância de `impl`, modo de atualização, `n_jobs`, identidade, origem, hash, classe customizada, caso fora da lista ou autorização científica falsa. Não grava qualquer estado científico.

## Prova de falha e prova positiva na MESMA OCI arquivada

Reutiliza apenas a imagem original exportada, já verificada por hash, com SHA256 `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8` e config ID `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`. Dois contêineres independentes são executados offline, não privilegiados, rootfs readonly, arquivo da árvore do repositório montado readonly, com quotas sintéticas pré-existentes:

- `PYTHONPATH=/workspace`: um caso P1 Cython deve ser **explicitamente bloqueado pelo novo guardião** com `BACKEND_IDENTITY_MISMATCH_OR_SILENT_FALLBACK`. Qualquer motivo de erro diferente é falha do CI, e o guardião nunca chama Numba no lugar dele.
- `PYTHONPATH=/workspace:/opt/p2-native:/opt/p2-source-helpers`: percorre os 72 IDs P1, validando **24 seleções exatas para cada backend**, além dos binários de produção e observador. O teste obtém ponteiros de função, mas não os chama.

Testes adversariais source-only precedem o processo OCI em quatro combinações Linux/Windows Python 3.10/3.12, seguidos de regressão independente. O único artefato exportado contém agregados e hashes públicos conhecidos. Sem acesso à UCI, sem `build_graph`, `set_graph`, `fit_predict`, RNG, Numba JIT, propagação Cython, uso de checkpoints ou resultados científicos.

## Limites e condição futura de integração

O guardião é um **pré-requisito de integridade do executor futuro**, não um executor nem uma autorização. Ele não substitui controle final de versões/ABI, fechamento de ferramentas de build e SDK, disponibilidade da imagem em backup durável, recursos máximos do fit completo, nem o manifesto P2-08 com permissão separada. Um futuro worker não pode usar apenas o nome do backend; deverá chamar o guardião antes de executar e falhar imediatamente em qualquer discrepância.

**NO-GO obrigatórios:** `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`.

Nenhuma aprovação de política P2-07C, merge ou execução científica é solicitada ou inferida por este estudo. Concluir a etapa exige sucesso do CI técnico e da regressão independente de produção, sem novos dados científicos.
