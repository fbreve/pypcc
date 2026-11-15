# PyPCC — P2-07O: compilação Numba dentro da OCI ORIGINAL, sem executar PCC (14/11/2026)

**PR DRAFT sobre P2-07N #288, sem merge no master nem mudanças de produção.**

## Lacuna investigada

O #287 comprovou as três construções de grafos originais dentro da imagem OCI original e o #288 confrontou esses picos de RSS com o modelo estático de arrays. Nenhum deles mediu o consumo da **compilação JIT Numba** nessa mesma OCI. O #273 mediu ajustes completos em **dados sintéticos**, não a compilação isolada na imagem OCI binária imutável.

O P2-07O importa a implementação original `pcc_numba.py` (Git blob `86d21d57ef535bbdb8566f10ace38cb54f93f741`) e, usando apenas **objetos de tipo Numba**, chama `_pcc_propagate_numba_sequential.compile(signature)` e `_seed_numba_rng.compile((int64,))`. O kernel interno `_pcc_step_numba_sequential` é compilado como dependência. A especialização é para os 27 argumentos originais, os layouts C/F e dtypes reais do caminho Numba de P1, com `c=2/3` não literal (variável escalar `int64`). Não há nenhum vetor com valores científicos nem execução ou chamada de um `CPUDispatcher`.

## Isolamento e resultado esperado

O worker roda na imagem arquivada P2-01I/J, gzip SHA256 `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`, ID `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`. O host revalida os bytes e importa por `docker load`, sem reconstruir nem acessar registry. O contêiner usa raiz somente leitura, rede desligada, usuário não privilegiado, capabilities removidas, CPU de 1 núcleo, limite proposto de 2 GiB RAM e `/tmp` temporário de 128 MiB com cache JIT. O script fica montado como fonte readonly, sem dados UCI.

A medição registra duração de CPU/wall **da compilação de assinaturas**, pico de RSS total do processo e inventário por hashes dos arquivos ELF/DSO mapeados antes e depois do JIT. Não arquiva códigos objeto gerados, caches, partes da OCI, matrizes, máscaras, logs brutos nem caminhos de usuário. Apenas um relatório JSON de agregados é preservado. A imagem original e o `master` não mudam.

O teste adversarial de contrato (11 casos por plataforma) precede a execução, cobrindo Linux/Windows Python 3.10/3.12. Apenas o job Linux executa a **compilação Numba**, sem propagação.

## Limitações científicas

- O tempo e a memória obtidos nesta etapa são **apenas do JIT em processo novo**. Não se mede o pico conjunto de JIT, preparação de dados/grafo, partículas, oito checkpoints, cópias ou propagação por 50.000 iterações. Não converter resultados em limites superiores de `fit_predict`.
- `compile(signature)` pode especializar os tipos originais, mas não prova que nenhuma outra especialização, biblioteca lazy, cache ou transiente aparecerá durante uma execução real.
- Há hashes de binários e bibliotecas nativas efetivamente mapeadas, mas não fechamento completo das fontes, compilador, cabeçalhos SDK, libc, bibliotecas latentes e host original de compilação.
- O teto de 2 GiB continua **proposto e não aprovado**, mesmo com `PASS`. Não é válido somar máximos de processos separados como se fossem uma medição de um ajuste completo.
- A política P2-07C e os 72 casos P1 permanecem congelados. Nenhuma aprovação P2-08 pode ser inferida de um SHA de revisão técnica.

**Gates inalterados:** `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`, 72 casos `NOT_RUN_NO_APPROVAL`.

A etapa será considerada tecnicamente concluída somente após os cinco jobs de CI e a regressão independente de produção terminarem com sucesso. Em caso de falha, analisar a causa sanitizada e preservar todos os bloqueios; não atribuir `PASS` ao escopo não testado.
