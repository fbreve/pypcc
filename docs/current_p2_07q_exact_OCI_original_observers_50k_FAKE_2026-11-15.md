# PyPCC — P2-07Q: oito checkpoints originais durante 50.000 iterações na OCI exata, seis ajustes artificiais (15/11/2026)

**PR experimental empilhado sobre P2-07P #293. A produção, os três backends originais, a imagem OCI e o protocolo P1 permanecem imutáveis.** A etapa usa dados inteiramente fictícios; os 72 ajustes científicos reais **NÃO** são executados.

## Lacuna que o P2-07Q resolve

- O P2-07D já mediu 50.000 iterações artificiais com observadores originais, mas em plataformas com dependências recompiladas. O P2-07P comprovou a execução `build_graph` + `fit_predict` completa na **imagem OCI binária imutável** que serve de candidata a ambiente final; entretanto, não invocou os observadores de marcos.
- O P2-07Q reúne as duas dimensões: na mesma OCI exata, execução normal e execução observada **em cada um dos três backends originais**, duas vezes por backend (seis ajustes completos, sem retries), com os oito checkpoints P1 `0, 1, 10, 100, 1.000, 5.000, 25.000, 50.000`.
- As três implementações científicas originais observáveis foram **recuperadas literalmente** das árvores históricas de #253/#255/#256, com os Git blobs inalterados. Os dois helpers históricos de fingerprint NumPy e Numba também foram recuperados sem modificação; nenhum observador foi reimplementado nem o código de produção instrumentado.

## Fontes originais byte a byte

| Fonte de pesquisa histórica | Git blob SHA-1 inalterado |
|---|---|
| `experiments/live_numpy_milestones.py` | `d642bd0f329eceb4f3dc3111b5f8c91f0e085cac` |
| `experiments/live_numba_milestones.py` | `e10942e96be623d47d5537d45bd365f366bba548` |
| `experiments/live_cython_milestones.py` | `281daf0e66a567e587d0bd3e21dc34329a234b95` |
| `experiments/bench_live_numpy_fullfit.py` | `cc35e3e3107b2f24b419a6d04efe35ebd8b2d4d5` |
| `experiments/bench_live_numba_fullfit.py` | `241a0500ac4ed33d8c348ac29bb74d62cb98bf4b` |

Não há modificações em `pcc.py`, `pcc_numba.py`, `pcc_numpy.py`, `pcc_step.pyx` ou `pcc_step_live_research.pyx`. O observador Cython **compilado anteriormente e congelado** é importado de `/opt/p2-native`, e seu binário é conferido por SHA256.

## Preparação e controles

Reutiliza o fixture estritamente artificial de P2-07D/P2-07P: **900 nós × 7 atributos fictícios**, duas classes fictícias, **seis rótulos inventados** (três por classe), e `seed=71` (exclusivo do ensaio artificial). Para cada backend:

1. `ParticleCompetitionAndCooperation.build_graph(X,k_nn=10,n_jobs=1)` original gera o mesmo grafo **simétrico**, de união kNN, com `neib_list` `int64` densa e padding `-1`. A simetria é conferida para todas as arestas; o SHA do grafo artificial deve ser idêntico entre os três backends e entre os dois braços de cada par.
2. Seleção de função do backend `numpy`, `numba` ou `cython` é validada por **identidade de objeto**, sem fallback, antes de instalar um wrapper **somente na instância experimental**. Esse wrapper original retorna `propagation_info` para conferir de fato **50.000 iterações**.
3. **Braço original:** `fit_predict` completo, `early_stop=False`, 50 mil iterações, sem checkpoints.
4. **Braço observado:** segundo `fit_predict` completo, mesmas condições e `seed`, utilizando a função de observação histórica que captura os oito marcos com teto **64 MiB de arrays**. Um único propagador original/observado é chamado por ajuste, sem executar os oito prefixos separadamente (o que reiniciaria o RNG).
5. Verifica correspondência byte a byte dos estados mutáveis finais das partículas/nós e predições; compara também estado NumPy ou próximo sorteio nativo Numba quando disponíveis. **O binário Cython original não exporta seu RNG final**: o observador registra o estado/ próximo xorshift nativo, mas **não se declara paridade de RNG original-versus-observado Cython**, somente paridade dos estados e predições.
6. Os snapshots permanecem somente em memória durante o processo e são conferidos como somente leitura, com dimensões e pontos de captura exatos. O `owndeg` pós-`fit_predict` é normalizado pela produção, portanto não se compara diretamente ao checkpoint do último passo anterior à normalização (evita falsa inconsistência).

O sistema é Linux amd64 com `--network none`, contêiner sem privilégios, rootfs readonly, memória `--memory=2g` e `memory.peak` de cgroup v2, `--cpus=1`, `/tmp` descartável de 128 MiB e limite sintético de 300 segundos por subprocesso **pareado**. São limites de segurança do teste, **não aprovados para os casos científicos reais**. Coletam-se CPU/tempo separados dos dois ajustes, pico RSS do processo pareado, pico do cgroup e uso final de `/tmp`.

Um relatório de agregados registra apenas tempos, memória, bytes de snapshots e aprovação de paridade. **Não arquiva dados UCI, matrizes `X`, grafos, rótulos, estado de partículas, checkpoints, predições por linha, hashes de estados mutáveis nem sementes P1**.

## Condições de aceitação e limites

Antes de iniciar os três processos OCI, o CI verifica os hashes dos cinco fontes históricos e do P1 com **12 testes adversariais de biblioteca padrão × quatro plataformas Linux/Windows Python 3.10/3.12**. O CI principal valida que foram seis ajustes **apenas artificiais**, dois por backend, com oito marcos cada, total de memória copiada dentro dos limites e paridade final por backend; uma regressão de produção independente deve passar.

O sucesso comprova a **funcionalidade integrada dos observadores e custos numa carga artificial** na OCI original, mas **não é um teto de CPU/wall/RSS/disco** para os grafos UCI reais ou máscaras P1. Isso ainda depende da escolha de host, reprodução/fechamento de toolchain e aprovação explícita da política de recursos e de um eventual piloto técnico com dados reais.

Estado científico obrigatório, inclusive após PASS: `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`. Nenhum merge para `master` nem execução dos 72 casos científicos.
