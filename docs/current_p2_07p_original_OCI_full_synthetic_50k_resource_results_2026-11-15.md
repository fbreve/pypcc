# PyPCC — P2-07P: 50.000 iterações completas dos três backends originais na mesma OCI, com dados inteiramente artificiais (15/11/2026)

**PR DRAFT de pesquisa, sobre P2-01M #292. Nenhuma alteração em `pcc.py`, `pcc_graph.py`, propagadores, observadores, imagem OCI, protocolo P1 ou `master`.** Os 72 experimentos UCI originais permanecem **`NOT_RUN_NO_APPROVAL`**.

## Lacuna operacional distinta dos experimentos anteriores

- P2-07D já validou `50.000` iterações e observadores do PCC em dados artificiais com dependências/bibliotecas reconstruídas para Linux e Windows, mas **não nos mesmos bytes da imagem OCI única cuja transferência P2-01I/J foi validada**.
- P2-07M mediu a construção dos **três grafos originais UCI**, sem sequer invocar propagação ou gerar uma máscara.
- P2-07O mediu a compilação **apenas** dos tipos Numba no mesmo ambiente OCI, sem chamar os kernels.
- P2-01K/L corrigiram/verificaram `PYTHONPATH` e rejeição de fallback Cython→Numba, mas **não executaram o corpo das funções originais**.
- P2-01M validou o procedimento de arquivamento da imagem no Windows, sem executar PCC.

**P2-07P integra pela primeira vez uma construção real de grafo `build_graph` seguida de `fit_predict` original completo, nos três backends nativos da exata imagem OCI histórica**, em um ensaio estritamente artificial. O resultado é uma medição de viabilidade técnica dos três entrypoints e de uso efetivo de CPU, memória de processo, pico do cgroup e armazenamento temporário num grafo **sintético**, não a validação do limite superior das cargas científicas UCI.

## Dados artificiais e preservação científica

A fonte é **somente** `synthetic_features_and_labels(900,7,2)` definida e validada no P2-07D. Produz 900 linhas fictícias determinísticas, 7 atributos fictícios, 2 classes fictícias, exatamente 3 rótulos sintéticos por classe, todos os demais `-1`. O seed **71** é o fixture sintético anterior, não a fórmula de nenhuma das 72 sementes P1. Não há download de dados UCI, leitura de ARFF/XLSX/CSV oficiais, aplicação de máscaras científicas nem consulta aos rótulos verdadeiros.

Cada processo isolado reconstrói o **mesmo grafo simétrico de união kNN (`k=10`) por `ParticleCompetitionAndCooperation.build_graph()` original**, com `neib_list` densa `int64`, contagem `neib_qt` e padding `-1` — a representação intencional por desempenho não é substituída por CSR. Todos os arcos ativos são conferidos por simetria, não duplicidade e ausência de loops. O grafo artificial deve ter o mesmo hash/forma nos três processos, mas seu hash/grau pode (e normalmente deve) **diferir** dos grafos reais P1.

`fit_predict()` recebe `impl` exato `numpy`, `numba` ou `cython`, parâmetros `p_grd=0.5`, `delta_v=0.1`, `deltap=1.0`, `dexp=2.0`, `max_iter=50000`, `early_stop=False`, `es_chk=2000`, `seed=71`. O novo guardião P2-01L verifica identidade da função retornada, inclusive Cython verdadeiro via `/opt/p2-native`, antes de construir o grafo. Um processo fresco por backend, **uma execução sintética completa por backend** (três ao todo), sem reexecutar casos para escolher o melhor.

**Esta etapa NÃO executa observadores de milestones ou checkpoints**. Portanto, seus consumos de memória/CPU/disco não incluem os custos dos observadores P2-06 necessários para o experimento original. Ela não produz o resultado científico principal de 5.000→50.000, não avalia RNG/checkpoints, não informa acurácia e não deve ser usada para justificar o envelope integral do estudo.

## Ambiente e guardrails (candidatos, NÃO aprovados)

- Exatamente a imagem Docker `p2-01h-exact-runtime-image.tar.gz`, gzip SHA-256 `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`, config ID `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`, transferida sem rebuild, pull ou reempacotamento.
- Linux amd64, CPython **3.12.16** no contêiner original, Cython original binário de produção **`cec70629e10b1f6e9f53274d571a52ad2177d31b89d2c189840f3abda2944a8a`** e observador binário isolado **`1ea9f1c151df795071b3fe449df2a3e35ab72bb79156b073df9f8297c66d899f`** somente para conferir o hash e import.
- `PYTHONPATH=/workspace:/opt/p2-native:/opt/p2-source-helpers`, rede desligada, rootfs readonly, UID/GID `65534`, `--cap-drop ALL`, memória Docker/cgroup v2 limitada a **2 GiB**, `--cpus 1`, `/tmp` descartável com **128 MiB**, sem permissões elevadas. A memória de 2 GiB é de uma proposta P2-07C ainda **não aprovada**.
- Limite adicional de segurança **300 segundos wall por processo sintético** no supervisor pai, seguido de `docker rm -f`, se estourar. Limites sintéticos **não validam** custo do ajuste científico real, nem devem ser chamados de timeouts aprovados para P1.
- Amostras observadas em cada backend: CPU e wall da construção de grafo, CPU e wall da chamada completa `fit_predict()`, pico de RSS do processo (`VmHWM`), `memory.peak` do cgroup v2, bytes de arquivos temporários presentes ao fim. Não converter em um teto superior da execução original com observador.
- Saída externa contém apenas agregados de dados **artificiais**, hash do grafo fictício, identidade dos binários e flags obrigatórios de `NO-GO`. Não salva `X` fictício, vizinhos, rótulos, dominâncias, estado das partículas, predições por nó, cópias da OCI ou arquivos originais UCI.

## CI e critérios de avaliação

O CI executa **11 testes adversariais de contrato source-only × quatro ambientes** Linux/Windows Python 3.10/3.12 antes da execução sintética da OCI original. Não se faz replay de imagens reconstruídas, e o worker OCI falha se algum dos backends não for o exato original, se o grafo for dirigido, se o arquivo Cython ou o P1 mudarem, se a execução ultrapassar o teto de proteção ou se surgir qualquer afirmação falsa de autorização científica.

Mesmo com **3/3 fits sintéticos PASS**, isso **não prova** que o limite de 2 GiB, 30 minutos/ajuste ou disco seja seguro para **dados reais**. Os grafos e posições rotuladas fictícios não reproduzem a topologia, dinâmica ou número real de transições dos dados UCI. O estudo original exige ainda observadores de checkpoints e monitor de recursos integrado em um host aprovado.

Estado obrigatório sempre: `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`.

**Próxima decisão substantiva depois do CI:** selecionar o host definitivo, aprovar explicitamente a política de recursos e avaliar um piloto **separadamente autorizado** na topologia real com checkpoints; somente esse ensaio pode ajudar a validar recursos para o estudo original. Nenhum `prossiga` genérico é interpretado como aprovação do piloto real P1.
