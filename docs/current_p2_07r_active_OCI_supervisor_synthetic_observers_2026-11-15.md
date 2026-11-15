# PyPCC — P2-07R: supervisor ativo do PCC artificial completo na OCI imutável (15/11/2026)

**Pesquisa experimental DRAFT baseada no P2-07Q #294. Não modifica nenhum kernel de produção, arquivo de dados original, código dos observadores, a imagem OCI ou o P1.**

## Lacuna concreta encontrada

O P2-07Q comprovou paridade dos três backends com oito observações em seis ajustes fictícios completos de 50.000 iterações, mediu tempo e pico de memória e isolou cada processo no cgroup de 2 GiB. Contudo, o supervisor pai usava `subprocess.run(timeout=300)` e **não monitorava nem interrompia o consumo acumulado de CPU**. A flag `--cpus=1` do Docker regula a *taxa* de CPU, mas não a quantidade *total* de CPU-s por ajuste. Isso não deve ser confundido com o limite candidato P2-07C de 3.600 CPU-s.

O P2-07H já havia testado um supervisor de contêiner com medidas de cgroup v2 e interrupções de CPU, wall, OOM, scratch e limpeza de processos, mas só em carga de processos artificiais sem PCC. O **P2-07R conecta a infraestrutura ativa ao trabalhador real de propagação artificial do P2-07Q**, preservando o código original dos três kernels, sem tocar na UCI.

## Execução estritamente artificial

O ensaio usa apenas `synthetic_features_and_labels(900,7,2)` do P2-07D, com 900 nós artificiais, 7 atributos artificiais, duas classes inventadas, seis rótulos fictícios, semente **71 artificial** e grafo `kNN=10` simetrizado pelo `build_graph` original, representado em `neib_list` **densa com padding `-1`**.

Há **cinco contêineres sequenciais**, todos usando a exata imagem OCI original P2-01J, SHA-256 `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`, config ID `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`:

1. `numpy`: **um ajuste artificial normal e outro observado** de 50.000 iterações com oito marcos.
2. `numba`: mesmo par original/observado.
3. `cython`: mesmo par original/observado, usando a biblioteca Cython nativa original e seu observador isolado congelado.
4. `cpu_fault`: carga **somente CPU artificial, sem PCC**, que deve receber `SAMPLED_CGROUP_CPU_BUDGET_KILL`.
5. `wall_fault`: pausa **somente wall artificial, sem PCC**, que deve receber `SAMPLED_WALL_BUDGET_KILL`.

O trabalhador de cada contêiner inicia bloqueado antes de qualquer `fit_predict` em uma **barreira dentro do tmpfs**. O supervisor pai só o libera após conferir a identidade Docker, o PID de origem do processo (PID + starttime de `/proc`), o `HostConfig` do Docker e os limites reais `memory.max`, `memory.swap.max`, `cpu.max`, `pids.max` e espaço `/tmp` através do cgroup v2. O serviço executa sem rede, como UID/GID 65534, rootfs readonly, com `--cap-drop ALL`, quota de 1 núcleo, pids 128 e 2 GiB de memória.

Após a liberação, o supervisor **amostra o consumo acumulado de CPU do cgroup** e o tempo wall monotônico. Exceder um limite de tempo exige `docker kill` do contêiner exato; o supervisor confirma término não zero, remove o contêiner e compara a identidade PID/starttime original para impedir falso sucesso após reutilização de PID. Ocorreu OOM real? O supervisor também recusa qualquer relatório como PASS. A limpeza é incondicional, mesmo em erro.

**Os limites deste ensaio não foram aprovados para o estudo científico:** cargas artificiais completas são limitadas a 120 CPU-s e 240 wall-s; a injeção CPU é interrompida a 0,7 CPU-s, e a wall a 1,0 wall-s. Há ainda um teto de segurança de 285 wall-s. Esses mecanismos são **watchdogs por amostragem**, não tetos rigorosos de kernel para tempo CPU/wall, embora a quota de memória/cgroup seja imposta pelo SO. A CPU observada inclui overhead de `docker exec` para leitura do cgroup.

## Integridade e correspondência

Cada caso completo reutiliza `q.worker(backend)` do P2-07Q e os cinco fontes históricos dos observadores e helpers, já validados byte a byte. O resultado normal deve comprovar os **oito checkpoints** `0,1,10,100,1000,5000,25000,50000`, paridade integral de arrays e predições por backend e RNG verificável em NumPy/Numba. O RNG original interno Cython permanece não exposto e **não recebe declaração falsa de paridade**.

O relatório técnico só registra o resultado do supervisor, tempos e picos agregados, e valores booleanos de paridade. Nenhum arquivo de dados UCI, máscara original, grafo, série de partículas, snapshot, resultado por nó ou imagem OCI é exportado para o artefato.

O workflow executa **12 testes adversariais só de fonte × Linux/Windows Python 3.10/3.12**, depois o ensaio OCI exato de cinco contêineres e regressão independente de produção. Este trabalho é o primeiro *ensaio de integração do watchdog ativo de CPU e wall com a propagação artificial real dos três backends originais*; não prova viabilidade do estudo científico nos grafos UCI reais.

## Limitações inalteradas e próximo passo

- A fonte UCI, as oito máscaras e sementes científicas e os **72 ajustes reais P1 não foram usados nem executados**.
- O host definitivo continua não selecionado, e a imagem OCI não tem fechamento reprodutível byte a byte de compilador/SDK/SO. Seu backup local/NAS externo ainda não foi confirmado.
- O limite real de CPU/wall/RSS/disco dos **ajustes científicos com checkpoints** não pode ser deduzido diretamente destes 900 nós artificiais.
- A política P2-07C (2 GiB RSS, 1.800 wall-s, 3.600 CPU-s por ajuste, 1 GiB scratch etc.) ainda precisa de revisão e autorização expressa do mantenedor.

**NO-GO científico permanente:** `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`.

O teste só pode declarar o **supervisor de execução artificial tecnicamente validado**, nunca aprovar um worker científico real ou criar um SHA de autorização P2-08. Manter o PR em DRAFT e não fazer merge em `master`.
