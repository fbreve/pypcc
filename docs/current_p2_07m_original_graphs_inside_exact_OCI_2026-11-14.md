# PyPCC — P2-07M: três grafos reais dentro da OCI original — nenhum PCC executado (14/11/2026)

**Pesquisa DRAFT, baseada no P2-07L #286.** Esta etapa compara a execução de `build_graph` dos três conjuntos UCI originais no substrato **exato, previamente arquivado** no P2-01I/J: gzip SHA256 `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`, configuração OCI `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`. Não reconstrói a imagem, baixa outra tag, publica contêiner, compila bibliotecas nem executa propagação.

## Procedimento preciso

No host efêmero Linux/Python 3.12.16, revalida os três ZIPs UCI oficiais e os arquivos selecionados pelo SHA256 do P1, com parsers originais e dependências P2 fixadas. Constrói **somente as matrizes X e Z** em float64, sem criar qualquer grafo no host desta etapa, sem máscara ou semente. Compara os mesmos SHA-256 de Z usados em #267 e #286. Grava temporariamente as três matrizes Z em arquivos `.npy` não exportáveis, juntamente com metadados de feature/forma/hash para conferência independente dentro da OCI.

Cada conjunto é executado separadamente, usando a OCI carregada offline de um artefato já existente e identificada pelo config SHA. O contêiner tem rede desabilitada, rootfs somente leitura, acesso às matrizes por bind mount readonly, usuário não privilegiado, capabilities removidas, limite de processos 128, memória **proposta** de 2 GiB e swap igual à memória (nenhum swap adicional), CPU quota 1 e tmpfs temporário de 32 MiB. O código de produção e o script de revisão são montados somente para leitura. Nenhum ajuste ou tarefa científica é incluído na imagem.

O worker importa a função original, chama `ParticleCompetitionAndCooperation(impl="numpy", n_jobs=1).build_graph(Z,k_nn=10,n_jobs=1)` exatamente uma vez por UCI, valida grafo simétrico `neib_list` denso int64, com padding -1, confirma identidade das três matrizes e os SHA-256 completos dos grafos aprovados em P2-04/05. O modelo não recebe máscaras, não cria partículas nem executa `fit_predict`. O resultado expõe apenas identificador UCI, hash de X e grafo, dimensões agregadas, pico VmHWM/VmRSS do processo e tempo de construção de grafo em CPU/wall.

A pasta temporária de matrizes é descartada antes de salvar o relatório. Apenas o pequeno JSON agregado é preservado. Falhas em hash, OCI, origem, cleanup ou orçamento são **NO-GO**, sem novas tentativas dentro da execução. O SHA do novo relatório é de evidência técnica, **não** SHA do manifesto P2-08.

## Escopo das observações

- O P2-07L #286 já comprovou em Linux CI a construção dos grafos originais com picos de processo de 268.296.192, 275.972.096 e 267.862.016 bytes para UCI 212, 850 e 292, respectivamente. Esses valores **não** são benchmarks reproduzíveis garantidos na OCI, pois o ambiente é diferente; o P2-07M coleta observações independentes na OCI.
- Mesmo que todos os grafos possam ser preparados abaixo de 2 GiB, isso **não** demonstra viabilidade de 50.000 iterações de PCC real, JIT do Numba, operadores Cython, estados das partículas, custo de snapshots, ou custo de um estudo com 72 casos.
- A imagem OCI é *byte-locked* e importada em hosts independentes, mas ainda **não** documenta o fechamento completo do compilador, SDK, base host, runtime do sistema ou procedência do rebuild; a decisão de ambiente definitivo segue pendente.
- O candidato P2-07C continua sendo apenas uma proposta de limites de tempo/RSS/CPU/disk e política de exportação, não uma autorização científica.

## Gates e validação

Quatorze testes unitários adversariais source-only são executados nas quatro combinações Linux/Windows e Python 3.10/3.12, seguidos de um job Linux de construção dos três grafos reais na OCI. A regressão de produção (separada) deverá passar. Nunca declarar validação antes do CI e da leitura dos artefatos.

Manter sempre `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false` e `real_PCC_fits_executed=0`. Nenhum dos 72 ajustes científicos foi autorizado nem executado aqui; manter PR DRAFT, sem merge no master.

**Próximo passo somente depois da prova técnica:** revisar fechamento de dependências no runtime selecionado e preparar critérios de aceitação dos limites operacionais para ajustes completos; não inferir GO a partir da construção de grafos.
