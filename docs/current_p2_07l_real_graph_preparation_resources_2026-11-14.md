# PyPCC — P2-07L: medição REAL de recursos da preparação dos três grafos, ZERO ajustes PCC

**Pesquisa em rascunho, empilhada sobre #285; sem merge nem mudança dos algoritmos originais.** Esta etapa é propositalmente diferente dos vários ensaios com grafos artificiais: usa as três fontes UCI oficiais imutáveis do P1 para construir os três grafos reais sob a rota exata aprovada **somente até build_graph**. Mede pico de memória física do processo (VmHWM), RSS no fim, tempo de CPU e wall da construção em três subprocessos Linux independentes. Nenhuma chamada a fit_predict, nenhum movimento de partículas, nenhuma máscara nem semente científica criada, nenhum dos 72 ajustes reais executado.

## Procedimento

1. Conferir SHA-256 original do P1 e dos três ZIPs/membros UCI (212, 850, 292) e usar os mesmos fontes e parsers congelados de #267, com versões numéricas candidatas fixas.
2. Validar os hashes das matrizes X originais e transformadas, sem exportar qualquer linha. Instanciar modelo original NumPy e chamar somente build_graph(Z, k_nn=10, n_jobs=1).
3. Exigir grafos não direcionados com neib_list densa int64 com padding -1 e neib_qt original. Reproduzir os três SHA-256 completos já validados em #267: UCI 212 `434ec3bb811222964304ae163afa1bf3eb25f767139107b5bb0f05dc67db27f2`, 850 `b0e260ae198037e89e5e8b0691da15b4c7bfc4cf17e41adb1fe8209a7a975384`, 292 `63e502a5d3f981a20c8c1a80884aa740a25fa7ea7f13335ad7554ad6e7824ca6`.
4. Um processo novo por dataset, sem repetições/supressão de falhas. Capturar VmHWM/VmRSS do Linux e tempos CPU/wall só para construção do grafo (e wall total da aquisição/preparação). Os scripts não imprimem nem salvam linhas originais, índices rotulados, vizinhos, máscaras ou atributos por amostra.
5. Emitir somente dados agregados e seu hash; manter explicitamente todos os gates científicos bloqueados.

## Interpretação obrigatória

A memória medida é **pico do processo inteiro durante a preparação**, incluindo importações e bibliotecas Python. Não é uma medição isolada da alocação do grafo e **não é o pico de um fit de 50.000 iterações**. Tempo de construção do grafo também não é tempo de propagação. Os três subprocessos usam o ambiente Linux/Python 3.12 candidato do GitHub Actions com pacotes exatos, **não a imagem OCI final imutavelmente aprovada**.

A comparação de VmHWM com a proposta de 2 GiB é **apenas um sinal de prontidão para preparação**. Um resultado abaixo de 2 GiB não autoriza concluir que o ajuste PCC, especialização Numba, estruturas Cython ou checkpoints caibam nesse limite. Mesmo com a medição concluída, P2-07 continua bloqueado até obter viabilidade defensável dos ajustes reais completos, host fechado, aprovação dos limites e SHA de autorização P2-08 distinto.

CI: doze testes adversariais source-only em cada uma de quatro combinações de Linux/Windows com Python 3.10/3.12; um job adicional Linux constrói apenas os três grafos reais sem chamar qualquer propagação. Os resultados não devem ser declarados aprovados antes da conferência dos jobs e do artefato exato.

**Invariantes:** P1 SHA f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166; 72 casos NOT_RUN_NO_APPROVAL; P2_01_final_execution_toolchain_lock_pass=false; P2_07_full_resources_gate_pass=false; P2_execution_preflight_sha256=null; P2_execution_authorized=false; real_PCC_fits_executed=0.
