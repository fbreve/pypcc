# PyPCC — P2-07N: separar o RSS medido do grafo e o orçamento estático dos 50.000 passos (14/11/2026)

**PR científico em DRAFT, empilhado sobre P2-07M #287; sem merge ou alterações em produção.** O objetivo é evitar interpretação incorreta do pico de memória observado e transformar duas fontes de evidência já existentes em uma decisão operacional verificável.

## Evidências verificadas, sem repetições

- **P2-07A — workflow #43478839756**, artefato Linux original `14064456453`: contabilidade teórica das matrizes permanentes dos três grafos, buffers, estados dos nós/partículas, três observadores, seus oito checkpoints por ajuste e 72 posições no protocolo P1. Maior carga hipotética de arrays co-residentes: **650.012 bytes**, não um limite de RSS. Checkpoints planejados para os 72 casos: **14.079.360 bytes**. A contabilidade reutiliza diretamente as fórmulas originais, sem ajustá-las ao resultado de medição.
- **P2-07M — workflow #43664754999**, artefato original `14142136186`: build_graph dos **três grafos UCI reais dentro da OCI original com bytes verificados**; pico de RSS por processo de 189.612.032 bytes (UCI 212), 189.870.080 (UCI 850) e 188.956.672 (UCI 292). Hash canônico do relatório `0cb2b85ec100c558837042bc0b147f13597e6182a6453bab60a540f8fc0c9b97`. **Nenhuma propagação ou fit** nessa etapa.
- O P1 conserva SHA `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. A OCI conserva gzip SHA `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8` e config ID `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`.

O P2-07N baixa **somente os dois pequenos JSONs originais**. Não importa a OCI, não consulta a UCI, não cria grafos, não chama qualquer backend, não usa máscaras nem consome números aleatórios, não executa Numba/JIT e não gera checkpoints reais. Confere a concordância entre P2-07A e P2-07M, as três identidades UCI e hashes exatos de grafos, as fórmulas e cada um dos três observadores, a dimensão e o padding de `neib_list`. Testes adversariais garantem que nenhum gate pode ser declarado satisfeito.

## Interpretação científica

A diferença entre o limite **proposto, ainda não aprovado, de 2 GiB por ajuste** e o pico de RSS do processo no final de `build_graph` indica somente **espaço aritmético não utilizado no ponto medido**. Ela **não é a margem garantida de um fit PCC**. Também **não se deve somar** esse pico de RSS à projeção de arrays permanentes/checkpoints e chamar o resultado de limite de RSS: o pico já inclui algumas estruturas e as fases temporárias/JIT/compilação ainda são desconhecidas.

Um `PASS` demonstra apenas consistência entre as duas evidências arquivadas e ausência de regressão no contrato de recurso. **Não** demonstra custo de CPU/wall, pico de memória após criar partículas, uso transiente do Numba JIT, Cython, restrições de espaço de saída ou os 50.000 passos reais.

## Decisões pendentes

1. Evidência defensável de RSS/cgroup **do ajuste real completo** por backend, dataset e observador.
2. Evidência de CPU/wall real de 50.000 iterações e definição de política de interrupção.
3. Fechamento das dependências nativas, compilação JIT e bibliotecas carregadas sob o processo científico.
4. Viabilidade de scratch/checkpoints/logs, além das próprias matrizes.
5. Seleção final de host/OCI com fechamento de fontes, SDK, compilador e dependências de sistema.
6. Aprovação expressa da política operacional e do **manifesto técnico P2-08 final com hash distinto**, após os gates.

A execução dos 72 ajustes científicos segue condicionada à autorização específica da versão técnica P2-08. Um pedido genérico de continuidade não altera esses limites. Não realizar pilotos não autorizados, mesmo de apenas um dos 72 casos.

**Estado exigido:** `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`; 72 casos `NOT_RUN_NO_APPROVAL`.

Ao concluir o P2-07N, a próxima ação útil deve ser uma decisão explícita do mantenedor sobre qual ambiente será a plataforma definitiva e quais verificações autorizadas podem medir um ajuste completo. Não há benefício científico em criar um novo lote de ensaios artificiais ou repetir a mesma contabilidade.
