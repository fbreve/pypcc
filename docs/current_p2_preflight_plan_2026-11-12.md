# P2 — preflight técnico separado, ainda NÃO executado (12/11/2026)

O P1 já foi aprovado **cientificamente** para um estudo **exploratório**, com SHA-256 de protocolo `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. O [manifesto científico](current_p1_scientific_sha_manifest_2026-11-12.json) preserva a aprovação explícita S1–S6 e as limitações. **Essa aprovação não autoriza sequer o primeiro ajuste real.**

Este documento inicia a **especificação do P2**, sem ainda executar preflight nem produzir um SHA técnico. O [plano estruturado](current_p2_preflight_plan_2026-11-12.json) é separado do payload científico P1.

## Critérios de passagem — oito gates

| Gate | Evidência obrigatória antes de qualquer ajuste |
| --- | --- |
| **P2-01 Fontes e versões** | SHA exato de fonte para NumPy/Numba/Cython e observadores isolados #253/#255/#256, além do preflight #257; Python, NumPy, SciPy, scikit-learn, Numba, Cython, compilador, binários, sistema e ABI. Falha de backend não autoriza fallback silencioso. |
| **P2-02 Arquivos oficiais** | ZIP e membro UCI 212/850/292 conferidos com hashes científicos; versão três classes de Vertebral, planilha exata Raisin, Channel derivado no Wholesale, linhas/colunas/rótulos originais. |
| **P2-03 Transformações X** | `float64`, z-score `ddof=0` e `log1p` apenas no Wholesale; verificar finitude, variância, forma, ordem de registros e SHA de X bruto e transformado. Sem remoção de linhas. |
| **P2-04 Grafo** | `k=10`, Minkowski `p=2`, simétrico, não direcionado; hash de `neib_qt` e da matriz **densa de `neib_list` com padding `-1`**, vizinhos ativos sem self-loop, ordenação e identidade entre backends. O CSR temporário do construtor não substitui a representação do PCC. |
| **P2-05 Máscaras e seeds** | 24 máscaras únicas, 3 rótulos por classe, ordenação original de observações, PCG64 independente da propagação, 72 sementes nativas não nulas, SHA de cada máscara e emparelhamento nos três backends. **Sem propagação de partículas.** |
| **P2-06 Observadores** | Build nativo isolado, paridade de checkpoints e estados com os testes sintéticos, validação da implementação realmente selecionada, sem fallback. Os PRs de pesquisa permanecem separados da produção. |
| **P2-07 Orçamento** | Medir/estimar memória de grafo, partículas, dominância, checkpoints, buffers Python/nativos, disco e CPU; teto do desenho para snapshots **64 MiB por ajuste**, mas sem confundir isso com RAM total. Verificar capacidade dos 72 casos antes da execução. |
| **P2-08 Selo e autorização** | Relatório técnico imutável, SHA **P2 separado** e aprovação explícita do mantenedor **para esse hash exato** antes do primeiro ajuste real; nenhum dado individual de pacientes/clientes é publicado no CI. |

**Situação atual:** oito gates **não executados**. O plano é um compromisso com verificações futuras, **não um relatório de preflight aprovado**. Ainda não existem binários P2 verificados, máscaras e grafos derivados das três fontes, orçamento global validado ou SHA P2.

A próxima implementação deverá ser isolada em um branch/PR técnico **NO-FIT**, mantendo verificações de bytes, versões, recursos e reprodutibilidade independentemente do resultado do PCC. A primeira propagação real só ocorrerá depois da aprovação explícita do SHA P2. Nenhum merge experimental ao `master` é necessário para preparar esse preflight.
