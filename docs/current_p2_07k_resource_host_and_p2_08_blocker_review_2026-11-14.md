# PyPCC — P2-07K: gate de recursos e ambiente, baseado em evidência original (14/11/2026)

**DRAFT de pesquisa empilhado sobre P2-07J #284. Não altera a produção, não faz merge no master, não libera execução científica.** O objetivo é transformar os resultados verificados do supervisor artificial em um registro explícito daquilo que está comprovado e daquilo que **ainda não pode ser aprovado**. Não há downloads UCI, geração de grafos/máscaras/sementes, propagação PCC sintética ou real, nem execução de Docker nesta etapa.

## Evidências usadas e distinções científicas

- **P2-07J:** CI técnico original [#43550965075](https://github.com/fbreve/pypcc/actions/runs/43550965075), **5/5 PASS**. Fonte arquivada no artefato GitHub Actions **14095798755** (expiração prevista para 14/12/2026). O relatório canônico original tem SHA-256 **1d1d9302d4a21d204f2c13ba6db971b4af660d7a4fe6a24083975e8b141b651f**, e o histórico administrativo completo (504 registros) tem SHA-256 **0da808928330769ea234f5cc0a2a4db4e0492e29ae2300b3a34965c94a475c01**.
- A nova auditoria confere **todos os cinco arquivos de histórico H**, não apenas os hashes do relatório, e compara exatamente o conteúdo do histórico administrativo com os sete cenários já aprovados tecnicamente, incluindo duas interrupções administrativas simuladas e cinco processos artificiais reais.
- A [regressão de produção #43550965006](https://github.com/fbreve/pypcc/actions/runs/43550965006) passou **4/4** em Linux/Windows e Python 3.10/3.12, mas não valida os recursos do PCC real. O identificador da regressão é preservado como proveniência, sem reivindicar que a etapa de auditoria executou novamente os kernels.
- O P1 segue imutável, SHA-256 **f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166**, com 72 casos na ordem original e 50.000 iterações por ajuste.
- O P2-07C original permanece uma **proposta**, SHA-256 **b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04**. Nem o CI nem esta documentação aprovam limites originalmente propostos.
- A OCI reaproveitada mantém gzip SHA-256 **008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8** e config ID **sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55**. O P2-07K não baixa seus ~291 MB nem a recompila.

## Limites propostos — situação real de aprovação

| Parâmetro | Proposta P2-07C | Situação |
| --- | --- | --- |
| Host | RAM física ≥8 GiB e espaço livre ≥12 GiB | Host final ainda não escolhido/aprovado |
| Simultaneidade | Um ajuste por vez | Ordem administrativa validada, nenhum ajuste científico executado |
| Memória de processo por ajuste | Máximo 2 GiB RSS/working set | cgroup de processo artificial testado; carga PCC real não medida |
| Tempo de ajuste | 1.800 s wall e 3.600 CPU-s | Supervisor artificial testado, limites reais de custo desconhecidos |
| Estudo completo | 172.800 s (48 horas) | Cálculo hipotético: 129.600 s de slots wall + 43.200 s de folga; não é prazo de execução observado |
| Espaço temporário por ajuste | Máximo 1 GiB | Falha ENOSPC artificial sob tmpfs 1 MiB testada; não é limite demonstrado para um PCC real |
| Checkpoints em memória | Máximo 64 MiB | Estimativa estática de arrays, não pico total RSS |
| Arquivos JSON / log por caso | Máximo 1 MiB / 2 MiB | Aritmética: máximo de 216 MiB para os 72 casos |
| Evidência total persistida | Máximo 1 GiB | Política de exportação ainda não aprovada |

Os testes com processos artificiais **demonstram funcionamento de mecanismos do SO**, não viabilidade de um ajuste de 50.000 iterações nos três grafos UCI e nos três backends. Em particular, o watchdog de CPU/wall é por amostragem. Nossos resultados não constituem limite superior empírico de memória, tempo, disco ou custos de especialização JIT para os dados reais.

## Seis pendências explícitas

1. Selecionar um host final e registrar fechamento reprodutível/imutável das dependências de SO, compilador, SDK, módulos nativos, runtime e bibliotecas carregadas.
2. Obter evidência defensável de viabilidade de recursos para 50.000 iterações nos **grafos originais**, sem extrapolar diretamente de casos artificiais.
3. Revisar e aprovar (ou ajustar com novo hash) os limites e a política de exportação do P2-07C.
4. Verificar a integração do supervisor com um executor científico original sob política de recursos aprovada; isto não é realizado neste PR.
5. Assegurar retenção durável independente da OCI e dos registros de evidência. Existe uma cópia temporária estendida do OCI até **11/02/2027**, mas isso **não prova** backup NAS/local.
6. Só após os gates anteriores, produzir manifesto **final** P2-08 com SHA distinto e solicitar autorização expressa do mantenedor para esse SHA exato.

A política científica mantém **todos os 72 casos em NOT_RUN_NO_APPROVAL**, sem repetições, substituições, mudanças na lista densa de vizinhos, alterações de máscaras ou mudança dos oito marcos. Um simples pedido genérico de continuidade **não** autoriza os experimentos.

## Testes e artefatos previstos

O P2-07K adiciona doze testes adversariais somente de metadados, em quatro ambientes (Linux/Windows, Python 3.10/3.12), seguidos por uma auditoria independente dos arquivos completos originais J preservados. O relatório final contém o conjunto de IDs, a aritmética das propostas, as evidências originais por hash e as pendências, mas nenhum arquivo de dados UCI ou estado por nó. O hash desse relatório é de **revisão, não de preflight executável**.

**NO-GO obrigatório:** P2_01_final_execution_toolchain_lock_pass=false; P2_07_full_resources_gate_pass=false; P2_execution_preflight_sha256=null; P2_execution_authorized=false; real_PCC_fits_executed=0. O PR deve permanecer DRAFT e empilhado.
