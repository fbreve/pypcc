# PyPCC P2-01A — identidade do código-fonte e metadados de ambiente (12/11/2026)

**Resultado permitido neste estágio:** comprovar a identidade dos objetos Git que compõem os três observadores, sem importar NumPy/Numba/Cython, compilar extensões, ler datasets, construir grafos, gerar máscaras ou executar PCC. **Não equivale a aprovar o gate completo P2-01, nem o P2 de execução.**

Este ramo técnico se origina exatamente de [PR #258](https://github.com/fbreve/pypcc/pull/258), cujo protocolo P1 de estudo **exploratório** está cientificamente congelado pelo SHA-256 **`f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`**. O (SHA) de P1 não é autorização para executar fits.

## Fontes conferidas

| Implementação | PR experimental, commit | Objeto Git do observador | Dependências nativas |
| --- | --- | --- | --- |
| **NumPy** | [#253](https://github.com/fbreve/pypcc/pull/253), `6d908e78878c4922674277e3313dbfe7f7924146` | `experiments/live_numpy_milestones.py` blob `d642bd0f329eceb4f3dc3111b5f8c91f0e085cac` | Usa `_pcc_step_numpy_sequential_unchecked` de produção; sem módulo compilado específico para o observador |
| **Numba** | [#255](https://github.com/fbreve/pypcc/pull/255), `4c8e6df4e01d7a4dfc4efc31eb73db04d6e212e5` | `experiments/live_numba_milestones.py` blob `e10942e96be623d47d5537d45bd365f366bba548` | JIT `@njit` e dispatcher de `pcc_numba.py`; signatures, caches e LLVM nativo **ainda não verificados em P2** |
| **Cython** | [#256](https://github.com/fbreve/pypcc/pull/256), `af34f50ebabb0a7204d3f30679adaf3f00169742` | `experiments/live_cython_milestones.py` blob `281daf0e66a567e587d0bd3e21dc34329a234b95`; gerador `experiments/build_live_cython.py` blob `d30e9b85419b8f31cd5f677f848357fb121d316f` | Módulo nativo **isolado** `pcc_step_live_research`, não o `pcc_step` normal; **não compilado nem identificado no P2 atual** |
| **Inventário anterior P0** | [#257](https://github.com/fbreve/pypcc/pull/257), `fa59ef20c235cf03075bdcc09d1d182bd5ca06fd` | `experiments/preflight_three_backend_sources.py` blob `31f7ab76ac68d1a688a60e7f64c0d6045b38866c` | Auditoria somente de código; **não contém build/ABI de execução** |

Todos os observadores descendem do mesmo `master` `ea69f690208620d5563c8efece6b2e04369f52c0`. A comparação de árvores Git mostrou os **mesmos dez blobs de produção** em cada um desses commits, inclusive `pcc_step.pyx` `021c9a2c34f912e0d8382f05d0e21b9713ffd5fc`, `pcc_graph.py` `3e8943462555b4feafbd162125b51e41f29d7ab7`, `pcc_numpy.py` `edf09a4ac5487327d8bd2b63415b5dd6564209f5` e `pcc_numba.py` `86d21d57ef535bbdb8566f10ace38cb54f93f741`. Nenhuma alteração de produção é autorizada ou realizada.

A [lista exata de blobs e dependências](current_p2_01_source_identity_2026-11-12.json) está registrada em JSON. O verificador [`p2_01_source_platform_preflight.py`](../experiments/p2_01_source_platform_preflight.py) consulta, **sem checkout alternativo ou compilação**, os commits e hashes originais usando `git rev-parse`, verifica ancestralidade e `git diff`, além de `git ls-remote` para rejeitar uma branch remota cujo head tenha se alterado. Lê somente metadados do Python, do OS, dos compiladores **encontráveis** e das distribuições disponíveis. Não importa o pacote: ter `numpy` ou `Cython` instalado no runner não prova o binário nativo aprovado.

## O que os arquivos de build declaram — e o que não declaram

O `pyproject.toml` mestre declara `setuptools>=61`, `wheel`, `Cython>=3.2,<4` e `numpy>=2,<3` como dependências de build. O `setup.py` declara Python `>=3.10`, `numpy>=2,<3`, `scipy`, `scikit-learn`, Numba como extra opcional, e flags de compilação `/O2` para MSVC e `-O3` para compiladores Unix/MinGW. **Não há lockfile de versões efetivamente resolvidas neste inventário.** A versão de pacote mestre indicada pelo arquivo `VERSION` é `1.3.0`; isso também não identifica um wheel compilado.

**P2-01 continua parcialmente aberto:** faltam fixar as versões reais completas da stack; versão de compilador, ABI, arquitetura, flags efetivas, hashes do wheel e do `pcc_step_live_research`, assinaturas nativas Numba e confirmar disponibilidade dos três observadores num mesmo ambiente de execução. Os testes de paridade **sintéticos** já realizados nos PRs de pesquisa são evidência histórica, mas não substituem a verificação desses artefatos binários na futura máquina P2. Em especial, nunca presumir que selecionar `impl="cython"` garante Cython sem fallback: a implementação de produção possui resolução alternativa e uma execução científica deve falhar caso escolha um backend diferente.

## Resultado esperado e próximo gate

A matriz CI executa Linux/Windows × Python 3.10/3.12, sem pacotes externos instalados deliberadamente, para produzir quatro arquivos de **identidade de fonte e ambiente**, não um selo executável. Espera-se `SOURCE_GIT_PINS_PASS_RUNTIME_NATIVE_P2_NOT_READY`, `P2_01_entire_gate_passed=false`, `P2_preflight_sha256=null`, `P2_execution_authorized=false` e **zero fits**.

Próximo: completar **P2-01B**, em ambiente técnico reproduzível, com builds independentes e hashes dos observadores/Cython/Numba, sem dados reais. Depois seguir para a importação e integridade das fontes, transformação, grafos e máscaras conforme P2-02 a P2-08. Cada etapa deve produzir evidência nova e distinta, não reutilizar hashes científicos P1 como autorização de execução.

**Estado:** ramo experimental, sem merge, `master` intacto. Este relatório não pretende que todo o P2-01 tenha sido concluído.
