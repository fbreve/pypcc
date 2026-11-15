# PyPCC — P2-01M: teste integral do backup original Windows, OCI real, sem execução PCC (15/11/2026)

**Pesquisa operacional somente. PR DRAFT sobre P2-01L #291. Não altera a imagem OCI, o código de produção nem o protocolo P1.**

## Problema concreto

O P2-01J já produziu uma cópia temporária, com retenção verificada de 90 dias, de uma imagem OCI binária transportável. Há também o script PowerShell `scripts/p2_01j_archive_original_image_locally.ps1` para salvar uma cópia em pasta local ou NAS **escolhida pelo proprietário**, mas sua execução real de download → validação → cópia → validação → recibo ainda não havia sido exercitada integralmente em uma máquina Windows com os bytes originais.

O P2-01M valida **esse mecanismo já existente** em Windows GitHub Actions com um destino descartável no `RUNNER_TEMP`; NÃO escolhe qualquer caminho de backup da máquina do usuário, NÃO acessa NAS, NÃO cria registro público ou privado de imagens, e NÃO publica nova cópia grande em Actions.

## Fontes originais fixas

- GitHub Actions [run #43531054368](https://github.com/fbreve/pypcc/actions/runs/43531054368), artefato de 90 dias **14087551705**, cujo conteúdo já foi confirmado em segundo runner independente; vencimento: **11/02/2027 23:59:18 UTC**.
- Arquivo **`p2-01h-exact-runtime-image.tar.gz`**, conteúdo OCI SHA256 **`008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`** (diferente do SHA do ZIP que encapsula o artefato do Actions).
- Config ID Docker **`sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`**.
- O script usa `gh run download` autenticado pelo `GH_TOKEN` do workflow em escopo somente `contents:read`/`actions:read`. Nenhum token é salvo em relatório.

## Validação real no Windows

O runner descarta o diretório temporário ao final do job. O teste `scripts/p2_01m_validate_original_OCI_windows_backup_ephemeral.ps1` faz o seguinte:

1. Executa o script original P2-01J contra pasta **previamente criada** e baixa o **artefato original real** do Actions. Confirma SHA256 do arquivo binário original e o recibo local completo (fonte, artifact ID, config ID, tamanho, indicadores NO-GO).
2. Executa `-VerifyExistingOnly` no arquivo copiado. Repete o modo normal para provar comportamento idempotente **sem substituição do binário já correto**.
3. Cria apenas um **arquivo de teste pequeno e deliberadamente adulterado**, com o mesmo nome do OCI, em outro diretório descartável. Confirma que tanto o modo de verificação quanto o modo normal **recusam o arquivo e não o sobrescrevem nem geram recibo de backup válido**.
4. Confirma que um caminho de destino inexistente é recusado sem criar diretórios por conta própria.
5. Apaga a cópia temporária após o ensaio. Exporta somente um JSON pequeno de indicadores e hashes previamente conhecidos, validado por **outro job Linux**. Nenhuma cópia de 291 MB é arquivada, replicada ou publicada por este workflow.

O ensaio não usa Docker, OCI run, GHCR, releases, instalação de pacotes, importações científicas, origem UCI, grafos, máscaras, gerador aleatório, Numba JIT, propagação ou ajuste PCC.

## Como concluir o backup durável no seu Windows/NAS

O único passo que pode realmente fechar a lacuna de retenção durável é o proprietário escolher uma pasta local/NAS com espaço suficiente, já existente, e executar o **script original**, não o ensaio descartável P2-01M:

```powershell
# No checkout do repositório; substitua por uma pasta local ou UNC EXISTENTE
.\scripts\p2_01j_archive_original_image_locally.ps1 -Destination 'D:\ResearchBackups\PyPCC'

# Verificação independente, sem baixar de novo
.\scripts\p2_01j_archive_original_image_locally.ps1 -Destination 'D:\ResearchBackups\PyPCC' -VerifyExistingOnly
```

Para NAS Synology acessível por SMB, `-Destination '\\servidor\compartilhamento\PyPCC'` também funciona **se a pasta já existir** e o usuário tiver acesso. Estes caminhos são somente exemplos. Antes de executar, configurar o `gh` com autenticação na conta autorizada e evitar repositórios públicos, releases e registries sem revisão de licenças da imagem.

**Não afirmar backup local ou NAS concluído** a partir de um teste em runner temporário. O material armazenado contém binários de terceiros, inclusive Python e bibliotecas, cujo direito de redistribuição externa não foi analisado aqui.

## Bloqueios científicos que continuam

Mesmo com PASS de backup Windows: não há host definitivo selecionado, manifesto de source-build de Python/Numba/Cython/compilador/SDK/OS não fechado, limite de CPU/tempo/memória/disco de ajustes reais de 50.000 iterações não comprovado e política P2-07C não aprovada.

Sempre: `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`. Todos os 72 ajustes científicos permanecem `NOT_RUN_NO_APPROVAL`.

A etapa é considerada tecnicamente encerrada somente se o download/cópia real Windows, os testes adversariais locais e a revisão independente de um recibo fonte-livre passarem. O teste isolado não autoriza P2-08, não seleciona host, nem cria backup durável fora do Actions.
