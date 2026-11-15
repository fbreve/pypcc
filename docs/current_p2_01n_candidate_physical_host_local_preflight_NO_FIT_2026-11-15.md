# PyPCC — P2-01N: verificação local de um host candidato real, sem ajustar PCC (15/11/2026)

**PR DRAFT empilhado sobre P2-07R #295, com código-fonte original e P1 inalterados.** Este teste é executável tanto em **Windows com Docker Desktop/WSL2 (contêiner Linux/amd64)** como em Linux nativo, mas não escolhe o host pelo usuário, não acessa dados oficiais e não executa qualquer propagação PCC.

## Por que este teste é diferente das revisões anteriores

Os ensaios P2-07Q/R executaram com sucesso seis ajustes artificiais de 50.000 iterações com oito observações e um supervisor ativo de CPU/wall. Entretanto, foram executados em runners temporários do GitHub, não no **host físico/VM que executará o estudo**. O P2-01M testou somente a *possibilidade de fazer backup* da imagem OCI original em Windows temporário, e não verificou se o computador/NAS do proprietário pode efetivamente executar a OCI.

O novo `experiments/p2_01n_candidate_host_readiness_no_fit.py` inspeciona um **host local escolhido para avaliação, sem acesso remoto**. O objetivo é um recibo pequeno de capacidade REAL, sem afirmar que o host ou os limites estejam aprovados.

### Verificações em um host Windows/Linux

1. **Imagem original já armazenada**, escolhida por caminho explícito: verifica os **290.796.771 bytes** e o SHA-256 da OCI `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`. **Não baixa nem copia** arquivos e não executa `docker load`; a imagem precisa estar carregada previamente no Docker do próprio computador.
2. **Capacidade física e disco de trabalho**: RAM real do host **≥ 8 GiB**, espaço livre na pasta de trabalho **≥ 12 GiB**. Essa pasta precisa existir; não pode ser substituída automaticamente por outra partição.
3. **Docker/WSL2 real, não remoto**: valida `docker context inspect` e `DOCKER_HOST` local, recusando SSH/TCP remoto; exige `OSType=linux`, `Architecture=amd64`, `CgroupVersion=2`, memória disponível na VM/engine Docker **≥ 8 GiB** (não apenas os 128 GiB que a máquina física possa ter), versão do engine e o **Docker image config ID exato** `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`.
4. **Um único contêiner somente de importação e resolução de backends**: `--network none`, `--read-only`, UID 65534, sem capabilities, memória 2 GiB e swap zero pelo cgroup, quota de uma CPU, 128 PIDs e tmpfs descartável de 128 MiB. Confirma os valores **de fato expostos por `/sys/fs/cgroup` dentro do contêiner**, em vez de confiar somente na linha de comando.
5. Dentro da OCI original, chama o P2-01L **já validado**, que verifica o **SHA integral do P1, fontes originais, dois binários Cython por SHA** e as identidades das **72 seleções de função** (24 NumPy/24 Numba/24 Cython), sem sequer invocar os propagadores. Qualquer fallback Cython→Numba é recusado. Testa a preservação do estado científico `NO-GO`.
6. O recibo contém somente números de RAM/disco, versão do daemon e flags/hashes de identidade já conhecidos. Não registra nome do computador, username, endereço, caminho da imagem, pasta de trabalho, IP, host Docker ou número de série. O arquivo de saída **é local e não é enviado automaticamente ao GitHub**. Pode ser discutido separadamente com dados sensíveis removidos.

### Execução no seu computador Windows

Requisitos: checkout atual do repositório contendo P2-01N, Python com biblioteca padrão, Docker Desktop configurado com **contêineres Linux/WSL2**, `docker` no PATH, pasta local existente com pelo menos 12 GiB disponíveis e uma **cópia do arquivo OCI exato**.

A cópia durável pode ser feita pelo script já testado do P2-01J, se não existir:

```powershell
# Exemplo: substitua por uma pasta de backup já existente, local ou SMB.
.\scripts\p2_01j_archive_original_image_locally.ps1 -Destination 'D:\Backups\PyPCC'
.\scripts\p2_01j_archive_original_image_locally.ps1 -Destination 'D:\Backups\PyPCC' -VerifyExistingOnly
```

Depois, o proprietário deve carregar **explicitamente** no seu Docker a mesma OCI verificada (isso altera apenas o armazenamento local do Docker, sem fazer download/rebuild):

```powershell
docker load --input 'D:\Backups\PyPCC\p2-01h-exact-runtime-image.tar.gz'
```

Finalmente, no checkout do PR P2-01N, escolha uma pasta de trabalho local **já existente**, com espaço suficiente. Um arquivo de recibo **não pode existir anteriormente**, para evitar sobrescrita:

```powershell
python -m experiments.p2_01n_candidate_host_readiness_no_fit `
  --archive 'D:\Backups\PyPCC\p2-01h-exact-runtime-image.tar.gz' `
  --work-dir 'D:\Backups\PyPCC' `
  --output 'D:\Backups\PyPCC\p2-01n-host-candidate-NO-FIT.json'
```

Os caminhos são **ilustrativos**. O procedimento não exige executar em um NAS; um NAS pode ser apenas o **destino do arquivo preservado**, enquanto o PC com Docker é o host de execução. Não se deve presumir que o NAS Synology execute a OCI x86-64 sem conferir processador, engine e limites reais.

## Resultado possível e o que ele NÃO significa

**PASS:** uma máquina **real candidata** demonstrou RAM física e do Docker, espaço livre na pasta escolhida, Docker Linux/amd64 cgroup v2, memória/swap/CPU/PIDs/tmpfs efetivamente restringidos e todas as 72 seleções de backend com os arquivos originais. Não há fit. Isso é uma **verificação técnica local**, não aprovação da máquina.

**STOP:** arquivo OCI ausente/hash incorreto, arquivo diferente carregado, pasta sem 12 GiB, memória da VM Docker insuficiente, daemon Docker remoto, contêiner Windows, cgroup v1, memória/swap/CPU/PIDs sem isolamento, falta de acesso aos arquivos originais, fallback de Cython ou importações quebradas. Em caso de STOP, não há substituição de OCI, reconstrução ou tentativa de executar o PCC.

**Não medido nesta etapa:** espaço livre interno do armazenamento de volumes/imagens da VM Docker, manutenção durável do backup em storage independente, cadeia de compilação/SDK/SO reproduzível a partir de fontes, dependências lazy `dlopen` fora do caminho de importação, host futuro idêntico ao observado, limites efetivos dos ajustes **reais** completos de 50.000 iterações com snapshots, política P2-07C aprovada ou manifesto final P2-08.

Em toda saída (inclusive PASS local), permanecem:

`P2_01_final_execution_toolchain_lock_pass=false`

`P2_07_full_resources_gate_pass=false`

`P2_execution_preflight_sha256=null`

`P2_execution_authorized=false`

`real_PCC_fits_executed=0`

Os **72 casos originais** continuam `NOT_RUN_NO_APPROVAL`. Uma execução integral de `docker load`/teste de importação **não é** autorização para uma propagação real, nem mesmo um único caso de piloto.

## Testes técnicos

O PR inclui testes adversariais de biblioteca padrão **15 × Linux/Windows × Python 3.10/3.12**, sem execução de Docker/PCC nesses quatro jobs. Um quinto job em Linux GitHub Actions testa a mesma interface host-versus-OCI com os **bytes originais arquivados**, carregando a imagem já conhecida apenas em runner descartável. O CI também executa a regressão de produção, sem uso de UCI. Passar nesse runner demonstra apenas funcionamento do **procedimento**, **não capacidade da máquina do usuário**. Nunca publicar um recibo contendo dados do host do usuário.
