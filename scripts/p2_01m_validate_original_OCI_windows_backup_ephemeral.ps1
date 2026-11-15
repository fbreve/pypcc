# Research-only P2-01M — validate REAL Windows/PowerShell archival path end-to-end
# in ephemeral Windows GitHub Actions storage. It does NOT back up anything to
# the user's PC/NAS, create a release/registry object, run Docker or PCC.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [string]$TemporaryRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedSha = '008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8'
$expectedImageId = 'sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55'
$expectedFilename = 'p2-01h-exact-runtime-image.tar.gz'
$scriptPath = Join-Path $PSScriptRoot 'p2_01j_archive_original_image_locally.ps1'

if (-not $IsWindows -or -not (Test-Path -LiteralPath $TemporaryRoot -PathType Container)) {
    throw 'P2-01M requires an existing ephemeral Windows runner temporary directory'
}
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw 'Exact original P2-01J Windows backup helper absent'
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'Authenticated GitHub CLI required to test original artifact download'
}
if (-not $env:GH_TOKEN) {
    throw 'GH_TOKEN is required to test exact prior Actions artifact; no interactive login'
}
if (-not $env:GITHUB_ACTIONS -or $env:GITHUB_ACTIONS -ne 'true') {
    throw 'P2-01M destructive negative tests are restricted to disposable GitHub Actions runner'
}

$work = Join-Path $TemporaryRoot ('p2-01m-' + [Guid]::NewGuid().ToString('N'))
$good = Join-Path $work 'good'
$bad = Join-Path $work 'bad'
New-Item -ItemType Directory -Path $good -Force | Out-Null
New-Item -ItemType Directory -Path $bad -Force | Out-Null

function Require([bool]$condition, [string]$description) {
    if (-not $condition) { throw $description }
}

function Expect-Blocked([scriptblock]$operation, [string]$description) {
    $blocked = $false
    try { & $operation | Out-Null } catch { $blocked = $true }
    Require $blocked "Expected original backup helper to reject: $description"
}

try {
    # The actual Windows code path downloads the preexisting original
    # archived image, not a mock image with a forged SHA.
    & $scriptPath -Destination $good | Out-Null
    $image = Join-Path $good $expectedFilename
    $receiptPath = Join-Path $good 'p2-01j-local-original-image-receipt.json'
    Require (Test-Path -LiteralPath $image -PathType Leaf) 'Original OCI file not backed up'
    Require (Test-Path -LiteralPath $receiptPath -PathType Leaf) 'Local backup JSON receipt missing'
    $imageSha = (Get-FileHash -LiteralPath $image -Algorithm SHA256).Hash.ToLowerInvariant()
    Require ($imageSha -ceq $expectedSha) 'Original archived OCI SHA mismatch after real Windows copy'
    $actualBytes = (Get-Item -LiteralPath $image).Length
    Require ($actualBytes -gt 200MB -and $actualBytes -lt 2GB) 'Original exact archive size invalid'

    $receipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding utf8 | ConvertFrom-Json
    Require ($receipt.schema -ceq 'pypcc-P2-01J-local-OCI-image-copy-receipt/v1') 'Receipt schema drifted'
    Require ($receipt.source_repo -ceq 'fbreve/pypcc') 'Original GH repository drifted'
    Require ($receipt.source_run_id -eq 43531054368) 'Source GH Actions run changed'
    Require ($receipt.source_artifact_id -eq 14087551705) 'Original verified 90-day artifact changed'
    Require ($receipt.exact_sha256 -ceq $expectedSha) 'Backup receipt falsely claimed source SHA'
    Require ($receipt.original_image_config_id -ceq $expectedImageId) 'Backup receipt OCI ID changed'
    Require ($receipt.archive_length_bytes -eq $actualBytes) 'Actual image size and receipt differ'
    Require ($receipt.source_rebuild_performed -ceq $false) 'Unapproved binary rebuild recorded'
    Require ($receipt.artifact_published_to_registry -ceq $false) 'Unapproved publication recorded'
    Require ($null -eq $receipt.P2_execution_preflight_sha256) 'False P2-08 SHA in receipt'
    Require ($receipt.P2_execution_authorized -ceq $false) 'False P2 permission'
    Require ($receipt.real_PCC_fits_executed -eq 0) 'False scientific execution in receipt'

    # Existing-byte verification works WITHOUT downloading again.
    & $scriptPath -Destination $good -VerifyExistingOnly | Out-Null

    # Normal repeat must not overwrite a correct OCI image.
    & $scriptPath -Destination $good | Out-Null
    Require ((Get-FileHash -LiteralPath $image -Algorithm SHA256).Hash.ToLowerInvariant() -ceq $expectedSha) 'Idempotent repeat changed original OCI'

    # A corrupt file with the correct name must never be admitted.
    $fake = Join-Path $bad $expectedFilename
    [IO.File]::WriteAllBytes($fake,[Text.Encoding]::ASCII.GetBytes('WRONG-OCI-IMAGE'))
    $fakeShaBefore=(Get-FileHash -LiteralPath $fake -Algorithm SHA256).Hash.ToLowerInvariant()
    Expect-Blocked { & $scriptPath -Destination $bad -VerifyExistingOnly } 'modified archive in verify mode'
    Expect-Blocked { & $scriptPath -Destination $bad } 'existing incorrect archive must never be replaced'
    Require ((Get-FileHash -LiteralPath $fake -Algorithm SHA256).Hash.ToLowerInvariant() -ceq $fakeShaBefore) 'Bad archive was unexpectedly overwritten'
    Require (-not (Test-Path -LiteralPath (Join-Path $bad 'p2-01j-local-original-image-receipt.json'))) 'Bad archive generated false receipt'

    # Missing destination must never be silently created.
    $missing = Join-Path $work 'destination-does-not-exist'
    Expect-Blocked { & $scriptPath -Destination $missing } 'missing backup destination'
    Require (-not (Test-Path -LiteralPath $missing)) 'Backup unexpectedly created missing destination'

    # Export ONLY source-free, infrastructure-free result counters.
    $safe = [ordered]@{
        schema = 'pypcc-P2-01M-original-Windows-OCI-backup-end-to-end-NO-PCC/v1'
        status = 'EXACT_ORIGINAL_OCI_WINDOWS_BACKUP_COPIED_AND_VERIFIED_EPHEMERAL_ONLY'
        original_image_sha256 = $expectedSha
        original_image_config_id = $expectedImageId
        original_source_actions_run_id = 43531054368
        original_source_actions_artifact_id = 14087551705
        original_archive_bytes = $actualBytes
        real_Windows_download_and_copy_verified = $true
        idempotent_rebackup_and_verify_only_pass = $true
        wrong_content_rejected_without_overwrite = $true
        missing_directory_rejected_without_creation = $true
        archive_released_or_uploaded_to_long_term_storage = $false
        actual_durable_local_or_NAS_user_backup_verified = $false
        physical_execution_host_selected = $false
        P2_01_final_execution_toolchain_lock_pass = $false
        P2_07_full_resources_gate_pass = $false
        P2_execution_preflight_sha256 = $null
        P2_execution_authorized = $false
        real_PCC_fits_executed = 0
    }
    $safe | ConvertTo-Json -Depth 5 -Compress
}
finally {
    # No Actions artifact copy or durable backup is produced here.
    Remove-Item -LiteralPath $work -Force -Recurse -ErrorAction SilentlyContinue
}
