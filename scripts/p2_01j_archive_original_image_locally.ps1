# P2-01J: explicitly initiated LOCAL/NAS immutable-image backup; not called by CI.
# Usage: .\scripts\p2_01j_archive_original_image_locally.ps1 -Destination '\\server\share\PyPCC'
# Requires gh CLI authenticated to access fbreve/pypcc Actions artifacts.
# NO rebuild/push/package install/PCC invocation. Never overwrite another image.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [string]$Destination,
    [switch]$VerifyExistingOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$sha = "008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8"
$imageId = "sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55"
$filename = "p2-01h-exact-runtime-image.tar.gz"
# Prefer the independently rehashed 90-day copy, not the 30-day source.
# Exact new artifact 14087551705 expires 2027-02-11T23:59:18Z.
$run = 43531054368
$name = "p2-01j-verified-EXACT-existing-OCI-binary-and-retention-receipt"
$sourceArtifactId = 14087551705

if (-not (Test-Path -LiteralPath $Destination -PathType Container)) {
    throw "Destination directory must ALREADY exist. Explicitly choose a local/NAS backup directory: $Destination"
}
$destDir = (Resolve-Path -LiteralPath $Destination).ProviderPath
$target = Join-Path $destDir $filename
$manifest = Join-Path $destDir "p2-01j-local-original-image-receipt.json"

function Verify-CandidateImage([string]$path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing original OCI archive: $path"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -cne $sha) {
        throw "SHA-256 mismatch; abort without changing file. Got $actual"
    }
    $length = (Get-Item -LiteralPath $path).Length
    if ($length -lt 200MB -or $length -ge 2GB) {
        throw "Archive length outside expected validated image bound: $length"
    }
    return $length
}

if ($VerifyExistingOnly) {
    $bytes = Verify-CandidateImage $target
    Write-Output "VERIFIED existing LOCAL/NAS OCI archive; bytes=$bytes; sha256=$sha"
    return
}
if (Test-Path -LiteralPath $target) {
    $bytes = Verify-CandidateImage $target
    Write-Output "Original archive already exists and matches SHA; no download or overwrite."
} else {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        throw "gh CLI not found. Install/authenticate GitHub CLI and rerun, or use -VerifyExistingOnly for existing archive."
    }
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("p2-01j-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $temp | Out-Null
    try {
        & gh run download $run --repo fbreve/pypcc --name $name --dir $temp
        if ($LASTEXITCODE -ne 0) { throw "gh download failed; original archive may have expired" }
        $candidates = @(Get-ChildItem -LiteralPath $temp -File -Recurse -Filter $filename)
        if ($candidates.Count -ne 1) { throw "Original GitHub Actions artifact must contain exactly one $filename" }
        $bytes = Verify-CandidateImage $candidates[0].FullName
        # Same-volume if possible; never delete a preexisting destination after check.
        if (Test-Path -LiteralPath $target) { throw "Destination unexpectedly created in parallel; refusing overwrite" }
        Copy-Item -LiteralPath $candidates[0].FullName -Destination $target -ErrorAction Stop
        try { [void](Verify-CandidateImage $target) }
        catch {
            Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
            throw
        }
    } finally {
        Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
$record = [ordered]@{
    schema = 'pypcc-P2-01J-local-OCI-image-copy-receipt/v1'
    source_repo = 'fbreve/pypcc'
    source_run_id = $run
    source_artifact_name = $name
    source_artifact_id = $sourceArtifactId
    source_artifact_expiry_utc = '2027-02-11T23:59:18Z'
    exact_source_image_filename = $filename
    exact_sha256 = $sha
    original_image_config_id = $imageId
    archive_length_bytes = $bytes
    destination_file = $target
    verified_utc = [DateTime]::UtcNow.ToString("o")
    source_rebuild_performed = $false
    artifact_published_to_registry = $false
    P2_execution_preflight_sha256 = $null
    P2_execution_authorized = $false
    real_PCC_fits_executed = 0
}
# Receipt can be overwritten only if original archive remains byte-identical.
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifest -Encoding utf8
Write-Output "LOCAL/NAS BACKUP VERIFIED: $target"
Write-Output "SHA-256: $sha"
Write-Output "Receipt: $manifest"
