<#
.SYNOPSIS
    Remove everything the pipeline generated, so `dvc repro` starts from nothing.

.DESCRIPTION
    Deletes stage outputs, dvc.lock, MLflow state and the Optuna study, then
    optionally garbage-collects the DVC cache.

    NEVER touches data/raw -- that is source data copied in by hand, not
    something any stage produced. Re-copying it costs gigabytes off another
    drive, which is also why `git clean -xdf` is the wrong tool here: DVC
    gitignores its outputs, so git treats the raw data as disposable.

    Dry run by default. Nothing is deleted until you pass -Execute.

.EXAMPLE
    .\scripts\clean_artifacts.ps1
    Lists what would be removed, with sizes. Deletes nothing.

.EXAMPLE
    .\scripts\clean_artifacts.ps1 -Execute
    Actually removes them.

.EXAMPLE
    .\scripts\clean_artifacts.ps1 -Execute -CollectCache
    Also runs `dvc gc --workspace` to reclaim the cache.
#>
[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$CollectCache
)

$ErrorActionPreference = 'Stop'

# Refuse to run anywhere that isn't the project root. A recursive delete
# rooted in the wrong directory is not a recoverable mistake.
if (-not (Test-Path 'dvc.yaml') -or -not (Test-Path 'params.yaml')) {
    throw "Run this from the repo root (no dvc.yaml/params.yaml in $PWD)."
}

$targets = @(
    # DVC pipeline state -- removing this alone marks every stage as never-run
    'dvc.lock',

    # Stage outputs
    'data/interim',
    'data/processed',
    'data_pipeline/splits',
    'metrics',
    'deployment/exported_model',

    # Checkpoints: train plus every experiment grid
    'training/checkpoints',
    'training/checkpoints_tune',
    'training/checkpoints_benchmark',
    'training/checkpoints_capacity',
    'training/checkpoints_loss',
    'training/checkpoints_batching',

    # MLflow: the tracking DB and the artifact store holding the pickled models.
    # mlruns/ lands next to the working directory because no artifact root was
    # configured; it is usually the largest item here.
    'training/mlflow.db',
    'mlruns',

    # Optuna study
    'training/optuna_study.db'
)

function Get-SizeMB {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    $bytes = (Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue |
              Measure-Object -Property Length -Sum).Sum
    if ($null -eq $bytes) { $bytes = (Get-Item $Path).Length }
    return [math]::Round($bytes / 1MB, 1)
}

Write-Host ""
Write-Host ("{0,-42} {1,10}" -f 'TARGET', 'SIZE (MB)')
Write-Host ("-" * 54)

$found = @()
foreach ($t in $targets) {
    if (Test-Path $t) {
        $size = Get-SizeMB $t
        Write-Host ("{0,-42} {1,10}" -f $t, $size)
        $found += $t
    } else {
        Write-Host ("{0,-42} {1,10}" -f $t, '-') -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "KEPT (not touched by this script):" -ForegroundColor Green
Write-Host "  data/raw/            source images and masks"
Write-Host "  data/raw/*.dvc       pins for the raw data, if you ran dvc add"
Write-Host "  .dvc/config          repo config"
Write-Host "  all code, dvc.yaml, params.yaml"
Write-Host ""

if (-not $Execute) {
    Write-Host "DRY RUN -- nothing deleted. Re-run with -Execute to remove the above." -ForegroundColor Yellow
    exit 0
}

foreach ($t in $found) {
    Write-Host "removing $t"
    Remove-Item -LiteralPath $t -Recurse -Force
}

if ($CollectCache) {
    # Runs AFTER dvc.lock is gone, so the pipeline outputs now count as
    # unreferenced and are collected. -w keeps whatever the remaining .dvc
    # files still point at, which is how data/raw survives this.
    # dvc gc only ever touches .dvc/cache, never files in the workspace.
    Write-Host ""
    Write-Host "running dvc gc --workspace"
    dvc gc --workspace --force
}

Write-Host ""
Write-Host "Done. Next: dvc repro evaluate" -ForegroundColor Green
