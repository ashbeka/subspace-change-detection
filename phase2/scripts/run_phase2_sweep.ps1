# run_phase2_sweep.ps1
# Runs a Phase 2 experiment sweep (train + eval) from the repo root.
#
# Examples (from repo root):
#   powershell -ExecutionPolicy Bypass -File phase2/scripts/run_phase2_sweep.ps1 -Preset core -Epochs 50 -Seeds 1234
#   powershell -ExecutionPolicy Bypass -File phase2/scripts/run_phase2_sweep.ps1 -Preset full -Epochs 150 -Seeds "1234,1235,1236" -OutputTag overnight
#
# Notes:
# - This script writes a per-run patched config `config_used.yaml` into each output dir.
# - It avoids editing your tracked YAML configs.
# - Use -ProgressBars for native interactive tqdm batch bars. In that mode,
#   transcript and per-run console logs are disabled so Python owns the terminal.

param(
    [ValidateSet("core", "full", "full+eig")]
    [string]$Preset = "core",

    [int]$Epochs = 50,

    # Seeds can be passed as:
    # - a comma-separated string:  -Seeds "1234,1235,1236"   (works with `powershell -File ...`)
    # - a space-separated string:  -Seeds "1234 1235 1236"
    # - or as extra args after -Seeds (PowerShell -File quirk): -Seeds 1234 1235 1236
    [string]$Seeds = "1234",

    [string]$OutputTag = "",

    # Output directory root (absolute or repo-relative). Results go under: <OutputRoot>/sweep_<OutputTag>/...
    [string]$OutputRoot = "phase2\\outputs",

    # Optional overrides (0 / -1 means "leave as config default").
    [int]$BatchSize = 0,
    [int]$EvalBatchSize = 0,
    [int]$NumWorkers = -1,
    [int]$ValEvery = 0,

    # Retention policy to limit output growth.
    [ValidateSet("full", "compact", "metrics_only")]
    [string]$Retention = "full",

    # Prefer live tqdm progress bars over per-run Tee-Object console logs.
    [switch]$ProgressBars,

    # Dataset caching controls (recommended with NumWorkers=0).
    [switch]$NoCacheCities,
    [int]$CacheMaxCities = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $repoRoot

# Prefer the repo-root virtualenv Python if present; otherwise fall back to PATH python.
$python = "python"
$venvPyWin = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$venvPyUnix = Join-Path $repoRoot ".venv\\bin\\python"
if (Test-Path $venvPyWin) {
    $python = $venvPyWin
} elseif (Test-Path $venvPyUnix) {
    $python = $venvPyUnix
}

# Data roots.
$oscdRoot = Join-Path $repoRoot "data\\OSCD"
$changeRootResidualCandidates = @(
    (Join-Path $repoRoot "phase1\\outputs\\oscd_saved_priors_fast\\oscd_change_maps"),
    (Join-Path $repoRoot "phase1\\outputs\\oscd_saved\\oscd_change_maps")
)
$changeRootEigCandidates = @(
    (Join-Path $repoRoot "phase1\\outputs\\oscd_saved_priors_fast_eig\\oscd_change_maps"),
    (Join-Path $repoRoot "phase1\\outputs\\oscd_saved_eig\\oscd_change_maps")
)

$changeRootResidual = $changeRootResidualCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$changeRootEig = $changeRootEigCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not (Test-Path $oscdRoot)) {
    throw "Missing OSCD root at $oscdRoot"
}
if (-not $changeRootResidual) {
    throw "Missing Phase 1 change maps (looked for: $($changeRootResidualCandidates -join ', ')). Generate with Phase 1 --save_change_maps."
}
if (($Preset -eq "full+eig") -and (-not $changeRootEig)) {
    throw "Missing eig Phase 1 change maps (looked for: $($changeRootEigCandidates -join ', ')). Generate eig priors first."
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$PythonArgs,
        [string]$LogPath
    )
    Write-Host "[$Name] $python -u $($PythonArgs -join ' ')" -ForegroundColor Cyan

    # Native executables that write to stderr can be treated as non-terminating errors
    # in Windows PowerShell; avoid breaking the sweep on warnings by forcing Continue.
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($LogPath) {
            New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null
            if ($ProgressBars) {
                Write-Host "[$Name] ProgressBars mode: native Python/tqdm terminal output; no per-run console log." -ForegroundColor DarkGray
                & $python -u @PythonArgs
            } else {
                & $python -u @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath
            }
        } else {
            & $python -u @PythonArgs
        }
    } finally {
        $ErrorActionPreference = $oldEap
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed (exit $LASTEXITCODE): $Name"
    }
}

function Write-PatchedConfig {
    param(
        [string]$BaseConfig,
        [string]$OutConfig,
        [int]$EpochsValue,
        [int]$SeedValue,
        [string]$ChangeRoot,
        [string]$ExperimentTag,
        [int]$BatchSizeValue,
        [int]$EvalBatchSizeValue,
        [int]$NumWorkersValue,
        [int]$ValEveryValue,
        [bool]$CacheCitiesValue,
        [int]$CacheMaxCitiesValue
    )
    $py = @'
import sys
from pathlib import Path
import yaml

base_cfg_path, out_cfg_path, epochs_s, seed_s, change_root, exp_tag, batch_s, eval_bs_s, num_workers_s, val_every_s, cache_cities_s, cache_max_s = sys.argv[1:13]
epochs = int(epochs_s)
seed = int(seed_s)
batch = int(batch_s)
eval_bs = int(eval_bs_s)
num_workers = int(num_workers_s)
val_every = int(val_every_s)
cache_cities = str(cache_cities_s).lower() in ("1", "true", "yes", "y")
cache_max = int(cache_max_s)

cfg = yaml.safe_load(Path(base_cfg_path).read_text(encoding="utf-8"))
cfg.setdefault("dataset", {})
cfg["dataset"]["root"] = "data/OSCD"

cfg.setdefault("phase1", {})
cfg["phase1"]["change_maps_root"] = str(change_root)

cfg.setdefault("training", {})
cfg["training"]["epochs"] = epochs
cfg["training"]["seed"] = seed
if batch > 0:
    cfg["training"]["batch_size"] = batch
if num_workers >= 0:
    cfg["training"]["num_workers"] = num_workers
if val_every > 0:
    cfg["training"]["val_every"] = val_every

if eval_bs > 0:
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["batch_size"] = eval_bs

cfg.setdefault("dataset", {})
cfg.setdefault("dataset", {}).setdefault("cache", {})
cfg["dataset"]["cache"]["cities"] = bool(cache_cities)
cfg["dataset"]["cache"]["max_cities"] = int(cache_max)

sch = cfg["training"].get("scheduler", {}) or {}
if str(sch.get("name", "cosine")).lower() == "cosine":
    sch["T_max"] = epochs
    cfg["training"]["scheduler"] = sch

cfg["experiment_tag"] = str(exp_tag)

Path(out_cfg_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
'@
$py | & $python -u - $BaseConfig $OutConfig "$EpochsValue" "$SeedValue" $ChangeRoot $ExperimentTag "$BatchSizeValue" "$EvalBatchSizeValue" "$NumWorkersValue" "$ValEveryValue" "$CacheCitiesValue" "$CacheMaxCitiesValue"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to write patched config: $OutConfig"
    }
}

# Build experiment list.
$experiments = @(
    @{ Tag = "E0_raw_unet"; Config = "phase2/configs/oscd/core/E0_raw_unet.yaml"; ChangeRoot = $changeRootResidual },
    @{ Tag = "S0_siamese"; Config = "phase2/configs/oscd/core/S0_raw_siamese.yaml"; ChangeRoot = $changeRootResidual },
    @{ Tag = "E1_raw_ds"; Config = "phase2/configs/oscd/core/E1_raw_ds_unet.yaml"; ChangeRoot = $changeRootResidual },
    @{ Tag = "E1b_raw_ds_cross"; Config = "phase2/configs/oscd/core/E1b_raw_ds_cross_unet.yaml"; ChangeRoot = $changeRootResidual },
    @{ Tag = "E2_raw_pca"; Config = "phase2/configs/oscd/core/E2_raw_pca_unet.yaml"; ChangeRoot = $changeRootResidual },
    @{ Tag = "E3_raw_ds_pca"; Config = "phase2/configs/oscd/core/E3_raw_ds_pca_unet.yaml"; ChangeRoot = $changeRootResidual }
)

if ($Preset -eq "full" -or $Preset -eq "full+eig") {
    $experiments += @(
        @{ Tag = "E0_raw_resnet"; Config = "phase2/configs/oscd/extended/E0_raw_resnet.yaml"; ChangeRoot = $changeRootResidual },
        @{ Tag = "E3_raw_ds_pca_resnet"; Config = "phase2/configs/oscd/extended/E3_raw_ds_pca_resnet.yaml"; ChangeRoot = $changeRootResidual },
        @{ Tag = "E3_raw_ds_pca_fusion"; Config = "phase2/configs/oscd/extended/E3_raw_ds_pca_fusion.yaml"; ChangeRoot = $changeRootResidual }
    )
}

if ($Preset -eq "full+eig") {
    $experiments += @(
        @{ Tag = "E1_raw_ds_eig"; Config = "phase2/configs/oscd/core/E1_raw_ds_unet.yaml"; ChangeRoot = $changeRootEig },
        @{ Tag = "E3_raw_ds_pca_eig"; Config = "phase2/configs/oscd/core/E3_raw_ds_pca_unet.yaml"; ChangeRoot = $changeRootEig }
    )
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputTag)) { $OutputTag = $ts }
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = "phase2\\outputs" }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot $OutputRoot
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$runRoot = Join-Path $OutputRoot ("sweep_" + $OutputTag)
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

$summaryPath = Join-Path $runRoot "sweep_test_summary.csv"
if (Test-Path $summaryPath) { Remove-Item -Force $summaryPath }

Write-Host "Repo root: $repoRoot" -ForegroundColor Cyan
Write-Host "Python: $python" -ForegroundColor Cyan
Write-Host "OSCD: $oscdRoot" -ForegroundColor Cyan
Write-Host "Phase1 change maps (residual): $changeRootResidual" -ForegroundColor Cyan
if ($changeRootEig) {
    Write-Host "Phase1 change maps (eig): $changeRootEig" -ForegroundColor Cyan
}

$seedList = @()
foreach ($tok in ($Seeds -split "[,\\s]+")) {
    if ([string]::IsNullOrWhiteSpace($tok)) { continue }
    $seedList += [int]$tok
}
if ($args -and $args.Count -gt 0) {
    foreach ($tok in $args) {
        if ([string]::IsNullOrWhiteSpace($tok)) { continue }
        try {
            $seedList += [int]$tok
        } catch {
            throw "Unexpected extra argument '$tok'. If you meant seeds, pass only integers after -Seeds."
        }
    }
}
if (-not $seedList -or $seedList.Count -eq 0) {
    throw "No valid seeds parsed from -Seeds '$Seeds'. Example: -Seeds '1234,1235,1236'"
}

Write-Host "Preset: $Preset | Epochs: $Epochs | Seeds: $($seedList -join ',')" -ForegroundColor Cyan
Write-Host "Outputs: $runRoot" -ForegroundColor Cyan

$transcriptStarted = $false
if ($ProgressBars) {
    Write-Host "ProgressBars mode: transcript disabled so native tqdm can render in this terminal." -ForegroundColor Yellow
} else {
    Start-Transcript -Path (Join-Path $runRoot "sweep_transcript.txt") -Force | Out-Null
    $transcriptStarted = $true
}

try {
    $totalScheduled = $seedList.Count * $experiments.Count
    $runIndex = 0
    foreach ($seed in $seedList) {
        foreach ($exp in $experiments) {
            $runIndex += 1
            $tag = $exp.Tag
            $baseCfg = Join-Path $repoRoot $exp.Config
            $changeRoot = $exp.ChangeRoot
            $priorsVariant = Split-Path (Split-Path $changeRoot -Parent) -Leaf

            if (-not (Test-Path $baseCfg)) {
                Write-Host "Skipping missing config: $baseCfg" -ForegroundColor Yellow
                continue
            }

            $outDir = Join-Path $runRoot ("{0}__seed{1}" -f $tag, $seed)
            New-Item -ItemType Directory -Force -Path $outDir | Out-Null

            $evalCsv = Join-Path $outDir "eval\\oscd_seg_eval_summary.csv"
            if (Test-Path $evalCsv) {
                Write-Host ("Skipping existing run (found eval): {0}" -f $outDir) -ForegroundColor DarkGray
                continue
            }

            $cfgUsed = Join-Path $outDir "config_used.yaml"
            $expTag = "{0}__seed{1}__ep{2}" -f $tag, $seed, $Epochs
            Write-PatchedConfig -BaseConfig $baseCfg -OutConfig $cfgUsed -EpochsValue $Epochs -SeedValue $seed -ChangeRoot $changeRoot -ExperimentTag $expTag -BatchSizeValue $BatchSize -EvalBatchSizeValue $EvalBatchSize -NumWorkersValue $NumWorkers -ValEveryValue $ValEvery -CacheCitiesValue (-not $NoCacheCities) -CacheMaxCitiesValue $CacheMaxCities

            $trainArgs = @(
                "-m", "phase2.train.train_oscd_seg",
                "--config", $cfgUsed,
                "--oscd_root", $oscdRoot,
                "--phase1_change_maps_root", $changeRoot,
                "--output_dir", $outDir,
                "--device", "cuda"
            )
            if ($ProgressBars) {
                $trainArgs += @("--progress_style", "tqdm")
            }
            $evalArgs = @(
                "-m", "phase2.eval.evaluate_oscd_seg",
                "--config", $cfgUsed,
                "--oscd_root", $oscdRoot,
                "--phase1_change_maps_root", $changeRoot,
                "--checkpoint", (Join-Path $outDir "best.ckpt"),
                "--output_dir", (Join-Path $outDir "eval"),
                "--device", "cuda"
            )

            try {
                Write-Host ("==== [{0}/{1}] {2} (seed {3}, priors {4}, epochs {5}) ====" -f $runIndex, $totalScheduled, $tag, $seed, $priorsVariant, $Epochs) -ForegroundColor Green
                Run-Step -Name "$tag train" -PythonArgs $trainArgs -LogPath (Join-Path $outDir "train_console.log.txt")
                Run-Step -Name "$tag eval" -PythonArgs $evalArgs -LogPath (Join-Path $outDir "eval_console.log.txt")

                if (Test-Path $evalCsv) {
                    $rows = Import-Csv $evalCsv | Where-Object { $_.split -eq "test" }
                    foreach ($r in $rows) {
                        $obj = [PSCustomObject]@{
                            tag = $tag
                            config = $exp.Config
                            seed = $seed
                            epochs = $Epochs
                            priors_variant = $priorsVariant
                            output_dir = $outDir
                            model = $r.model
                            experiment_tag = $r.experiment_tag
                            threshold = $r.threshold
                            mean_iou = $r.mean_iou
                            mean_f1 = $r.mean_f1
                            mean_auroc = $r.mean_auroc
                            mean_pr_auc = $r.mean_pr_auc
                        }
                        $obj | Export-Csv -Path $summaryPath -NoTypeInformation -Append
                    }
                } else {
                    Write-Host "Missing eval summary CSV at $evalCsv" -ForegroundColor Yellow
                }

                if ($Retention -ne "full") {
                    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $outDir "train_console.log.txt")
                    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $outDir "eval_console.log.txt")
                }
                if ($Retention -eq "metrics_only") {
                    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $outDir "best.ckpt")
                }
            } catch {
                Write-Host ("FAILED: {0} seed {1} ({2}) -> {3}" -f $tag, $seed, $priorsVariant, $_.Exception.Message) -ForegroundColor Red
                continue
            }
        }
    }

    Write-Host "Sweep finished." -ForegroundColor Green
    Write-Host "Test summary: $summaryPath" -ForegroundColor Cyan
} finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
