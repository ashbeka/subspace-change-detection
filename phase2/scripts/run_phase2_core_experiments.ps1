# run_phase2_core_experiments.ps1
# Runs Phase 2 core experiments (train + eval) from the repo root.

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

# Data roots (absolute paths).
$oscdRoot = Join-Path $repoRoot "data\\OSCD"
$phase1ChangeMapsRoot = Join-Path $repoRoot "phase1\\outputs\\oscd_saved_priors_fast\\oscd_change_maps"
if (-not (Test-Path $phase1ChangeMapsRoot)) {
    $alt = Join-Path $repoRoot "phase1\\outputs\\oscd_saved\\oscd_change_maps"
    if (Test-Path $alt) {
        $phase1ChangeMapsRoot = $alt
    }
}

# Optional quick mode (e.g., 5 on CPU). Use 50 for the full configs.
$maxEpochs = 50

function Run-ExeWithRetry {
    param(
        [string]$Exe,
        [string[]]$Args,
        [int]$MaxRetries = 1,
        [string]$Name = ""
    )
    $attempt = 0
    while ($attempt -le $MaxRetries) {
        $attemptNum = $attempt + 1
        $joined = ($Args | ForEach-Object { if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ } }) -join ' '
        Write-Host "[$Name] Running (attempt $attemptNum/$($MaxRetries+1)):`n $Exe $joined" -ForegroundColor Cyan
        & $Exe @Args
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$Name] Succeeded." -ForegroundColor Green
            return $true
        } else {
            Write-Host "[$Name] Failed with exit code $LASTEXITCODE." -ForegroundColor Yellow
        }
        $attempt++
    }
    Write-Host "[$Name] Giving up after $($MaxRetries+1) attempts." -ForegroundColor Red
    return $false
}

function Run-Experiment {
    param(
        [string]$ConfigPath,
        [string]$OutputDir,
        [string]$Tag
    )

    $ckptPath = Join-Path $OutputDir "best.ckpt"
    $evalOutDir = Join-Path $OutputDir "eval"

    $trainArgs = @(
        "-m", "phase2.train.train_oscd_seg",
        "--config", $ConfigPath,
        "--oscd_root", $oscdRoot,
        "--phase1_change_maps_root", $phase1ChangeMapsRoot,
        "--output_dir", $OutputDir,
        "--device", "cuda",
        "--max_epochs", "$maxEpochs"
    )
    $evalArgs = @(
        "-m", "phase2.eval.evaluate_oscd_seg",
        "--config", $ConfigPath,
        "--oscd_root", $oscdRoot,
        "--phase1_change_maps_root", $phase1ChangeMapsRoot,
        "--checkpoint", $ckptPath,
        "--output_dir", $evalOutDir,
        "--device", "cuda"
    )

    $okTrain = Run-ExeWithRetry -Exe $python -Args $trainArgs -MaxRetries 1 -Name "$Tag train"
    if (-not $okTrain) {
        Write-Host "[$Tag] Training failed after retries; skipping evaluation." -ForegroundColor Red
        return
    }

    $okEval = Run-ExeWithRetry -Exe $python -Args $evalArgs -MaxRetries 1 -Name "$Tag eval"
    if (-not $okEval) {
        Write-Host "[$Tag] Evaluation failed after retries." -ForegroundColor Red
    }
}

Write-Host "Repo root: $repoRoot" -ForegroundColor Cyan
Write-Host "Python: $python" -ForegroundColor Cyan
Write-Host "OSCD root: $oscdRoot" -ForegroundColor Cyan
Write-Host "Phase1 change maps: $phase1ChangeMapsRoot" -ForegroundColor Cyan

$phase2Out = Join-Path $repoRoot "phase2\\outputs"

# E0: raw only (U-Net)
$outE0 = Join-Path $phase2Out "oscd_seg_E0_raw_v2"
Run-Experiment -ConfigPath (Join-Path $repoRoot "phase2\\configs\\oscd_seg_baseline.yaml") -OutputDir $outE0 -Tag "E0_raw_only"

# S0: Siamese baseline (raw only)
$outS0 = Join-Path $phase2Out "oscd_seg_siamese_v2"
Run-Experiment -ConfigPath (Join-Path $repoRoot "phase2\\configs\\oscd_seg_siamese.yaml") -OutputDir $outS0 -Tag "S0_siamese_raw_only"

# E1: raw + DS projection
$outE1 = Join-Path $phase2Out "oscd_seg_E1_raw_ds_v2"
Run-Experiment -ConfigPath (Join-Path $repoRoot "phase2\\configs\\oscd_seg_E1_raw_ds.yaml") -OutputDir $outE1 -Tag "E1_raw+ds"

# E1b: raw + DS cross-residual
$outE1b = Join-Path $phase2Out "oscd_seg_E1b_raw_ds_cross_v2"
Run-Experiment -ConfigPath (Join-Path $repoRoot "phase2\\configs\\oscd_seg_E1b_raw_ds_cross.yaml") -OutputDir $outE1b -Tag "E1b_raw+ds_cross"

# E2: raw + PCA-diff
$outE2 = Join-Path $phase2Out "oscd_seg_E2_raw_pca_v2"
Run-Experiment -ConfigPath (Join-Path $repoRoot "phase2\\configs\\oscd_seg_E2_raw_pca.yaml") -OutputDir $outE2 -Tag "E2_raw+pca"

# E3: raw + DS projection + PCA-diff
$outE3 = Join-Path $phase2Out "oscd_seg_E3_raw_ds_pca_v2"
Run-Experiment -ConfigPath (Join-Path $repoRoot "phase2\\configs\\oscd_seg_priors.yaml") -OutputDir $outE3 -Tag "E3_raw+ds+pca"

Write-Host "All scheduled experiments attempted." -ForegroundColor Cyan
