# Watch a Phase 2 sweep while it is running.
#
# Examples from repo root:
#   powershell -ExecutionPolicy Bypass -File phase2/scripts/watch_phase2_sweep.ps1
#   powershell -ExecutionPolicy Bypass -File phase2/scripts/watch_phase2_sweep.ps1 -RunRoot phase2/outputs/sweep_core_150ep_E0_E1_repro_20260503_042530 -RefreshSeconds 15

param(
    [string]$RunRoot = "",
    [int]$RefreshSeconds = 15,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $latest = Get-ChildItem "phase2\outputs" -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "sweep_*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No sweep_* directory found under phase2\outputs."
    }
    $RunRoot = $latest.FullName
} elseif (-not [System.IO.Path]::IsPathRooted($RunRoot)) {
    $RunRoot = Join-Path $repoRoot $RunRoot
}

if (-not (Test-Path -LiteralPath $RunRoot)) {
    throw "RunRoot does not exist: $RunRoot"
}

function Get-TargetEpoch {
    param([string]$ConfigPath)
    if (-not (Test-Path -LiteralPath $ConfigPath)) { return $null }
    $m = Select-String -LiteralPath $ConfigPath -Pattern '^\s*epochs:\s*(\d+)' | Select-Object -First 1
    if ($m -and $m.Matches.Count -gt 0) {
        return [int]$m.Matches[0].Groups[1].Value
    }
    return $null
}

function Get-LastRecord {
    param([string]$TrainLog)
    if (-not (Test-Path -LiteralPath $TrainLog)) { return $null }
    try {
        $records = Get-Content -LiteralPath $TrainLog -Raw | ConvertFrom-Json
        if ($null -eq $records) { return $null }
        if ($records -is [System.Array]) {
            if ($records.Length -eq 0) { return $null }
            return $records[($records.Length - 1)]
        }
        return $records
    } catch {
        return $null
    }
}

while ($true) {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host ""
    Write-Host "[$now] Watching: $RunRoot" -ForegroundColor Cyan

    $runDirs = Get-ChildItem -LiteralPath $RunRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name

    if (-not $runDirs) {
        Write-Host "No run directories yet." -ForegroundColor Yellow
    }

    foreach ($d in $runDirs) {
        $cfg = Join-Path $d.FullName "config_used.yaml"
        $target = Get-TargetEpoch -ConfigPath $cfg
        $trainLog = Join-Path $d.FullName "train_log.json"
        $evalCsv = Join-Path $d.FullName "eval\oscd_seg_eval_summary.csv"
        $last = Get-LastRecord -TrainLog $trainLog

        if (Test-Path -LiteralPath $evalCsv) {
            $test = Import-Csv -LiteralPath $evalCsv | Where-Object { $_.split -eq "test" } | Select-Object -First 1
            if ($test) {
                Write-Host ("DONE  {0,-34} test IoU={1} F1={2} AUROC={3} PR-AUC={4}" -f $d.Name, $test.mean_iou, $test.mean_f1, $test.mean_auroc, $test.mean_pr_auc) -ForegroundColor Green
            } else {
                Write-Host ("DONE  {0,-34} eval CSV present" -f $d.Name) -ForegroundColor Green
            }
        } elseif ($last) {
            $epoch = [int]$last.epoch
            $denom = if ($target) { $target } else { "?" }
            $pct = if ($target) { "{0:N1}%" -f (($epoch / [double]$target) * 100.0) } else { "?" }
            $loss = if ($last.train.loss -ne $null) { "{0:N4}" -f [double]$last.train.loss } else { "?" }
            $iou = if ($last.val.iou -ne $null) { "{0:N4}" -f [double]$last.val.iou } else { "?" }
            $f1 = if ($last.val.f1 -ne $null) { "{0:N4}" -f [double]$last.val.f1 } else { "?" }
            Write-Host ("RUN   {0,-34} epoch {1}/{2} ({3}) loss={4} val_iou={5} val_f1={6}" -f $d.Name, $epoch, $denom, $pct, $loss, $iou, $f1) -ForegroundColor Yellow
        } else {
            Write-Host ("WAIT  {0,-34} no train log yet" -f $d.Name) -ForegroundColor DarkGray
        }
    }

    $summary = Join-Path $RunRoot "sweep_test_summary.csv"
    if (Test-Path -LiteralPath $summary) {
        $count = @(Import-Csv -LiteralPath $summary).Count
        Write-Host ("Summary rows written: {0}" -f $count) -ForegroundColor Cyan
    }

    if ($Once) {
        break
    }
    Start-Sleep -Seconds $RefreshSeconds
}
