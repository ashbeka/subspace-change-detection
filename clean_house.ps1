[CmdletBinding(SupportsShouldProcess = $true)]
param(
  # Keep these Phase 2 output folders under phase2/outputs (by folder name).
  [string[]]$KeepPhase2 = @("sweep_overnight_full_eig_3seeds_150ep_v2"),

  # Keep these Phase 1 output folders under phase1/outputs (by folder name).
  # Needed for Phase 2 priors training:
  # - oscd_saved_priors_fast (residual priors)
  # - oscd_saved_priors_fast_eig (eig priors)
  [string[]]$KeepPhase1 = @("oscd_saved_priors_fast", "oscd_saved_priors_fast_eig"),

  # When set, removes all phase1/outputs and phase2/outputs content except what's in KeepPhase*.
  [switch]$Aggressive
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path $PSScriptRoot).Path
Set-Location $repoRoot

function Remove-PathSafe {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return }
  if ($PSCmdlet.ShouldProcess($Path, "Remove")) {
    Remove-Item -LiteralPath $Path -Recurse -Force
  }
}

Write-Host "Repo: $repoRoot" -ForegroundColor Cyan
$mode = if ($Aggressive) { "aggressive" } else { "safe" }
Write-Host "Mode: $mode" -ForegroundColor Cyan

############################
# Generated web context pack
############################

Remove-PathSafe (Join-Path $repoRoot "web_context_pack")
Remove-PathSafe (Join-Path $repoRoot "web_context_pack_min.zip")
Remove-PathSafe (Join-Path $repoRoot "web_reference_papers.zip")

################
# Phase 2 outputs
################

$phase2Out = Join-Path $repoRoot "phase2\\outputs"
if (Test-Path $phase2Out) {
  $keep2 = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  foreach ($k in $KeepPhase2) { [void]$keep2.Add($k) }

  $dirs = Get-ChildItem -Path $phase2Out -Directory -Force
  foreach ($d in $dirs) {
    if ($keep2.Contains($d.Name)) { continue }
    if (-not $Aggressive) {
      if ($d.Name -match '^(?:_smoke_|_bench_|sweep__debug_)') {
        Remove-PathSafe $d.FullName
      }
    } else {
      Remove-PathSafe $d.FullName
    }
  }
}

################
# Phase 1 outputs
################

$phase1Out = Join-Path $repoRoot "phase1\\outputs"
if (Test-Path $phase1Out) {
  $keep1 = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  foreach ($k in $KeepPhase1) { [void]$keep1.Add($k) }

  $dirs = Get-ChildItem -Path $phase1Out -Directory -Force
  foreach ($d in $dirs) {
    if ($keep1.Contains($d.Name)) { continue }
    if (-not $Aggressive) {
      # Remove common throwaway outputs (visualizations / eval-only runs).
      if ($d.Name -match '^(?:oscd_run|oscd_run_validation|multisenge_viz|oscd_figs_all|oscd_previews|oscd_previews_eig)$') {
        Remove-PathSafe $d.FullName
      }
    } else {
      Remove-PathSafe $d.FullName
    }
  }
}

Write-Host "Done." -ForegroundColor Green
