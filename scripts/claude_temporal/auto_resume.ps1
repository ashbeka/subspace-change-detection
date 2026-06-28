# Auto-resume runner: launch a headless Claude turn in the worktree to run the next queued experiment.
# Enable via Windows Task Scheduler (see docs/CRUX_PROMPT.md / chat for the schtasks command).
# CAVEATS: spends tokens autonomously; only useful after the 5-hour rate window has reset; needs the
# `claude` CLI on PATH + authenticated; GEE OAuth cached creds are used (local). Commits to claude/temporal-ds.

$ErrorActionPreference = "Continue"
$wt = "E:\research_projects\sccd-claude"
Set-Location $wt
$log = Join-Path $wt ".auto_resume.log"
"`n===== auto_resume run (Task Scheduler) =====" | Out-File -Append $log

$prompt = @'
You are resuming autonomously on branch claude/temporal-ds in this worktree. Steps:
1. Read docs/CONTINUITY.md (master state), docs/EXPERIMENT_QUEUE.md, and the memory files.
2. Pick the TOP unchecked item in docs/EXPERIMENT_QUEUE.md. Run it end-to-end: implement, run, debug once if
   it errors. Follow the methodology + discipline in CONTINUITY (pre-register the construction + falsifier;
   ALWAYS report the trivial/standard null e.g. SAM/CVA/IR-MAD; never overclaim; be blunt about negatives).
3. Write a short report to docs/experiment_reports/, commit results to claude/temporal-ds with a clear message,
   then check the item off in docs/EXPERIMENT_QUEUE.md (with a one-line result) and commit that.
4. If the queue has no unchecked items, OR you appear rate-limited, do nothing further and exit cleanly.
Do exactly one queue item per run. Be rigorous; prefer a smart workaround over stopping.
'@

# Adjust the flag/path if your CLI differs: --dangerously-skip-permissions == bypass permission prompts (autonomy).
claude -p $prompt --dangerously-skip-permissions *>> $log 2>&1
"exit code: $LASTEXITCODE" | Out-File -Append $log
