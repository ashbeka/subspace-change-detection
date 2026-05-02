# Next Step Decision Memo

Date: 2026-05-02  
Basis: `docs/ADVERSARIAL_REENTRY_AUDIT.md` and repo files only.

## Decision

Run one no-output Phase 2 forward-pass smoke test. Do not code, train, clean outputs, read more papers, or distill old chats first.

Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase D - Minimal Reproduction Plan", "Step 0 - Environment And Forward-Pass Smoke".

## 1. Five Most Important Findings

1. The implemented project is OSCD binary change segmentation with unsupervised Sentinel-2 prior channels, not end-to-end disaster damage segmentation.  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 1" and "Bottom Line"; `phase2/train/train_oscd_seg.py`.

2. Phase 1 implements DS and classical change-score methods; Phase 2 implements supervised OSCD segmentation with raw bands plus optional priors.  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 1", "Phase A - 4"; `phase1/eval/run_oscd_eval.py`, `phase2/data/oscd_seg_dataset.py`.

3. The safest thesis framing is "interpretable unsupervised multispectral change priors for supervised Sentinel-2 change segmentation."  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 7", "Phase C", "Bottom Line".

4. Reported Phase 2 results are not reproduced by the audit and are mostly single-seed.  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C", rows on Phase 2 E1/raw+DS and "Phase B", C26.

5. Some narrative docs are stale or contradictory, especially around damage framing, priors-only experiments, Siamese status, and old Phase 2 bugs.  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C", rows on damage framing, priors-only checks, Siamese, and `CODEBASE_AUDIT.md`.

## 2. Five Biggest Risks Or Unverified Assumptions

1. Damage claims are currently unsafe because the integrated xBD/xBD-S12 damage pipeline is missing.  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C", damage-mapping row; `phase2/data/damage_dataset_adapter.py`.

2. The key Phase 2 result claim, raw+DS beating raw-only, is not freshly reproduced and is single-seed.  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C", Phase 2 E1 row; "Phase B", C26.

3. Phase 1 runtime comparisons are unsafe because timing is not isolated per method.  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 6" and "Phase C", runtime row; `phase1/eval/run_oscd_eval.py`.

4. CVA and pixel difference should not be treated as independent methods without checking the implementation.  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 4" and "Phase C", CVA row.

5. CUDA-based rerun assumptions are unsafe in the recorded environment; the rerun log showed CPU-only PyTorch.  
   Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase C", CUDA row; `PIPELINE_RERUN_LOG.txt`; `phase2/train/train_oscd_seg.py`.

## 3. Single Safest Next Action

Run the no-output environment/data/model forward-pass smoke test from the audit.

This is the safest action because it validates the minimum live chain: Python environment, imports, OSCD data layout, config parsing, dataset tensor creation, channel inference, model construction, and one CPU forward pass. It does not train, overwrite outputs, depend on CUDA, or modify code.

Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase D - Step 0"; `phase2/data/oscd_seg_dataset.py`; `phase2/train/train_oscd_seg.py`.

## 4. Why This Beats Coding, Papers, Or Old Chats Right Now

Coding now would risk fixing the wrong thing before proving the current pipeline can still load data and execute. Reading papers now would improve context but would not test whether the repo is alive. Distilling old chats now would amplify stale claims before the basic code/data contract is checked.

Recommendation source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase A - 6", "Phase C", and "Phase D - Step 0".

## 5. Exact Commands To Run Next

Run exactly this command block from the repo root:

```powershell
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

@'
from pathlib import Path
import yaml
import torch

from phase2.data.oscd_seg_dataset import OSCDSegmentationDataset
from phase2.train.train_oscd_seg import infer_channel_counts, build_model

cfg = yaml.safe_load(Path("phase2/configs/oscd_seg_baseline.yaml").read_text(encoding="utf-8"))

# Keep this as a liveness check, not a reported experiment.
cfg["dataset"]["patch_size"] = 128
cfg["dataset"]["patch_overlap"] = 0
cfg["model"]["base_channels"] = 8
cfg["model"]["depth"] = 2

ds = OSCDSegmentationDataset(
    oscd_root=Path("data/OSCD"),
    split="val",
    cfg=cfg,
    phase1_change_maps_root=Path("phase1/outputs/oscd_saved_priors_fast/oscd_change_maps"),
)
sample = ds[0]
counts = infer_channel_counts(cfg)
model = build_model(cfg, counts["total"]).eval()
with torch.no_grad():
    logits = model(sample["x"].unsqueeze(0))

print("dataset_len", len(ds))
print("x", tuple(sample["x"].shape), "y", tuple(sample["y"].shape), "valid", tuple(sample["valid"].shape))
print("channels", counts)
print("logits", tuple(logits.shape))
'@ | & $py -
```

Command source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase D - Step 0"; `phase2/configs/oscd_seg_baseline.yaml`.

## 6. Expected Output Files

None.

This command should only print terminal lines such as `dataset_len`, tensor shapes, channel counts, and `logits`.

Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase D - Step 0".

## 7. What Success Looks Like

Success means the command exits cleanly and prints:

- a positive `dataset_len`
- `x`, `y`, and `valid` tensor shapes
- channel counts
- a `logits` shape

Interpretation: the current environment, OSCD loader, baseline config, Phase 2 dataset, and U-Net forward path are alive.

Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase D - Step 0"; `phase2/data/oscd_seg_dataset.py`; `phase2/models/unet2d.py`.

## 8. What Failure Looks Like

Failure means one of these:

- import error: environment or requirements issue
- file-not-found error: OSCD layout or missing local data
- raster read error: data/dependency issue
- shape/channel error: config/model/dataset contract issue
- runtime device error: environment issue

Interpretation: fix the failed minimum contract before trusting old results or doing any new research work.

Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase D - Step 0"; `requirements.txt`; `phase2/train/train_oscd_seg.py`.

## 9. What Not To Do Yet

Do not run training, sweeps, cleanup scripts, CUDA commands, xBD/xBD-S12 damage work, or old-output deletion. Do not start coding fixes unless the smoke test fails and the failure is clearly localized.

Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Phase D - What Not To Run Yet"; `phase2/scripts/run_phase2_core_experiments.ps1`; `phase2/scripts/run_phase2_sweep.ps1`; `clean_house.ps1`.

## 10. Old Chat Distillation Timing

Old chat distillation should happen after this smoke test, not before.

Reason: the audit found that confident narrative sources can be stale or contradictory. A live code/data check should anchor the next discussion before old chats are summarized.

Source: `docs/ADVERSARIAL_REENTRY_AUDIT.md`, "Evidence Policy", "Phase C", and "Bottom Line".

## Recommended Next Action

Run only the no-output Phase 2 forward-pass smoke test in section 5.
