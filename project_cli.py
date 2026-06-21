from __future__ import annotations

import argparse
import fnmatch
import importlib
import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent

PHASE1_CONFIGS = {
    "default": "phase1/configs/oscd_default.yaml",
    "fast": "phase1/configs/oscd_priors_fast.yaml",
    "fast-eig": "phase1/configs/oscd_priors_fast_eig.yaml",
    "canonical": "phase1/configs/oscd_priors_canonical.yaml",
    "eig": "phase1/configs/oscd_variant_eig.yaml",
    "geodesic": "phase1/configs/oscd_geodesic_priors.yaml",
    "multisenge": "phase1/configs/multisenge_default.yaml",
    "multisenge-geodesic": "phase1/configs/multisenge_temporal_geodesic.yaml",
}

PHASE2_PRIOR_ROOTS = {
    "legacy-residual": "phase1/outputs/oscd_saved_priors_fast/oscd_change_maps",
    "legacy-eig": "phase1/outputs/oscd_saved_priors_fast_eig/oscd_change_maps",
    "full": "phase1/outputs/oscd_saved_full/oscd_change_maps",
    "geodesic": "phase1/outputs/oscd_geodesic_priors/oscd_change_maps",
}

PHASE2_CONFIGS = {
    "e0-raw": "phase2/configs/oscd/core/E0_raw_unet.yaml",
    "e1-ds": "phase2/configs/oscd/core/E1_raw_ds_unet.yaml",
    "e1b-ds-cross": "phase2/configs/oscd/core/E1b_raw_ds_cross_unet.yaml",
    "e2-pca": "phase2/configs/oscd/core/E2_raw_pca_unet.yaml",
    "e3-ds-pca": "phase2/configs/oscd/core/E3_raw_ds_pca_unet.yaml",
    "s0-siamese": "phase2/configs/oscd/core/S0_raw_siamese.yaml",
    "e0-resnet": "phase2/configs/oscd/extended/E0_raw_resnet.yaml",
    "e3-resnet": "phase2/configs/oscd/extended/E3_raw_ds_pca_resnet.yaml",
    "e3-fusion": "phase2/configs/oscd/extended/E3_raw_ds_pca_fusion.yaml",
    "e4-pixel": "phase2/configs/oscd/extended/E4_raw_pixel_unet.yaml",
    "e5-celik": "phase2/configs/oscd/extended/E5_raw_celik_unet.yaml",
    "e6-irmad": "phase2/configs/oscd/extended/E6_raw_irmad_unet.yaml",
    "g0-geodesic": "phase2/configs/oscd/extended/G0_raw_geodesic_unet.yaml",
    "xbd-template": "phase2/configs/future_damage/xbd_template.yaml",
}

KEEP_PHASE1 = [
    "oscd_saved_priors_fast",
    "oscd_saved_priors_fast_eig",
    "oscd_saved_full",
    "oscd_priors_canonical_*",
    "venus_kds_*",
    "sensei_message_assets_*",
]

KEEP_PHASE2 = [
    "runs_gpu_150ep_20251215_233309",
    "smoke_e0_e1_20260503_040723",
    "sweep_core_150ep_repro_v5_20260503_052422",
    "sweep_overnight_full_eig_3seeds_150ep_v2",
]


@dataclass(frozen=True)
class CheckResult:
    label: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class CommandInfo:
    name: str
    category: str
    purpose: str
    example: list[str]
    note: str = ""


COMMANDS: list[CommandInfo] = [
    CommandInfo("doctor", "readiness", "Check environment, imports, data roots, and key files.", ["doctor", "--strict"]),
    CommandInfo("test", "readiness", "Run formula/code correctness tests.", ["test"]),
    CommandInfo("list", "inventory", "List commands, configs, scripts, outputs, or everything.", ["list", "all"]),
    CommandInfo("phase1-oscd-priors", "phase1", "Generate OSCD unsupervised prior maps.", ["phase1-oscd-priors", "--config", "canonical"]),
    CommandInfo("phase1-oscd-geodesic", "phase1", "Generate OSCD local geodesic prior maps.", ["phase1-oscd-geodesic"]),
    CommandInfo("phase1-subspace-inspect", "phase1", "Inspect OSCD PCA/DS construction for one city.", ["phase1-subspace-inspect", "--city", "beirut"]),
    CommandInfo("phase1-spatial-subspace-compare", "phase1", "Compare global/window/patch/band-image DS score maps on one OSCD city.", ["phase1-spatial-subspace-compare", "--city", "beirut"]),
    CommandInfo("phase1-spatial-subspace-sweep", "phase1", "Run spatial DS comparison across multiple cities/configs and aggregate results.", ["phase1-spatial-subspace-sweep", "--cities", "core5"]),
    CommandInfo("phase1-score-calibration", "phase1", "Fit score-map changed-area fractions on OSCD train cities and evaluate them on test cities.", ["phase1-score-calibration", "--sweep-root", "<saved_npy_sweep>"]),
    CommandInfo("phase1-subspace-audit", "phase1", "Compatibility alias for phase1-subspace-inspect.", ["phase1-subspace-audit", "--city", "beirut"]),
    CommandInfo("phase1-venus", "phase1", "Run the Venus DS/KDS/KGDS diagnostic demo.", ["phase1-venus"]),
    CommandInfo("phase1-viz-examples", "visualization", "Draw OSCD pre/post/GT/diff/DS example figures.", ["phase1-viz-examples", "--cities", "test"]),
    CommandInfo("phase1-viz-method-grid", "visualization", "Draw saved Phase 1 prior maps side by side.", ["phase1-viz-method-grid", "--prior-root", "full"]),
    CommandInfo("phase1-multisenge-manifest", "multisenge", "Build a small MultiSenGE temporal manifest.", ["phase1-multisenge-manifest"]),
    CommandInfo("phase1-multisenge-viz", "multisenge", "Run exploratory MultiSenGE DS visualization.", ["phase1-multisenge-viz"]),
    CommandInfo("phase1-multisenge-temporal-dynamics", "multisenge", "Measure first/second DS and geodesic components over MultiSenGE dates.", ["phase1-multisenge-temporal-dynamics", "phase1-multisenge-geodesic"]),
    CommandInfo("phase1-multisenge-temporal-injections", "multisenge", "Compare temporal DS with radiometric/registration nuisance controls.", ["phase1-multisenge-temporal-injections"]),
    CommandInfo("phase1-multisenge-order-aware-interventions", "multisenge", "Stress-test unordered, difference, and trajectory subspaces on real temporal backgrounds.", ["phase1-multisenge-order-aware-interventions"]),
    CommandInfo("phase1-multiscale-order-aware-interventions", "multisenge", "Localize controlled temporal events with cell-wise unordered and trajectory subspaces.", ["phase1-multiscale-order-aware-interventions"]),
    CommandInfo("phase1-registered-sequence-dynamics", "multisenge", "Analyze first/second DS on a dated registered TIFF sequence.", ["phase1-registered-sequence-dynamics"]),
    CommandInfo("phase1-multiscale-sequence-dynamics", "multisenge", "Analyze local/multiscale temporal DS on a dated registered TIFF sequence.", ["phase1-multiscale-sequence-dynamics"]),
    CommandInfo("phase1-temporal-context-ds", "multisenge", "Compare backward/forward temporal-context DS on a dated registered TIFF sequence.", ["phase1-temporal-context-ds"]),
    CommandInfo("phase1-temporal-context-injections", "multisenge", "Stress-test temporal-context DS with controlled change and nuisance injections.", ["phase1-temporal-context-injections"]),
    CommandInfo("phase1-temporal-registration-curve", "multisenge", "Measure temporal-context sensitivity to subpixel registration error and low-frequency controls.", ["phase1-temporal-registration-curve"]),
    CommandInfo("phase1-seasonal-regime-study", "multisenge", "Stress-test seasonal observation DS on abrupt, gradual, and nuisance trajectories.", ["phase1-seasonal-regime-study"]),
    CommandInfo("phase1-rtw-invariance-gate", "multisenge", "Test RTW timing/tempo invariance against marginal-matched seasonal-shape changes.", ["phase1-rtw-invariance-gate"]),
    CommandInfo("phase1-breizhcrops-download", "temporal", "Download and verify official BreizhCrops 2017 L2A geographic partitions.", ["phase1-breizhcrops-download", "--regions", "frh01,frh04"]),
    CommandInfo("phase1-rtw-breizhcrops-transfer", "temporal", "Test frozen RTW on geographically held-out natural crop-phenology labels and killer controls.", ["phase1-rtw-breizhcrops-transfer"]),
    CommandInfo("phase1-irrigation-data-feasibility", "multisenge", "Check IrrMapper and Sentinel-2 temporal coverage before data acquisition.", ["phase1-irrigation-data-feasibility"]),
    CommandInfo("phase2-train", "phase2", "Train one OSCD segmentation config.", ["phase2-train", "--config", "e0-raw"]),
    CommandInfo("phase2-eval", "phase2", "Evaluate one trained checkpoint.", ["phase2-eval", "--config", "e0-raw", "--checkpoint", "<best.ckpt>"]),
    CommandInfo("phase2-sweep", "phase2", "Run a controlled train/eval sweep.", ["phase2-sweep", "--preset", "core", "--progress-bars"]),
    CommandInfo("phase2-watch", "phase2", "Watch a running or latest sweep.", ["phase2-watch"]),
    CommandInfo("phase2-analyze", "phase2", "Analyze completed sweep artifacts.", ["phase2-analyze", "--sweep-root", "<sweep_dir>"]),
    CommandInfo("phase2-compare", "phase2", "Compare explicit eval summary files.", ["phase2-compare", "--summaries", "<a.json>", "<b.json>", "--tags", "A", "B", "--output", "<out.csv>"]),
    CommandInfo("phase2-viz-predictions", "visualization", "Visualize segmentation predictions for one checkpoint.", ["phase2-viz-predictions", "--config", "e1-ds", "--checkpoint", "<best.ckpt>"]),
    CommandInfo("phase2-viz-combined", "visualization", "Visualize Phase 1 priors plus Phase 2 predictions.", ["phase2-viz-combined", "--config", "e3-ds-pca", "--checkpoint", "<best.ckpt>"]),
    CommandInfo("cleanup", "maintenance", "Preview or apply generated-output cleanup.", ["cleanup"], "Preview by default; --apply is destructive."),
    CommandInfo("run-python-script", "escape hatch", "Run any repo Python script through the project venv.", ["run-python-script", "phase1/scripts/inspect_oscd_subspace.py"]),
    CommandInfo("run-module", "escape hatch", "Run any Python module through the project venv.", ["run-module", "phase2.eval.analyze_sweep_results"]),
    CommandInfo("run-ps1", "escape hatch", "Run any repo PowerShell script.", ["run-ps1", "phase2/scripts/run_phase2_sweep.ps1"]),
]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def venv_python() -> Path:
    win = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    unix = REPO_ROOT / ".venv" / "bin" / "python"
    if win.exists():
        return win
    if unix.exists():
        return unix
    return Path(sys.executable)


def powershell_exe(prefer_pwsh: bool = True) -> str:
    candidates = ["pwsh", "powershell"] if prefer_pwsh else ["powershell", "pwsh"]
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "powershell"


def print_command(cmd: list[str]) -> None:
    print("Command:")
    print("  " + subprocess.list2cmdline([str(x) for x in cmd]))


def print_cli_command(argv: list[str]) -> None:
    print("Project CLI command:")
    print("  " + subprocess.list2cmdline([str(venv_python()), "project_cli.py", *argv]))


def run_command(cmd: list[str], *, dry_run: bool = False) -> int:
    print_command(cmd)
    if dry_run:
        print("Dry run: command not executed.")
        return 0
    return subprocess.run([str(x) for x in cmd], cwd=REPO_ROOT, check=False).returncode


def check_path(label: str, path: str | Path, *, kind: str = "any") -> CheckResult:
    p = repo_path(path)
    if kind == "file":
        ok = p.is_file()
    elif kind == "dir":
        ok = p.is_dir()
    else:
        ok = p.exists()
    return CheckResult(label, ok, str(p))


def check_import(module: str) -> CheckResult:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:
        return CheckResult(f"import {module}", False, repr(exc))
    version = getattr(mod, "__version__", "")
    return CheckResult(f"import {module}", True, str(version) if version else "ok")


def latest_change_maps(pattern: str) -> str:
    roots = []
    out_root = repo_path("phase1/outputs")
    if out_root.exists():
        for path in out_root.glob(pattern):
            maps = path / "oscd_change_maps"
            if maps.exists():
                roots.append(maps)
    if not roots:
        return ""
    roots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(roots[0].relative_to(REPO_ROOT))


def phase1_config(value: str) -> str:
    return PHASE1_CONFIGS.get(value, value)


def phase2_config(value: str) -> str:
    return PHASE2_CONFIGS.get(value, value)


def resolve_prior_root(value: str) -> str:
    if value == "canonical-latest":
        latest = latest_change_maps("oscd_priors_canonical_*")
        if latest:
            return latest
    return PHASE2_PRIOR_ROOTS.get(value, value)


def print_command_catalog() -> None:
    print("Project CLI command catalog")
    categories = []
    for item in COMMANDS:
        if item.category not in categories:
            categories.append(item.category)
    for category in categories:
        print(f"\n[{category}]")
        for item in [x for x in COMMANDS if x.category == category]:
            example = subprocess.list2cmdline([str(x) for x in item.example])
            print(f"  {item.name:<28} {item.purpose}")
            print(f"  {'':<28} example: project_cli.py {example}")
            if item.note:
                print(f"  {'':<28} note: {item.note}")


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[CheckResult] = [
        CheckResult("repo root", True, str(REPO_ROOT)),
        CheckResult("python", True, f"{sys.executable} ({platform.python_version()})"),
        check_path("root venv python", venv_python(), kind="file"),
        check_path("OSCD data root", "data/OSCD", kind="dir"),
        check_path("Venus TPAMI data root", "data/venus_tpami2015", kind="dir"),
        check_path("phase1 configs", "phase1/configs", kind="dir"),
        check_path("phase2 configs", "phase2/configs", kind="dir"),
        check_path("organized Chrome bookmarks", "docs/source_records/final_organization_2026-06-12/chrome_bookmarks_organized_all_2026-06-20.html", kind="file"),
        check_path("retained non-public PDF", "references/reference_papers/MVA_2025_human_motion_analysis.pdf", kind="file"),
    ]
    for module in ["numpy", "yaml", "torch", "rasterio", "sklearn", "h5py", "phase1", "phase2"]:
        checks.append(check_import(module))

    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "none"
        checks.append(CheckResult("torch cuda", cuda, gpu))
    except Exception as exc:
        checks.append(CheckResult("torch cuda", False, repr(exc)))

    print("Project readiness check")
    failures = 0
    for check in checks:
        mark = "PASS" if check.ok else "FAIL"
        if not check.ok:
            failures += 1
        print(f"{mark:4} {check.label:<32} {check.detail}")

    if args.strict and failures:
        return 1
    return 0


def iter_files(root: str, suffixes: tuple[str, ...]) -> Iterable[Path]:
    base = repo_path(root)
    if not base.exists():
        return []
    return sorted((p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in suffixes), key=lambda p: str(p).lower())


def cmd_list(args: argparse.Namespace) -> int:
    if args.kind in {"commands", "all"}:
        print_command_catalog()
    if args.kind in {"configs", "all"}:
        print("Phase 1 config aliases:")
        for name, path in PHASE1_CONFIGS.items():
            print(f"  {name:<20} {path}")
        print("\nPhase 2 config aliases:")
        for name, path in PHASE2_CONFIGS.items():
            print(f"  {name:<20} {path}")
        print("\nPhase 2 configs:")
        for path in iter_files("phase2/configs", (".yaml", ".yml")):
            print(f"  {path.relative_to(REPO_ROOT)}")
    if args.kind in {"scripts", "all"}:
        print("\nPython scripts:")
        for root in ["phase1/scripts", "phase1/eval", "phase2/train", "phase2/eval", "phase2/viz"]:
            for path in iter_files(root, (".py",)):
                print(f"  {path.relative_to(REPO_ROOT)}")
        print("\nPowerShell scripts:")
        for path in iter_files("phase2/scripts", (".ps1",)):
            print(f"  {path.relative_to(REPO_ROOT)}")
    if args.kind in {"outputs", "all"}:
        print("\nKnown prior roots:")
        for name, path in PHASE2_PRIOR_ROOTS.items():
            exists = "exists" if repo_path(path).exists() else "missing"
            print(f"  {name:<18} {path} ({exists})")
        canonical_latest = latest_change_maps("oscd_priors_canonical_*")
        if canonical_latest:
            print(f"  {'canonical-latest':<18} {canonical_latest} (exists)")
        for root in ["phase1/outputs", "phase2/outputs"]:
            p = repo_path(root)
            print(f"\n{root}:")
            if not p.exists():
                print("  missing")
                continue
            for child in sorted((x for x in p.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
                print(f"  {child.name}")
    return 0


def cmd_phase1_oscd_priors(args: argparse.Namespace) -> int:
    config = phase1_config(args.config)
    out = args.output_dir
    if not out:
        stem = Path(config).stem
        out = f"phase1/outputs/{stem}_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.eval.run_oscd_eval",
        "--config",
        config,
        "--oscd_root",
        args.oscd_root,
        "--output_dir",
        out,
    ]
    if args.no_window:
        cmd.append("--no_window")
    if args.disable_celik:
        cmd.append("--disable_celik")
    if args.save_change_maps:
        cmd.append("--save_change_maps")
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_oscd_geodesic(args: argparse.Namespace) -> int:
    config = phase1_config(args.config)
    out = args.output_dir or f"phase1/outputs/oscd_geodesic_priors_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.eval.run_oscd_geodesic_priors",
        "--config",
        config,
        "--oscd_root",
        args.oscd_root,
        "--output_dir",
        out,
    ]
    if args.max_cities:
        cmd += ["--max_cities", str(args.max_cities)]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_subspace_inspect(args: argparse.Namespace) -> int:
    cmd = [
        str(venv_python()),
        "phase1/scripts/inspect_oscd_subspace.py",
        "--oscd_root",
        args.oscd_root,
        "--city",
        args.city,
        "--split",
        args.split,
        "--rank",
        str(args.rank),
        "--sample_pixels",
        str(args.sample_pixels),
        "--seed",
        str(args.seed),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_spatial_subspace_compare(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/oscd_spatial_subspace_compare_{args.city}_{timestamp()}"
    cmd = [
        str(venv_python()),
        "phase1/scripts/compare_oscd_spatial_subspaces.py",
        "--oscd_root",
        args.oscd_root,
        "--stats_path",
        args.stats_path,
        "--city",
        args.city,
        "--split",
        args.split,
        "--rank",
        str(args.rank),
        "--methods",
        args.methods,
        "--output_dir",
        out,
        "--seed",
        str(args.seed),
        "--max_fit_samples",
        str(args.max_fit_samples),
        "--score_chunk_size",
        str(args.score_chunk_size),
    ]
    if not args.save_npy:
        cmd.append("--no-save-npy")
    if args.save_band_attribution:
        cmd.append("--save-band-attribution")
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_spatial_subspace_sweep(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/oscd_spatial_subspace_sweep_{timestamp()}"
    cmd = [
        str(venv_python()),
        "phase1/scripts/sweep_oscd_spatial_subspaces.py",
        "--oscd_root",
        args.oscd_root,
        "--stats_path",
        args.stats_path,
        "--cities",
        args.cities,
        "--configs",
        args.configs,
        "--split",
        args.split,
        "--output_dir",
        out,
        "--seed",
        str(args.seed),
        "--max_fit_samples",
        str(args.max_fit_samples),
        "--score_chunk_size",
        str(args.score_chunk_size),
    ]
    if args.save_npy:
        cmd.append("--save-npy")
    else:
        cmd.append("--no-save-npy")
    if args.resume:
        cmd.append("--resume")
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.dry_run:
        cmd.append("--dry-run")
    return run_command(cmd, dry_run=False)


def cmd_phase1_score_calibration(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/oscd_split_safe_calibration_{timestamp()}"
    cmd = [
        str(venv_python()),
        "phase1/scripts/evaluate_oscd_split_calibration.py",
        "--sweep-root",
        args.sweep_root,
        "--oscd-root",
        args.oscd_root,
        "--output-dir",
        out,
        "--methods",
        args.methods,
        "--grid-size",
        str(args.grid_size),
        "--max-fraction",
        str(args.max_fraction),
        "--visual-cities",
        args.visual_cities,
    ]
    if args.config:
        cmd += ["--config", args.config]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_multisenge_manifest(args: argparse.Namespace) -> int:
    out = args.output_path or f"phase1/outputs/multisenge_manifest_{args.max_patches}p_{args.min_s2_dates}dates_{timestamp()}.json"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.build_multisenge_manifest",
        "--multisenge_root",
        args.multisenge_root,
        "--output_path",
        out,
        "--min_s2_dates",
        str(args.min_s2_dates),
        "--max_patches",
        str(args.max_patches),
        "--seed",
        str(args.seed),
    ]
    if args.include_s1:
        cmd.append("--include_s1")
    if args.no_require_ground_reference:
        cmd.append("--no_require_ground_reference")
    if args.patch_ids:
        cmd += ["--patch_ids", args.patch_ids]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_multisenge_viz(args: argparse.Namespace) -> int:
    config = phase1_config(args.config)
    out = args.output_dir or f"phase1/outputs/multisenge_viz_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.eval.run_multisenge_viz",
        "--config",
        config,
        "--multisenge_root",
        args.multisenge_root,
        "--output_dir",
        out,
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_multisenge_geodesic(args: argparse.Namespace) -> int:
    config = phase1_config(args.config)
    out = args.output_dir or f"phase1/outputs/multisenge_temporal_dynamics_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.eval.run_multisenge_temporal_geodesic",
        "--config",
        config,
        "--multisenge_root",
        args.multisenge_root,
        "--manifest",
        args.manifest,
        "--output_dir",
        out,
    ]
    if args.max_patches:
        cmd += ["--max_patches", str(args.max_patches)]
    if args.patch_ids:
        cmd += ["--patch_ids", args.patch_ids]
    if args.rank:
        cmd += ["--rank", str(args.rank)]
    if args.preprocessing:
        cmd += ["--preprocessing", args.preprocessing]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_multisenge_temporal_injections(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/multisenge_temporal_injections_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.evaluate_temporal_band_subspace_injections",
        "--multisenge_root",
        args.multisenge_root,
        "--manifest",
        args.manifest,
        "--output_dir",
        out,
        "--target_date",
        args.target_date,
        "--rank",
        str(args.rank),
        "--preprocessing",
        args.preprocessing,
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_multisenge_order_aware_interventions(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/multisenge_order_aware_interventions_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.evaluate_multisenge_order_aware_interventions",
        "--multisenge_root",
        args.multisenge_root,
        "--manifest",
        args.manifest,
        "--output_dir",
        out,
        "--crop_size",
        str(args.crop_size),
        "--repeats",
        str(args.repeats),
        "--ranks",
        args.ranks,
        "--representations",
        args.representations,
        "--preprocessing",
        args.preprocessing,
        "--bootstrap",
        str(args.bootstrap),
        "--seed",
        str(args.seed),
        "--max_patches",
        str(args.max_patches),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_multiscale_order_aware_interventions(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/multiscale_order_aware_interventions_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.evaluate_multiscale_order_aware_interventions",
        "--multisenge_root",
        args.multisenge_root,
        "--manifest",
        args.manifest,
        "--output_dir",
        out,
        "--crop_size",
        str(args.crop_size),
        "--repeats",
        str(args.repeats),
        "--grids",
        args.grids,
        "--representations",
        args.representations,
        "--rank",
        str(args.rank),
        "--preprocessing",
        args.preprocessing,
        "--spatial_smoothing_sigma",
        str(args.spatial_smoothing_sigma),
        "--bootstrap",
        str(args.bootstrap),
        "--seed",
        str(args.seed),
        "--max_patches",
        str(args.max_patches),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_registered_sequence_dynamics(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/registered_sequence_dynamics_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.analyze_registered_multispectral_sequence",
        "--sequence_dir",
        args.sequence_dir,
        "--glob",
        args.glob,
        "--output_dir",
        out,
        "--rank",
        str(args.rank),
        "--preprocessing",
        args.preprocessing,
        "--nodata",
        str(args.nodata),
        "--balanced_gap_ratio",
        str(args.balanced_gap_ratio),
        "--top_k",
        str(args.top_k),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_multiscale_sequence_dynamics(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/multiscale_sequence_dynamics_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.analyze_multiscale_temporal_subspaces",
        "--sequence_dir",
        args.sequence_dir,
        "--glob",
        args.glob,
        "--output_dir",
        out,
        "--grids",
        args.grids,
        "--rank",
        str(args.rank),
        "--preprocessing",
        args.preprocessing,
        "--nodata",
        str(args.nodata),
        "--min-valid-locations",
        str(args.min_valid_locations),
        "--reference-event-frames",
        args.reference_event_frames,
        "--evaluation-frame-count",
        str(args.evaluation_frame_count),
        "--figure-frame-count",
        str(args.figure_frame_count),
    ]
    if args.reference_map_dir:
        cmd += ["--reference-map-dir", args.reference_map_dir]
    if args.reference_crop:
        cmd += ["--reference-crop", args.reference_crop]
    if args.reference_lognfa_dir:
        cmd += [
            "--reference-lognfa-dir",
            args.reference_lognfa_dir,
            "--reference-logepsilon",
            str(args.reference_logepsilon),
        ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_temporal_context_ds(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/temporal_context_ds_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.analyze_bidirectional_temporal_context_ds",
        "--sequence_dir",
        args.sequence_dir,
        "--glob",
        args.glob,
        "--output_dir",
        out,
        "--context_sizes",
        args.context_sizes,
        "--ranks",
        args.ranks,
        "--factorizations",
        args.factorizations,
        "--preprocessing",
        args.preprocessing,
        "--nodata",
        str(args.nodata),
        "--scale_divisor",
        str(args.scale_divisor),
        "--figure_count",
        str(args.figure_count),
    ]
    if args.figure_config:
        cmd += ["--figure_config", args.figure_config]
    if args.reference_lognfa_dir:
        cmd += [
            "--reference_lognfa_dir",
            args.reference_lognfa_dir,
            "--reference_logepsilon",
            str(args.reference_logepsilon),
        ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_temporal_context_injections(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/temporal_context_injections_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.evaluate_temporal_context_injections",
        "--multisenge_root",
        args.multisenge_root,
        "--manifest",
        args.manifest,
        "--output_dir",
        out,
        "--target_date",
        args.target_date,
        "--context_size",
        str(args.context_size),
        "--rank",
        str(args.rank),
        "--factorization",
        args.factorization,
        "--preprocessing",
        args.preprocessing,
        "--repeats",
        str(args.repeats),
        "--max_patches",
        str(args.max_patches),
        "--seed",
        str(args.seed),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_temporal_registration_curve(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/temporal_registration_curve_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.evaluate_temporal_context_registration_curve",
        "--multisenge_root",
        args.multisenge_root,
        "--manifest",
        args.manifest,
        "--output_dir",
        out,
        "--target_date",
        args.target_date,
        "--context_size",
        str(args.context_size),
        "--rank",
        str(args.rank),
        "--preprocessing",
        args.preprocessing,
        "--shifts",
        args.shifts,
        "--strategies",
        args.strategies,
        "--local_strength",
        str(args.local_strength),
        "--window_size",
        str(args.window_size),
        "--repeats",
        str(args.repeats),
        "--max_patches",
        str(args.max_patches),
        "--seed",
        str(args.seed),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_seasonal_regime_study(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/seasonal_regime_subspace_study_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.evaluate_seasonal_regime_subspaces",
        "--output_dir",
        out,
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--ranks",
        args.ranks,
        "--preprocessing",
        args.preprocessing,
        "--representations",
        args.representations,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--noise",
        str(args.noise),
        "--bootstrap",
        str(args.bootstrap),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_spacenet7_temporal_subspaces(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/spacenet7_temporal_subspaces_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.evaluate_spacenet7_temporal_subspaces",
        "--aoi_root",
        args.aoi_root,
        "--output_dir",
        out,
        "--window",
        str(args.window),
        "--rank",
        str(args.rank),
        "--grids",
        args.grids,
        "--representations",
        args.representations,
        "--preprocessing",
        args.preprocessing,
        "--radiometric_normalization",
        args.radiometric_normalization,
        "--min_valid_pixels",
        str(args.min_valid_pixels),
        "--min_new_building_pixels",
        str(args.min_new_building_pixels),
        "--bootstrap",
        str(args.bootstrap),
        "--seed",
        str(args.seed),
        "--all_touched_labels" if args.all_touched_labels else "--no-all_touched_labels",
    ]
    if args.controls_only:
        cmd.append("--controls_only")
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_rtw_invariance_gate(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/multisenge_rtw_invariance_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.evaluate_multisenge_rtw_invariance",
        "--multisenge_root",
        args.multisenge_root,
        "--manifest",
        args.manifest,
        "--output_dir",
        out,
        "--crop_size",
        str(args.crop_size),
        "--repeats",
        str(args.repeats),
        "--development_patches",
        str(args.development_patches),
        "--max_patches",
        str(args.max_patches),
        "--subsequence_lengths",
        args.subsequence_lengths,
        "--n_samples",
        args.n_samples,
        "--ranks",
        args.ranks,
        "--preprocessing",
        args.preprocessing,
        "--rtw_replicates",
        str(args.rtw_replicates),
        "--screening_repeats",
        str(args.screening_repeats),
        "--screening_rtw_replicates",
        str(args.screening_rtw_replicates),
        "--finalists",
        str(args.finalists),
        "--bootstrap",
        str(args.bootstrap),
        "--seed",
        str(args.seed),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_rtw_breizhcrops_transfer(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/breizhcrops_rtw_transfer_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.evaluate_breizhcrops_rtw_transfer",
        "--data_root",
        args.data_root,
        "--output_dir",
        out,
        "--development_region",
        args.development_region,
        "--holdout_region",
        args.holdout_region,
        "--max_fields_per_class",
        str(args.max_fields_per_class),
        "--anchors_per_class",
        str(args.anchors_per_class),
        "--min_steps",
        str(args.min_steps),
        "--quality_threshold",
        str(args.quality_threshold),
        "--rtw_replicates",
        str(args.rtw_replicates),
        "--bootstrap",
        str(args.bootstrap),
        "--seed",
        str(args.seed),
    ]
    if args.search_rtw:
        cmd.extend([
            "--search_rtw",
            "--rtw_search_anchors_per_class",
            str(args.rtw_search_anchors_per_class),
            "--rtw_search_subsequence_lengths",
            args.rtw_search_subsequence_lengths,
            "--rtw_search_n_samples",
            args.rtw_search_n_samples,
            "--rtw_search_ranks",
            args.rtw_search_ranks,
            "--rtw_search_preprocessing",
            args.rtw_search_preprocessing,
            "--rtw_finalists",
            str(args.rtw_finalists),
        ])
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_breizhcrops_download(args: argparse.Namespace) -> int:
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.download_breizhcrops_l2a",
        "--data_root",
        args.data_root,
        "--regions",
        args.regions,
    ]
    if args.keep_archives:
        cmd.append("--keep_archives")
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_spacenet7_hybrid_analysis(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/spacenet7_hybrid_analysis_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.analyze_spacenet7_hybrid_validation",
        "--input_root",
        args.input_root,
        "--geometry_glob",
        args.geometry_glob,
        "--controls_glob",
        args.controls_glob,
        "--output_dir",
        out,
        "--min_new_building_pixels",
        str(args.min_new_building_pixels),
        "--bootstrap",
        str(args.bootstrap),
        "--seed",
        str(args.seed),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_irrigation_data_feasibility(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/irrigation_regime_data_feasibility_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.scripts.check_irrigation_regime_data",
        "--output_dir",
        out,
        f"--bbox={args.bbox}",
        "--start",
        args.start,
        "--end",
        args.end,
        "--max_cloud",
        str(args.max_cloud),
    ]
    if args.ee_project:
        cmd += ["--ee_project", args.ee_project]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_viz_examples(args: argparse.Namespace) -> int:
    config = phase1_config(args.config)
    out = args.output_dir or f"phase1/outputs/oscd_examples_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.eval.visualize_oscd_examples",
        "--config",
        config,
        "--oscd_root",
        args.oscd_root,
        "--output_dir",
        out,
        "--cities",
        args.cities,
    ]
    if args.metrics_json:
        cmd += ["--metrics_json", args.metrics_json]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_viz_method_grid(args: argparse.Namespace) -> int:
    config = phase1_config(args.config)
    prior_root = resolve_prior_root(args.prior_root)
    out = args.output_dir or f"phase1/outputs/oscd_method_grid_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase1.eval.visualize_oscd_methods_grid",
        "--config",
        config,
        "--oscd_root",
        args.oscd_root,
        "--change_maps_root",
        prior_root,
        "--output_dir",
        out,
        "--cities",
        args.cities,
        "--methods",
        args.methods,
        "--dpi",
        str(args.dpi),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase1_venus(args: argparse.Namespace) -> int:
    out = args.output_dir or f"phase1/outputs/venus_kds_audit_{timestamp()}"
    cmd = [
        str(venv_python()),
        "phase1/scripts/venus_kds_demo.py",
        "--venus_root",
        args.venus_root,
        "--output_dir",
        out,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--linear_rank",
        str(args.linear_rank),
        "--kds_subspace_rank",
        str(args.kds_subspace_rank),
        "--kds_rank",
        str(args.kds_rank),
        "--kgds_subspace_rank",
        str(args.kgds_subspace_rank),
        "--kgds_rank",
        str(args.kgds_rank),
        "--sigma2",
        str(args.sigma2),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase2_sweep(args: argparse.Namespace) -> int:
    tag = args.output_tag or f"{args.preset}_{args.epochs}ep_{timestamp()}"
    shell = powershell_exe(prefer_pwsh=True)
    cmd = [
        shell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "phase2/scripts/run_phase2_sweep.ps1",
        "-Preset",
        args.preset,
        "-Epochs",
        str(args.epochs),
        "-Seeds",
        args.seeds,
        "-OutputTag",
        tag,
        "-OutputRoot",
        args.output_root,
        "-Retention",
        args.retention,
    ]
    if args.batch_size:
        cmd += ["-BatchSize", str(args.batch_size)]
    if args.eval_batch_size:
        cmd += ["-EvalBatchSize", str(args.eval_batch_size)]
    if args.num_workers is not None:
        cmd += ["-NumWorkers", str(args.num_workers)]
    if args.val_every:
        cmd += ["-ValEvery", str(args.val_every)]
    if args.progress_bars:
        cmd.append("-ProgressBars")
    if args.no_cache_cities:
        cmd.append("-NoCacheCities")
    if args.cache_max_cities:
        cmd += ["-CacheMaxCities", str(args.cache_max_cities)]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase2_watch(args: argparse.Namespace) -> int:
    shell = powershell_exe(prefer_pwsh=False)
    cmd = [
        shell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "phase2/scripts/watch_phase2_sweep.ps1",
        "-RefreshSeconds",
        str(args.refresh_seconds),
    ]
    if args.run_root:
        cmd += ["-RunRoot", args.run_root]
    if args.once:
        cmd.append("-Once")
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase2_train(args: argparse.Namespace) -> int:
    config = phase2_config(args.config)
    out = args.output_dir or f"phase2/outputs/{Path(config).stem}_{timestamp()}"
    prior_root = resolve_prior_root(args.prior_root)
    cmd = [
        str(venv_python()),
        "-m",
        "phase2.train.train_oscd_seg",
        "--config",
        config,
        "--oscd_root",
        args.oscd_root,
        "--phase1_change_maps_root",
        prior_root,
        "--output_dir",
        out,
        "--device",
        args.device,
    ]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.max_epochs:
        cmd += ["--max_epochs", str(args.max_epochs)]
    if args.resume:
        cmd.append("--resume")
    if args.resume_ckpt:
        cmd += ["--resume_ckpt", args.resume_ckpt]
    if args.overwrite_output_dir:
        cmd.append("--overwrite_output_dir")
    if args.progress_style:
        cmd += ["--progress_style", args.progress_style]
    if args.progress_leave:
        cmd.append("--progress_leave")
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase2_eval(args: argparse.Namespace) -> int:
    config = phase2_config(args.config)
    prior_root = resolve_prior_root(args.prior_root)
    out = args.output_dir or f"{Path(args.checkpoint).parent}/eval_cli_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase2.eval.evaluate_oscd_seg",
        "--config",
        config,
        "--oscd_root",
        args.oscd_root,
        "--phase1_change_maps_root",
        prior_root,
        "--checkpoint",
        args.checkpoint,
        "--output_dir",
        out,
        "--device",
        args.device,
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase2_compare(args: argparse.Namespace) -> int:
    cmd = [
        str(venv_python()),
        "-m",
        "phase2.eval.compare_priors_effect",
        "--summaries",
        *args.summaries,
        "--output",
        args.output,
        "--tags",
        *args.tags,
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase2_viz_predictions(args: argparse.Namespace) -> int:
    config = phase2_config(args.config)
    prior_root = resolve_prior_root(args.prior_root)
    out = args.output_dir or f"{Path(args.checkpoint).parent}/figs_seg_cli_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase2.viz.viz_seg_predictions",
        "--config",
        config,
        "--oscd_root",
        args.oscd_root,
        "--phase1_change_maps_root",
        prior_root,
        "--checkpoint",
        args.checkpoint,
        "--output_dir",
        out,
        "--cities",
        args.cities,
        "--device",
        args.device,
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase2_viz_combined(args: argparse.Namespace) -> int:
    config = phase2_config(args.config)
    prior_root = resolve_prior_root(args.prior_root)
    out = args.output_dir or f"{Path(args.checkpoint).parent}/figs_combined_cli_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase2.viz.viz_oscd_combined",
        "--config",
        config,
        "--oscd_root",
        args.oscd_root,
        "--phase1_change_maps_root",
        prior_root,
        "--checkpoint",
        args.checkpoint,
        "--output_dir",
        out,
        "--cities",
        args.cities,
        "--device",
        args.device,
    ]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_phase2_analyze(args: argparse.Namespace) -> int:
    cmd = [str(venv_python()), "-m", "phase2.eval.analyze_sweep_results", "--sweep_root", args.sweep_root]
    if args.output_dir:
        cmd += ["--output_dir", args.output_dir]
    if args.baseline_tag:
        cmd += ["--baseline_tag", args.baseline_tag]
    if args.compare_tags:
        cmd += ["--compare_tags", args.compare_tags]
    return run_command(cmd, dry_run=args.dry_run)


def cmd_test(args: argparse.Namespace) -> int:
    cmd = [str(venv_python()), "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    return run_command(cmd, dry_run=args.dry_run)


def is_within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def cleanup_candidates(aggressive: bool) -> list[Path]:
    candidates: list[Path] = []
    for item in ["web_context_pack", "web_context_pack_min.zip", "web_reference_papers.zip"]:
        p = repo_path(item)
        if p.exists():
            candidates.append(p)

    phase2_out = repo_path("phase2/outputs")
    if phase2_out.exists():
        for child in phase2_out.iterdir():
            if not child.is_dir():
                continue
            if aggressive:
                if not matches_any(child.name, KEEP_PHASE2):
                    candidates.append(child)
            elif child.name.startswith(("_smoke_", "_bench_", "sweep__debug_")):
                candidates.append(child)

    phase1_out = repo_path("phase1/outputs")
    safe_phase1 = {
        "oscd_run",
        "oscd_run_validation",
        "multisenge_viz",
        "oscd_figs_all",
        "oscd_previews",
        "oscd_previews_eig",
    }
    if phase1_out.exists():
        for child in phase1_out.iterdir():
            if not child.is_dir():
                continue
            if aggressive:
                if not matches_any(child.name, KEEP_PHASE1):
                    candidates.append(child)
            elif child.name in safe_phase1:
                candidates.append(child)
    return candidates


def cmd_cleanup(args: argparse.Namespace) -> int:
    candidates = cleanup_candidates(args.aggressive)
    print("Cleanup mode:", "aggressive" if args.aggressive else "safe")
    print("Action:", "delete" if args.apply else "preview only")
    if args.aggressive:
        print("Phase 1 keep patterns:", ", ".join(KEEP_PHASE1))
        print("Phase 2 keep patterns:", ", ".join(KEEP_PHASE2))
    if not candidates:
        print("No cleanup candidates found.")
        return 0
    for path in candidates:
        print(("DELETE " if args.apply else "WOULD  ") + str(path.relative_to(REPO_ROOT)))
    if args.apply:
        for path in candidates:
            resolved = path.resolve()
            if not is_within_repo(resolved):
                raise RuntimeError(f"Refusing to delete outside repo: {resolved}")
            if resolved == REPO_ROOT:
                raise RuntimeError("Refusing to delete repo root")
            if resolved.is_dir():
                shutil.rmtree(resolved)
            elif resolved.exists():
                resolved.unlink()
    return 0


def cmd_run_python_script(args: argparse.Namespace) -> int:
    cmd = [str(venv_python()), args.script] + list(args.script_args)
    return run_command(cmd, dry_run=args.dry_run)


def cmd_run_module(args: argparse.Namespace) -> int:
    cmd = [str(venv_python()), "-m", args.module] + list(args.module_args)
    return run_command(cmd, dry_run=args.dry_run)


def cmd_run_ps1(args: argparse.Namespace) -> int:
    shell = powershell_exe(prefer_pwsh=True)
    cmd = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", args.script] + list(args.script_args)
    return run_command(cmd, dry_run=args.dry_run)


def prompt_choice(title: str, choices: list[tuple[str, str, list[str]]]) -> tuple[str, str, list[str]] | None:
    print(f"\n{title}")
    for index, (_, label, _) in enumerate(choices, start=1):
        print(f"  {index}. {label}")
    print("  0. Back / exit")
    raw = input(f"Choose 0-{len(choices)}: ").strip()
    if not raw:
        return None
    if raw == "0":
        return None
    if not raw.isdigit() or not (1 <= int(raw) <= len(choices)):
        print("Invalid choice.")
        return None
    return choices[int(raw) - 1]


def split_extra_args(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return shlex.split(raw, posix=(os.name != "nt"))


def show_subcommand_help(command: str) -> None:
    parser = build_parser()
    try:
        parser.parse_args([command, "--help"])
    except SystemExit:
        pass


def fill_placeholders(argv: list[str]) -> list[str] | None:
    filled = []
    for part in argv:
        if part.startswith("<") and part.endswith(">"):
            value = input(f"Value for {part}: ").strip()
            if not value:
                print("Required value omitted; returning to menu.")
                return None
            filled.append(value)
        else:
            filled.append(part)
    return filled


def interactive_execute(command_args: list[str]) -> int:
    current = list(command_args)
    while True:
        print("")
        print_cli_command(current)
        print("Actions: [r] run, [d] dry-run/preview, [a] append options, [h] help for this command, [b] back")
        raw = input("Action [b]: ").strip().lower()
        if raw in {"", "b", "back"}:
            return 0
        if raw in {"h", "help"}:
            show_subcommand_help(current[0])
            continue
        if raw in {"a", "append"}:
            extra = input("Append CLI options, for example: --epochs 5 --device cpu\n> ")
            current.extend(split_extra_args(extra))
            continue
        if raw in {"d", "dry", "dry-run", "preview"}:
            filled = fill_placeholders(current)
            if filled is None:
                continue
            current = filled
            if "--dry-run" not in current and current[0] not in {"doctor", "list", "cleanup"}:
                current = [*current, "--dry-run"]
            return main(current)
        if raw in {"r", "run", "y", "yes"}:
            filled = fill_placeholders(current)
            if filled is None:
                continue
            current = filled
            return main(current)
        print("Unknown action.")


def interactive_command_catalog() -> int:
    while True:
        categories = []
        seen = set()
        for item in COMMANDS:
            if item.category not in seen:
                categories.append(item.category)
                seen.add(item.category)
        choices = [(cat, cat, ["list", "commands"]) for cat in categories]
        choice = prompt_choice("Command categories", choices)
        if choice is None:
            return 0
        category = choice[0]
        items = [item for item in COMMANDS if item.category == category]
        subchoices = [(item.name, f"{item.name}: {item.purpose}", item.example) for item in items]
        selected = prompt_choice(f"{category} commands", subchoices)
        if selected is None:
            continue
        _, _, argv = selected
        code = interactive_execute(argv)
        if code:
            return code


def cmd_interactive(args: argparse.Namespace) -> int:
    print("Subspace Change Detection command center")
    print(f"Repo: {REPO_ROOT}")
    print("Use this as the main interactive surface. Every action prints the exact command before it runs.")
    while True:
        choices = [
            ("readiness", "Readiness: doctor, tests", ["list", "commands"]),
            ("inventory", "Inventory: commands, configs, scripts, outputs", ["list", "all"]),
            ("phase1", "Phase 1 / priors: OSCD, DS audit, Venus", ["list", "commands"]),
            ("multisenge", "MultiSenGE: manifest and temporal/geodesic exploration", ["list", "commands"]),
            ("phase2", "Phase 2: train, eval, sweep, analyze", ["list", "commands"]),
            ("visualization", "Visualization: Phase 1 and Phase 2 figures", ["list", "commands"]),
            ("maintenance", "Maintenance: cleanup preview/apply", ["cleanup"]),
            ("all", "Browse all command categories", ["list", "commands"]),
        ]
        choice = prompt_choice("Main menu", choices)
        if choice is None:
            return 0
        key, _, argv = choice
        if key == "inventory":
            code = interactive_execute(argv)
        elif key == "all":
            code = interactive_command_catalog()
        else:
            items = [item for item in COMMANDS if item.category == key]
            subchoices = [(item.name, f"{item.name}: {item.purpose}", item.example) for item in items]
            selected = prompt_choice(f"{key} commands", subchoices)
            if selected is None:
                continue
            _, _, selected_argv = selected
            code = interactive_execute(selected_argv)
        if code:
            return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Central CLI for subspace-based satellite change-detection research workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="Check environment, imports, data roots, and key files.")
    p.add_argument("--strict", action="store_true", help="Return non-zero if any check fails.")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("list", help="List command catalog, configs, scripts, or outputs.")
    p.add_argument("kind", choices=["commands", "configs", "scripts", "outputs", "all"], nargs="?", default="all")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("interactive", help="Open the interactive command center.")
    p.set_defaults(func=cmd_interactive)

    p = sub.add_parser("test", help="Run formula-verification tests.")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_test)

    p = sub.add_parser("phase1-oscd-priors", help="Generate/evaluate OSCD Phase 1 prior maps.")
    p.add_argument("--config", default="canonical", help="Config alias or path.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--output-dir", default="")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--disable-celik", action="store_true")
    p.add_argument("--save-change-maps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_oscd_priors)

    p = sub.add_parser("phase1-oscd-geodesic", help="Generate OSCD local geodesic prior maps.")
    p.add_argument("--config", default="geodesic", help="Phase 1 config alias or path.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--output-dir", default="")
    p.add_argument("--max-cities", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_oscd_geodesic)

    p = sub.add_parser("phase1-subspace-inspect", help="Inspect OSCD PCA/DS construction for one city.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--city", default="beirut")
    p.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    p.add_argument("--rank", type=int, default=6)
    p.add_argument("--sample-pixels", type=int, default=200000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_subspace_inspect)

    p = sub.add_parser("phase1-spatial-subspace-compare", help="Compare global/window/patch/band-image DS score maps on one OSCD city.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--stats-path", default="phase1/data/oscd_band_stats.json")
    p.add_argument("--city", default="beirut")
    p.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    p.add_argument("--rank", type=int, default=6)
    p.add_argument(
        "--methods",
        default="global_pixel,window128,patch3,patch5",
        help=(
            "Comma list, e.g. global_pixel,patch3,patch5,window128s64mean,"
            "band_image_ds,band_image_norm,band_image_ratio,band_image_residual. "
            "Legacy alias flatbands is still accepted."
        ),
    )
    p.add_argument("--output-dir", default="")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-fit-samples", type=int, default=20000)
    p.add_argument("--score-chunk-size", type=int, default=25000)
    p.add_argument("--save-band-attribution", action="store_true", help="For band-image DS methods, save per-band projected-energy attribution PNG/CSV.")
    p.add_argument("--save-npy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_spatial_subspace_compare)

    p = sub.add_parser("phase1-spatial-subspace-sweep", help="Run spatial DS comparison across multiple OSCD cities/configs.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--stats-path", default="phase1/data/oscd_band_stats.json")
    p.add_argument("--cities", default="core5", help="Comma list, 'core5' for beirut,dubai,lasvegas,milano,norcia, or 'all' for the 24 local OSCD cities.")
    p.add_argument(
        "--configs",
        default="rank4_core:4:global_pixel+patch3+patch5;rank6_spatial:6:global_pixel+window128+patch3+patch5;rank8_core:8:global_pixel+patch3+patch5",
        help="Semicolon-separated configs as name:rank:method+method. Method names can include band_image_ds variants; legacy flatbands is accepted.",
    )
    p.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    p.add_argument("--output-dir", default="")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-fit-samples", type=int, default=20000)
    p.add_argument("--score-chunk-size", type=int, default=25000)
    p.add_argument("--save-npy", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_spatial_subspace_sweep)

    p = sub.add_parser("phase1-score-calibration", help="Fit changed-area fractions on OSCD train cities and evaluate them unchanged on test cities.")
    p.add_argument("--sweep-root", required=True, help="Spatial sweep root generated with --save-npy.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--output-dir", default="")
    p.add_argument("--methods", default="raw_l2,pca_diff,band_image_norm,ir_mad,rank_fusion_pca_band_irmad")
    p.add_argument("--config", default="")
    p.add_argument("--grid-size", type=int, default=300)
    p.add_argument("--max-fraction", type=float, default=0.5)
    p.add_argument("--visual-cities", default="brasilia,dubai,norcia")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_score_calibration)

    p = sub.add_parser("phase1-subspace-audit", help="Compatibility alias for phase1-subspace-inspect.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--city", default="beirut")
    p.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    p.add_argument("--rank", type=int, default=6)
    p.add_argument("--sample-pixels", type=int, default=200000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_subspace_inspect)

    p = sub.add_parser("phase1-multisenge-manifest", help="Build a small MultiSenGE temporal manifest.")
    p.add_argument("--multisenge-root", default="data/MultiSenGE")
    p.add_argument("--output-path", default="")
    p.add_argument("--min-s2-dates", type=int, default=5)
    p.add_argument("--max-patches", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--include-s1", action="store_true")
    p.add_argument("--no-require-ground-reference", action="store_true")
    p.add_argument("--patch-ids", default="", help="Optional comma-separated patch IDs to select explicitly.")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_multisenge_manifest)

    p = sub.add_parser("phase1-multisenge-viz", help="Run exploratory MultiSenGE DS visualization.")
    p.add_argument("--config", default="multisenge", help="Phase 1 config alias or path.")
    p.add_argument("--multisenge-root", default="data/MultiSenGE/s2")
    p.add_argument("--output-dir", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_multisenge_viz)

    p = sub.add_parser(
        "phase1-multisenge-temporal-dynamics",
        aliases=["phase1-multisenge-geodesic"],
        help="Measure first/second DS and geodesic components over MultiSenGE dates.",
    )
    p.add_argument("--config", default="multisenge-geodesic", help="Phase 1 config alias or path.")
    p.add_argument("--multisenge-root", default="data/MultiSenGE")
    p.add_argument("--manifest", default="phase1/outputs/multisenge_manifest_50p_5dates.json")
    p.add_argument("--output-dir", default="")
    p.add_argument("--max-patches", type=int, default=0)
    p.add_argument("--patch-ids", default="", help="Optional comma-separated patch IDs from the manifest.")
    p.add_argument("--rank", type=int, default=0, help="Override the configured subspace rank.")
    p.add_argument("--preprocessing", default="", choices=["", "uncentered", "centered", "band_l2", "centered_band_l2"])
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_multisenge_geodesic)

    p = sub.add_parser(
        "phase1-multisenge-temporal-injections",
        help="Compare temporal DS response to local synthetic change and nuisance controls.",
    )
    p.add_argument("--multisenge-root", default="data/MultiSenGE")
    p.add_argument("--manifest", default="phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json")
    p.add_argument("--output-dir", default="")
    p.add_argument("--target-date", default="20200909")
    p.add_argument("--rank", type=int, default=6)
    p.add_argument("--preprocessing", default="centered_band_l2", choices=["uncentered", "centered", "band_l2", "centered_band_l2"])
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_multisenge_temporal_injections)

    p = sub.add_parser(
        "phase1-multisenge-order-aware-interventions",
        help="Stress-test unordered, difference, and trajectory subspaces on real MultiSenGE backgrounds.",
    )
    p.add_argument("--multisenge-root", default="data/MultiSenGE")
    p.add_argument("--manifest", default="phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json")
    p.add_argument("--output-dir", default="")
    p.add_argument("--crop-size", type=int, default=32)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--ranks", default="1,2")
    p.add_argument("--representations", default="unordered,difference,trajectory2,trajectory3")
    p.add_argument("--preprocessing", default="feature_centered_observation_l2")
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-patches", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_multisenge_order_aware_interventions)

    p = sub.add_parser(
        "phase1-multiscale-order-aware-interventions",
        help="Localize controlled temporal events with cell-wise unordered and trajectory subspaces.",
    )
    p.add_argument("--multisenge-root", default="data/MultiSenGE")
    p.add_argument("--manifest", default="phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json")
    p.add_argument("--output-dir", default="")
    p.add_argument("--crop-size", type=int, default=32)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--grids", default="2,4")
    p.add_argument("--representations", default="unordered,trajectory2")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--preprocessing", default="feature_centered_observation_l2")
    p.add_argument("--spatial-smoothing-sigma", type=float, default=0.0)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-patches", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_multiscale_order_aware_interventions)

    p = sub.add_parser(
        "phase1-registered-sequence-dynamics",
        help="Analyze first/second DS and spatial contributions on registered dated TIFFs.",
    )
    p.add_argument("--sequence-dir", required=True)
    p.add_argument("--glob", default="*.tif")
    p.add_argument("--output-dir", default="")
    p.add_argument("--rank", type=int, default=0, help="0 keeps the full band-image span.")
    p.add_argument("--preprocessing", default="centered", choices=["uncentered", "centered", "band_l2", "centered_band_l2"])
    p.add_argument("--nodata", type=float, default=0.0)
    p.add_argument("--balanced-gap-ratio", type=float, default=1.5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_registered_sequence_dynamics)

    p = sub.add_parser(
        "phase1-multiscale-sequence-dynamics",
        help="Analyze local/multiscale temporal DS on registered dated TIFFs.",
    )
    p.add_argument("--sequence-dir", required=True)
    p.add_argument("--glob", default="*.tif")
    p.add_argument("--output-dir", default="")
    p.add_argument("--grids", default="1,2,4,8")
    p.add_argument("--rank", type=int, default=0, help="0 keeps the full band-image span.")
    p.add_argument("--preprocessing", default="centered", choices=["uncentered", "centered", "band_l2", "centered_band_l2"])
    p.add_argument("--nodata", type=float, default=0.0)
    p.add_argument("--min-valid-locations", type=int, default=64)
    p.add_argument("--reference-event-frames", default="")
    p.add_argument("--evaluation-frame-count", type=int, default=0)
    p.add_argument("--figure-frame-count", type=int, default=5)
    p.add_argument("--reference-map-dir", default="")
    p.add_argument("--reference-crop", default="")
    p.add_argument("--reference-lognfa-dir", default="")
    p.add_argument("--reference-logepsilon", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_multiscale_sequence_dynamics)

    p = sub.add_parser(
        "phase1-temporal-context-ds",
        help="Compare backward/forward temporal-context DS on registered dated TIFFs.",
    )
    p.add_argument("--sequence-dir", required=True)
    p.add_argument("--glob", default="*.tif")
    p.add_argument("--output-dir", default="")
    p.add_argument("--context-sizes", default="3,5,7")
    p.add_argument("--ranks", default="1,2,3")
    p.add_argument("--factorizations", default="per_band,joint")
    p.add_argument(
        "--preprocessing",
        default="centered_column_l2",
        choices=["uncentered", "centered", "column_l2", "centered_column_l2"],
    )
    p.add_argument("--nodata", type=float, default=0.0)
    p.add_argument("--scale-divisor", type=float, default=1.0)
    p.add_argument("--reference-lognfa-dir", default="")
    p.add_argument("--reference-logepsilon", type=float, default=1.0)
    p.add_argument("--figure-config", default="")
    p.add_argument("--figure-count", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_temporal_context_ds)

    p = sub.add_parser(
        "phase1-temporal-context-injections",
        help="Stress-test temporal-context DS with controlled MultiSenGE interventions.",
    )
    p.add_argument("--multisenge-root", default="data/MultiSenGE")
    p.add_argument("--manifest", default="phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json")
    p.add_argument("--output-dir", default="")
    p.add_argument("--target-date", default="20200909")
    p.add_argument("--context-size", type=int, default=3)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--factorization", default="per_band", choices=["per_band", "joint"])
    p.add_argument("--preprocessing", default="centered_column_l2")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--max-patches", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_temporal_context_injections)

    p = sub.add_parser(
        "phase1-temporal-registration-curve",
        help="Measure temporal-context sensitivity to subpixel translation and scale-space controls.",
    )
    p.add_argument("--multisenge-root", default="data/MultiSenGE")
    p.add_argument("--manifest", default="phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json")
    p.add_argument("--output-dir", default="")
    p.add_argument("--target-date", default="20200909")
    p.add_argument("--context-size", type=int, default=3)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preprocessing", default="centered_column_l2")
    p.add_argument("--shifts", default="0.25,0.5,1,2")
    p.add_argument("--strategies", default="native,gaussian1,gaussian2,pool2,pool4,phase_align")
    p.add_argument("--local-strength", type=float, default=0.25)
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-patches", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_temporal_registration_curve)

    p = sub.add_parser(
        "phase1-seasonal-regime-study",
        help="Stress-test seasonal observation DS on abrupt, gradual, and nuisance trajectories.",
    )
    p.add_argument("--output-dir", default="")
    p.add_argument("--repeats", type=int, default=80)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ranks", default="1,2,4,8")
    p.add_argument(
        "--preprocessing",
        default="uncentered,feature_centered,feature_centered_observation_l2",
    )
    p.add_argument(
        "--representations",
        default="unordered",
        help="Comma-separated unordered,difference,trajectory2,trajectory3,...",
    )
    p.add_argument("--height", type=int, default=16)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--noise", type=float, default=0.008)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_seasonal_regime_study)

    p = sub.add_parser(
        "phase1-rtw-invariance-gate",
        help="Test RTW timing/tempo invariance against marginal-matched seasonal-shape changes.",
    )
    p.add_argument("--multisenge-root", default="data/MultiSenGE")
    p.add_argument("--manifest", default="phase1/outputs/multisenge_manifest_32TLT_5patches_23dates.json")
    p.add_argument("--output-dir", default="")
    p.add_argument("--crop-size", type=int, default=32)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--development-patches", type=int, default=3)
    p.add_argument("--max-patches", type=int, default=5)
    p.add_argument("--subsequence-lengths", default="4,8,12")
    p.add_argument("--n-samples", default="64,256")
    p.add_argument("--ranks", default="2,5")
    p.add_argument(
        "--preprocessing",
        default="raw,reference_zscore,per_sequence_zscore",
    )
    p.add_argument("--rtw-replicates", type=int, default=3)
    p.add_argument("--screening-repeats", type=int, default=2)
    p.add_argument("--screening-rtw-replicates", type=int, default=1)
    p.add_argument("--finalists", type=int, default=6)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_rtw_invariance_gate)

    p = sub.add_parser(
        "phase1-breizhcrops-download",
        help="Download and verify official BreizhCrops 2017 L2A partitions.",
    )
    p.add_argument("--data-root", default="data/BreizhCrops")
    p.add_argument("--regions", default="frh01,frh02,frh03,frh04")
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_breizhcrops_download)

    p = sub.add_parser(
        "phase1-rtw-breizhcrops-transfer",
        help="Test frozen RTW on geographically held-out natural crop-phenology labels and killer controls.",
    )
    p.add_argument("--data-root", default="data/BreizhCrops")
    p.add_argument("--output-dir", default="")
    p.add_argument("--development-region", default="frh01")
    p.add_argument("--holdout-region", default="frh04")
    p.add_argument("--max-fields-per-class", type=int, default=80)
    p.add_argument("--anchors-per-class", type=int, default=40)
    p.add_argument("--min-steps", type=int, default=12)
    p.add_argument("--quality-threshold", type=float, default=0.5)
    p.add_argument("--rtw-replicates", type=int, default=3)
    p.add_argument("--search-rtw", action="store_true")
    p.add_argument("--rtw-search-anchors-per-class", type=int, default=8)
    p.add_argument("--rtw-search-subsequence-lengths", default="2,4,8,12")
    p.add_argument("--rtw-search-n-samples", default="32,64,128")
    p.add_argument("--rtw-search-ranks", default="2,5")
    p.add_argument(
        "--rtw-search-preprocessing", default="raw,per_sequence_zscore"
    )
    p.add_argument("--rtw-finalists", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2718)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_rtw_breizhcrops_transfer)

    p = sub.add_parser(
        "phase1-spacenet7-temporal-subspaces",
        help="Evaluate rolling first/second DS maps on labeled monthly SpaceNet 7 construction.",
    )
    p.add_argument(
        "--aoi-root",
        default="data/SpaceNet7_sample/L15-1203E-1203N_4815_3378_13",
    )
    p.add_argument("--output-dir", default="")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--grids", default="8,16")
    p.add_argument("--representations", default="unordered,difference,trajectory2")
    p.add_argument("--preprocessing", default="feature_centered")
    p.add_argument(
        "--radiometric-normalization",
        default="none",
        choices=["none", "per_date_channel_standardize", "per_date_channel_quantile"],
    )
    p.add_argument("--min-valid-pixels", type=int, default=16)
    p.add_argument("--min-new-building-pixels", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--all-touched-labels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--controls-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_spacenet7_temporal_subspaces)

    p = sub.add_parser(
        "phase1-spacenet7-hybrid-analysis",
        help="Evaluate the frozen geometry+radiometry rank fusion across SpaceNet 7 AOIs.",
    )
    p.add_argument("--input-root", default="phase1/outputs")
    p.add_argument("--geometry-glob", required=True)
    p.add_argument("--controls-glob", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--min-new-building-pixels", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=90210)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_spacenet7_hybrid_analysis)

    p = sub.add_parser(
        "phase1-irrigation-data-feasibility",
        help="Check IrrMapper and Sentinel-2 temporal coverage before data acquisition.",
    )
    p.add_argument("--output-dir", default="")
    p.add_argument("--ee-project", default="")
    p.add_argument("--bbox", default="-112.60,45.20,-112.40,45.35")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--max-cloud", type=float, default=60.0)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_irrigation_data_feasibility)

    p = sub.add_parser("phase1-viz-examples", help="Visualize OSCD examples with raw diffs and DS maps.")
    p.add_argument("--config", default="canonical", help="Phase 1 config alias or path.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--output-dir", default="")
    p.add_argument("--cities", default="test")
    p.add_argument("--metrics-json", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_viz_examples)

    p = sub.add_parser("phase1-viz-method-grid", help="Visualize saved Phase 1 method maps side by side.")
    p.add_argument("--config", default="default", help="Phase 1 config alias or path.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--prior-root", default="full", help="Known prior alias or path.")
    p.add_argument("--output-dir", default="")
    p.add_argument("--cities", default="test")
    p.add_argument("--methods", default="pixel_diff,pca_diff,ds_projection,ds_cross_residual,celik,ir_mad")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_viz_method_grid)

    p = sub.add_parser("phase1-venus", help="Run Venus DS/KDS/KGDS diagnostic demo.")
    p.add_argument("--venus-root", default="data/venus_tpami2015")
    p.add_argument("--output-dir", default="")
    p.add_argument("--width", type=int, default=63)
    p.add_argument("--height", type=int, default=48)
    p.add_argument("--linear-rank", type=int, default=100)
    p.add_argument("--kds-subspace-rank", type=int, default=100)
    p.add_argument("--kds-rank", type=int, default=100)
    p.add_argument("--kgds-subspace-rank", type=int, default=150)
    p.add_argument("--kgds-rank", type=int, default=300)
    p.add_argument("--sigma2", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_venus)

    p = sub.add_parser("phase2-sweep", help="Run wrapped Phase 2 sweep script.")
    p.add_argument("--preset", choices=["core", "full", "full+eig"], default="core")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seeds", default="1234,1235,1236")
    p.add_argument("--output-tag", default="")
    p.add_argument("--output-root", default="phase2/outputs")
    p.add_argument("--retention", choices=["full", "compact", "metrics_only"], default="full")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--val-every", type=int, default=0)
    p.add_argument("--progress-bars", action="store_true")
    p.add_argument("--no-cache-cities", action="store_true")
    p.add_argument("--cache-max-cities", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase2_sweep)

    p = sub.add_parser("phase2-watch", help="Watch latest or specified Phase 2 sweep.")
    p.add_argument("--run-root", default="")
    p.add_argument("--refresh-seconds", type=int, default=15)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase2_watch)

    p = sub.add_parser("phase2-train", help="Run one Phase 2 training config.")
    p.add_argument("--config", default="e0-raw", help="Phase 2 config alias or path.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--prior-root", default="legacy-residual", help="Known prior alias or path.")
    p.add_argument("--output-dir", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-ckpt", default="")
    p.add_argument("--overwrite-output-dir", action="store_true")
    p.add_argument("--progress-style", choices=["tqdm", "none"], default="tqdm")
    p.add_argument("--progress-leave", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase2_train)

    p = sub.add_parser("phase2-eval", help="Evaluate one Phase 2 checkpoint.")
    p.add_argument("--config", required=True, help="Phase 2 config alias or path.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--prior-root", default="legacy-residual", help="Known prior alias or path.")
    p.add_argument("--output-dir", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase2_eval)

    p = sub.add_parser("phase2-compare", help="Compare explicit Phase 2 evaluation summaries.")
    p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase2_compare)

    p = sub.add_parser("phase2-viz-predictions", help="Visualize segmentation predictions for one checkpoint.")
    p.add_argument("--config", required=True, help="Phase 2 config alias or path.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--prior-root", default="legacy-residual", help="Known prior alias or path.")
    p.add_argument("--output-dir", default="")
    p.add_argument("--cities", default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase2_viz_predictions)

    p = sub.add_parser("phase2-viz-combined", help="Visualize Phase 1 priors plus Phase 2 predictions.")
    p.add_argument("--config", required=True, help="Phase 2 config alias or path.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--prior-root", default="legacy-residual", help="Known prior alias or path.")
    p.add_argument("--output-dir", default="")
    p.add_argument("--cities", default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase2_viz_combined)

    p = sub.add_parser("phase2-analyze", help="Analyze completed Phase 2 sweep artifacts.")
    p.add_argument("--sweep-root", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--baseline-tag", default="")
    p.add_argument("--compare-tags", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase2_analyze)

    p = sub.add_parser("cleanup", help="Preview or apply generated-output cleanup.")
    p.add_argument("--aggressive", action="store_true", help="Delete output dirs except keep patterns.")
    p.add_argument("--apply", action="store_true", help="Actually delete. Default is preview only.")
    p.set_defaults(func=cmd_cleanup)

    p = sub.add_parser("run-python-script", help="Run any repo Python script through the project venv.")
    p.add_argument("script")
    p.add_argument("script_args", nargs=argparse.REMAINDER)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run_python_script)

    p = sub.add_parser("run-module", help="Run any Python module through the project venv.")
    p.add_argument("module")
    p.add_argument("module_args", nargs=argparse.REMAINDER)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run_module)

    p = sub.add_parser("run-ps1", help="Run any repo PowerShell script.")
    p.add_argument("script")
    p.add_argument("script_args", nargs=argparse.REMAINDER)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run_ps1)

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["interactive"]
    parser = build_parser()
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
