from __future__ import annotations

import argparse
import fnmatch
import importlib
import os
import platform
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


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[CheckResult] = [
        CheckResult("repo root", True, str(REPO_ROOT)),
        CheckResult("python", True, f"{sys.executable} ({platform.python_version()})"),
        check_path("root venv python", venv_python(), kind="file"),
        check_path("OSCD data root", "data/OSCD", kind="dir"),
        check_path("Venus TPAMI data root", "data/venus_tpami2015", kind="dir"),
        check_path("phase1 configs", "phase1/configs", kind="dir"),
        check_path("phase2 configs", "phase2/configs", kind="dir"),
        check_path("organized Chrome bookmarks", "docs/source_records/chrome_bookmarks_organized_research_2026-06-07.html", kind="file"),
        check_path("retained non-public PDF", "references/reference_papers/MVA_2025_human_motion_analysis.pdf", kind="file"),
    ]
    for module in ["numpy", "yaml", "torch", "rasterio", "sklearn", "phase1", "phase2"]:
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
    if args.kind in {"configs", "all"}:
        print("Phase 1 config aliases:")
        for name, path in PHASE1_CONFIGS.items():
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
        for root in ["phase1/outputs", "phase2/outputs"]:
            p = repo_path(root)
            print(f"\n{root}:")
            if not p.exists():
                print("  missing")
                continue
            for child in sorted((x for x in p.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
                print(f"  {child.name}")
    return 0


def phase1_config(value: str) -> str:
    return PHASE1_CONFIGS.get(value, value)


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


def cmd_phase1_subspace_audit(args: argparse.Namespace) -> int:
    cmd = [
        str(venv_python()),
        "phase1/scripts/audit_oscd_subspace.py",
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
    out = args.output_dir or f"phase2/outputs/{Path(args.config).stem}_{timestamp()}"
    prior_root = PHASE2_PRIOR_ROOTS.get(args.prior_root, args.prior_root)
    cmd = [
        str(venv_python()),
        "-m",
        "phase2.train.train_oscd_seg",
        "--config",
        args.config,
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
    prior_root = PHASE2_PRIOR_ROOTS.get(args.prior_root, args.prior_root)
    out = args.output_dir or f"{Path(args.checkpoint).parent}/eval_cli_{timestamp()}"
    cmd = [
        str(venv_python()),
        "-m",
        "phase2.eval.evaluate_oscd_seg",
        "--config",
        args.config,
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


def prompt_choice(title: str, choices: list[tuple[str, str, list[str]]]) -> tuple[str, str, list[str]]:
    print(f"\n{title}")
    for index, (_, label, _) in enumerate(choices, start=1):
        print(f"  {index}. {label}")
    raw = input(f"Choose 1-{len(choices)} [1]: ").strip()
    if not raw:
        return choices[0]
    if not raw.isdigit() or not (1 <= int(raw) <= len(choices)):
        print("Invalid choice; using default.")
        return choices[0]
    return choices[int(raw) - 1]


def cmd_interactive(args: argparse.Namespace) -> int:
    choices = [
        ("doctor", "Check environment readiness", ["doctor"]),
        ("list", "List configs, scripts, and outputs", ["list", "all"]),
        ("tests", "Run formula-verification tests", ["test"]),
        ("subspace", "Audit OSCD subspace implementation on Beirut", ["phase1-subspace-audit"]),
        ("venus", "Run Venus KDS/KGDS demo", ["phase1-venus"]),
        ("priors", "Generate canonical OSCD prior maps", ["phase1-oscd-priors", "--config", "canonical"]),
        ("sweep", "Run Phase 2 core sweep with tqdm bars", ["phase2-sweep", "--progress-bars"]),
        ("watch", "Watch latest Phase 2 sweep once", ["phase2-watch", "--once"]),
        ("cleanup", "Preview cleanup candidates", ["cleanup"]),
    ]
    _, _, command_args = prompt_choice("Project CLI interactive menu", choices)
    print("")
    return main(command_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Central CLI for DS damage segmentation research workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="Check environment, imports, data roots, and key files.")
    p.add_argument("--strict", action="store_true", help="Return non-zero if any check fails.")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("list", help="List configs, scripts, or outputs.")
    p.add_argument("kind", choices=["configs", "scripts", "outputs", "all"], nargs="?", default="all")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("interactive", help="Open a small interactive launcher menu.")
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

    p = sub.add_parser("phase1-subspace-audit", help="Audit OSCD PCA/DS construction for one city.")
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--city", default="beirut")
    p.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    p.add_argument("--rank", type=int, default=6)
    p.add_argument("--sample-pixels", type=int, default=200000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase1_subspace_audit)

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
    p.add_argument("--config", default="phase2/configs/oscd/core/E0_raw_unet.yaml")
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
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--oscd-root", default="data/OSCD")
    p.add_argument("--prior-root", default="legacy-residual", help="Known prior alias or path.")
    p.add_argument("--output-dir", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_phase2_eval)

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
    parser = build_parser()
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
