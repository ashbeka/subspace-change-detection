"""
Run a controlled multi-city, multi-configuration OSCD spatial DS sweep.

Source/provenance:
- Wraps `compare_oscd_spatial_subspaces.py`, which implements canonical
  Fukui/Maki DS with three Sentinel-2 sample definitions: global pixels,
  local windows, flattened local patches, and band-image vectors.
- This file adds experiment management only: repeated city/config runs,
  aggregate CSV summaries, and a human-readable report.

Allowed claim:
- The sweep can show whether a spatial DS construction is stable across a small
  OSCD city set and rank choices. It cannot prove a general satellite-specific
  subspace method without broader datasets and follow-up baselines.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CITIES = "beirut,dubai,lasvegas,milano,norcia"
ALL_CITIES = (
    "abudhabi,aguasclaras,beihai,beirut,bercy,bordeaux,brasilia,chongqing,"
    "cupertino,dubai,hongkong,lasvegas,milano,montpellier,mumbai,nantes,"
    "norcia,paris,pisa,rennes,rio,saclay_e,saclay_w,valencia"
)
DEFAULT_CONFIGS = (
    "rank4_core:4:global_pixel+patch3+patch5;"
    "rank6_spatial:6:global_pixel+window128+patch3+patch5;"
    "rank8_core:8:global_pixel+patch3+patch5"
)
SUMMARY_METRICS = ["auroc", "average_precision", "best_f1", "best_iou", "otsu_f1", "otsu_iou", "raw_l2_corr", "runtime_sec"]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    rank: int
    methods: str


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_configs(value: str) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for raw in [x.strip() for x in value.split(";") if x.strip()]:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Each config must look like name:rank:method+method. "
                f"Got {raw!r} from {value!r}."
            )
        name, rank_text, methods_text = parts
        methods = ",".join([x.strip() for x in methods_text.split("+") if x.strip()])
        if not name or not methods:
            raise ValueError(f"Invalid config entry: {raw!r}")
        configs.append(SweepConfig(name=name, rank=int(rank_text), methods=methods))
    if not configs:
        raise ValueError("No sweep configs were provided.")
    return configs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run OSCD spatial DS comparison across multiple cities and configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--oscd_root", "--oscd-root", default="data/OSCD")
    ap.add_argument("--stats_path", "--stats-path", default="phase1/data/oscd_band_stats.json")
    ap.add_argument("--cities", default=DEFAULT_CITIES, help="Comma-separated city list, 'core5', or 'all'.")
    ap.add_argument(
        "--configs",
        default=DEFAULT_CONFIGS,
        help=(
            "Semicolon-separated configs as name:rank:method+method. "
            "Example: rank6:6:global_pixel+window128+patch3+patch5"
        ),
    )
    ap.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    ap.add_argument("--output_dir", "--output-dir", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_fit_samples", "--max-fit-samples", type=int, default=20000)
    ap.add_argument("--score_chunk_size", "--score-chunk-size", type=int, default=25000)
    ap.add_argument("--celik_downsample_max_side", "--celik-downsample-max-side", type=int, default=384)
    ap.add_argument("--celik_feature_mode", "--celik-feature-mode", choices=("spectral_norm", "multiband_patch"), default="spectral_norm")
    ap.add_argument("--celik_max_fit_samples", "--celik-max-fit-samples", type=int, default=20000)
    ap.add_argument("--ir_mad_iters", "--ir-mad-iters", type=int, default=10)
    ap.add_argument("--ir_mad_downsample_max_pixels", "--ir-mad-downsample-max-pixels", type=int, default=200000)
    ap.add_argument("--save_npy", "--save-npy", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--resume", action="store_true", help="Skip a city/config run if its metrics CSV already exists.")
    ap.add_argument("--continue_on_error", "--continue-on-error", action="store_true")
    ap.add_argument("--dry_run", "--dry-run", action="store_true")
    return ap.parse_args()


def run_and_log(cmd: list[str], log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = subprocess.list2cmdline(cmd)
    print(command_text, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {command_text}\n")
        if dry_run:
            log.write("Dry run: command not executed.\n")
            return 0
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return proc.wait()


def read_metrics_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def float_or_nan(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean(values: Iterable[float]) -> float:
    vals = [x for x in values if x == x]
    return sum(vals) / len(vals) if vals else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_config_method(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["config"]), str(row["rank"]), str(row["method"]))].append(row)

    summary: list[dict[str, object]] = []
    for (config, rank, method), items in sorted(grouped.items()):
        out: dict[str, object] = {
            "config": config,
            "rank": int(rank),
            "method": method,
            "n_city_runs": len(items),
        }
        for metric in SUMMARY_METRICS:
            out[f"mean_{metric}"] = mean(float_or_nan(x.get(metric)) for x in items)
        summary.append(out)
    return summary


def paired_method_comparisons(
    rows: list[dict[str, object]],
    metrics: tuple[str, ...] = ("average_precision", "auroc", "best_f1", "otsu_f1"),
    seed: int = 1234,
) -> list[dict[str, object]]:
    """Compare methods on matched city/config observations.

    The bootstrap interval estimates uncertainty in the mean paired delta. The
    Wilcoxon test supplies a distribution-free paired significance check; both
    are descriptive evidence, not proof of cross-dataset generalization.
    """
    by_config: dict[str, dict[str, dict[str, dict[str, object]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_config[str(row["config"])][str(row["method"])][str(row["city"])] = row

    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for config, by_method in sorted(by_config.items()):
        methods = sorted(by_method)
        for metric in metrics:
            for index, method_a in enumerate(methods):
                for method_b in methods[index + 1 :]:
                    common = sorted(set(by_method[method_a]) & set(by_method[method_b]))
                    paired: list[tuple[str, float]] = []
                    for city in common:
                        a = float_or_nan(by_method[method_a][city].get(metric))
                        b = float_or_nan(by_method[method_b][city].get(metric))
                        if a == a and b == b:
                            paired.append((city, a - b))
                    if not paired:
                        continue
                    deltas = np.asarray([item[1] for item in paired], dtype=np.float64)
                    tolerance = 1e-12
                    nonzero = deltas[np.abs(deltas) > tolerance]
                    if nonzero.size:
                        p_value = float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)
                        ranks = rankdata(np.abs(nonzero))
                        rank_sum = float(np.sum(ranks))
                        rank_biserial = float((np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0])) / rank_sum)
                    else:
                        p_value = 1.0
                        rank_biserial = 0.0

                    if deltas.size > 1:
                        samples = rng.choice(deltas, size=(10000, deltas.size), replace=True).mean(axis=1)
                        ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
                    else:
                        ci_low = ci_high = deltas[0]
                    output.append(
                        {
                            "config": config,
                            "metric": metric,
                            "method_a": method_a,
                            "method_b": method_b,
                            "n_cities": int(deltas.size),
                            "mean_delta_a_minus_b": float(np.mean(deltas)),
                            "median_delta_a_minus_b": float(np.median(deltas)),
                            "bootstrap_95ci_low": float(ci_low),
                            "bootstrap_95ci_high": float(ci_high),
                            "wins_a": int(np.sum(deltas > tolerance)),
                            "ties": int(np.sum(np.abs(deltas) <= tolerance)),
                            "wins_b": int(np.sum(deltas < -tolerance)),
                            "wilcoxon_p_two_sided": p_value,
                            "matched_rank_biserial": rank_biserial,
                        }
                    )
    return output


def best_rows(rows: list[dict[str, object]], metric: str, *, ds_only: bool = False) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if ds_only and str(row.get("family")) == "baseline":
            continue
        grouped[(str(row["city"]), str(row["config"]))].append(row)
    winners: list[dict[str, object]] = []
    for (city, config), items in sorted(grouped.items()):
        valid = [x for x in items if float_or_nan(x.get(metric)) == float_or_nan(x.get(metric))]
        if not valid:
            continue
        winner = max(valid, key=lambda x: float_or_nan(x.get(metric)))
        winners.append(
            {
                "city": city,
                "config": config,
                "rank": winner.get("rank"),
                "metric": metric,
                "method": winner.get("method"),
                "family": winner.get("family"),
                "value": winner.get(metric),
                "run_dir": winner.get("run_dir"),
            }
        )
    return winners


def fmt(value: object, ndigits: int = 4) -> str:
    x = float_or_nan(value)
    return "nan" if x != x else f"{x:.{ndigits}f}"


def top_summary_lines(summary: list[dict[str, object]], metric: str, n: int = 8) -> list[str]:
    sorted_rows = sorted(summary, key=lambda x: float_or_nan(x.get(f"mean_{metric}")), reverse=True)
    lines = []
    for row in sorted_rows[:n]:
        lines.append(
            f"| {row['config']} | {row['rank']} | {row['method']} | "
            f"{fmt(row.get('mean_auroc'))} | {fmt(row.get('mean_average_precision'))} | "
            f"{fmt(row.get('mean_otsu_f1'))} | {fmt(row.get('mean_best_f1'))} |"
        )
    return lines


def write_report(
    path: Path,
    args: argparse.Namespace,
    configs: list[SweepConfig],
    cities: list[str],
    summary: list[dict[str, object]],
    best_ap: list[dict[str, object]],
    best_ap_ds: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("# OSCD Spatial Subspace Sweep Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "Test whether spatial support improves Difference Subspace (DS) score maps on OSCD, "
        "rather than relying on one-city Beirut evidence."
    )
    lines.append("")
    lines.append("## 2. Sweep Design")
    lines.append("")
    lines.append(f"- Cities: `{', '.join(cities)}`")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append(f"- Save NPY maps: `{args.save_npy}`")
    lines.append("- Configurations:")
    for cfg in configs:
        lines.append(f"  - `{cfg.name}`: rank `{cfg.rank}`, methods `{cfg.methods}`")
    lines.append("")
    lines.append("## 3. Mean Results By Config And Method")
    lines.append("")
    lines.append("| config | rank | method | mean AUROC | mean AP | mean Otsu F1 | mean best F1 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    lines.extend(top_summary_lines(summary, "average_precision", n=999))
    lines.append("")
    lines.append("## 4. Best Method Per City/Config By Average Precision")
    lines.append("")
    lines.append("| city | config | rank | method | family | AP |")
    lines.append("|---|---|---:|---|---|---:|")
    for row in best_ap:
        lines.append(
            f"| {row['city']} | {row['config']} | {row['rank']} | {row['method']} | "
            f"{row['family']} | {fmt(row['value'])} |"
        )
    lines.append("")
    lines.append("## 5. Best DS-Family Method Per City/Config By Average Precision")
    lines.append("")
    lines.append("| city | config | rank | method | AP |")
    lines.append("|---|---|---:|---|---:|")
    for row in best_ap_ds:
        lines.append(f"| {row['city']} | {row['config']} | {row['rank']} | {row['method']} | {fmt(row['value'])} |")
    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")
    pca_rows = [x for x in summary if x.get("method") == "pca_diff"]
    ds_rows = [x for x in summary if x.get("method") not in {"raw_l2", "pca_diff"}]
    best_pca = max(pca_rows, key=lambda x: float_or_nan(x.get("mean_average_precision"))) if pca_rows else None
    best_ds = max(ds_rows, key=lambda x: float_or_nan(x.get("mean_average_precision"))) if ds_rows else None
    if best_pca and best_ds:
        ds_ap = float_or_nan(best_ds.get("mean_average_precision"))
        pca_ap = float_or_nan(best_pca.get("mean_average_precision"))
        lines.append(
            f"- Best DS-family mean AP: `{best_ds['method']}` in `{best_ds['config']}` "
            f"with AP `{fmt(ds_ap)}`."
        )
        lines.append(
            f"- Best PCA-diff mean AP: `{best_pca['config']}` with AP `{fmt(pca_ap)}`."
        )
        if ds_ap > pca_ap:
            lines.append("- Preliminary read: a DS-family spatial construction beats PCA-diff on mean AP in this sweep. Verify maps and per-city stability before claiming improvement.")
        else:
            lines.append("- Preliminary read: PCA-diff remains stronger on mean AP. Spatial DS may still be useful for interpretation or specific cities, but it is not yet a performance winner.")
    lines.append("- Treat this as Phase 1 score-map evidence only. It does not prove Phase 2 neural segmentation improvement.")
    lines.append("- Inspect `comparison_grid.png` files for qualitative failure modes before deciding whether to train U-Net with any new prior.")
    lines.append("")
    lines.append("## 7. Paired Average-Precision Pressure Tests")
    lines.append("")
    lines.append("Positive delta means method A is better. Confidence intervals and Wilcoxon p-values are paired over cities.")
    lines.append("")
    lines.append("| config | method A | method B | n | mean delta | 95% CI | W/T/L | p |")
    lines.append("|---|---|---|---:|---:|---|---:|---:|")
    pressure_methods = {"band_image_norm", "pca_diff", "raw_l2", "celik_pca_kmeans", "ir_mad"}
    for row in pairwise:
        if row["metric"] != "average_precision":
            continue
        if row["method_a"] not in pressure_methods or row["method_b"] not in pressure_methods:
            continue
        lines.append(
            f"| {row['config']} | {row['method_a']} | {row['method_b']} | {row['n_cities']} | "
            f"{fmt(row['mean_delta_a_minus_b'])} | [{fmt(row['bootstrap_95ci_low'])}, {fmt(row['bootstrap_95ci_high'])}] | "
            f"{row['wins_a']}/{row['ties']}/{row['wins_b']} | {fmt(row['wilcoxon_p_two_sided'])} |"
        )
    lines.append("")
    if failures:
        lines.append("## 8. Failures")
        lines.append("")
        for failure in failures:
            lines.append(f"- `{failure['city']}` / `{failure['config']}` returned code `{failure['returncode']}`. See `{failure['log']}`.")
        lines.append("")
    lines.append("## 9. Output Files")
    lines.append("")
    lines.append(f"- Full row summary: `{path.parent / 'sweep_metrics_all.csv'}`")
    lines.append(f"- Mean summary: `{path.parent / 'sweep_summary_by_config_method.csv'}`")
    lines.append(f"- Best AP rows: `{path.parent / 'sweep_best_ap_by_city_config.csv'}`")
    lines.append(f"- Best DS AP rows: `{path.parent / 'sweep_best_ds_ap_by_city_config.csv'}`")
    lines.append(f"- Paired city-wise comparisons: `{path.parent / 'sweep_pairwise_method_comparisons.csv'}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    city_key = args.cities.lower()
    if city_key == "core5":
        cities = parse_csv_list(DEFAULT_CITIES)
    elif city_key == "all":
        cities = parse_csv_list(ALL_CITIES)
    else:
        cities = parse_csv_list(args.cities)
    configs = parse_configs(args.configs)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "phase1" / "outputs" / f"oscd_spatial_subspace_sweep_{timestamp()}"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    logs_dir = output_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cities": cities,
        "configs": [asdict(x) for x in configs],
        "args": vars(args),
        "output_dir": str(output_dir),
    }
    (output_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    total = len(cities) * len(configs)
    completed = 0

    for city in cities:
        for cfg in configs:
            completed += 1
            run_name = f"{cfg.name}__{city}"
            run_dir = runs_dir / run_name
            metrics_path = run_dir / "spatial_subspace_metrics.csv"
            log_path = logs_dir / f"{run_name}.log"
            print()
            print(f"[{completed}/{total}] {city} / {cfg.name} rank={cfg.rank} methods={cfg.methods}", flush=True)
            if args.resume and metrics_path.exists():
                print(f"Skipping existing run: {metrics_path}", flush=True)
                returncode = 0
            else:
                cmd = [
                    sys.executable,
                    "phase1/scripts/compare_oscd_spatial_subspaces.py",
                    "--oscd_root",
                    args.oscd_root,
                    "--stats_path",
                    args.stats_path,
                    "--city",
                    city,
                    "--split",
                    args.split,
                    "--rank",
                    str(cfg.rank),
                    "--methods",
                    cfg.methods,
                    "--output_dir",
                    str(run_dir),
                    "--seed",
                    str(args.seed),
                    "--max_fit_samples",
                    str(args.max_fit_samples),
                    "--score_chunk_size",
                    str(args.score_chunk_size),
                    "--celik_downsample_max_side",
                    str(args.celik_downsample_max_side),
                    "--celik_feature_mode",
                    str(args.celik_feature_mode),
                    "--celik_max_fit_samples",
                    str(args.celik_max_fit_samples),
                    "--ir_mad_iters",
                    str(args.ir_mad_iters),
                    "--ir_mad_downsample_max_pixels",
                    str(args.ir_mad_downsample_max_pixels),
                ]
                if not args.save_npy:
                    cmd.append("--no-save-npy")
                start = time.perf_counter()
                returncode = run_and_log(cmd, log_path, dry_run=args.dry_run)
                elapsed = time.perf_counter() - start
                print(f"Run finished with code {returncode} in {elapsed:.1f}s", flush=True)

            if returncode != 0:
                failures.append({"city": city, "config": cfg.name, "returncode": returncode, "log": str(log_path)})
                if not args.continue_on_error:
                    raise SystemExit(returncode)
                continue
            if args.dry_run:
                continue
            if not metrics_path.exists():
                failures.append({"city": city, "config": cfg.name, "returncode": "missing_metrics", "log": str(log_path)})
                if not args.continue_on_error:
                    raise RuntimeError(f"Missing metrics after successful run: {metrics_path}")
                continue
            for row in read_metrics_csv(metrics_path):
                enriched: dict[str, object] = {
                    "city": city,
                    "config": cfg.name,
                    "rank": cfg.rank,
                    "methods_requested": cfg.methods,
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    **row,
                }
                all_rows.append(enriched)

    if args.dry_run:
        print(f"Dry run complete. Manifest: {output_dir / 'sweep_manifest.json'}")
        return

    write_csv(output_dir / "sweep_metrics_all.csv", all_rows)
    summary = aggregate_by_config_method(all_rows)
    write_csv(output_dir / "sweep_summary_by_config_method.csv", summary)
    best_ap = best_rows(all_rows, "average_precision", ds_only=False)
    best_ap_ds = best_rows(all_rows, "average_precision", ds_only=True)
    pairwise = paired_method_comparisons(all_rows, seed=args.seed)
    write_csv(output_dir / "sweep_best_ap_by_city_config.csv", best_ap)
    write_csv(output_dir / "sweep_best_ds_ap_by_city_config.csv", best_ap_ds)
    write_csv(output_dir / "sweep_pairwise_method_comparisons.csv", pairwise)
    write_report(output_dir / "sweep_report.md", args, configs, cities, summary, best_ap, best_ap_ds, pairwise, failures)

    print()
    print(f"Wrote sweep output: {output_dir}")
    print(f"Wrote report: {output_dir / 'sweep_report.md'}")
    if failures:
        print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
