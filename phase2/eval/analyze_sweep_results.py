"""Analyze completed Phase 2 OSCD sweep artifacts.

This script reads existing per-run eval JSON files. It does not rerun model
inference, so threshold analysis is limited to proxy evidence from fixed
threshold metrics versus ranking metrics (AUROC/PR-AUC).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


SUMMARY_METRICS = ("mean_iou", "mean_f1", "mean_auroc", "mean_pr_auc")
CITY_METRICS = ("iou", "f1", "auroc", "pr_auc", "precision", "recall", "acc")
COUNT_METRICS = ("tp", "fp", "fn", "tn")
DEFAULT_COMPARE_TAGS = ("E1_raw_ds", "E1b_raw_ds_cross", "E2_raw_pca", "E3_raw_ds_pca", "S0_siamese")


def parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--sweep_root", type=Path, required=True, help="Completed sweep output folder.")
  ap.add_argument(
    "--output_dir",
    type=Path,
    default=None,
    help="Where to write analysis CSVs. Defaults to sweep_root/analysis_<timestamp>.",
  )
  ap.add_argument("--baseline_tag", default="E0_raw_unet")
  ap.add_argument("--compare_tags", default=",".join(DEFAULT_COMPARE_TAGS))
  return ap.parse_args()


def fnum(value: object) -> float:
  if value is None:
    return float("nan")
  try:
    return float(value)
  except (TypeError, ValueError):
    return float("nan")


def safe_mean(values: Iterable[float]) -> float:
  xs = [x for x in values if not math.isnan(x)]
  return mean(xs) if xs else float("nan")


def safe_stdev(values: Iterable[float]) -> float:
  xs = [x for x in values if not math.isnan(x)]
  return stdev(xs) if len(xs) >= 2 else 0.0


def fmt(value: float) -> str:
  if math.isnan(value):
    return "nan"
  return f"{value:.6f}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def read_eval_runs(sweep_root: Path) -> tuple[list[dict], list[dict]]:
  run_re = re.compile(r"^(?P<tag>.+)__seed(?P<seed>\d+)$")
  summary_rows: list[dict] = []
  city_rows: list[dict] = []

  for run_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
    match = run_re.match(run_dir.name)
    if not match:
      continue
    tag = match.group("tag")
    seed = int(match.group("seed"))
    eval_path = run_dir / "eval" / "oscd_seg_eval_results.json"
    meta_path = run_dir / "eval" / "run_metadata.json"
    if not eval_path.exists():
      continue

    with eval_path.open("r", encoding="utf-8") as f:
      results = json.load(f)

    model = ""
    experiment_tag = ""
    git_hash = ""
    if meta_path.exists():
      with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
      model = str(meta.get("config", {}).get("model", {}).get("type", ""))
      experiment_tag = str(meta.get("config", {}).get("experiment_tag", ""))
      git_hash = str(meta.get("git_hash", ""))

    for split, split_res in results.items():
      summary = split_res.get("summary", {})
      srow = {
        "split": split,
        "tag": tag,
        "seed": seed,
        "run": run_dir.name,
        "model": model,
        "experiment_tag": experiment_tag,
        "git_hash": git_hash,
        "threshold": fnum(summary.get("threshold")),
      }
      for metric in SUMMARY_METRICS:
        srow[metric] = fnum(summary.get(metric))
      summary_rows.append(srow)

      for city, stats in split_res.get("per_city", {}).items():
        crow = {
          "split": split,
          "tag": tag,
          "seed": seed,
          "run": run_dir.name,
          "city": city,
        }
        for metric in CITY_METRICS + COUNT_METRICS:
          crow[metric] = fnum(stats.get(metric))
        city_rows.append(crow)

  return summary_rows, city_rows


def group_summary_by_tag(summary_rows: list[dict]) -> list[dict]:
  grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
  for row in summary_rows:
    grouped[(row["split"], row["tag"])].append(row)

  out: list[dict] = []
  for (split, tag), rows in sorted(grouped.items()):
    orow = {"split": split, "tag": tag, "n": len(rows)}
    for metric in SUMMARY_METRICS:
      vals = [fnum(r.get(metric)) for r in rows]
      orow[f"{metric}_mean"] = safe_mean(vals)
      orow[f"{metric}_std"] = safe_stdev(vals)
      orow[f"{metric}_min"] = min([v for v in vals if not math.isnan(v)], default=float("nan"))
      orow[f"{metric}_max"] = max([v for v in vals if not math.isnan(v)], default=float("nan"))
    out.append(orow)
  return out


def pairwise_summary_deltas(summary_rows: list[dict], baseline_tag: str, compare_tags: list[str]) -> list[dict]:
  by_key = {(r["split"], r["seed"], r["tag"]): r for r in summary_rows}
  rows: list[dict] = []
  splits = sorted({r["split"] for r in summary_rows})
  seeds = sorted({r["seed"] for r in summary_rows})
  for split in splits:
    for seed in seeds:
      base = by_key.get((split, seed, baseline_tag))
      if not base:
        continue
      for tag in compare_tags:
        target = by_key.get((split, seed, tag))
        if not target:
          continue
        row = {"split": split, "seed": seed, "baseline_tag": baseline_tag, "tag": tag}
        for metric in SUMMARY_METRICS:
          row[f"baseline_{metric}"] = fnum(base.get(metric))
          row[f"{tag}_{metric}"] = fnum(target.get(metric))
          row[f"delta_{metric}"] = fnum(target.get(metric)) - fnum(base.get(metric))
        rows.append(row)
  return rows


def pairwise_city_deltas(city_rows: list[dict], baseline_tag: str, compare_tags: list[str]) -> list[dict]:
  by_key = {(r["split"], r["seed"], r["city"], r["tag"]): r for r in city_rows}
  rows: list[dict] = []
  splits = sorted({r["split"] for r in city_rows})
  seeds = sorted({r["seed"] for r in city_rows})
  cities = sorted({r["city"] for r in city_rows})
  for split in splits:
    for seed in seeds:
      for city in cities:
        base = by_key.get((split, seed, city, baseline_tag))
        if not base:
          continue
        for tag in compare_tags:
          target = by_key.get((split, seed, city, tag))
          if not target:
            continue
          row = {"split": split, "seed": seed, "city": city, "baseline_tag": baseline_tag, "tag": tag}
          for metric in CITY_METRICS:
            row[f"baseline_{metric}"] = fnum(base.get(metric))
            row[f"{tag}_{metric}"] = fnum(target.get(metric))
            row[f"delta_{metric}"] = fnum(target.get(metric)) - fnum(base.get(metric))
          rows.append(row)
  return rows


def city_delta_summary(delta_rows: list[dict]) -> list[dict]:
  grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
  for row in delta_rows:
    grouped[(row["split"], row["tag"], row["city"])].append(row)

  out: list[dict] = []
  for (split, tag, city), rows in sorted(grouped.items()):
    orow = {"split": split, "tag": tag, "city": city, "n": len(rows)}
    for metric in ("iou", "f1", "auroc", "pr_auc", "precision", "recall"):
      vals = [fnum(r.get(f"delta_{metric}")) for r in rows]
      orow[f"delta_{metric}_mean"] = safe_mean(vals)
      orow[f"delta_{metric}_std"] = safe_stdev(vals)
      orow[f"wins_{metric}"] = sum(1 for v in vals if not math.isnan(v) and v > 0)
    out.append(orow)
  return out


def threshold_proxy_candidates(delta_rows: list[dict]) -> list[dict]:
  out: list[dict] = []
  for row in delta_rows:
    diou = fnum(row.get("delta_iou"))
    df1 = fnum(row.get("delta_f1"))
    dauroc = fnum(row.get("delta_auroc"))
    dpr = fnum(row.get("delta_pr_auc"))
    pattern = None
    if diou < 0 and df1 < 0 and (dauroc > 0 or dpr > 0):
      pattern = "ranking_better_threshold_worse"
    elif diou > 0 and df1 > 0 and dauroc < 0 and dpr < 0:
      pattern = "threshold_better_ranking_worse"
    elif diou > 0 and df1 > 0 and (dauroc > 0 or dpr > 0):
      pattern = "broadly_better"
    elif diou < 0 and df1 < 0 and (dauroc < 0 or dpr < 0):
      pattern = "broadly_worse"
    if pattern:
      out.append({**row, "threshold_proxy_pattern": pattern})
  return out


def metric_fields(prefix: str, metrics: Iterable[str]) -> list[str]:
  fields: list[str] = []
  for metric in metrics:
    fields.extend([f"baseline_{metric}", f"{prefix}_{metric}", f"delta_{metric}"])
  return fields


def write_report(
  path: Path,
  sweep_root: Path,
  summary_by_tag: list[dict],
  summary_deltas: list[dict],
  city_summary: list[dict],
  proxy_rows: list[dict],
) -> None:
  test_tag_rows = [r for r in summary_by_tag if r["split"] == "test"]
  test_delta_rows = [r for r in summary_deltas if r["split"] == "test"]
  test_city_rows = [r for r in city_summary if r["split"] == "test"]

  by_tag_delta: dict[str, list[dict]] = defaultdict(list)
  for row in test_delta_rows:
    by_tag_delta[row["tag"]].append(row)

  lines: list[str] = []
  lines.append("# Sweep Analysis Report")
  lines.append("")
  lines.append(f"Sweep root: `{sweep_root}`")
  lines.append("")
  lines.append("## Test Mean Metrics By Tag")
  lines.append("")
  lines.append("| tag | IoU mean | IoU std | F1 mean | AUROC mean | PR-AUC mean |")
  lines.append("|---|---:|---:|---:|---:|---:|")
  for row in sorted(test_tag_rows, key=lambda r: r["tag"]):
    lines.append(
      "| {tag} | {iou} | {iou_std} | {f1} | {auroc} | {prauc} |".format(
        tag=row["tag"],
        iou=fmt(fnum(row.get("mean_iou_mean"))),
        iou_std=fmt(fnum(row.get("mean_iou_std"))),
        f1=fmt(fnum(row.get("mean_f1_mean"))),
        auroc=fmt(fnum(row.get("mean_auroc_mean"))),
        prauc=fmt(fnum(row.get("mean_pr_auc_mean"))),
      )
    )

  lines.append("")
  lines.append("## Mean Test Deltas Versus E0")
  lines.append("")
  lines.append("| tag | delta IoU | delta F1 | delta AUROC | delta PR-AUC |")
  lines.append("|---|---:|---:|---:|---:|")
  for tag in sorted(by_tag_delta):
    rows = by_tag_delta[tag]
    lines.append(
      "| {tag} | {diou} | {df1} | {dauroc} | {dpr} |".format(
        tag=tag,
        diou=fmt(safe_mean(fnum(r.get("delta_mean_iou")) for r in rows)),
        df1=fmt(safe_mean(fnum(r.get("delta_mean_f1")) for r in rows)),
        dauroc=fmt(safe_mean(fnum(r.get("delta_mean_auroc")) for r in rows)),
        dpr=fmt(safe_mean(fnum(r.get("delta_mean_pr_auc")) for r in rows)),
      )
    )

  lines.append("")
  lines.append("## Worst And Best Test Cities By Mean Delta IoU")
  for tag in ("E1_raw_ds", "E3_raw_ds_pca", "S0_siamese"):
    rows = [r for r in test_city_rows if r["tag"] == tag]
    if not rows:
      continue
    worst = sorted(rows, key=lambda r: fnum(r.get("delta_iou_mean")))[:3]
    best = sorted(rows, key=lambda r: fnum(r.get("delta_iou_mean")), reverse=True)[:3]
    lines.append("")
    lines.append(f"### {tag}")
    lines.append("")
    lines.append("Worst:")
    for row in worst:
      lines.append(f"- {row['city']}: delta IoU {fmt(fnum(row.get('delta_iou_mean')))}, wins {row.get('wins_iou')}/{row.get('n')}")
    lines.append("")
    lines.append("Best:")
    for row in best:
      lines.append(f"- {row['city']}: delta IoU {fmt(fnum(row.get('delta_iou_mean')))}, wins {row.get('wins_iou')}/{row.get('n')}")

  lines.append("")
  lines.append("## Threshold Proxy")
  lines.append("")
  lines.append(
    "Existing v5 artifacts do not include probability maps, so this report cannot perform true validation-threshold tuning. "
    "Proxy rows flag cases where fixed-threshold IoU/F1 and ranking metrics disagree."
  )
  pattern_counts: dict[str, int] = defaultdict(int)
  for row in proxy_rows:
    if row.get("split") == "test":
      pattern_counts[str(row.get("threshold_proxy_pattern"))] += 1
  for pattern, count in sorted(pattern_counts.items()):
    lines.append(f"- {pattern}: {count} test city/seed comparisons")

  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  args = parse_args()
  sweep_root = args.sweep_root.resolve()
  if not sweep_root.exists():
    raise FileNotFoundError(sweep_root)

  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_dir = args.output_dir.resolve() if args.output_dir else sweep_root / f"analysis_{ts}"
  output_dir.mkdir(parents=True, exist_ok=True)

  compare_tags = [t.strip() for t in args.compare_tags.split(",") if t.strip()]
  summary_rows, city_rows = read_eval_runs(sweep_root)
  if not summary_rows:
    raise RuntimeError(f"No eval results found under {sweep_root}")

  summary_by_tag = group_summary_by_tag(summary_rows)
  summary_deltas = pairwise_summary_deltas(summary_rows, args.baseline_tag, compare_tags)
  city_deltas = pairwise_city_deltas(city_rows, args.baseline_tag, compare_tags)
  city_summary = city_delta_summary(city_deltas)
  proxy_rows = threshold_proxy_candidates(city_deltas)

  summary_fields = ["split", "tag", "seed", "run", "model", "experiment_tag", "git_hash", "threshold", *SUMMARY_METRICS]
  write_csv(output_dir / "summary_by_run.csv", summary_rows, summary_fields)

  tag_fields = ["split", "tag", "n"]
  for metric in SUMMARY_METRICS:
    tag_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"])
  write_csv(output_dir / "summary_by_tag.csv", summary_by_tag, tag_fields)

  delta_fields = ["split", "seed", "baseline_tag", "tag"]
  for metric in SUMMARY_METRICS:
    delta_fields.extend([f"baseline_{metric}", f"{args.baseline_tag}_{metric}", f"delta_{metric}"])
  # The target-value field names depend on tag, so use stable explicit fields instead.
  normalized_summary_deltas: list[dict] = []
  for row in summary_deltas:
    out = {k: row[k] for k in ("split", "seed", "baseline_tag", "tag")}
    for metric in SUMMARY_METRICS:
      out[f"baseline_{metric}"] = row.get(f"baseline_{metric}")
      out[f"target_{metric}"] = row.get(f"{row['tag']}_{metric}")
      out[f"delta_{metric}"] = row.get(f"delta_{metric}")
    normalized_summary_deltas.append(out)
  delta_fields = ["split", "seed", "baseline_tag", "tag"]
  for metric in SUMMARY_METRICS:
    delta_fields.extend([f"baseline_{metric}", f"target_{metric}", f"delta_{metric}"])
  write_csv(output_dir / "pairwise_seed_deltas.csv", normalized_summary_deltas, delta_fields)

  city_fields = ["split", "tag", "seed", "run", "city", *CITY_METRICS, *COUNT_METRICS]
  write_csv(output_dir / "per_city_metrics.csv", city_rows, city_fields)

  city_delta_fields = ["split", "seed", "city", "baseline_tag", "tag"]
  for metric in CITY_METRICS:
    city_delta_fields.extend([f"baseline_{metric}", f"target_{metric}", f"delta_{metric}"])
  normalized_city_deltas: list[dict] = []
  for row in city_deltas:
    out = {k: row[k] for k in ("split", "seed", "city", "baseline_tag", "tag")}
    for metric in CITY_METRICS:
      out[f"baseline_{metric}"] = row.get(f"baseline_{metric}")
      out[f"target_{metric}"] = row.get(f"{row['tag']}_{metric}")
      out[f"delta_{metric}"] = row.get(f"delta_{metric}")
    normalized_city_deltas.append(out)
  write_csv(output_dir / "per_city_deltas.csv", normalized_city_deltas, city_delta_fields)

  city_summary_fields = ["split", "tag", "city", "n"]
  for metric in ("iou", "f1", "auroc", "pr_auc", "precision", "recall"):
    city_summary_fields.extend([f"delta_{metric}_mean", f"delta_{metric}_std", f"wins_{metric}"])
  write_csv(output_dir / "per_city_delta_summary.csv", city_summary, city_summary_fields)

  proxy_fields = ["threshold_proxy_pattern", *city_delta_fields]
  normalized_proxy_rows = []
  for row in proxy_rows:
    out = {"threshold_proxy_pattern": row.get("threshold_proxy_pattern")}
    out.update({k: row[k] for k in ("split", "seed", "city", "baseline_tag", "tag")})
    for metric in CITY_METRICS:
      out[f"baseline_{metric}"] = row.get(f"baseline_{metric}")
      out[f"target_{metric}"] = row.get(f"{row['tag']}_{metric}")
      out[f"delta_{metric}"] = row.get(f"delta_{metric}")
    normalized_proxy_rows.append(out)
  write_csv(output_dir / "threshold_proxy_candidates.csv", normalized_proxy_rows, proxy_fields)

  write_report(
    output_dir / "analysis_report.md",
    sweep_root,
    summary_by_tag,
    normalized_summary_deltas,
    city_summary,
    normalized_proxy_rows,
  )

  manifest = {
    "sweep_root": str(sweep_root),
    "output_dir": str(output_dir),
    "baseline_tag": args.baseline_tag,
    "compare_tags": compare_tags,
    "n_summary_rows": len(summary_rows),
    "n_city_rows": len(city_rows),
    "note": "Threshold proxy only; v5 eval artifacts do not contain probability maps for true threshold tuning.",
  }
  (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
  main()
