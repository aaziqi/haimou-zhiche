#!/usr/bin/env python3
"""
Compute paired revision statistics from two per-image metric CSV files.

Expected CSV columns match the current project format:
key, inp_path, pred_path, ref_path, psnr, ssim, ...

Example:
  python revision_stats_from_csv.py ^
    --baseline metrics_cyclegan_euvp_epoch202.csv ^
    --candidate metrics_mpcgan_euvp_epoch202.csv ^
    --label-baseline CycleGAN-202 ^
    --label-candidate MP-CycleGAN-202 ^
    --output-json revision_stats_euvp.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


DEFAULT_METRICS = ("psnr", "ssim")


def load_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            key = (row.get("key") or "").strip()
            if not key:
                continue
            rows[key] = row
    if not rows:
        raise ValueError(f"No valid rows found in {csv_path}")
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _parse_metric_value(row: dict[str, str], metric: str) -> float | None:
    raw = (row.get(metric) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def wilcoxon_signed_rank(deltas: list[float]) -> dict[str, float | int | None]:
    nonzero = [float(x) for x in deltas if abs(float(x)) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return {
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "w_stat": 0.0,
            "z": 0.0,
            "p_two_sided": 1.0,
        }

    abs_vals = [abs(x) for x in nonzero]
    ranks = _average_ranks(abs_vals)
    w_plus = sum(rank for rank, delta in zip(ranks, nonzero) if delta > 0)
    w_minus = sum(rank for rank, delta in zip(ranks, nonzero) if delta < 0)
    w_stat = min(w_plus, w_minus)

    tie_counts = {}
    for v in abs_vals:
        tie_counts[v] = tie_counts.get(v, 0) + 1
    tie_correction = sum(t * (t + 1) * (2 * t + 1) for t in tie_counts.values() if t > 1)

    mu = n * (n + 1) / 4.0
    sigma_sq = (n * (n + 1) * (2 * n + 1) - tie_correction / 2.0) / 24.0
    sigma = math.sqrt(max(sigma_sq, 1e-12))
    cc = 0.5 if w_stat != mu else 0.0
    z = (w_stat - mu + cc) / sigma if w_stat < mu else (w_stat - mu - cc) / sigma
    p = max(0.0, min(1.0, 2.0 * min(_normal_cdf(z), 1.0 - _normal_cdf(z))))

    return {
        "n_nonzero": n,
        "w_plus": round(w_plus, 6),
        "w_minus": round(w_minus, 6),
        "w_stat": round(w_stat, 6),
        "z": round(z, 6),
        "p_two_sided": round(p, 8),
    }


def summarize_metric(
    baseline_rows: dict[str, dict[str, str]],
    candidate_rows: dict[str, dict[str, str]],
    metric: str,
) -> dict[str, object]:
    shared_keys = sorted(set(baseline_rows) & set(candidate_rows))
    if not shared_keys:
        raise ValueError("The two CSV files do not share any keys.")

    baseline_vals = []
    candidate_vals = []
    deltas = []
    improved = 0
    worsened = 0
    tied = 0
    skipped_missing = 0

    for key in shared_keys:
        b = _parse_metric_value(baseline_rows[key], metric)
        c = _parse_metric_value(candidate_rows[key], metric)
        if b is None or c is None:
            skipped_missing += 1
            continue
        d = c - b
        baseline_vals.append(b)
        candidate_vals.append(c)
        deltas.append(d)
        if d > 1e-12:
            improved += 1
        elif d < -1e-12:
            worsened += 1
        else:
            tied += 1

    if not baseline_vals:
        raise ValueError(f"No valid shared rows with metric '{metric}' were found.")

    wilcoxon = wilcoxon_signed_rank(deltas)

    return {
        "shared_keys": len(shared_keys),
        "matched_pairs": len(baseline_vals),
        "skipped_missing": skipped_missing,
        "baseline_mean": round(mean(baseline_vals), 6),
        "candidate_mean": round(mean(candidate_vals), 6),
        "mean_delta": round(mean(deltas), 6),
        "median_delta": round(float(median(deltas)), 6),
        "improved_count": improved,
        "worsened_count": worsened,
        "tied_count": tied,
        "win_rate": round(improved / len(baseline_vals), 6),
        "non_loss_rate": round((improved + tied) / len(baseline_vals), 6),
        "wilcoxon": wilcoxon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired revision statistics from two metric CSV files.")
    parser.add_argument("--baseline", required=True, type=Path, help="CSV path for the baseline model.")
    parser.add_argument("--candidate", required=True, type=Path, help="CSV path for the candidate model.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS), help="Metrics to compare, e.g. psnr ssim.")
    parser.add_argument("--label-baseline", default="Baseline", help="Display label for baseline model.")
    parser.add_argument("--label-candidate", default="Candidate", help="Display label for candidate model.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    baseline_rows = load_rows(args.baseline)
    candidate_rows = load_rows(args.candidate)

    result = {
        "baseline_csv": str(args.baseline),
        "candidate_csv": str(args.candidate),
        "label_baseline": args.label_baseline,
        "label_candidate": args.label_candidate,
        "metrics": {},
    }

    for metric in args.metrics:
        result["metrics"][metric] = summarize_metric(baseline_rows, candidate_rows, metric)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
