#!/usr/bin/env python3
"""
scripts/stats.py
================
Unified cross-run aggregator and plotting suite.
Auto-discovers results, computes exact streaming stats, and plots summaries.
"""

from __future__ import annotations

import argparse
import csv as _csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Headless-safe for HPC environments
import matplotlib.pyplot as plt

# Optional speedups
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats as _scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Constants & The Single Source of Truth
# ---------------------------------------------------------------------------
CHUNKSIZE = 200_000               
MIN_SAMPLES_PER_CELL = 50         
TIME_COLS = ("call_ns", "time_ns")
ALPHA = 0.05

# (h_id, desc, bench, lgn_solver, rival_solver, rival_label, expect_winner)
HYPOTHESES = [
    ("H01",          "lgn FK vs KDL FK",                  "fk",                "lgn",      "kdl",            "KDL",       "lgn"),
    ("H01p",         "lgn FK vs Pinocchio FK",            "fk",                "lgn",      "pinocchio",      "Pinocchio", "lgn"),
    ("H02p",         "lgn IK vs Pinocchio IK (idiomatic)","ik_posonly",        "lgn",      "pinocchio",      "Pinocchio", "lgn"),
    ("H02p_fair",    "lgn IK vs Pinocchio IK (fair math)","ik_posonly_fair",   "lgn_fair", "pinocchio_fair", "Pinocchio", "lgn"),
    ("H04",          "lgn Hv vs scalar vel (legacy)",     "vel_only",          "lgn",      "scalar",         "scalar",    "rival"),
    ("H04_alloc",    "lgn Hv vs scalar vel (alloc)",      "vel_only_alloc",    "lgn",      "scalar",         "scalar",    "rival"),
    ("H04_prealloc", "lgn Hv vs scalar vel (prealloc)",   "vel_only_prealloc", "lgn",      "scalar",         "scalar",    "rival"),
    ("H_M",          "lgn M vs Pinocchio CRBA",           "dyn_M",             "lgn",      "pinocchio",      "Pinocchio", "lgn"),
    ("H_C",          "lgn C vs Pinocchio NLE-G",          "dyn_C",             "lgn",      "pinocchio",      "Pinocchio", "lgn"),
    ("H_R",          "lgn RNEA vs Pinocchio RNEA",        "dyn_RNEA_eq",       "lgn",      "pinocchio",      "Pinocchio", "lgn"),
    ("H_A",          "lgn ABA vs Pinocchio ABA",          "dyn_ABA_eq",        "lgn",      "pinocchio",      "Pinocchio", "lgn"),
]

COLOR_LGN          = "#E69F4D"   
COLOR_UNIT_LINE    = "#222222"   

KINEMATICS_IDS = ["H01", "H01p", "H02p", "H02p_fair"]
HV_IDS         = ["H04_alloc", "H04_prealloc", "H04"]
DYN_IDS        = ["H_M", "H_C", "H_R", "H_A"]

# ---------------------------------------------------------------------------
# Math / Statistics
# ---------------------------------------------------------------------------
def welch_from_moments(mean_a, var_a, n_a, mean_b, var_b, n_b):
    if n_a < 2 or n_b < 2 or math.isnan(var_a) or math.isnan(var_b):
        return float("nan"), float("nan"), float("nan")

    se_a = var_a / n_a
    se_b = var_b / n_b
    se = math.sqrt(se_a + se_b)
    if se <= 0:
        if mean_a < mean_b: return float("-inf"), 0.0, float("inf")
        elif mean_a > mean_b: return float("inf"), 1.0, float("inf")
        else: return 0.0, 0.5, float("inf")

    t = (mean_a - mean_b) / se
    num = (se_a + se_b) ** 2
    denom = (se_a ** 2) / (n_a - 1) + (se_b ** 2) / (n_b - 1)
    df = num / denom if denom > 0 else float("inf")

    if HAS_SCIPY: p_one = float(_scipy_stats.t.cdf(t, df))
    else: p_one = 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    return t, p_one, df

def cohen_d_from_moments(mean_a, var_a, n_a, mean_b, var_b, n_b):
    if n_a < 2 or n_b < 2 or math.isnan(var_a) or math.isnan(var_b): return float("nan")
    denom = (n_a - 1) * var_a + (n_b - 1) * var_b
    if denom <= 0: return float("nan")
    pooled = math.sqrt(denom / (n_a + n_b - 2))
    return float("nan") if pooled <= 0 else (mean_a - mean_b) / pooled

def aggregate(values: list) -> dict:
    if not values: return {"n_runs": 0}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return {"n_runs": 0}
    n_runs = int(arr.size)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n_runs >= 2 else 0.0
    return {
        "n_runs": n_runs,
        "median": float(np.median(arr)),
        "mean": mean,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)) if n_runs >= 2 else mean,
        "p95": float(np.percentile(arr, 95)) if n_runs >= 2 else mean,
        "cv": (100.0 * sd / mean) if mean > 0 and n_runs >= 2 else float("nan"),
    }

# ---------------------------------------------------------------------------
# Streaming CSV Loader
# ---------------------------------------------------------------------------
def _pick_time_col(columns):
    for c in TIME_COLS:
        if c in columns: return c
    return None

def _extract_run_tag(path: Path) -> str:
    stem = path.stem
    m = re.match(r"^results_v2_raw_(.+)$", stem)
    return m.group(1) if m else stem

def stream_run(path: Path) -> dict | None:
    if HAS_PANDAS:
        try:
            head = pd.read_csv(path, nrows=0)
            tcol = _pick_time_col(head.columns)
            if not tcol: return None
            sum_x, sum_x2, count = defaultdict(float), defaultdict(float), defaultdict(int)
            reader = pd.read_csv(
                path, usecols=["solver", "benchmark", "n", tcol],
                dtype={"solver": "string", "benchmark": "string"}, chunksize=CHUNKSIZE, low_memory=False
            )
            for chunk in reader:
                chunk["n"] = pd.to_numeric(chunk["n"], errors="coerce")
                chunk[tcol] = pd.to_numeric(chunk[tcol], errors="coerce")
                chunk = chunk.dropna()
                if chunk.empty: continue
                chunk["n"] = chunk["n"].astype(int)
                g = chunk.groupby(["solver", "benchmark", "n"])[tcol]
                agg = g.agg(["sum", lambda s: float((s * s).sum()), "count"])
                for (solver, bench, n_dof), row in agg.iterrows():
                    key = (str(solver), str(bench), int(n_dof))
                    sum_x[key] += float(row["sum"])
                    sum_x2[key] += float(row.iloc[1]) 
                    count[key] += int(row["count"])
            return _finalize_moments(sum_x, sum_x2, count)
        except Exception as e:
            print(f"  !! {path.name}: pandas read failed ({e})", file=sys.stderr)
            return None
    else:
        sum_x, sum_x2, count = defaultdict(float), defaultdict(float), defaultdict(int)
        try:
            with open(path, newline="") as f:
                reader = _csv.DictReader(f)
                tcol = _pick_time_col(reader.fieldnames or ())
                if not tcol: return None
                for row in reader:
                    if row.get(tcol) == tcol or row.get("solver") == "solver": continue
                    try:
                        key = (row["solver"], row["benchmark"], int(row["n"]))
                        t = float(row[tcol])
                        sum_x[key] += t
                        sum_x2[key] += t * t
                        count[key] += 1
                    except (KeyError, ValueError, TypeError): continue
            return _finalize_moments(sum_x, sum_x2, count)
        except Exception as e:
            print(f"  !! {path.name}: csv read failed ({e})", file=sys.stderr)
            return None

def _finalize_moments(sum_x, sum_x2, count) -> dict:
    out = {}
    for key, c in count.items():
        if c < MIN_SAMPLES_PER_CELL: continue
        mean = sum_x[key] / c
        var = float("nan") if c < 2 else max(0.0, (sum_x2[key] - sum_x[key] * mean) / (c - 1))
        out[key] = (mean, var, c)
    return out

# ---------------------------------------------------------------------------
# Text Export Helpers
# ---------------------------------------------------------------------------
def fmt_ns(ns): return "-" if ns is None or math.isnan(ns) else f"{ns/1e6:.2f} ms" if ns >= 1e6 else f"{ns/1e3:.2f} us" if ns >= 1e3 else f"{ns:.1f} ns"
def fmt_x(x): return "-" if x is None or math.isnan(x) else f"{x:.2f}x"
def fmt_pct(x): return "-" if x is None or math.isnan(x) else f"{x:.1f}%"
def fmt_p(p): return "-" if p is None or math.isnan(p) else "<1e-6" if p < 1e-6 else f"{p:.3g}"

def print_table(rows, headers):
    if not rows: return print("  (no rows)")
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep + "\n| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |\n" + sep)
    for r in rows: print("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)

def write_markdown(out_path: Path, machine: str, n_runs: int, cell_means: dict, hyp_ratios: dict, hyp_pvals: dict):
    label = machine or "M?"
    lines = [
        f"% Cross-run aggregate over {n_runs} runs on {label}",
        f"% Each cell value is the median across {n_runs} runs.",
        f"% Ratios annotated with N/{n_runs} = how many runs' Welch tests rejected H0 (lgn < rival, alpha={ALPHA}).\n"
    ]

    def med_ns(solver, bench, n): return float(np.median(v)) if (v := cell_means.get((solver, bench, n), [])) else None
    def med_ratio(h_id, n): return float(np.median(v)) if (v := hyp_ratios.get((h_id, n), [])) else None
    def r_count(h_id, n): 
        v = hyp_pvals.get((h_id, n), [])
        return sum(1 for p in v if p is not None and not math.isnan(p) and p < ALPHA), len(v)

    # FK Table
    lines.extend([f"% --- Table II (FK) row block for {label} ---", rf"\multicolumn{{6}}{{l}}{{\textit{{{label}: {n_runs}-run median}}}} \\"])
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        if (lgn := med_ns("lgn", "fk", n)) is None: continue
        lines.append(f"{n:<4} & {lgn:>8.1f} & {med_ns('kdl', 'fk', n):>8.1f} & {med_ns('pinocchio', 'fk', n):>8.1f} & {med_ratio('H01', n):.2f}$\\times$ & {med_ratio('H01p', n):.2f}$\\times$ \\\\")
    
    # IK Table
    lines.extend(["", f"% --- Table III (IK) row block for {label} ---", rf"\multicolumn{{5}}{{l}}{{\textit{{{label}: {n_runs}-run median}}}} \\"])
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        if (lgn := med_ns("lgn", "ik_posonly", n)) is None: continue
        lines.append(f"{n:<4} & {lgn:>9,.0f} & {med_ns('pinocchio', 'ik_posonly', n):>9,.0f} & {med_ratio('H02p', n):.2f}$\\times$ & {med_ns('kdl', 'ik_posonly', n):>11,.0f} \\\\")
    
    # Verdicts
    lines.extend(["", f"% --- Cross-run hypothesis verdicts on {label} ---"])
    for h_id, _desc, _bench, _lgn, _riv, _label, _expect in HYPOTHESES:
        for n in sorted({k[1] for k in hyp_pvals if k[0] == h_id}):
            rej, tot = r_count(h_id, n)
            if tot > 0: lines.append(f"% {h_id:6s} n={n:3d}  reject {rej}/{tot}  median ratio {med_ratio(h_id, n):.2f}x")
    out_path.write_text("\n".join(lines))

# ---------------------------------------------------------------------------
# Plotting Engine 
# ---------------------------------------------------------------------------
def _hyp_by_id(h_id: str): return next((h for h in HYPOTHESES if h[0] == h_id), None)

def plot_ratio_summary(h_ids: list[str], per_run_means: dict[str, dict], out_path: Path, machine: str, n_runs: int, title: str):
    series = []
    for h_id in h_ids:
        if not (h := _hyp_by_id(h_id)): continue
        _id, desc, bench, lgn_s, rival_s, _rl, expect = h
        per_n_ratios: dict[int, list[float]] = defaultdict(list)
        
        for run_tag, means in per_run_means.items():
            common = {k[2] for k in means if k[0] == lgn_s and k[1] == bench} & {k[2] for k in means if k[0] == rival_s and k[1] == bench}
            for n in common:
                ma, mb = means[(lgn_s, bench, n)], means[(rival_s, bench, n)]
                if ma > 0 and mb > 0: per_n_ratios[n].append(mb / ma if expect == "lgn" else ma / mb)

        per_n_stats = {}
        for n, rs in per_n_ratios.items():
            if not rs: continue
            med = float(np.median(rs))
            per_n_stats[n] = (med, med - float(np.min(rs)) if len(rs) > 1 else 0.0, float(np.max(rs)) - med if len(rs) > 1 else 0.0)
        
        if per_n_stats: series.append((h_id, expect, desc, per_n_stats))

    if not series: return False

    all_ns = sorted({n for _id, _e, _d, s in series for n in s})
    x, w = np.arange(len(all_ns)), 0.82 / len(series)
    fig, ax = plt.subplots(figsize=(10.0, 5.4))

    legend_handles = []
    for i, (h_id, expect, desc, per_n) in enumerate(series):
        offset = -0.41 + w/2 + i * w
        for j, n in enumerate(all_ns):
            if n not in per_n: continue
            med, lo, hi = per_n[n]
            color = COLOR_LGN if (expect == "lgn" and med > 1.0) or (expect == "rival" and med < 1.0) else "#9C9C9C"
            ax.bar(x[j] + offset, med, w, color=color, edgecolor="#222", linewidth=0.4, yerr=[[lo], [hi]], capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": "#222"})
        
        meds = [per_n[n][0] for n in all_ns if n in per_n]
        wins = sum(1 for r in meds if (expect == "lgn" and r > 1.0) or (expect == "rival" and r < 1.0))
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_LGN if wins > len(meds)/2 else "#9C9C9C", label=f"{h_id}  ({desc.split(':')[0] if ':' in desc else desc})"))

    ax.axhline(1.0, color=COLOR_UNIT_LINE, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in all_ns])
    ax.set_ylabel("Speedup Ratio (>1 means Expected Winner Won)")
    ax.set_title(f"{title}  --  {machine}, {n_runs} run(s)")
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.95, fontsize=8.5, ncol=2)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    fig.text(0.5, 0.005, "Height: cross-run median | Whiskers: min/max | Orange: lgn beat comparison", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    # Auto-resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_results_dir = repo_root / "results_v2"
    
    ap = argparse.ArgumentParser(description="Unified stats aggregator & plotting tool")
    ap.add_argument("csv", nargs="*", help="Optional: manually specify CSVs. Otherwise auto-detects.")
    ap.add_argument("--machine", default="dt-hm10", help="Machine label")
    ap.add_argument("--min-runs", type=int, default=1, help="Min runs per cell")
    ap.add_argument("--out-dir", default=str(default_results_dir / "plots_10x"), help="Output dir")
    ap.add_argument("--markdown", default=str(default_results_dir / "kinematics_tables.tex"), help="LaTeX output")
    ap.add_argument("--include-dyn", action="store_true", help="Plot dynamics")
    ap.add_argument("--no-plots", action="store_true", help="Skip plots")
    args = ap.parse_args()

    # Auto-glob if no CSVs passed
    if not args.csv:
        paths = list(default_results_dir.glob("results_v2_raw_*.csv"))
        if not paths:
            print(f"ERROR: No arguments passed and no CSVs found in {default_results_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-detected {len(paths)} CSV(s) in {default_results_dir}")
    else:
        paths = [Path(p) for p in args.csv]

    runs: list = []
    print(f"Streaming {len(paths)} run(s)...", flush=True)
    for p in sorted(paths):
        tag = _extract_run_tag(p)
        print(f"  loading [{tag}]...", end="", flush=True)
        if not (moments := stream_run(p)):
            print(" FAILED"); continue
        print(f" {len(moments)} cells", flush=True)
        runs.append((tag, moments))

    if not (n_runs := len(runs)):
        print("ERROR: no runs loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 72}\n  CROSS-RUN AGGREGATE  --  {args.machine}  --  {n_runs} runs\n{'=' * 72}")

    cell_means, hyp_ratios, hyp_pvals = defaultdict(list), defaultdict(list), defaultdict(list)
    for _tag, moments in runs:
        for cell, (mean_ns, _var, _c) in moments.items(): cell_means[cell].append(mean_ns)
        for h_id, _desc, bench, lgn_s, rival_s, _label, _expect in HYPOTHESES:
            for n in sorted({k[2] for k in moments if k[0] == lgn_s and k[1] == bench} & {k[2] for k in moments if k[0] == rival_s and k[1] == bench}):
                ma, va, na = moments[(lgn_s, bench, n)]
                mb, vb, nb = moments[(rival_s, bench, n)]
                if ma > 0 and mb > 0:
                    hyp_ratios[(h_id, n)].append(mb / ma) 
                    _, p, _ = welch_from_moments(ma, va, na, mb, vb, nb)
                    hyp_pvals[(h_id, n)].append(p)

    # CLI Output Tables
    print("\n--- (i) Per-(solver, benchmark, n) Cross-Run Means -----------------\n")
    rows = []
    for cell, vals in sorted(cell_means.items()):
        if len(vals) < args.min_runs: continue
        agg = aggregate(vals)
        rows.append((*cell[:2], f"n={cell[2]}", f"{agg['n_runs']}/{n_runs}", fmt_ns(agg["median"]), fmt_ns(agg["min"]), fmt_ns(agg["max"]), fmt_pct(agg["cv"])))
    print_table(rows, ["solver", "bench", "n", "runs", "median", "min", "max", "CV%"])

    print("\n--- (ii) Per-hypothesis robust verdict ---------------------------\n")
    rows = []
    for h_id, _desc, _bench, _lgn, _riv, _label, _expect in HYPOTHESES:
        for n in sorted({k[1] for k in hyp_pvals if k[0] == h_id}):
            if len(pvals := hyp_pvals[(h_id, n)]) < args.min_runs: continue
            rej = sum(1 for p in pvals if not math.isnan(p) and p < ALPHA)
            rows.append((h_id, f"n={n}", f"{len(pvals)}", f"{rej}/{len(pvals)}", fmt_x(aggregate(hyp_ratios[(h_id, n)])["median"]), fmt_p(float(np.nanmedian(pvals)))))
    print_table(rows, ["H", "n", "runs", "reject", "median_ratio", "median_p"])
    
    # LaTeX Generation
    if args.markdown:
        out_path = Path(args.markdown)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(out_path, args.machine, n_runs, cell_means, hyp_ratios, hyp_pvals)
        print(f"\nWrote paper-ready tables -> {out_path}")

    # Plot Generation
    if args.no_plots: return print("\nSkipping plots (--no-plots).")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'-' * 72}\nGenerating Plots -> {out_dir}/")

    per_run_means = {tag: {k: m[0] for k, m in moments.items()} for tag, moments in runs}
    if plot_ratio_summary(KINEMATICS_IDS, per_run_means, out_dir / "summary_kinematics.png", args.machine, n_runs, "Kinematics Summary (FK & IK)"): print("  wrote summary_kinematics.png")
    if plot_ratio_summary(HV_IDS, per_run_means, out_dir / "summary_hv.png", args.machine, n_runs, "Velocity Summary (Hv)"): print("  wrote summary_hv.png")
    if args.include_dyn and plot_ratio_summary(DYN_IDS, per_run_means, out_dir / "summary_dyn.png", args.machine, n_runs, "Dynamics Summary"): print("  wrote summary_dyn.png")
    
    print("Done.")

if __name__ == "__main__":
    main()
