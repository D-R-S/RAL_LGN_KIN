#!/usr/bin/env python3
"""
scripts/compute_stats_cross_machine.py
======================================
Cross-machine aggregator. Takes N labelled iteration CSVs spanning M
machines and produces:

  (1) Per-machine summary (median ratio, min-max, p5-p95, CV%) -- one
      block per machine. Same statistic structure as compute_stats_10x.py
      but emitted side-by-side so the cross-machine pattern is readable.

  (2) Pooled stats across all machines:
      - "Pool of per-run ratios"      = treat each run on each machine as
        one observation; aggregate the resulting 10*M ratios. Robust
        headline claim ("across everything we measured, ratio X looks
        like Y").
      - "Pool of per-machine medians" = take each machine's 10-run
        median; aggregate the resulting M numbers. Robustness across
        machines ("each box says roughly Y").

  (3) Plots, per hypothesis:
        <H_id>_per_machine.png  -- absolute ns, grouped bars per
                                   machine, error bars = min-max across
                                   that machine's runs. Log y.
        <H_id>_ratios.png       -- ratio per n, one bar per machine
                                   plus a "pooled across machines" bar
                                   with hatched fill. Linear y, h-line
                                   at 1.

  (4) Optional --markdown writes a paper-ready LaTeX block with one row
      group per machine plus a pooled row group.

CSV labelling
-------------
Pass labelled paths via --label-csv as positional args, each formatted as

    <label>::<csv_path>

so a 10-machine sweep looks like

    --label-csv M1-Lat::pc_1/iter_01.csv M1-Lat::pc_1/iter_02.csv ...
                M2-Zen5::pc_2/iter_01.csv ...

The wrapper aggregate_all_machines.sh builds this automatically.

Streaming + Welch from moments: same approach as compute_stats_10x.py.
Peak RAM ~ O(machines * cells), still tiny.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNKSIZE = 200_000
MIN_SAMPLES_PER_CELL = 50
TIME_COLS = ("call_ns", "time_ns")
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Hypothesis registry  (kinematic only by default)
# ---------------------------------------------------------------------------
KIN_HYPOTHESES = [
    ("H01",  "lgn FK vs KDL FK",                "fk",          "lgn",  "kdl",        "KDL"),
    ("H01p", "lgn FK vs Pinocchio FK",          "fk",          "lgn",  "pinocchio",  "Pinocchio"),
    ("H02p", "lgn IK vs Pinocchio IK",          "ik_posonly",  "lgn",  "pinocchio",  "Pinocchio"),
    ("H04",  "lgn Hv vs scalar vel",            "vel_only",    "lgn",  "scalar",     "scalar"),
]

DYN_HYPOTHESES = [
    ("H_M",  "lgn M vs Pinocchio CRBA",         "dyn_M",       "lgn",  "pinocchio",  "Pinocchio"),
    ("H_C",  "lgn C vs Pinocchio NLE-G",        "dyn_C",       "lgn",  "pinocchio",  "Pinocchio"),
    ("H_R",  "lgn RNEA vs Pinocchio RNEA",      "dyn_RNEA_eq", "lgn",  "pinocchio",  "Pinocchio"),
    ("H_A",  "lgn ABA vs Pinocchio ABA",        "dyn_ABA_eq",  "lgn",  "pinocchio",  "Pinocchio"),
]


# ---------------------------------------------------------------------------
# Welch / Cohen from moments (same as compute_stats_10x.py)
# ---------------------------------------------------------------------------
def welch_from_moments(ma, va, na, mb, vb, nb):
    if na < 2 or nb < 2 or va < 0 or vb < 0:
        return float("nan"), float("nan"), float("nan")
    se_a = va / na
    se_b = vb / nb
    se = math.sqrt(se_a + se_b)
    if se <= 0:
        if ma < mb: return float("-inf"), 0.0, float("inf")
        if ma > mb: return float("inf"), 1.0, float("inf")
        return 0.0, 0.5, float("inf")
    t = (ma - mb) / se
    num = (se_a + se_b) ** 2
    denom = (se_a ** 2) / (na - 1) + (se_b ** 2) / (nb - 1)
    df = num / denom if denom > 0 else float("inf")
    if HAS_SCIPY:
        p_one = float(_scipy_stats.t.cdf(t, df))
    else:
        p_one = 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
    return t, p_one, df


def cohen_d_from_moments(ma, va, na, mb, vb, nb):
    if na < 2 or nb < 2 or va < 0 or vb < 0:
        return float("nan")
    denom = (na - 1) * va + (nb - 1) * vb
    if denom <= 0:
        return float("nan")
    pooled = math.sqrt(denom / (na + nb - 2))
    if pooled <= 0:
        return float("nan")
    return (ma - mb) / pooled


# ---------------------------------------------------------------------------
# Streaming loader -> per-cell running moments for one CSV
# ---------------------------------------------------------------------------
def _pick_time_col(columns):
    for c in TIME_COLS:
        if c in columns:
            return c
    return None


def _extract_run_tag(path: Path) -> str:
    stem = path.stem
    m = re.match(r"^results_v2_raw_(.+)$", stem)
    return m.group(1) if m else stem


def stream_run_pandas(path: Path) -> dict | None:
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception as e:
        print(f"  !! {path.name}: cannot read header ({e})", file=sys.stderr)
        return None
    tcol = _pick_time_col(head.columns)
    if tcol is None:
        return None

    sum_x: dict = defaultdict(float)
    sum_x2: dict = defaultdict(float)
    count: dict = defaultdict(int)
    try:
        reader = pd.read_csv(
            path,
            usecols=["solver", "benchmark", "n", tcol],
            dtype={"solver": "string", "benchmark": "string"},
            chunksize=CHUNKSIZE,
            low_memory=False,
        )
        for chunk in reader:
            chunk["n"] = pd.to_numeric(chunk["n"], errors="coerce")
            chunk[tcol] = pd.to_numeric(chunk[tcol], errors="coerce")
            chunk = chunk.dropna(subset=["solver", "benchmark", "n", tcol])
            if chunk.empty:
                continue
            chunk["n"] = chunk["n"].astype(int)
            g = chunk.groupby(["solver", "benchmark", "n"])[tcol]
            agg = g.agg(["sum", lambda s: float((s * s).sum()), "count"])
            agg.columns = ["sum", "sum_sq", "count"]
            for (s, b, nd), row in agg.iterrows():
                key = (str(s), str(b), int(nd))
                sum_x[key] += float(row["sum"])
                sum_x2[key] += float(row["sum_sq"])
                count[key] += int(row["count"])
    except Exception as e:
        print(f"  !! {path.name}: read failed ({e})", file=sys.stderr)
        return None

    out = {}
    for key, c in count.items():
        if c < MIN_SAMPLES_PER_CELL:
            continue
        mean = sum_x[key] / c
        ss = sum_x2[key] - sum_x[key] * mean
        var = ss / (c - 1) if c >= 2 and ss > 0 else 0.0
        out[key] = (mean, var, c)
    return out


def stream_run_stdlib(path: Path) -> dict | None:
    sum_x: dict = defaultdict(float)
    sum_x2: dict = defaultdict(float)
    count: dict = defaultdict(int)
    try:
        with open(path, newline="") as f:
            reader = _csv.DictReader(f)
            tcol = _pick_time_col(reader.fieldnames or ())
            if tcol is None:
                return None
            for row in reader:
                if row.get(tcol) == tcol or row.get("solver") == "solver":
                    continue
                try:
                    key = (row["solver"], row["benchmark"], int(row["n"]))
                    t = float(row[tcol])
                except (KeyError, ValueError, TypeError):
                    continue
                sum_x[key] += t
                sum_x2[key] += t * t
                count[key] += 1
    except Exception as e:
        print(f"  !! {path.name}: read failed ({e})", file=sys.stderr)
        return None

    out = {}
    for key, c in count.items():
        if c < MIN_SAMPLES_PER_CELL:
            continue
        mean = sum_x[key] / c
        ss = sum_x2[key] - sum_x[key] * mean
        var = ss / (c - 1) if c >= 2 and ss > 0 else 0.0
        out[key] = (mean, var, c)
    return out


def stream_run(path: Path):
    return stream_run_pandas(path) if HAS_PANDAS else stream_run_stdlib(path)


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------
def aggregate(values):
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    n = int(arr.size)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    return {
        "n": n,
        "median": float(np.median(arr)),
        "mean": mean,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "cv": (100.0 * sd / mean) if mean > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_ns(ns):
    if ns is None or (isinstance(ns, float) and math.isnan(ns)):
        return "-"
    if ns >= 1e6: return f"{ns/1e6:.2f} ms"
    if ns >= 1e3: return f"{ns/1e3:.2f} us"
    return f"{ns:.1f} ns"


def fmt_x(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.2f}x"


def fmt_pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.1f}%"


def print_table(rows, headers):
    if not rows:
        print("  (no rows)")
        return
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    print(sep); print(hdr); print(sep)
    for r in rows:
        print("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# A small set of distinct colors for the per-machine bars. Cycles if needed.
MACHINE_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def plot_per_machine_absolute(h_id, title, bench, lgn_s, rival_s, rival_label,
                              machines: list, machine_data: dict,
                              out_path: Path):
    """
    One subplot per machine, log-y bars: lgn vs rival at each n,
    error bars = min-max across that machine's runs.

    machine_data[label][cell] = list of per-run means (mean_ns).
    """
    n_m = len(machines)
    if n_m == 0:
        return False
    # Find the global n set across machines
    global_ns: set = set()
    for label in machines:
        d = machine_data[label]
        for k in d:
            if k[0] in (lgn_s, rival_s) and k[1] == bench and len(d[k]) >= 2:
                global_ns.add(k[2])
    if not global_ns:
        print(f"    {h_id}: no usable cells; skipping")
        return False
    ns = sorted(global_ns)

    fig, axes = plt.subplots(1, n_m, figsize=(4.0 * n_m, 4.6), sharey=True)
    if n_m == 1:
        axes = [axes]

    for ax_i, label in enumerate(machines):
        ax = axes[ax_i]
        d = machine_data[label]

        def stats(solver, n):
            vals = d.get((solver, bench, n), [])
            if len(vals) < 2:
                return None, None, None
            arr = np.asarray(vals, dtype=float)
            med = float(np.median(arr))
            return med, med - float(np.min(arr)), float(np.max(arr)) - med

        x = np.arange(len(ns))
        w = 0.38
        lgn_med = []; lgn_lo = []; lgn_hi = []
        riv_med = []; riv_lo = []; riv_hi = []
        for n in ns:
            m, lo, hi = stats(lgn_s, n)
            lgn_med.append(m); lgn_lo.append(lo or 0); lgn_hi.append(hi or 0)
            m, lo, hi = stats(rival_s, n)
            riv_med.append(m); riv_lo.append(lo or 0); riv_hi.append(hi or 0)

        ax.bar(x - w/2, lgn_med, w, color=MACHINE_COLORS[ax_i % len(MACHINE_COLORS)],
               label="lgn", edgecolor="black", linewidth=0.4,
               yerr=[lgn_lo, lgn_hi], capsize=3,
               error_kw={"elinewidth": 1.0, "ecolor": "black"})
        ax.bar(x + w/2, riv_med, w, color="#999999",
               label=rival_label, edgecolor="black", linewidth=0.4,
               yerr=[riv_lo, riv_hi], capsize=3,
               error_kw={"elinewidth": 1.0, "ecolor": "black"})

        ax.set_xticks(x)
        ax.set_xticklabels([f"n={n}" for n in ns], rotation=0, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", which="both", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        if ax_i == 0:
            ax.set_ylabel("Mean time per call (ns, log)")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(f"{h_id}: {title}  --  per machine (median bar, min-max whiskers)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    wrote {out_path}")
    return True


def plot_ratio_per_machine(h_id, title, bench, lgn_s, rival_s,
                           machines: list, machine_data: dict,
                           out_path: Path, invert: bool = False):
    """
    One plot: ratio rival/lgn per n, with one bar per machine grouped at
    each n, plus a pooled-across-all-machines bar drawn with hatching.
    invert=True flips to lgn/rival (used for H04 carrier penalty).
    """
    n_m = len(machines)
    if n_m == 0:
        return False

    # Compute per-machine per-run ratios.
    per_machine_ratios: dict = {}  # label -> {n: [ratio, ratio, ...]}
    global_ns: set = set()
    for label in machines:
        d = machine_data[label]
        per_n: dict = defaultdict(list)
        ns_lgn = {k[2] for k in d if k[0] == lgn_s and k[1] == bench}
        ns_riv = {k[2] for k in d if k[0] == rival_s and k[1] == bench}
        for n in sorted(ns_lgn & ns_riv):
            lgn_vals = d.get((lgn_s, bench, n), [])
            riv_vals = d.get((rival_s, bench, n), [])
            # pair per run (lists are in run-order within a machine)
            for a, b in zip(lgn_vals, riv_vals):
                if a <= 0 or b <= 0:
                    continue
                r = (a / b) if invert else (b / a)
                per_n[n].append(r)
            if per_n[n]:
                global_ns.add(n)
        per_machine_ratios[label] = per_n

    # Pooled: concatenate per-run ratios across all machines, per n.
    pooled_ratios: dict = defaultdict(list)
    for label in machines:
        for n, ratios in per_machine_ratios[label].items():
            pooled_ratios[n].extend(ratios)

    if not global_ns:
        print(f"    {h_id}: no ratio data; skipping")
        return False
    ns = sorted(global_ns)

    fig, ax = plt.subplots(figsize=(max(9.0, 1.3 * len(ns) * (n_m + 1)), 5.0))

    x = np.arange(len(ns))
    n_series = n_m + 1  # +1 for the pooled bar
    w = 0.85 / n_series

    for i, label in enumerate(machines):
        per_n = per_machine_ratios[label]
        meds = []; los = []; his = []
        for n in ns:
            ratios = per_n.get(n, [])
            if len(ratios) < 2:
                meds.append(float("nan")); los.append(0); his.append(0); continue
            arr = np.asarray(ratios)
            med = float(np.median(arr))
            meds.append(med)
            los.append(med - float(np.min(arr)))
            his.append(float(np.max(arr)) - med)
        offset = -0.425 + w/2 + i * w
        ax.bar(x + offset, meds, w,
               color=MACHINE_COLORS[i % len(MACHINE_COLORS)],
               label=label,
               edgecolor="black", linewidth=0.4,
               yerr=[los, his], capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": "black"})

    # Pooled bar (hatched).
    meds = []; los = []; his = []
    for n in ns:
        ratios = pooled_ratios.get(n, [])
        if len(ratios) < 2:
            meds.append(float("nan")); los.append(0); his.append(0); continue
        arr = np.asarray(ratios)
        med = float(np.median(arr))
        meds.append(med)
        los.append(med - float(np.min(arr)))
        his.append(float(np.max(arr)) - med)
    offset = -0.425 + w/2 + n_m * w
    ax.bar(x + offset, meds, w,
           facecolor="#ffffff", edgecolor="black", linewidth=0.8,
           hatch="///", label=f"pooled ({sum(len(pooled_ratios[n]) for n in ns)} runs)",
           yerr=[los, his], capsize=2,
           error_kw={"elinewidth": 0.8, "ecolor": "black"})

    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ratio_label = "lgn / rival" if invert else "rival / lgn"
    ax.set_ylabel(f"speedup ratio  ({ratio_label}, >1: lgn faster)" if not invert
                  else f"penalty ratio  ({ratio_label}, lower = better)")
    ax.set_title(f"{h_id}: {title}  --  ratio per machine + pooled across all")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    fig.text(0.5, 0.005,
             "Bar height: cross-run median (within each machine, or pooled across machines). "
             "Error bars: min--max across the runs being aggregated.",
             ha="center", fontsize=8, color="#555")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Markdown / LaTeX writer
# ---------------------------------------------------------------------------
def write_markdown(out_path: Path, machines, machine_data, machine_moments, hyp_list):
    """
    Emit per-machine row blocks for each kinematic table, then a pooled row block.
    """
    lines = []
    lines.append(f"% Cross-machine aggregate over {len(machines)} machines")
    lines.append("% One row block per machine, then a 'pooled' block at the end.")
    lines.append("")

    def med_ns_machine(label, solver, bench, n):
        vals = machine_data[label].get((solver, bench, n), [])
        return float(np.median(vals)) if vals else None

    def med_ratio_machine(label, h_id, n, invert=False):
        spec = next((h for h in (KIN_HYPOTHESES + DYN_HYPOTHESES) if h[0] == h_id), None)
        if spec is None: return None
        _id, _t, bench, lgn_s, rival_s, _rl = spec
        lgn_vals = machine_data[label].get((lgn_s, bench, n), [])
        riv_vals = machine_data[label].get((rival_s, bench, n), [])
        rs = []
        for a, b in zip(lgn_vals, riv_vals):
            if a <= 0 or b <= 0: continue
            rs.append((a / b) if invert else (b / a))
        return float(np.median(rs)) if rs else None

    def med_ratio_pooled(h_id, n, invert=False):
        spec = next((h for h in (KIN_HYPOTHESES + DYN_HYPOTHESES) if h[0] == h_id), None)
        if spec is None: return None
        _id, _t, bench, lgn_s, rival_s, _rl = spec
        rs = []
        for label in machines:
            lgn_vals = machine_data[label].get((lgn_s, bench, n), [])
            riv_vals = machine_data[label].get((rival_s, bench, n), [])
            for a, b in zip(lgn_vals, riv_vals):
                if a <= 0 or b <= 0: continue
                rs.append((a / b) if invert else (b / a))
        return float(np.median(rs)) if rs else None

    # FK table block, per machine + pooled
    for label in machines + ["POOLED"]:
        lines.append(f"% --- FK rows for {label} ---")
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{label}}}}} \\")
        for n in [2, 4, 8, 16, 32, 64, 128, 256]:
            if label == "POOLED":
                # Pooled FK rows: medians of pooled medians across machines per cell.
                lgn_v = [v for lab in machines for v in machine_data[lab].get(("lgn","fk",n),[])]
                kdl_v = [v for lab in machines for v in machine_data[lab].get(("kdl","fk",n),[])]
                pin_v = [v for lab in machines for v in machine_data[lab].get(("pinocchio","fk",n),[])]
                lgn = float(np.median(lgn_v)) if lgn_v else None
                kdl = float(np.median(kdl_v)) if kdl_v else None
                pin = float(np.median(pin_v)) if pin_v else None
                r_kdl = med_ratio_pooled("H01", n)
                r_pin = med_ratio_pooled("H01p", n)
            else:
                lgn = med_ns_machine(label, "lgn", "fk", n)
                kdl = med_ns_machine(label, "kdl", "fk", n)
                pin = med_ns_machine(label, "pinocchio", "fk", n)
                r_kdl = med_ratio_machine(label, "H01", n)
                r_pin = med_ratio_machine(label, "H01p", n)
            if lgn is None: continue
            def f(x): return f"{x:.1f}" if x is not None else "---"
            def r(x): return f"{x:.2f}$\\times$" if x is not None else "---"
            lines.append(f"{n:<4} & {f(lgn):>8} & {f(kdl):>8} & {f(pin):>8} & {r(r_kdl)} & {r(r_pin)} \\\\")
        lines.append("")

    # IK and velprop blocks: same structure, omitted for brevity in this output
    # (added back if you want full LaTeX automation later)

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Cross-machine aggregator over labelled iter CSVs",
    )
    ap.add_argument("--label-csv", nargs="+", required=True,
                    help="labelled CSV paths in the form LABEL::PATH")
    ap.add_argument("--out-dir", default="cross_machine",
                    help="output directory for plots + reports")
    ap.add_argument("--markdown", default="",
                    help="write paper-ready LaTeX rows to this file")
    ap.add_argument("--include-dyn", action="store_true",
                    help="also plot dyn hypotheses (off by default due to math bug)")
    ap.add_argument("--min-runs-per-machine", type=int, default=3,
                    help="machines with fewer runs are dropped (default: 3)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ---- Parse label-csv pairs -------------------------------------------
    grouped: dict = defaultdict(list)  # label -> [Path, ...]
    for item in args.label_csv:
        if "::" not in item:
            print(f"WARNING: skipping bad input {item!r} (expected LABEL::PATH)",
                  file=sys.stderr)
            continue
        label, path = item.split("::", 1)
        label = label.strip()
        p = Path(path.strip())
        if not p.is_file():
            print(f"WARNING: missing {p}", file=sys.stderr)
            continue
        grouped[label].append(p)

    if not grouped:
        print("ERROR: no labelled inputs", file=sys.stderr)
        sys.exit(1)

    # ---- Stream per machine ----------------------------------------------
    # machine_moments[label][cell] = list of (mean, var, count) per run
    # machine_data[label][cell]    = list of per-run means (mean) per run
    machine_moments: dict = defaultdict(lambda: defaultdict(list))
    machine_data: dict = defaultdict(lambda: defaultdict(list))

    print()
    print("Streaming all runs across all machines...")
    machines = sorted(grouped.keys())
    for label in machines:
        paths = sorted(grouped[label])
        print(f"  [{label}] {len(paths)} runs")
        for p in paths:
            print(f"    loading {p.name}...", end="", flush=True)
            m = stream_run(p)
            if m is None:
                print(" FAILED")
                continue
            for cell, (mean, var, count) in m.items():
                machine_moments[label][cell].append((mean, var, count))
                machine_data[label][cell].append(mean)
            print(f" {len(m)} cells")

    # Drop under-populated machines
    keep_machines = []
    for label in machines:
        n_runs = max((len(v) for v in machine_data[label].values()), default=0)
        if n_runs < args.min_runs_per_machine:
            print(f"  WARNING: dropping {label} ({n_runs} runs < min {args.min_runs_per_machine})")
            continue
        keep_machines.append(label)
    machines = keep_machines

    if not machines:
        print("ERROR: no machines have enough runs", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 72)
    print(f"  CROSS-MACHINE AGGREGATE  --  {len(machines)} machines")
    print("=" * 72)
    if not HAS_SCIPY:
        print("  (scipy not available -- Welch p uses asymptotic normal)")

    hypotheses = list(KIN_HYPOTHESES)
    if args.include_dyn:
        hypotheses += DYN_HYPOTHESES

    # ---- (1) Per-machine summary table -----------------------------------
    print("\n--- (1) Per-machine summary: per-hypothesis median ratio ---------\n")
    rows = []
    for label in machines:
        d = machine_data[label]
        for h_id, desc, bench, lgn_s, rival_s, _rl in hypotheses:
            ns_lgn = {k[2] for k in d if k[0] == lgn_s and k[1] == bench}
            ns_riv = {k[2] for k in d if k[0] == rival_s and k[1] == bench}
            for n in sorted(ns_lgn & ns_riv):
                lgn_vals = d[(lgn_s, bench, n)]
                riv_vals = d[(rival_s, bench, n)]
                if len(lgn_vals) < 2 or len(riv_vals) < 2:
                    continue
                ratios = [b/a for a, b in zip(lgn_vals, riv_vals) if a > 0 and b > 0]
                if not ratios:
                    continue
                agg = aggregate(ratios)
                rows.append((label, h_id, f"n={n}", f"{agg['n']}",
                             fmt_x(agg["median"]),
                             fmt_x(agg["min"]) + " - " + fmt_x(agg["max"]),
                             fmt_pct(agg["cv"])))
    print_table(rows, ["machine", "H", "n", "runs", "median ratio", "min-max", "CV%"])

    # ---- (2) Pooled across machines --------------------------------------
    print("\n--- (2a) Pooled per-run ratios across all machines ---------------\n")
    print("    Pools all per-run ratios from every machine into one set.\n")
    rows = []
    for h_id, desc, bench, lgn_s, rival_s, _rl in hypotheses:
        all_ns: set = set()
        for label in machines:
            d = machine_data[label]
            for k in d:
                if k[0] == lgn_s and k[1] == bench:
                    all_ns.add(k[2])
        for n in sorted(all_ns):
            ratios = []
            for label in machines:
                d = machine_data[label]
                lgn_vals = d.get((lgn_s, bench, n), [])
                riv_vals = d.get((rival_s, bench, n), [])
                for a, b in zip(lgn_vals, riv_vals):
                    if a > 0 and b > 0:
                        ratios.append(b / a)
            if len(ratios) < 2:
                continue
            agg = aggregate(ratios)
            rows.append((h_id, desc, f"n={n}", f"{agg['n']}",
                         fmt_x(agg["median"]),
                         fmt_x(agg["min"]) + " - " + fmt_x(agg["max"]),
                         fmt_x(agg["p5"]) + " - " + fmt_x(agg["p95"]),
                         fmt_pct(agg["cv"])))
    print_table(rows, ["H", "Description", "n", "runs",
                       "median", "min-max", "p5-p95", "CV%"])

    print("\n--- (2b) Per-machine medians, aggregated across machines ---------\n")
    print("    Each machine contributes ONE number (its 10-run median).\n")
    rows = []
    for h_id, desc, bench, lgn_s, rival_s, _rl in hypotheses:
        all_ns: set = set()
        for label in machines:
            d = machine_data[label]
            for k in d:
                if k[0] == lgn_s and k[1] == bench:
                    all_ns.add(k[2])
        for n in sorted(all_ns):
            per_machine_med = []
            for label in machines:
                d = machine_data[label]
                lgn_vals = d.get((lgn_s, bench, n), [])
                riv_vals = d.get((rival_s, bench, n), [])
                rs = [b/a for a, b in zip(lgn_vals, riv_vals) if a > 0 and b > 0]
                if rs:
                    per_machine_med.append(float(np.median(rs)))
            if len(per_machine_med) < 2:
                continue
            agg = aggregate(per_machine_med)
            rows.append((h_id, desc, f"n={n}", f"{agg['n']}",
                         fmt_x(agg["median"]),
                         fmt_x(agg["min"]) + " - " + fmt_x(agg["max"]),
                         fmt_pct(agg["cv"])))
    print_table(rows, ["H", "Description", "n", "machines",
                       "median of medians", "min-max", "CV% across machines"])

    # ---- (3) Plots --------------------------------------------------------
    print("\n--- (3) Plots ----------------------------------------------------\n")
    for h_id, desc, bench, lgn_s, rival_s, rival_label in hypotheses:
        plot_per_machine_absolute(
            h_id, desc, bench, lgn_s, rival_s, rival_label,
            machines, machine_data,
            plots_dir / f"{h_id}_{bench}_per_machine.png"
        )
        invert = (h_id == "H04")
        plot_ratio_per_machine(
            h_id, desc, bench, lgn_s, rival_s,
            machines, machine_data,
            plots_dir / f"{h_id}_{bench}_ratios.png",
            invert=invert,
        )

    # ---- (4) Optional LaTeX block ----------------------------------------
    if args.markdown:
        write_markdown(Path(args.markdown), machines, machine_data, machine_moments, hypotheses)
        print(f"\nWrote LaTeX rows -> {args.markdown}")


if __name__ == "__main__":
    main()
