#!/usr/bin/env python3
"""
scripts/bench_stats.py
======================
Cross-run aggregator + plotter for N bench runs on one machine.

Merges the prior compute_stats_10x.py and plot_stats_10x.py into a
single script with one source of truth: the HYPOTHESES registry
below, including each hypothesis's `expect` direction.

Orientation rule (single, global)
---------------------------------
Every ratio in this script is computed in the EXPECT direction:

    expect == "lgn"   :  ratio = rival_time / lgn_time
                         ratio > 1  <=>  lgn won (faster)
    expect == "rival" :  ratio = lgn_time / rival_time
                         ratio > 1  <=>  rival won (faster)

So across the whole pipeline -- numerical tables, LaTeX rows, bar
plots, summary plot -- a value > 1 always means "the side we expected
to win, won." There is no per-hypothesis special case in the code.

The one inversion that actually exists is in the registry: H04
(velocity propagation) has expect="rival" because the FLOP-count
bound predicts scalar will win, and we want to express the
hypothesis honestly. Everywhere else, expect="lgn".

Within-run Welch is also oriented:
    expect == "lgn"   :  H1 is  mean_lgn   < mean_rival
    expect == "rival" :  H1 is  mean_rival < mean_lgn
A "rejection" in the verdict table therefore always means "the
expected winner is significantly faster in this run."

Two layers, one streaming pass per CSV
--------------------------------------
Within-run  : per-cell Welch one-tailed t-test (expect-oriented) and
              Cohen's d, both computed in closed form from running
              (sum, sum_of_squares, count) -- streaming-compatible
              and exact for our sample sizes.

Cross-run   : per-(solver, bench, n) aggregate of per-run means;
              per-(hypothesis, n) aggregate of per-run expect-oriented
              ratios; per-(hypothesis, n) robust verdict (how many of
              N per-run Welch tests rejected H0 at alpha=0.05).

CLI
---
    # compute only
    python3 scripts/bench_stats.py compute  run1.csv run2.csv ... \\
        --machine AMD-9950X --markdown rows.tex

    # plots only
    python3 scripts/bench_stats.py plot     run1.csv run2.csv ... \\
        --machine AMD-9950X --out-dir results_v2/plots/

    # both (single streaming pass per CSV, reused)
    python3 scripts/bench_stats.py all      run1.csv run2.csv ... \\
        --machine Z5 --markdown rows.tex --out-dir ../results_v2/plots/

Dependencies
------------
numpy, matplotlib. pandas and scipy optional (faster CSV streaming
and exact Student's-t survival respectively; stdlib + asymptotic
normal fallback kick in otherwise).
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

# Matplotlib only needed for `plot` and `all` subcommands. Import
# lazily so `compute` works on a headless box without it.
_plt = None
def _lazy_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


# ── Constants ───────────────────────────────────────────────────────────────
CHUNKSIZE            = 200_000
MIN_SAMPLES_PER_CELL = 50
TIME_COLS            = ("call_ns", "time_ns")
ALPHA                = 0.05


# ── Single source of truth: hypotheses ──────────────────────────────────────
# Tuple layout:
#   (h_id, description, benchmark, lgn_solver, rival_solver,
#    rival_label_for_plots, expect)
#
# `expect` is "lgn" everywhere except H04 (velocity-only), where the
# FLOP-count bound predicts scalar will win. See module docstring.
HYPOTHESES = [
    ("H01",         "lgn FK vs KDL FK",
        "fk",                "lgn",      "kdl",            "KDL",        "lgn"),
    ("H01p",        "lgn FK vs Pinocchio FK",
        "fk",                "lgn",      "pinocchio",      "Pinocchio",  "lgn"),
    ("H02p",        "lgn IK vs Pinocchio IK (idiomatic)",
        "ik_posonly",        "lgn",      "pinocchio",      "Pinocchio",  "lgn"),
    ("H02p_fair",   "lgn IK vs Pinocchio IK (fair math)",
        "ik_posonly_fair",   "lgn_fair", "pinocchio_fair", "Pinocchio",  "lgn"),
    ("H04",         "lgn Hv vs scalar vel (legacy)",
        "vel_only",          "lgn",      "scalar",         "scalar",     "rival"),
    ("H04_alloc",   "lgn Hv vs scalar vel (alloc)",
        "vel_only_alloc",    "lgn",      "scalar",         "scalar",     "rival"),
    ("H04_prealloc","lgn Hv vs scalar vel (prealloc)",
        "vel_only_prealloc", "lgn",      "scalar",         "scalar",     "rival"),
    ("H_M",         "lgn M vs Pinocchio CRBA",
        "dyn_M",             "lgn",      "pinocchio",      "Pinocchio",  "lgn"),
    ("H_C",         "lgn C vs Pinocchio NLE-G",
        "dyn_C",             "lgn",      "pinocchio",      "Pinocchio",  "lgn"),
    ("H_R",         "lgn RNEA vs Pinocchio RNEA",
        "dyn_RNEA_eq",       "lgn",      "pinocchio",      "Pinocchio",  "lgn"),
    ("H_A",         "lgn ABA vs Pinocchio ABA",
        "dyn_ABA_eq",        "lgn",      "pinocchio",      "Pinocchio",  "lgn"),
]

def _hyp_by_id(h_id: str):
    for h in HYPOTHESES:
        if h[0] == h_id:
            return h
    return None

# Sanity check: only H04* should be expect=rival. If this fires you've
# almost certainly mis-edited the registry.
def _validate_registry():
    for h in HYPOTHESES:
        h_id, _desc, _bench, _lgn, _riv, _lbl, expect = h
        assert expect in ("lgn", "rival"), f"{h_id}: bad expect {expect!r}"
        if expect == "rival":
            assert h_id.startswith("H04"), \
                f"unexpected expect=rival for {h_id}; only H04* should invert"
_validate_registry()


# ── Plot palette ────────────────────────────────────────────────────────────
COLOR_LGN       = "#E69F4D"   # warm orange -- lgn
COLOR_LGN_DARK  = "#C97B2A"
COLOR_KDL       = "#7E8AAB"
COLOR_PINOCCHIO = "#5D8A66"
COLOR_SCALAR    = "#9080A8"
COLOR_NEUTRAL   = "#9C9C9C"
COLOR_UNIT_LINE = "#222222"

RIVAL_COLOR = {
    "kdl":             COLOR_KDL,
    "pinocchio":       COLOR_PINOCCHIO,
    "pinocchio_fair":  COLOR_PINOCCHIO,
    "scalar":          COLOR_SCALAR,
}

# Hypotheses shown in the ratio summary plot, in legend order.
SUMMARY_DEFAULT = ["H01", "H01p", "H02p", "H02p_fair",
                   "H04_alloc", "H04_prealloc", "H04"]

DYN_IDS = {"H_M", "H_C", "H_R", "H_A"}


# ── Welch + Cohen's d from running moments ──────────────────────────────────
def _welch_lower(mean_a, var_a, n_a, mean_b, var_b, n_b):
    """
    One-tailed Welch t-test, H1: mean_a < mean_b, from sample moments.
    Returns (t, p_lower_tail, df). var_* are sample variances (ddof=1).
    """
    if n_a < 2 or n_b < 2 or var_a < 0 or var_b < 0:
        return float("nan"), float("nan"), float("nan")
    se_a = var_a / n_a
    se_b = var_b / n_b
    se = math.sqrt(se_a + se_b)
    if se <= 0:
        if mean_a < mean_b: return float("-inf"), 0.0, float("inf")
        if mean_a > mean_b: return float("inf"),  1.0, float("inf")
        return 0.0, 0.5, float("inf")
    t = (mean_a - mean_b) / se
    num = (se_a + se_b) ** 2
    den = (se_a ** 2) / (n_a - 1) + (se_b ** 2) / (n_b - 1)
    df = num / den if den > 0 else float("inf")
    if HAS_SCIPY:
        p_one = float(_scipy_stats.t.cdf(t, df))
    else:
        # asymptotic normal; df is huge in practice (hundreds of thousands)
        p_one = 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
    return t, p_one, df


def welch_oriented(expect: str,
                   m_lgn, v_lgn, n_lgn,
                   m_riv, v_riv, n_riv):
    """
    Welch oriented by `expect`. H1 is always "expected winner is faster",
    so a small p means the expected winner did beat the other side.
    """
    if expect == "lgn":
        # H1: mean_lgn < mean_rival
        return _welch_lower(m_lgn, v_lgn, n_lgn, m_riv, v_riv, n_riv)
    elif expect == "rival":
        # H1: mean_rival < mean_lgn
        return _welch_lower(m_riv, v_riv, n_riv, m_lgn, v_lgn, n_lgn)
    raise ValueError(f"expect must be 'lgn' or 'rival', got {expect!r}")


def cohen_d_oriented(expect: str,
                     m_lgn, v_lgn, n_lgn,
                     m_riv, v_riv, n_riv):
    """
    Signed Cohen's d such that positive d means the expected winner is
    faster (smaller time). For expect=lgn that's (m_rival - m_lgn)/pooled;
    for expect=rival it's the reverse. This matches the sign of the
    expect-oriented ratio's deviation from 1.
    """
    if n_lgn < 2 or n_riv < 2 or v_lgn < 0 or v_riv < 0:
        return float("nan")
    denom = (n_lgn - 1) * v_lgn + (n_riv - 1) * v_riv
    if denom <= 0:
        return float("nan")
    pooled = math.sqrt(denom / (n_lgn + n_riv - 2))
    if pooled <= 0:
        return float("nan")
    if expect == "lgn":
        return (m_riv - m_lgn) / pooled
    else:
        return (m_lgn - m_riv) / pooled


def oriented_ratio(expect: str, lgn_time: float, rival_time: float) -> float:
    """Single point of truth for the ratio orientation."""
    if lgn_time <= 0 or rival_time <= 0:
        return float("nan")
    return (rival_time / lgn_time) if expect == "lgn" else (lgn_time / rival_time)


# ── Streaming CSV loaders ───────────────────────────────────────────────────
def _pick_time_col(columns):
    for c in TIME_COLS:
        if c in columns:
            return c
    return None


def _extract_run_tag(path: Path) -> str:
    stem = path.stem
    m = re.match(r"^results_v2_raw_(.+)$", stem)
    return m.group(1) if m else stem


def stream_moments_pandas(path: Path) -> dict | None:
    """
    Return {(solver, bench, n): (mean, var_ddof1, count)} for one CSV.
    """
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception as e:
        print(f"  !! {path.name}: cannot read header ({e})", file=sys.stderr)
        return None
    tcol = _pick_time_col(head.columns)
    if tcol is None:
        print(f"  !! {path.name}: no timing column", file=sys.stderr)
        return None

    sum_x:  dict = defaultdict(float)
    sum_x2: dict = defaultdict(float)
    count:  dict = defaultdict(int)
    try:
        reader = pd.read_csv(
            path,
            usecols=["solver", "benchmark", "n", tcol],
            dtype={"solver": "string", "benchmark": "string"},
            chunksize=CHUNKSIZE,
            low_memory=False,
        )
        for chunk in reader:
            chunk["n"]   = pd.to_numeric(chunk["n"],   errors="coerce")
            chunk[tcol]  = pd.to_numeric(chunk[tcol],  errors="coerce")
            chunk = chunk.dropna(subset=["solver", "benchmark", "n", tcol])
            if chunk.empty:
                continue
            chunk["n"] = chunk["n"].astype(int)
            g = chunk.groupby(["solver", "benchmark", "n"])[tcol]
            agg = g.agg(["sum", lambda s: float((s * s).sum()), "count"])
            agg.columns = ["sum", "sum_sq", "count"]
            for (solver, bench, n_dof), row in agg.iterrows():
                key = (str(solver), str(bench), int(n_dof))
                sum_x[key]  += float(row["sum"])
                sum_x2[key] += float(row["sum_sq"])
                count[key]  += int(row["count"])
    except Exception as e:
        print(f"  !! {path.name}: read failed ({e})", file=sys.stderr)
        return None
    return _finalize_moments(sum_x, sum_x2, count)


def stream_moments_stdlib(path: Path) -> dict | None:
    sum_x:  dict = defaultdict(float)
    sum_x2: dict = defaultdict(float)
    count:  dict = defaultdict(int)
    try:
        with open(path, newline="") as f:
            reader = _csv.DictReader(f)
            tcol = _pick_time_col(reader.fieldnames or ())
            if tcol is None:
                print(f"  !! {path.name}: no timing column", file=sys.stderr)
                return None
            for row in reader:
                if row.get(tcol) == tcol or row.get("solver") == "solver":
                    continue
                try:
                    key = (row["solver"], row["benchmark"], int(row["n"]))
                    t = float(row[tcol])
                except (KeyError, ValueError, TypeError):
                    continue
                sum_x[key]  += t
                sum_x2[key] += t * t
                count[key]  += 1
    except Exception as e:
        print(f"  !! {path.name}: read failed ({e})", file=sys.stderr)
        return None
    return _finalize_moments(sum_x, sum_x2, count)


def _finalize_moments(sum_x, sum_x2, count) -> dict:
    out = {}
    for key, c in count.items():
        if c < MIN_SAMPLES_PER_CELL:
            continue
        mean = sum_x[key] / c
        if c < 2:
            var = float("nan")
        else:
            ss = sum_x2[key] - sum_x[key] * mean   # = sum_x2 - sum_x^2/n
            var = ss / (c - 1) if ss > 0 else 0.0
        out[key] = (mean, var, c)
    return out


def stream_moments(path: Path) -> dict | None:
    return stream_moments_pandas(path) if HAS_PANDAS else stream_moments_stdlib(path)


# ── Cross-run aggregation ───────────────────────────────────────────────────
def aggregate(values: list) -> dict:
    if not values:
        return {"n_runs": 0}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n_runs": 0}
    n_runs = int(arr.size)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n_runs >= 2 else 0.0
    return {
        "n_runs": n_runs,
        "median": float(np.median(arr)),
        "mean":   mean,
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "p5":     float(np.percentile(arr, 5)),
        "p95":    float(np.percentile(arr, 95)),
        "cv":     (100.0 * sd / mean) if mean > 0 else float("nan"),
    }


# ── Formatting helpers ──────────────────────────────────────────────────────
def fmt_ns(ns):
    if ns is None or (isinstance(ns, float) and math.isnan(ns)):
        return "-"
    if ns >= 1e6: return f"{ns/1e6:.2f} ms"
    if ns >= 1e3: return f"{ns/1e3:.2f} us"
    return f"{ns:.1f} ns"

def fmt_x(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "-"
    return f"{x:.2f}x"

def fmt_pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "-"
    return f"{x:.1f}%"

def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)): return "-"
    if p < 1e-6: return "<1e-6"
    return f"{p:.3g}"

def fmt_d(d):
    if d is None or (isinstance(d, float) and math.isnan(d)): return "-"
    return f"{d:+.2f}"

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
        print("| " + " | ".join(str(r[i]).ljust(widths[i])
                                 for i in range(len(headers))) + " |")
    print(sep)


# ── Pairing per-run results by run_tag ──────────────────────────────────────
def collect_per_run(runs: list) -> tuple[dict, dict, dict, dict]:
    """
    runs : list[(run_tag, moments)]   moments[cell] = (mean, var, count)

    Returns:
      cell_means [(solver,bench,n)]          -> list[mean_ns]      (per run)
      hyp_ratios [(h_id,n)]                  -> list[expect-oriented ratio]
      hyp_pvals  [(h_id,n)]                  -> list[expect-oriented p]
      hyp_ds     [(h_id,n)]                  -> list[expect-oriented d]

    Pairing is by run_tag: a run only contributes to (h_id, n) if it
    has BOTH solvers for that bench/n. No zip-by-position bug.
    """
    cell_means: dict = defaultdict(list)
    hyp_ratios: dict = defaultdict(list)
    hyp_pvals:  dict = defaultdict(list)
    hyp_ds:     dict = defaultdict(list)

    for _tag, moments in runs:
        for cell, (mean_ns, _v, _c) in moments.items():
            cell_means[cell].append(mean_ns)

        for h in HYPOTHESES:
            h_id, _desc, bench, lgn_s, rival_s, _lbl, expect = h
            ns_l = {k[2] for k in moments if k[0] == lgn_s   and k[1] == bench}
            ns_r = {k[2] for k in moments if k[0] == rival_s and k[1] == bench}
            for n in sorted(ns_l & ns_r):
                m_l, v_l, n_l = moments[(lgn_s,   bench, n)]
                m_r, v_r, n_r = moments[(rival_s, bench, n)]
                if m_l <= 0 or m_r <= 0:
                    continue
                hyp_ratios[(h_id, n)].append(oriented_ratio(expect, m_l, m_r))
                _, p, _ = welch_oriented(expect, m_l, v_l, n_l, m_r, v_r, n_r)
                hyp_pvals[(h_id, n)].append(p)
                hyp_ds[(h_id, n)].append(
                    cohen_d_oriented(expect, m_l, v_l, n_l, m_r, v_r, n_r))

    return cell_means, hyp_ratios, hyp_pvals, hyp_ds


# ── Compute reporting ───────────────────────────────────────────────────────
def print_compute_report(runs, cell_means, hyp_ratios, hyp_pvals, hyp_ds,
                          machine: str, min_runs: int):
    n_runs = len(runs)
    print()
    print("=" * 72)
    if machine:
        print(f"  CROSS-RUN AGGREGATE  --  machine: {machine}  --  {n_runs} runs")
    else:
        print(f"  CROSS-RUN AGGREGATE  --  {n_runs} runs")
    if not HAS_SCIPY:
        print("  (scipy not available -- Welch p-values use asymptotic normal "
              "approximation)")
    print("  Ratios are EXPECT-oriented: >1 means the expected winner won.")
    print("  H04* expect=rival (scalar); all others expect=lgn.")
    print("=" * 72)

    # (i) per-cell means across runs
    print("\n--- (i) Per-(solver, benchmark, n) cross-run means -----------------\n")
    rows = []
    for cell in sorted(cell_means.keys()):
        solver, bench, n = cell
        vals = cell_means[cell]
        if len(vals) < min_runs:
            continue
        agg = aggregate(vals)
        rows.append((
            solver, bench, f"n={n}",
            f"{agg['n_runs']}/{n_runs}",
            fmt_ns(agg["median"]), fmt_ns(agg["mean"]),
            fmt_ns(agg["min"]),    fmt_ns(agg["max"]),
            fmt_pct(agg["cv"]),
        ))
    print_table(rows, ["solver", "bench", "n", "runs",
                       "median", "mean", "min", "max", "CV%"])

    # (ii) per-hypothesis ratios across runs (expect-oriented)
    print("\n--- (ii) Per-hypothesis cross-run ratio (expect-oriented) ----------\n")
    print("    ratio > 1 means the expected winner won.")
    print("    H04* (expect=rival): >1 means scalar won (carrier penalty).\n")
    rows = []
    for h in HYPOTHESES:
        h_id, desc, _b, _l, _r, _lbl, expect = h
        ns = sorted({k[1] for k in hyp_ratios if k[0] == h_id})
        for n in ns:
            ratios = hyp_ratios[(h_id, n)]
            if len(ratios) < min_runs:
                continue
            agg = aggregate(ratios)
            rows.append((
                h_id, f"[{expect}] {desc}", f"n={n}",
                f"{agg['n_runs']}/{n_runs}",
                fmt_x(agg["median"]),
                fmt_x(agg["min"]) + " - " + fmt_x(agg["max"]),
                fmt_x(agg["p5"])  + " - " + fmt_x(agg["p95"]),
                fmt_pct(agg["cv"]),
            ))
    print_table(rows, ["H", "Description", "n", "runs",
                       "median", "min-max", "p5-p95", "CV%"])

    # (iii) verdict: "expected winner won" count + Welch reject count
    print("\n--- (iii) Per-hypothesis robust verdict ----------------------------\n")
    print(f"    Per-run Welch (one-tailed, H1: expected winner is faster), alpha={ALPHA}")
    print(f"    'reject' = runs whose Welch test rejected H0 in favour of the")
    print(f"               expected winner")
    print(f"    'exp_won' = runs whose median ratio came out > 1\n")
    rows = []
    for h in HYPOTHESES:
        h_id, _desc, _b, _l, _r, _lbl, _expect = h
        ns = sorted({k[1] for k in hyp_pvals if k[0] == h_id})
        for n in ns:
            pvals  = hyp_pvals[(h_id, n)]
            ratios = hyp_ratios[(h_id, n)]
            ds     = hyp_ds[(h_id, n)]
            if len(pvals) < min_runs:
                continue
            n_reject  = sum(1 for p in pvals
                            if p is not None and not math.isnan(p) and p < ALPHA)
            n_exp_won = sum(1 for r in ratios if r > 1.0)
            med_p = float(np.nanmedian([p for p in pvals if not math.isnan(p)])) \
                    if pvals else float("nan")
            med_d = float(np.nanmedian([d for d in ds    if not math.isnan(d)])) \
                    if ds    else float("nan")
            rows.append((
                h_id, f"n={n}",
                f"{len(pvals)}",
                f"{n_reject}/{len(pvals)}",
                f"{n_exp_won}/{len(ratios)}",
                fmt_x(aggregate(ratios)["median"]),
                fmt_p(med_p),
                fmt_d(med_d),
            ))
    print_table(rows, ["H", "n", "runs", "reject", "exp_won",
                       "median_ratio", "median_p", "median_d"])


# ── LaTeX rows for the paper ────────────────────────────────────────────────
def write_markdown(out_path: Path, machine: str, n_runs: int,
                   cell_means: dict, hyp_ratios: dict, hyp_pvals: dict):
    label = machine or "M?"
    lines = []
    lines.append(f"% Cross-run aggregate over {n_runs} runs on {label}")
    lines.append(f"% Each cell value is the median across {n_runs} runs.")
    lines.append(f"% Ratios are expect-oriented: >1 means expected winner won.")
    lines.append(f"% H04* (velocity) is expect=rival; all others expect=lgn.")
    lines.append(f"% Rejection counts (N/{n_runs}): per-run Welch with H1 = "
                 f"expected winner faster, alpha={ALPHA}.")
    lines.append("")

    def med_ns(solver, bench, n):
        v = cell_means.get((solver, bench, n), [])
        return float(np.median(v)) if v else None

    def med_ratio(h_id, n):
        v = hyp_ratios.get((h_id, n), [])
        return float(np.median(v)) if v else None

    def fnum(x, digits=1):
        return f"{x:,.{digits}f}" if x is not None else "---"

    def rfmt(x):
        return f"{x:.2f}$\\times$" if x is not None else "---"

    # FK
    lines.append(f"% --- Table II (FK) row block for {label} ---")
    lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{label}: {n_runs}-run median}}}} \\")
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        lgn = med_ns("lgn", "fk", n)
        kdl = med_ns("kdl", "fk", n)
        pin = med_ns("pinocchio", "fk", n)
        if lgn is None:
            continue
        lines.append(
            f"{n:<4} & {fnum(lgn):>8} & {fnum(kdl):>8} & {fnum(pin):>8} & "
            f"{rfmt(med_ratio('H01', n))} & {rfmt(med_ratio('H01p', n))} \\\\"
        )
    lines.append("")

    # IK
    lines.append(f"% --- Table III (IK) row block for {label} ---")
    lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{label}: {n_runs}-run median}}}} \\")
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        lgn = med_ns("lgn",       "ik_posonly", n)
        pin = med_ns("pinocchio", "ik_posonly", n)
        kdl = med_ns("kdl",       "ik_posonly", n)
        if lgn is None:
            continue
        lines.append(
            f"{n:<4} & {fnum(lgn,0):>9} & {fnum(pin,0):>9} & "
            f"{rfmt(med_ratio('H02p', n))} & {fnum(kdl,0):>11} \\\\"
        )
    lines.append("")

    # Velocity propagation (expect=rival; ratio>1 means scalar won)
    lines.append(f"% --- Table IV (velprop) row block for {label} ---")
    lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{label}: {n_runs}-run median, "
                 rf"ratio>1 = scalar won}}}} \\")
    for n in [2, 4, 8, 16, 32, 64, 128]:
        lgn = med_ns("lgn",    "vel_only", n)
        sca = med_ns("scalar", "vel_only", n)
        if lgn is None or sca is None:
            continue
        # expect=rival, so ratio = lgn/scalar
        r = med_ratio("H04", n)
        rstr = f"{r:.2f}$\\times$" if r is not None else "---"
        lines.append(f"{n:<4} & {lgn:>8.1f} & {sca:>8.1f} & {rstr} \\\\")
    lines.append("")

    # Per-hypothesis verdicts
    lines.append(f"% --- Cross-run hypothesis verdicts on {label} ---")
    lines.append("% Format: H_id  n  reject/n_runs  median_ratio (expect-oriented)")
    for h in HYPOTHESES:
        h_id, _desc, _b, _l, _r, _lbl, _expect = h
        ns = sorted({k[1] for k in hyp_pvals if k[0] == h_id})
        for n in ns:
            pvals = hyp_pvals.get((h_id, n), [])
            if not pvals:
                continue
            n_rej = sum(1 for p in pvals
                        if p is not None and not math.isnan(p) and p < ALPHA)
            med = med_ratio(h_id, n)
            med_str = f"{med:.2f}x" if med is not None else "---"
            lines.append(f"% {h_id:12s} n={n:3d}  reject {n_rej}/{len(pvals)}  "
                         f"median ratio {med_str}")
    lines.append("")
    out_path.write_text("\n".join(lines))


# ── Plotting ────────────────────────────────────────────────────────────────
def _ratio_color_for_outcome(expect: str, ratio: float) -> str:
    """
    Color a ratio bar in the summary plot by whether lgn actually beat
    the comparison (regardless of expect direction):

      expect=lgn   :  ratio > 1  -> lgn won as expected  -> orange
                      ratio < 1  -> lgn lost              -> neutral
      expect=rival :  ratio > 1  -> rival won as expected -> neutral
                      ratio < 1  -> lgn beat expectation  -> orange

    The "lgn upset" case in expect=rival is rare and worth flagging.
    """
    if math.isnan(ratio):
        return "#cccccc"
    lgn_actually_won = (expect == "lgn"   and ratio > 1.0) \
                    or (expect == "rival" and ratio < 1.0)
    return COLOR_LGN if lgn_actually_won else COLOR_NEUTRAL


def plot_hypothesis_bars(h, per_run_means: dict[str, dict],
                         out_path: Path, machine: str) -> bool:
    """
    lgn (orange) vs rival, log-y, cross-run median bars with min-max
    whiskers. One pair per n. Pairing per run_tag.
    """
    plt = _lazy_plt()
    h_id, desc, bench, lgn_s, rival_s, rival_label, expect = h

    runs_with_both: list[str] = []
    lgn_by_n: dict[int, list[float]] = defaultdict(list)
    riv_by_n: dict[int, list[float]] = defaultdict(list)
    for run_tag, means in per_run_means.items():
        ns_l = {k[2] for k in means if k[0] == lgn_s   and k[1] == bench}
        ns_r = {k[2] for k in means if k[0] == rival_s and k[1] == bench}
        common = ns_l & ns_r
        if not common:
            continue
        runs_with_both.append(run_tag)
        for n in common:
            lgn_by_n[n].append(means[(lgn_s,   bench, n)])
            riv_by_n[n].append(means[(rival_s, bench, n)])

    ns = sorted(n for n in lgn_by_n if len(lgn_by_n[n]) >= 2)
    if not ns:
        print(f"    {h_id}: <2 runs with both solvers; skipping")
        return False

    def stats(vals):
        arr = np.asarray(vals, dtype=float)
        med = float(np.median(arr))
        return med, med - float(np.min(arr)), float(np.max(arr)) - med

    lgn_med, lgn_lo, lgn_hi = zip(*(stats(lgn_by_n[n]) for n in ns))
    riv_med, riv_lo, riv_hi = zip(*(stats(riv_by_n[n]) for n in ns))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(ns))
    w = 0.38
    rival_color = RIVAL_COLOR.get(rival_s, COLOR_NEUTRAL)

    ax.bar(x - w/2, lgn_med, w, color=COLOR_LGN, label="lgn",
           edgecolor=COLOR_LGN_DARK, linewidth=0.6,
           yerr=[list(lgn_lo), list(lgn_hi)], capsize=4,
           error_kw={"elinewidth": 1.0, "ecolor": "#222"})
    ax.bar(x + w/2, riv_med, w, color=rival_color, label=rival_label,
           edgecolor="#444", linewidth=0.4,
           yerr=[list(riv_lo), list(riv_hi)], capsize=4,
           error_kw={"elinewidth": 1.0, "ecolor": "#222"})

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_yscale("log")
    ax.set_ylabel("Mean time per call (ns, log)")
    n_runs = len(runs_with_both)
    suffix = f"  --  {machine}, {n_runs} runs" if machine else f"  --  {n_runs} runs"
    expect_note = "expect lgn" if expect == "lgn" else "expect rival (carrier penalty)"
    ax.set_title(f"{h_id}: {desc}{suffix}   [{expect_note}]")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", which="both", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    fig.text(0.5, 0.005,
             "Error bars: min--max across runs  |  bar height: cross-run median",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    wrote {out_path}")
    return True


def plot_ratio_summary(h_ids: list[str],
                       per_run_means: dict[str, dict],
                       out_path: Path, machine: str,
                       n_runs_loaded: int) -> bool:
    """
    One bar per (hypothesis, n). Bar height = cross-run median of the
    expect-oriented ratio. >1 always means the expected winner won.
    Bar color flags whether lgn actually beat the comparison (orange)
    or not (neutral gray), regardless of expect direction.
    """
    plt = _lazy_plt()
    series = []
    for h_id in h_ids:
        h = _hyp_by_id(h_id)
        if h is None:
            continue
        _id, desc, bench, lgn_s, rival_s, _rl, expect = h

        per_n: dict[int, list[float]] = defaultdict(list)
        for _tag, means in per_run_means.items():
            ns_l = {k[2] for k in means if k[0] == lgn_s   and k[1] == bench}
            ns_r = {k[2] for k in means if k[0] == rival_s and k[1] == bench}
            for n in (ns_l & ns_r):
                r = oriented_ratio(expect,
                                   means[(lgn_s,   bench, n)],
                                   means[(rival_s, bench, n)])
                if not math.isnan(r):
                    per_n[n].append(r)

        per_n_stats: dict[int, tuple[float, float, float]] = {}
        for n, rs in per_n.items():
            if len(rs) < 2:
                continue
            arr = np.asarray(rs, dtype=float)
            med = float(np.median(arr))
            per_n_stats[n] = (med,
                              med - float(np.min(arr)),
                              float(np.max(arr)) - med)
        if per_n_stats:
            series.append((h_id, expect, desc, per_n_stats))

    if not series:
        print("    ratio summary: no data; skipping")
        return False

    all_ns = sorted({n for _id, _e, _d, s in series for n in s})
    x = np.arange(len(all_ns))

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    k = len(series)
    w = 0.82 / k

    legend_handles = []
    for i, (h_id, expect, desc, per_n) in enumerate(series):
        offset = -0.41 + w/2 + i * w
        for j, n in enumerate(all_ns):
            if n not in per_n:
                continue
            med, lo, hi = per_n[n]
            color = _ratio_color_for_outcome(expect, med)
            ax.bar(x[j] + offset, med, w, color=color,
                   edgecolor="#222", linewidth=0.4,
                   yerr=[[lo], [hi]], capsize=2.5,
                   error_kw={"elinewidth": 0.9, "ecolor": "#222"})

        meds = [per_n[n][0] for n in all_ns if n in per_n]
        lgn_wins_count = sum(1 for r in meds
                             if (expect == "lgn"   and r > 1.0)
                             or (expect == "rival" and r < 1.0))
        legend_color = COLOR_LGN if lgn_wins_count > len(meds) / 2 \
                                 else COLOR_NEUTRAL
        short = desc.split(":")[0] if ":" in desc else desc
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=legend_color,
                          label=f"{h_id}  ({short})")
        )

    ax.axhline(1.0, color=COLOR_UNIT_LINE, linewidth=1.2,
               linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in all_ns])
    ax.set_ylabel("expect-oriented ratio (>1 = expected winner won)")
    title = "Cross-run ratios per hypothesis"
    if machine:
        title += f"  --  {machine}, {n_runs_loaded} runs"
    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper left",
              framealpha=0.95, fontsize=8.5, ncol=2)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    fig.text(0.5, 0.005,
             "Height: cross-run median ratio (oriented by hypothesis `expect`). "
             "Whiskers: min--max across runs. "
             "Orange: lgn actually beat the comparison. "
             "Gray: expected outcome held with no lgn upset.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    wrote {out_path}")
    return True


def per_run_means_from_runs(runs: list) -> dict[str, dict]:
    """Strip variance/count -> just per-cell means, keyed by run_tag."""
    out: dict[str, dict] = {}
    for tag, moments in runs:
        out[tag] = {cell: mean for cell, (mean, _v, _c) in moments.items()}
    return out


def run_plots(runs: list, machine: str, out_dir: Path,
              include_dyn: bool, summary_only: bool):
    per_run_means = per_run_means_from_runs(runs)

    print()
    print(f"Drawing plots for {machine or 'unlabelled'}  "
          f"({len(per_run_means)} runs)")
    print("-" * 72)

    if not summary_only:
        for h in HYPOTHESES:
            h_id = h[0]
            if h_id in DYN_IDS and not include_dyn:
                continue
            bench = h[2]
            plot_hypothesis_bars(h, per_run_means,
                                 out_dir / f"{h_id}_{bench}.png", machine)

    summary_ids = [h_id for h_id in SUMMARY_DEFAULT
                   if _hyp_by_id(h_id) is not None]
    if include_dyn:
        summary_ids += [h_id for h_id in ("H_M", "H_C", "H_R", "H_A")
                        if _hyp_by_id(h_id) is not None]
    plot_ratio_summary(summary_ids, per_run_means,
                       out_dir / "ratios_summary.png",
                       machine, len(per_run_means))

    print("-" * 72)
    print(f"Done. Plots in {out_dir}/")


# ── Shared loader ───────────────────────────────────────────────────────────
def load_all_runs(paths: list[Path]) -> list:
    print(f"Streaming {len(paths)} run(s)  (CHUNKSIZE={CHUNKSIZE:,})...",
          flush=True)
    runs = []
    for p in sorted(paths):
        tag = _extract_run_tag(p)
        print(f"  loading [{tag}]...", end="", flush=True)
        moments = stream_moments(p)
        if moments is None:
            print(" FAILED")
            continue
        print(f" {len(moments)} cells", flush=True)
        runs.append((tag, moments))
    return runs


# ── CLI ─────────────────────────────────────────────────────────────────────
def _add_common_io_args(sp):
    sp.add_argument("csv", nargs="+",
                    help="one CSV per run, e.g. results_v2_raw_<RUN_TAG>.csv")
    sp.add_argument("--machine", default="",
                    help="machine label for headers/titles")


def main():
    ap = argparse.ArgumentParser(
        description="Cross-run aggregator + plotter for N bench runs. "
                    "All ratios are expect-oriented: >1 means the expected "
                    "winner won. H04* expects rival (scalar); all others "
                    "expect lgn."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_c = sub.add_parser("compute", help="numerical aggregate only")
    _add_common_io_args(sp_c)
    sp_c.add_argument("--markdown", default="",
                      help="write LaTeX-style table rows for the paper")
    sp_c.add_argument("--min-runs", type=int, default=3,
                      help="minimum runs per cell to report (default: 3)")

    sp_p = sub.add_parser("plot", help="plots only")
    _add_common_io_args(sp_p)
    sp_p.add_argument("--out-dir", default="plots_10x",
                      help="output directory for PNGs (default: plots_10x/)")
    sp_p.add_argument("--include-dyn", action="store_true",
                      help="also plot dynamics hypotheses (H_M/H_C/H_R/H_A)")
    sp_p.add_argument("--summary-only", action="store_true",
                      help="emit only the ratio summary plot")

    sp_a = sub.add_parser("all", help="compute + plot from a single load")
    _add_common_io_args(sp_a)
    sp_a.add_argument("--markdown", default="")
    sp_a.add_argument("--min-runs", type=int, default=3)
    sp_a.add_argument("--out-dir", default="plots_10x")
    sp_a.add_argument("--include-dyn", action="store_true")
    sp_a.add_argument("--summary-only", action="store_true")

    args = ap.parse_args()

    paths = [Path(p) for p in args.csv]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"ERROR: file(s) not found: {missing}", file=sys.stderr)
        sys.exit(1)

    runs = load_all_runs(paths)
    if not runs:
        print("ERROR: no runs loaded.", file=sys.stderr)
        sys.exit(1)

    if args.cmd in ("compute", "all"):
        if len(runs) < args.min_runs:
            print(f"WARNING: only {len(runs)} runs loaded "
                  f"(--min-runs={args.min_runs})", file=sys.stderr)
        cm, hr, hp, hd = collect_per_run(runs)
        print_compute_report(runs, cm, hr, hp, hd,
                             machine=args.machine, min_runs=args.min_runs)
        if args.markdown:
            write_markdown(Path(args.markdown), args.machine, len(runs),
                           cm, hr, hp)
            print(f"\nWrote paper-ready table rows -> {args.markdown}")

    if args.cmd in ("plot", "all"):
        if len(runs) < 2:
            print("ERROR: need at least 2 runs to draw error bars.",
                  file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_plots(runs, args.machine, out_dir,
                  include_dyn=args.include_dyn,
                  summary_only=args.summary_only)


if __name__ == "__main__":
    main()
