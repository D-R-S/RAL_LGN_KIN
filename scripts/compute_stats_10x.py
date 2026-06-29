#!/usr/bin/env python3
"""
scripts/compute_stats_10x.py
============================
Cross-run aggregator over N bench runs on one machine.

Two-layer statistics, restored from the original design:

  Within-run  : per-(solver, benchmark, n) Welch one-tailed t-test,
                Cohen's d, lgn vs rival. Computed from running
                (sum, sum_of_squares, count), so streaming-compatible
                and exact.

  Cross-run   : per-(solver, bench, n) cross-run aggregate of per-run
                means; per-(hypothesis, n) cross-run aggregate of
                per-run speedup ratios; per-(hypothesis, n) robust
                verdict = how many of the N runs' Welch tests rejected
                H0 in favour of lgn at alpha = 0.05.

Streaming design
----------------
Each input CSV is processed once, in chunks of CHUNKSIZE rows. We
accumulate per (solver, benchmark, n):

    sum_x        running sum of times          (for mean)
    sum_x2       running sum of squared times  (for variance)
    count        number of samples

These three numbers per cell are sufficient for Welch's t and Cohen's d
in closed form -- we never need to hold the per-sample array.

Peak RAM is O(cells_per_run) = a few hundred floats, not O(samples).

Output
------
  (i)   Per-(solver, bench, n) cross-run aggregate of per-run means.
  (ii)  Per-(hypothesis, n) cross-run aggregate of per-run speedup ratios.
  (iii) Per-(hypothesis, n) robust verdict, including per-run Welch
        rejection count out of N.

Plus --markdown writes paper-ready LaTeX table rows.

Dependencies
------------
numpy. scipy is optional (only the survival function of the t
distribution is needed; we use a numpy-only Welch--Satterthwaite +
asymptotic-normal fallback if scipy is missing). pandas is optional;
the stdlib csv loader works without it.
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

# scipy for an exact Student's t survival function. Fall back to
# numpy-only Welch (asymptotic normal) if absent -- accurate for the
# huge sample sizes we deal with.
try:
    from scipy import stats as _scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNKSIZE = 200_000               # rows per streaming chunk
MIN_SAMPLES_PER_CELL = 50         # drop crashed-binary partial cells
TIME_COLS = ("call_ns", "time_ns")
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Hypothesis registry  (matches compute_stats.py)
# ---------------------------------------------------------------------------
HYPOTHESES = [
    ("H01",  "lgn FK vs KDL FK",                "fk",          "lgn",  "kdl",        "KDL"),
    ("H01p", "lgn FK vs Pinocchio FK",          "fk",          "lgn",  "pinocchio",  "Pinocchio"),
    ("H02p", "lgn IK vs Pinocchio IK",          "ik_posonly",  "lgn",  "pinocchio",  "Pinocchio"),
    ("H04",  "Scalar vs lgn vel",            "vel_only",    "lgn",  "scalar",     "scalar"),
    ("H_M",  "lgn M vs Pinocchio CRBA",         "dyn_M",       "lgn",  "pinocchio",  "Pinocchio"),
    ("H_C",  "lgn C vs Pinocchio NLE-G",        "dyn_C",       "lgn",  "pinocchio",  "Pinocchio"),
    ("H_R",  "lgn RNEA vs Pinocchio RNEA",      "dyn_RNEA_eq", "lgn",  "pinocchio",  "Pinocchio"),
    ("H_A",  "lgn ABA vs Pinocchio ABA",        "dyn_ABA_eq",  "lgn",  "pinocchio",  "Pinocchio"),
    ("H02pf", "lgn IK vs Pinocchio IK (fair math)","ik_posonly_fair", "lgn_fair", "pinocchio_fair", "Pinocchio"),
]


# ---------------------------------------------------------------------------
# Welch's t-test from running moments
# ---------------------------------------------------------------------------
def welch_from_moments(mean_a, var_a, n_a, mean_b, var_b, n_b):
    """
    One-tailed Welch t-test (H1: mean_a < mean_b) from sample moments.

    var_a, var_b must be the sample variance (ddof=1) of the two arrays.
    Returns (t_stat, p_one_tailed, df).

    Closed form, exact: this is what scipy.ttest_ind(equal_var=False)
    computes internally; we just feed it the moments rather than the
    full arrays. For our 100k+ sample sizes the cost of running
    scipy on the array would be ~ms per call -- not a problem at
    the call rate we have. We use the moment form purely because it
    composes with the streaming pass, no other reason.
    """
    if n_a < 2 or n_b < 2 or var_a < 0 or var_b < 0:
        return float("nan"), float("nan"), float("nan")

    se_a = var_a / n_a
    se_b = var_b / n_b
    se = math.sqrt(se_a + se_b)
    if se <= 0:
        # Both samples are constant. If they differ, p is effectively 0;
        # if they don't, t is 0 and p is 0.5.
        if mean_a < mean_b:
            return float("-inf"), 0.0, float("inf")
        elif mean_a > mean_b:
            return float("inf"), 1.0, float("inf")
        else:
            return 0.0, 0.5, float("inf")

    t = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (se_a + se_b) ** 2
    denom = (se_a ** 2) / (n_a - 1) + (se_b ** 2) / (n_b - 1)
    df = num / denom if denom > 0 else float("inf")

    if HAS_SCIPY:
        # Exact Student's t survival function. One-tailed lower-tail.
        p_one = float(_scipy_stats.t.cdf(t, df))
    else:
        # Fall back to asymptotic normal: for df > 30 the agreement
        # with the exact t is at the 0.1% level. With our sample sizes
        # we are well above that.
        p_one = 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    return t, p_one, df


def cohen_d_from_moments(mean_a, var_a, n_a, mean_b, var_b, n_b):
    """Cohen's d from sample moments (pooled SD, ddof=1)."""
    if n_a < 2 or n_b < 2 or var_a < 0 or var_b < 0:
        return float("nan")
    denom = (n_a - 1) * var_a + (n_b - 1) * var_b
    if denom <= 0:
        return float("nan")
    pooled = math.sqrt(denom / (n_a + n_b - 2))
    if pooled <= 0:
        return float("nan")
    return (mean_a - mean_b) / pooled


# ---------------------------------------------------------------------------
# Streaming CSV reader -> per-cell running moments
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
    """
    Stream a CSV in chunks; return per-cell running moments:
        {(solver, bench, n): (sum_x, sum_x2, count)}
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
            # Per-chunk groupby on small chunk -> fast.
            g = chunk.groupby(["solver", "benchmark", "n"])[tcol]
            agg = g.agg(["sum", lambda s: float((s * s).sum()), "count"])
            agg.columns = ["sum", "sum_sq", "count"]
            for (solver, bench, n_dof), row in agg.iterrows():
                key = (str(solver), str(bench), int(n_dof))
                sum_x[key] += float(row["sum"])
                sum_x2[key] += float(row["sum_sq"])
                count[key] += int(row["count"])
    except Exception as e:
        print(f"  !! {path.name}: read failed ({e})", file=sys.stderr)
        return None

    return _finalize_moments(sum_x, sum_x2, count)


def stream_run_stdlib(path: Path) -> dict | None:
    """Pandas-free fallback."""
    sum_x: dict = defaultdict(float)
    sum_x2: dict = defaultdict(float)
    count: dict = defaultdict(int)
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
                sum_x[key] += t
                sum_x2[key] += t * t
                count[key] += 1
    except Exception as e:
        print(f"  !! {path.name}: read failed ({e})", file=sys.stderr)
        return None

    return _finalize_moments(sum_x, sum_x2, count)


def _finalize_moments(sum_x, sum_x2, count) -> dict:
    """
    Convert running moments to {cell: (mean, var, count)}.

    Variance is the unbiased sample variance (ddof=1) computed in a
    numerically-stable two-pass way: we have sum_x and sum_x2 so the
    one-pass formula  var = (sum_x2 - sum_x^2/n) / (n-1)  applies. For
    the sample sizes we deal with (hundreds of thousands) the
    cancellation error in this form is fine; the alternative would be
    Welford's online algorithm which would require touching every
    sample (we don't, by design).
    """
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


def stream_run(path: Path) -> dict | None:
    if HAS_PANDAS:
        return stream_run_pandas(path)
    return stream_run_stdlib(path)


# ---------------------------------------------------------------------------
# Cross-run aggregation
# ---------------------------------------------------------------------------
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
        "mean": mean,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "cv": (100.0 * sd / mean) if mean > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_ns(ns):
    if ns is None or (isinstance(ns, float) and math.isnan(ns)):
        return "-"
    if ns >= 1e6:
        return f"{ns/1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.2f} us"
    return f"{ns:.1f} ns"


def fmt_x(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.2f}x"


def fmt_pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.1f}%"


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    if p < 1e-6:
        return "<1e-6"
    return f"{p:.3g}"


def fmt_d(d):
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return "-"
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
        print("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)


# ---------------------------------------------------------------------------
# LaTeX block writer
# ---------------------------------------------------------------------------
def write_markdown(out_path: Path, machine: str, n_runs: int,
                   cell_means: dict, hyp_ratios: dict, hyp_pvals: dict):
    """Emit LaTeX rows for the three kinematic tables, with footnote on rejection rate."""
    label = machine or "M?"
    lines = []
    lines.append(f"% Cross-run aggregate over {n_runs} runs on {label}")
    lines.append(f"% Each cell value is the median across {n_runs} runs.")
    lines.append(f"% Ratios annotated with N/{n_runs} = how many runs' Welch tests rejected H0 (lgn < rival, alpha={ALPHA}).")
    lines.append("")

    def med_ns(solver, bench, n):
        v = cell_means.get((solver, bench, n), [])
        return float(np.median(v)) if v else None

    def med_ratio(h_id, n):
        v = hyp_ratios.get((h_id, n), [])
        return float(np.median(v)) if v else None

    def rejection_count(h_id, n):
        v = hyp_pvals.get((h_id, n), [])
        if not v:
            return None, 0
        n_rej = sum(1 for p in v if p is not None and not math.isnan(p) and p < ALPHA)
        return n_rej, len(v)

    # FK
    lines.append(f"% --- Table II (FK) row block for {label} ---")
    lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{label}: {n_runs}-run median}}}} \\")
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        lgn = med_ns("lgn", "fk", n)
        kdl = med_ns("kdl", "fk", n)
        pin = med_ns("pinocchio", "fk", n)
        r_kdl = med_ratio("H01", n)
        r_pin = med_ratio("H01p", n)
        if lgn is None:
            continue
        def f(x):
            return f"{x:.1f}" if x is not None else "---"
        def r(x):
            return f"{x:.2f}$\\times$" if x is not None else "---"
        lines.append(f"{n:<4} & {f(lgn):>8} & {f(kdl):>8} & {f(pin):>8} & {r(r_kdl)} & {r(r_pin)} \\\\")
    lines.append("")

    # IK
    lines.append(f"% --- Table III (IK) row block for {label} ---")
    lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{label}: {n_runs}-run median}}}} \\")
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        lgn = med_ns("lgn", "ik_posonly", n)
        pin = med_ns("pinocchio", "ik_posonly", n)
        kdl = med_ns("kdl", "ik_posonly", n)
        r_pin = med_ratio("H02p", n)
        if lgn is None:
            continue
        def fnum(x, digits=0):
            return f"{x:,.{digits}f}" if x is not None else "---"
        def r(x):
            return f"{x:.2f}$\\times$" if x is not None else "---"
        lines.append(f"{n:<4} & {fnum(lgn):>9} & {fnum(pin):>9} & {r(r_pin)} & {fnum(kdl):>11} \\\\")
    lines.append("")

    # Velocity propagation
    lines.append(f"% --- Table IV (velprop) row block for {label} ---")
    lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{label}: {n_runs}-run median}}}} \\")
    for n in [2, 4, 8, 16, 32, 64, 128]:
        lgn = med_ns("lgn", "vel_only", n)
        sca = med_ns("scalar", "vel_only", n)
        if lgn is None or sca is None:
            continue
        ratio_lgn_over_scalar = (lgn / sca) if sca > 0 else float("nan")
        lines.append(f"{n:<4} & {lgn:>8.1f} & {sca:>8.1f} & {ratio_lgn_over_scalar:.2f}$\\times$ \\\\")
    lines.append("")

    # Per-hypothesis rejection summary as a separate block
    lines.append(f"% --- Cross-run hypothesis verdicts on {label} ---")
    lines.append("% Format: H_id  n  reject_count/n_runs  median_ratio")
    for h_id, _desc, _bench, _lgn, _riv, _label in HYPOTHESES:
        ns = sorted({k[1] for k in hyp_pvals if k[0] == h_id})
        for n in ns:
            n_rej, n_tot = rejection_count(h_id, n)
            med = med_ratio(h_id, n)
            if n_tot == 0:
                continue
            lines.append(f"% {h_id:6s} n={n:3d}  reject {n_rej}/{n_tot}  median ratio {med:.2f}x")
    lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Cross-run aggregator over N bench runs on one machine "
                    "(streaming, Welch t-test preserved via moment form)"
    )
    ap.add_argument("csv", nargs="+",
                    help="one CSV per run, e.g. results_v2_raw_<RUN_TAG>.csv")
    ap.add_argument("--machine", default="",
                    help="machine label for the report header (e.g. M2-Zen5)")
    ap.add_argument("--markdown", default="",
                    help="write LaTeX-style table rows for the paper to this file")
    ap.add_argument("--min-runs", type=int, default=3,
                    help="minimum runs per cell to report it (default: 3)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.csv]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"ERROR: file(s) not found: {missing}", file=sys.stderr)
        sys.exit(1)

    # ---- Stream all runs --------------------------------------------------
    # runs[i] = (run_tag, moments_dict)
    # moments_dict[cell] = (mean, var, count)
    runs: list = []
    print(f"Streaming {len(paths)} run(s)  (CHUNKSIZE={CHUNKSIZE:,})...", flush=True)
    for p in sorted(paths):
        tag = _extract_run_tag(p)
        print(f"  loading [{tag}]...", end="", flush=True)
        moments = stream_run(p)
        if moments is None:
            print(" FAILED")
            continue
        print(f" {len(moments)} cells", flush=True)
        runs.append((tag, moments))

    if not runs:
        print("ERROR: no runs loaded.", file=sys.stderr)
        sys.exit(1)
    if len(runs) < args.min_runs:
        print(f"WARNING: only {len(runs)} runs loaded (--min-runs={args.min_runs})",
              file=sys.stderr)

    n_runs = len(runs)
    print()
    print("=" * 72)
    if args.machine:
        print(f"  CROSS-RUN AGGREGATE  --  machine: {args.machine}  --  {n_runs} runs")
    else:
        print(f"  CROSS-RUN AGGREGATE  --  {n_runs} runs")
    if not HAS_SCIPY:
        print("  (scipy not available -- Welch p-values use asymptotic normal approximation)")
    print("=" * 72)

    # ---- Per-(solver, bench, n) cross-run means ---------------------------
    cell_means: dict = defaultdict(list)
    for _tag, moments in runs:
        for cell, (mean_ns, _var, _c) in moments.items():
            cell_means[cell].append(mean_ns)

    # ---- Per-hypothesis: per-run ratios, p-values, d ---------------------
    hyp_ratios: dict = defaultdict(list)
    hyp_pvals: dict = defaultdict(list)
    hyp_ds: dict = defaultdict(list)

    for _tag, moments in runs:
        for h_id, _desc, bench, lgn_s, rival_s, _label in HYPOTHESES:
            ns_lgn = {k[2] for k in moments if k[0] == lgn_s and k[1] == bench}
            ns_riv = {k[2] for k in moments if k[0] == rival_s and k[1] == bench}
            for n in sorted(ns_lgn & ns_riv):
                ma, va, na = moments[(lgn_s, bench, n)]
                mb, vb, nb = moments[(rival_s, bench, n)]
                if ma <= 0 or mb <= 0:
                    continue
                hyp_ratios[(h_id, n)].append(mb / ma)
                _, p, _ = welch_from_moments(ma, va, na, mb, vb, nb)
                hyp_pvals[(h_id, n)].append(p)
                hyp_ds[(h_id, n)].append(cohen_d_from_moments(ma, va, na, mb, vb, nb))

    # ---- Print (i) ---------------------------------------------------------
    print("\n--- (i) Per-(solver, benchmark, n) cross-run means -----------------\n")
    rows = []
    for cell in sorted(cell_means.keys()):
        solver, bench, n = cell
        vals = cell_means[cell]
        if len(vals) < args.min_runs:
            continue
        agg = aggregate(vals)
        rows.append((
            solver, bench, f"n={n}",
            f"{agg['n_runs']}/{n_runs}",
            fmt_ns(agg["median"]),
            fmt_ns(agg["mean"]),
            fmt_ns(agg["min"]),
            fmt_ns(agg["max"]),
            fmt_pct(agg["cv"]),
        ))
    print_table(rows, ["solver", "bench", "n", "runs",
                       "median", "mean", "min", "max", "CV%"])

    # ---- Print (ii) --------------------------------------------------------
    print("\n--- (ii) Per-hypothesis cross-run ratio (rival/lgn) ----------------\n")
    print("    ratio > 1 means lgn is faster\n")
    rows = []
    for h_id, desc, _bench, _lgn, _riv, _label in HYPOTHESES:
        ns = sorted({k[1] for k in hyp_ratios if k[0] == h_id})
        for n in ns:
            ratios = hyp_ratios[(h_id, n)]
            if len(ratios) < args.min_runs:
                continue
            agg = aggregate(ratios)
            rows.append((
                h_id, desc, f"n={n}",
                f"{agg['n_runs']}/{n_runs}",
                fmt_x(agg["median"]),
                fmt_x(agg["min"]) + " - " + fmt_x(agg["max"]),
                fmt_x(agg["p5"]) + " - " + fmt_x(agg["p95"]),
                fmt_pct(agg["cv"]),
            ))
    print_table(rows, ["H", "Description", "n", "runs",
                       "median", "min-max", "p5-p95", "CV%"])

    # ---- Print (iii) -------------------------------------------------------
    print("\n--- (iii) Per-hypothesis robust verdict ---------------------------\n")
    print(f"    Within-run Welch (one-tailed, H1: lgn < rival), alpha={ALPHA}")
    print(f"    'reject' = number of runs whose Welch test rejected H0\n")
    rows = []
    for h_id, _desc, _bench, _lgn, _riv, _label in HYPOTHESES:
        ns = sorted({k[1] for k in hyp_pvals if k[0] == h_id})
        for n in ns:
            pvals = hyp_pvals[(h_id, n)]
            ratios = hyp_ratios[(h_id, n)]
            ds = hyp_ds[(h_id, n)]
            if len(pvals) < args.min_runs:
                continue
            n_reject = sum(1 for p in pvals if p is not None and not math.isnan(p) and p < ALPHA)
            n_lgn_faster = sum(1 for r in ratios if r > 1.0)
            med_p = float(np.nanmedian([p for p in pvals if not math.isnan(p)])) if pvals else float("nan")
            med_d = float(np.nanmedian([d for d in ds if not math.isnan(d)])) if ds else float("nan")
            rows.append((
                h_id, f"n={n}",
                f"{len(pvals)}",
                f"{n_reject}/{len(pvals)}",
                f"{n_lgn_faster}/{len(ratios)}",
                fmt_x(aggregate(ratios)["median"]),
                fmt_p(med_p),
                fmt_d(med_d),
            ))
    print_table(rows, ["H", "n", "runs", "reject", "lgn_faster",
                       "median_ratio", "median_p", "median_d"])

    # ---- Optional: paper-ready LaTeX --------------------------------------
    if args.markdown:
        write_markdown(Path(args.markdown), args.machine, n_runs,
                       cell_means, hyp_ratios, hyp_pvals)
        print(f"\nWrote paper-ready table rows -> {args.markdown}")


if __name__ == "__main__":
    main()
