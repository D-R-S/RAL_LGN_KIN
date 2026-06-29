#!/usr/bin/env python3
"""
scripts/compute_stats_multi.py
==============================
Multi-machine version of compute_stats_v2. Takes one or more
(--label, csv_glob) pairs and emits a wide table:

    one row per (hypothesis, n)
    one COLUMN GROUP per machine (speedup, p, Cohen d)
    one final POOLED row per (hypothesis, n) that combines all machines

Designed for the paper figure where you want to show the result
holds across architectures, not just one box.

Usage:
    python3 scripts/compute_stats_multi.py \\
        --label "Ryzen 9950X"   "results_ryzen/results_v2_raw_*.csv" \\
        --label "i7-1255U"      "results_latitude/results_v2_raw_*.csv" \\
        --label "NUC15"         "results_nuc15/results_v2_raw_*.csv" \\
        --label "Orin NX"       "results_orin/results_v2_raw_*.csv" \\
        --only H01p,H02p \\
        --format both \\
        --out-md  table.md \\
        --out-csv table.csv

Note: each --label can take MULTIPLE csv files via a shell glob.
Quote the glob to avoid early expansion if you want python to expand it.
The script accepts both styles - shell-expanded list or a literal glob.

Output formats:
    text      ASCII table (default)
    md        Markdown
    csv       CSV (for LaTeX / pandas downstream)
    both      text + md + csv (uses --out-* paths if given)
"""

from __future__ import annotations

import argparse
import csv as _csv
import glob as _glob
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("ERROR: numpy + scipy are required for this script.", file=sys.stderr)
    print("       sudo apt install python3-numpy python3-scipy", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================================
#  Hypothesis registry (shared with compute_stats_v2)
# ============================================================================
HYPOTHESES = [
    ("H01",  "lgn FK vs KDL FK",                "fk",          "lgn", "kdl",       "KDL"),
    ("H01p", "lgn FK vs Pinocchio FK",          "fk",          "lgn", "pinocchio", "Pinocchio"),
    ("H02p", "lgn IK vs Pinocchio IK",          "ik_posonly",  "lgn", "pinocchio", "Pinocchio"),
    ("H04",  "lgn Hv vs scalar vel",            "vel_only",    "lgn", "scalar",    "scalar"),
    ("H_M",  "lgn M vs Pinocchio CRBA",         "dyn_M",       "lgn", "pinocchio", "Pinocchio"),
    ("H_C",  "lgn C vs Pinocchio NLE-G",        "dyn_C",       "lgn", "pinocchio", "Pinocchio"),
    ("H_R",  "lgn RNEA vs Pinocchio RNEA",      "dyn_RNEA_eq", "lgn", "pinocchio", "Pinocchio"),
    ("H_A",  "lgn ABA vs Pinocchio ABA",        "dyn_ABA_eq",  "lgn", "pinocchio", "Pinocchio"),
]
HYP_BY_ID = {h[0]: h for h in HYPOTHESES}
ALPHA = 0.05
TIME_COLS = ("call_ns", "time_ns")


# ============================================================================
#  CSV loading (per machine)
# ============================================================================

def _pick_time_col(columns):
    for c in TIME_COLS:
        if c in columns:
            return c
    return None


def load_one_csv(path: str) -> dict:
    """Return {(solver, benchmark, n): np.array([t, ...])} for one file."""
    if HAS_PANDAS:
        head = pd.read_csv(path, nrows=0)
        tcol = _pick_time_col(head.columns)
        if tcol is None:
            return {}
        df = pd.read_csv(
            path,
            usecols=lambda c: c in {"solver", "benchmark", "n", tcol},
            dtype={"solver": "string", "benchmark": "string"},
            low_memory=False,
        )
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
        df = df.dropna(subset=["solver", "benchmark", "n", tcol])
        df["n"] = df["n"].astype(int)
        grouped = df.groupby(["solver", "benchmark", "n"])[tcol]
        return {key: arr.to_numpy() for key, arr in grouped}
    else:
        buckets = defaultdict(list)
        with open(path, newline="") as f:
            reader = _csv.DictReader(f)
            tcol = _pick_time_col(reader.fieldnames or ())
            if tcol is None:
                return {}
            for row in reader:
                if row.get("solver") == "solver":  # stray header row
                    continue
                try:
                    n = int(row["n"]); t = float(row[tcol])
                    buckets[(row["solver"], row["benchmark"], n)].append(t)
                except (KeyError, ValueError, TypeError):
                    continue
        return {k: np.asarray(v) for k, v in buckets.items()}


def load_machine(label: str, csv_args: list[str]) -> dict:
    """
    csv_args may be already-expanded filenames OR literal globs.
    Returns merged dict across all CSVs for this machine.
    """
    paths = []
    for a in csv_args:
        # Already a real file?
        if Path(a).is_file():
            paths.append(a)
        else:
            # Treat as glob
            matched = sorted(_glob.glob(a))
            if matched:
                paths.extend(matched)
            else:
                print(f"  WARNING: '{a}' matched no files for {label}",
                      file=sys.stderr)
    if not paths:
        return {}

    merged = defaultdict(list)
    for p in paths:
        for key, arr in load_one_csv(p).items():
            merged[key].extend(arr.tolist() if hasattr(arr, "tolist") else list(arr))
    return {k: np.asarray(v) for k, v in merged.items()}


# ============================================================================
#  Stats helpers
# ============================================================================

def welch_t_one_tail(a, b):
    """One-tailed Welch test, H1: mean(a) < mean(b)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p2 = scipy_stats.ttest_ind(a, b, equal_var=False)
    p1 = p2/2 if t < 0 else 1.0 - p2/2
    return float(t), float(p1)


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    denom = (na-1)*sa*sa + (nb-1)*sb*sb
    if denom <= 0:
        return float("nan")
    pooled = math.sqrt(denom / (na+nb-2))
    return (ma - mb) / pooled if pooled > 0 else float("nan")


def speedup(a, b):
    ma = float(np.mean(a)) if len(a) else float("nan")
    mb = float(np.mean(b)) if len(b) else float("nan")
    return mb / ma if ma and not math.isnan(mb) else float("nan")


# ============================================================================
#  Formatting
# ============================================================================

def fmt_x(v):
    return "-" if math.isnan(v) else f"{v:.2f}x"

def fmt_p(p):
    if math.isnan(p): return "-"
    if p < 1e-6: return "<1e-6"
    if p > 0.9999: return "1.0"
    return f"{p:.3g}"

def fmt_d(d):
    return "-" if math.isnan(d) else f"{d:+.2f}"


# ============================================================================
#  Core analysis: per-machine + pooled
# ============================================================================

def analyse_one_cell(data: dict, hyp_tuple, n: int):
    """
    Compute (speedup, p, cohen_d, n_lgn, n_rival) for a single (hyp, n, machine).
    Returns None if either side has no samples in this dataset.
    """
    _, _, bench, lgn_s, riv_s, _ = hyp_tuple
    a = data.get((lgn_s, bench, n))
    b = data.get((riv_s, bench, n))
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return None
    sp = speedup(a, b)
    _, p = welch_t_one_tail(a, b)
    d = cohen_d(a, b)
    return sp, p, d, len(a), len(b)


def pool_machines(machines: dict, hyp_tuple, n: int):
    """Concatenate samples across all machines for one (hyp, n)."""
    _, _, bench, lgn_s, riv_s, _ = hyp_tuple
    a_chunks, b_chunks = [], []
    for label, data in machines.items():
        a = data.get((lgn_s, bench, n))
        b = data.get((riv_s, bench, n))
        if a is not None and len(a):
            a_chunks.append(a)
        if b is not None and len(b):
            b_chunks.append(b)
    if not a_chunks or not b_chunks:
        return None
    a = np.concatenate(a_chunks)
    b = np.concatenate(b_chunks)
    sp = speedup(a, b)
    _, p = welch_t_one_tail(a, b)
    d = cohen_d(a, b)
    return sp, p, d, len(a), len(b)


def collect_all_ns(machines: dict, hyp_tuple):
    """All n values that appear for this hypothesis across any machine."""
    _, _, bench, lgn_s, riv_s, _ = hyp_tuple
    ns = set()
    for data in machines.values():
        for solver, b, n in data.keys():
            if b == bench and solver in (lgn_s, riv_s):
                ns.add(n)
    return sorted(ns)


# ============================================================================
#  Rendering
# ============================================================================

def render_text(rows, machine_labels, headers_top, headers_sub) -> str:
    """ASCII table with two header rows: machine labels, then per-col subhdrs."""
    # Compute column widths from headers + data
    ncols = len(headers_sub)
    widths = [len(h) for h in headers_sub]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    # Top header row spans subcolumns; build group spans
    top_cells = []
    fixed_cols = 2  # H, n
    top_cells.append(("", widths[0]))
    top_cells.append(("", widths[1]))
    span = 3  # speedup, p, d per machine
    for label in machine_labels + ["POOLED"]:
        group_w = sum(widths[fixed_cols : fixed_cols + span]) + (span - 1) * 3  # padding
        top_cells.append((label, group_w))
        fixed_cols += span

    out = []
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    # Top header
    parts = ["|"]
    parts.append(" " + "".ljust(widths[0]) + " |")
    parts.append(" " + "".ljust(widths[1]) + " |")
    fixed_cols = 2
    for label in machine_labels + ["POOLED"]:
        group_w = sum(widths[fixed_cols : fixed_cols + span]) + (span - 1) * 3
        parts.append(" " + label.center(group_w) + " |")
        fixed_cols += span
    out.append(sep)
    out.append("".join(parts))

    # Sub header
    out.append(sep)
    out.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers_sub)) + " |")
    out.append(sep)

    for r in rows:
        out.append("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(ncols)) + " |")
    out.append(sep)
    return "\n".join(out)


def render_markdown(rows, machine_labels, headers_sub) -> str:
    """
    Markdown table with one column per (machine, metric) plus pooled.
    Top-row uses multi-col headers via concatenation since GFM
    doesn't support real multi-col -- we just prefix the metric.
    """
    out = []
    # GFM has no col-span; rebuild headers with machine prefix
    pretty = [headers_sub[0], headers_sub[1]]
    span = 3
    fixed_cols = 2
    for label in machine_labels + ["POOLED"]:
        for sub in headers_sub[fixed_cols : fixed_cols + span]:
            pretty.append(f"{label} — {sub}")
        fixed_cols += span

    out.append("| " + " | ".join(pretty) + " |")
    out.append("|" + "|".join("---" for _ in pretty) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def render_csv(rows, machine_labels, headers_sub) -> str:
    pretty = [headers_sub[0], headers_sub[1]]
    span = 3
    fixed_cols = 2
    for label in machine_labels + ["POOLED"]:
        for sub in headers_sub[fixed_cols : fixed_cols + span]:
            pretty.append(f"{label}_{sub}")
        fixed_cols += span
    out = [",".join(pretty)]
    for r in rows:
        out.append(",".join(str(x) for x in r))
    return "\n".join(out)


# ============================================================================
#  Main
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-machine + pooled stats table for the IK paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--label", action="append", nargs="+", default=[],
        metavar=("NAME", "CSV"),
        help="machine label followed by one or more CSV paths/globs; "
             "repeat --label for each machine",
    )
    ap.add_argument("--only", default="",
                    help="comma-sep hypothesis IDs to include (default: all)")
    ap.add_argument("--format", choices=["text", "md", "csv", "both"],
                    default="text",
                    help="'both' writes all three; combine with --out-* flags")
    ap.add_argument("--out-md", default="",
                    help="write markdown to this path")
    ap.add_argument("--out-csv", default="",
                    help="write csv to this path")
    ap.add_argument("--out-text", default="",
                    help="write ASCII to this path")
    args = ap.parse_args()
    if not args.label:
        ap.error("at least one --label NAME csv... is required")
    return args


def main():
    args = parse_args()

    # Parse the --label invocations into ordered (label, csvs) pairs
    machine_specs = []
    for entry in args.label:
        if len(entry) < 2:
            print(f"ERROR: --label needs a name and at least one CSV: {entry}",
                  file=sys.stderr)
            sys.exit(1)
        label = entry[0]
        csvs = entry[1:]
        machine_specs.append((label, csvs))

    # Load every machine
    machines: dict[str, dict] = {}
    for label, csvs in machine_specs:
        print(f"Loading machine '{label}' from {len(csvs)} pattern(s)...",
              file=sys.stderr)
        data = load_machine(label, csvs)
        if not data:
            print(f"  WARNING: no rows for '{label}' - skipping", file=sys.stderr)
            continue
        samples = sum(len(v) for v in data.values())
        groups = len(data)
        print(f"  loaded {samples:,} samples across {groups} (solver,bench,n) groups",
              file=sys.stderr)
        machines[label] = data

    if not machines:
        print("ERROR: no machine produced usable data", file=sys.stderr)
        sys.exit(1)

    machine_labels = list(machines.keys())

    only_set = {h.strip() for h in args.only.split(",") if h.strip()} or None

    # Build the wide table
    headers_sub = ["H", "n"]
    for _ in machine_labels + ["POOLED"]:
        headers_sub.extend(["speedup", "p", "d"])

    rows = []
    for hyp in HYPOTHESES:
        h_id = hyp[0]
        if only_set and h_id not in only_set:
            continue
        ns = collect_all_ns(machines, hyp)
        if not ns:
            continue
        for n in ns:
            row = [h_id, f"n={n}"]
            for label in machine_labels:
                cell = analyse_one_cell(machines[label], hyp, n)
                if cell is None:
                    row.extend(["-", "-", "-"])
                else:
                    sp, p, d, _, _ = cell
                    row.extend([fmt_x(sp), fmt_p(p), fmt_d(d)])
            pooled = pool_machines(machines, hyp, n)
            if pooled is None:
                row.extend(["-", "-", "-"])
            else:
                sp, p, d, _, _ = pooled
                row.extend([fmt_x(sp), fmt_p(p), fmt_d(d)])
            rows.append(row)

    if not rows:
        print("No rows produced (check --only filter and CSV contents).",
              file=sys.stderr)
        sys.exit(1)

    headers_top = []  # filled in renderer

    # Render
    text_out = render_text(rows, machine_labels, headers_top, headers_sub)
    md_out   = render_markdown(rows, machine_labels, headers_sub)
    csv_out  = render_csv(rows, machine_labels, headers_sub)

    # Decide what to emit
    fmt = args.format
    if fmt == "text":
        print(text_out)
    elif fmt == "md":
        print(md_out)
    elif fmt == "csv":
        print(csv_out)
    elif fmt == "both":
        # Always print text to stdout; write the others to files if specified
        print(text_out)
        if not (args.out_md or args.out_csv):
            print("\n# (use --out-md / --out-csv to also write those formats)")

    if args.out_text:
        Path(args.out_text).write_text(text_out)
        print(f"\n# wrote {args.out_text}", file=sys.stderr)
    if args.out_md:
        Path(args.out_md).write_text(md_out)
        print(f"# wrote {args.out_md}", file=sys.stderr)
    if args.out_csv:
        Path(args.out_csv).write_text(csv_out)
        print(f"# wrote {args.out_csv}", file=sys.stderr)

    print(f"\n# alpha = {ALPHA}  (one-tailed Welch; H1: lgn < rival)", file=sys.stderr)
    print(f"# pooled row concatenates samples across all machines", file=sys.stderr)


if __name__ == "__main__":
    main()
