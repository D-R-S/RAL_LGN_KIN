#!/usr/bin/env python3
"""
scripts/compute_stats_v2.py
===========================
Read the flat per-sample CSV emitted by bench_v2_lgn_kdl / bench_v2_pin /
bench_v2_lgn_only (or the concatenated results_v2_raw.csv) and compute,
for each hypothesis pair, Welch's t-test, Cohen's d, and speedup.

CSV schema expected (header row required):
    solver,benchmark,n,call_ns[,...optional columns...]

Where:
    solver     in {lgn, kdl, pinocchio, scalar}
    benchmark  in {fk, ik_posonly, vel_only, dyn_M, dyn_C,
                   dyn_RNEA_eq, dyn_ABA_eq, ...}
    n          DOF count (int)
    call_ns    one timing sample in nanoseconds (float)
               (also accepts the legacy alias `time_ns`)

Note: the concatenated results_v2_raw.csv may contain multiple header
rows (one per source CSV). Non-numeric rows are skipped silently.

Usage:
    python3 scripts/compute_stats_v2.py results_v2/results_v2_raw.csv
    python3 scripts/compute_stats_v2.py file_a.csv file_b.csv
    python3 scripts/compute_stats_v2.py results_v2/results_v2_raw.csv \
            --plots --plot-dir plots/
    python3 scripts/compute_stats_v2.py results_v2/results_v2_raw.csv \
            --perf results_v2/perf_stat_lgn.txt

Hypothesis map:
    H01   lgn FK  vs  KDL FK         (per n)
    H01p  lgn FK  vs  Pinocchio FK   (per n)
    H02p  lgn IK  vs  Pinocchio IK   (per n)
    H04   lgn Hv  vs  scalar vel     (per n)
    H_M   lgn M   vs  Pinocchio CRBA (per n)
    H_C   lgn C   vs  Pinocchio NLE-G (per n)
    H_R   lgn RNEA vs Pinocchio RNEA (per n)
    H_A   lgn ABA  vs Pinocchio ABA  (per n)

Dependencies:
    pip install --user numpy scipy
    pip install --user pandas matplotlib   # optional (faster CSV load + plots)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# --- numpy + scipy ----------------------------------------------------------
try:
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: numpy/scipy not found - statistical tests skipped.",
          file=sys.stderr)
    print("  pip install --user numpy scipy", file=sys.stderr)

# --- pandas (optional speedup) ----------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# --- matplotlib (optional) --------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
#  Hypothesis registry
# ============================================================================
#
# Each hypothesis: (id, description, benchmark_name, lgn_solver, rival_solver, rival_label)
# benchmark_name matches the CSV `benchmark` column.
# lgn_solver and rival_solver match the `solver` column.
#
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

ALPHA = 0.05


# ============================================================================
#  Data loading
# ============================================================================

# Accepted column names for the timing column, in priority order
TIME_COLS = ("call_ns", "time_ns")


def _pick_time_col(columns) -> str | None:
    for c in TIME_COLS:
        if c in columns:
            return c
    return None


def load_csv_pandas(paths: list[str]) -> dict:
    """Return {(solver, benchmark, n): np.array([time_ns, ...])}"""
    frames = []
    for p in paths:
        # First peek at the header to find the timing column
        head = pd.read_csv(p, nrows=0)
        tcol = _pick_time_col(head.columns)
        if tcol is None:
            raise ValueError(
                f"{p}: no timing column found "
                f"(expected one of {TIME_COLS}, got {list(head.columns)})"
            )
        wanted = {"solver", "benchmark", "n", tcol}
        df = pd.read_csv(
            p,
            usecols=lambda c: c in wanted,
            dtype={"solver": "string", "benchmark": "string"},
            low_memory=False,
        )
        # Concatenated CSVs may contain repeated header rows -> non-numeric.
        # Coerce and drop those.
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
        df = df.dropna(subset=["solver", "benchmark", "n", tcol])
        df["n"] = df["n"].astype(int)
        df = df.rename(columns={tcol: "time_ns"})
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)
    grouped = big.groupby(["solver", "benchmark", "n"])["time_ns"]
    return {key: arr.to_numpy() for key, arr in grouped}


def load_csv_stdlib(paths: list[str]) -> dict:
    """Fallback loader using only the stdlib + numpy (slower but no pandas)."""
    buckets: dict = defaultdict(list)
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            tcol = _pick_time_col(reader.fieldnames or ())
            if tcol is None:
                print(f"WARNING: {p}: no timing column "
                      f"(expected one of {TIME_COLS}); skipping",
                      file=sys.stderr)
                continue
            for row in reader:
                # Skip stray header rows that landed in the data (concat CSVs)
                if row.get(tcol) == tcol or row.get("solver") == "solver":
                    continue
                try:
                    solver = row["solver"]
                    benchmark = row["benchmark"]
                    n = int(row["n"])
                    t = float(row[tcol])
                except (KeyError, ValueError, TypeError):
                    continue
                buckets[(solver, benchmark, n)].append(t)
    if HAS_SCIPY:
        return {k: np.asarray(v) for k, v in buckets.items()}
    return {k: list(v) for k, v in buckets.items()}


def load_csv(paths: list[str]) -> dict:
    if HAS_PANDAS:
        try:
            return load_csv_pandas(paths)
        except Exception as e:
            print(f"WARNING: pandas load failed ({e}); falling back to stdlib",
                  file=sys.stderr)
    return load_csv_stdlib(paths)


# ============================================================================
#  Perf telemetry parser
# ============================================================================

# Events we care about. perf_avx2.sh emits either Intel-named or bare events
# depending on the CPU. We accept either spelling for each metric.
PERF_EVENT_ALIASES = {
    "instructions": ["cpu_core/instructions/", "instructions"],
    "cycles":       ["cpu_core/cycles/",       "cycles"],
    "cache_refs":   ["cpu_core/cache-references/", "cache-references"],
    "cache_misses": ["cpu_core/cache-misses/",     "cache-misses"],
    "avx256":       ["cpu_core/fp_arith_inst_retired.256b_packed_double/",
                     "fp_arith_inst_retired.256b_packed_double"],
    "avx128":       ["cpu_core/fp_arith_inst_retired.128b_packed_double/",
                     "fp_arith_inst_retired.128b_packed_double"],
    "scalar":       ["cpu_core/fp_arith_inst_retired.scalar_double/",
                     "fp_arith_inst_retired.scalar_double"],
}


def parse_perf(path: str) -> dict:
    """
    Parse `perf stat` output (whether --repeat aggregated or single-shot).
    Returns {metric_name: int_count}. Missing metrics simply absent.
    """
    import re as _re
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return {}

    out: dict = {}
    # `perf stat` lines look like:
    #     20,285,833,915      cpu_core/fp_arith_inst_retired.256b_packed_double/   ...
    # or:
    #     1,234,567           instructions    #  3.13 insn per cycle
    # We grab the first number and match the event token on the same line.
    for line in text.splitlines():
        m_num = _re.match(r"\s*([\d,]+)\s+(\S+)", line)
        if not m_num:
            continue
        raw_count, event = m_num.group(1), m_num.group(2)
        try:
            count = int(raw_count.replace(",", ""))
        except ValueError:
            continue
        for key, aliases in PERF_EVENT_ALIASES.items():
            if event in aliases and key not in out:
                out[key] = count
                break
    return out


def _fmt_count(n: int) -> str:
    """Group thousands with commas for readability."""
    return f"{n:>18,}"


def print_perf_banner(metrics: dict):
    print()
    print("=" * 68)
    print("  HARDWARE TELEMETRY  (cpu_core, P-core only)")
    print("=" * 68)
    if not metrics:
        print("  no perf data found")
        return

    if "instructions" in metrics:
        print(f"  Instructions:    {_fmt_count(metrics['instructions'])}")
    if "cycles" in metrics:
        print(f"  Cycles:          {_fmt_count(metrics['cycles'])}")
    if "instructions" in metrics and "cycles" in metrics and metrics["cycles"] > 0:
        ipc = metrics["instructions"] / metrics["cycles"]
        # Alder Lake P-core peaks at ~6 IPC; >3 is healthy for AVX workload.
        flag = ""
        if ipc < 1.5:
            flag = "  (LOW - check for stalls)"
        elif ipc >= 3.0:
            flag = "  (healthy)"
        print(f"  IPC:                          {ipc:6.2f}{flag}")

    if "cache_refs" in metrics and "cache_misses" in metrics and metrics["cache_refs"] > 0:
        miss = 100.0 * metrics["cache_misses"] / metrics["cache_refs"]
        flag = "  (high)" if miss > 5.0 else ""
        print(f"  Cache miss rate:              {miss:6.2f}% of refs{flag}")

    have_fp = any(k in metrics for k in ("avx256", "avx128", "scalar"))
    if have_fp:
        avx256 = metrics.get("avx256", 0)
        avx128 = metrics.get("avx128", 0)
        scalar = metrics.get("scalar", 0)
        total = avx256 + avx128 + scalar
        if total > 0:
            print()
            print("  FP op breakdown (double-precision):")
            print(f"    AVX2   (256b lane):  {_fmt_count(avx256)}  "
                  f"({100.0*avx256/total:5.1f}%)")
            print(f"    SSE2   (128b lane):  {_fmt_count(avx128)}  "
                  f"({100.0*avx128/total:5.1f}%)")
            print(f"    Scalar           :   {_fmt_count(scalar)}  "
                  f"({100.0*scalar/total:5.1f}%)")
            # Headroom heuristic: lots of scalar = vectorisation opportunity.
            scalar_pct = 100.0 * scalar / total
            wide_pct   = 100.0 * avx256 / total
            print()
            if scalar_pct > 50.0:
                print("  Vectorisation flag:  >50% scalar - significant headroom")
                print("    Pinocchio's FK lead is plausibly closeable by widening")
                print("    the hot loop to 256-bit lanes.")
            elif scalar_pct > 30.0:
                print("  Vectorisation flag:  >30% scalar - moderate headroom")
            elif wide_pct >= 70.0:
                print("  Vectorisation flag:  well vectorised (>=70% AVX2)")
            else:
                print("  Vectorisation flag:  mixed; see breakdown above")


# ============================================================================
#  Statistics
# ============================================================================

def welch_t(a, b) -> tuple[float, float]:
    """One-tailed Welch t-test: H0 means equal, H1 mean(a) < mean(b)."""
    if not HAS_SCIPY or len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p_two = stats.ttest_ind(a, b, equal_var=False)
    # one-sided test: lower-tail
    p_one = p_two / 2.0 if t < 0 else 1.0 - p_two / 2.0
    return float(t), float(p_one)


def cohen_d(a, b) -> float:
    if not HAS_SCIPY or len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    denom = (na - 1) * sa * sa + (nb - 1) * sb * sb
    if denom <= 0:
        return float("nan")
    pooled = math.sqrt(denom / (na + nb - 2))
    return (ma - mb) / pooled if pooled > 0 else float("nan")


def mean(arr) -> float:
    if HAS_SCIPY:
        return float(np.mean(arr)) if len(arr) else float("nan")
    return sum(arr) / len(arr) if arr else float("nan")


def speedup(a, b) -> float:
    """mean(b) / mean(a) - how many times faster is a than b."""
    ma, mb = mean(a), mean(b)
    return mb / ma if ma > 0 else float("nan")


# ============================================================================
#  Formatting
# ============================================================================

def fmt_ns(ns: float) -> str:
    if math.isnan(ns):
        return "-"
    if ns >= 1e6:
        return f"{ns/1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.2f} us"
    return f"{ns:.1f} ns"


def fmt_x(x: float) -> str:
    return "-" if math.isnan(x) else f"{x:.2f}x"


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "-"
    if p < 1e-6:
        return "<1e-6"
    return f"{p:.4g}"


def fmt_d(d: float) -> str:
    return "-" if math.isnan(d) else f"{d:+.2f}"


def reject(p: float, direction: str) -> str:
    """direction in {'lgn_faster', 'rival_faster', 'tie'}"""
    if math.isnan(p):
        return "-"
    if p < ALPHA and direction == "lgn_faster":
        return "REJECT (lgn wins)"
    if p < ALPHA and direction == "rival_faster":
        return "REJECT (lgn loses)"
    return "retain"


def print_table(rows, headers):
    if not rows:
        print("  (no rows)")
        return
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        print("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)


# ============================================================================
#  Main analysis
# ============================================================================

def analyse(data: dict, only_hyp: set | None = None) -> list:
    """Returns a list of result rows."""
    rows = []
    headers = ["H", "Description", "n",
               "lgn mean", "rival mean", "speedup",
               "samples (l/r)", "p (one-tail)", "Cohen d", "verdict"]

    for h_id, desc, bench, lgn_solver, rival_solver, rival_label in HYPOTHESES:
        if only_hyp and h_id not in only_hyp:
            continue

        # Find all n values where BOTH solvers have data for this benchmark
        ns_lgn   = {k[2] for k in data if k[0] == lgn_solver   and k[1] == bench}
        ns_rival = {k[2] for k in data if k[0] == rival_solver and k[1] == bench}
        ns_common = sorted(ns_lgn & ns_rival)

        if not ns_common:
            rows.append((h_id, desc, "-", "-", "-", "-", "-", "-", "-",
                         "no data"))
            continue

        for n in ns_common:
            a = data[(lgn_solver,   bench, n)]
            b = data[(rival_solver, bench, n)]
            ma, mb = mean(a), mean(b)
            sp = speedup(a, b)
            _, p = welch_t(a, b)
            d = cohen_d(a, b)
            direction = "lgn_faster" if ma < mb else "rival_faster"
            rows.append((
                h_id, desc, f"n={n}",
                fmt_ns(ma), fmt_ns(mb),
                fmt_x(sp),
                f"{len(a)}/{len(b)}",
                fmt_p(p), fmt_d(d),
                reject(p, direction)
            ))

    print_table(rows, headers)
    return rows


# ============================================================================
#  Plots
# ============================================================================

def make_plots(data: dict, out_dir: Path):
    if not HAS_MPL:
        print("(matplotlib unavailable - skipping plots)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    for h_id, desc, bench, lgn_solver, rival_solver, rival_label in HYPOTHESES:
        ns_lgn   = sorted(k[2] for k in data if k[0] == lgn_solver   and k[1] == bench)
        ns_rival = sorted(k[2] for k in data if k[0] == rival_solver and k[1] == bench)
        ns = sorted(set(ns_lgn) & set(ns_rival))
        if not ns:
            continue

        means_lgn   = [mean(data[(lgn_solver,   bench, n)]) for n in ns]
        means_rival = [mean(data[(rival_solver, bench, n)]) for n in ns]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = list(range(len(ns)))
        w = 0.35
        ax.bar([xi - w/2 for xi in x], means_lgn,   w, label="lgn",        color="#4C72B0")
        ax.bar([xi + w/2 for xi in x], means_rival, w, label=rival_label,  color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels([f"n={n}" for n in ns])
        ax.set_yscale("log")
        ax.set_ylabel("Mean time (ns, log)")
        ax.set_title(f"{h_id}: {desc}")
        ax.legend()
        ax.grid(True, axis="y", which="both", alpha=0.25)
        fig.tight_layout()
        out_path = out_dir / f"{h_id}_{bench}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"  wrote {out_path}")


# ============================================================================
#  Entry
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Stats on lgn bench CSVs (compute_stats_v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("csv", nargs="+", help="one or more CSV files to union")
    ap.add_argument("--plots", action="store_true",
                    help="emit per-hypothesis bar plots (requires matplotlib)")
    ap.add_argument("--plot-dir", default="plots",
                    help="directory for plot output (default: plots/)")
    ap.add_argument("--only", default="",
                    help="comma-separated hypothesis IDs to restrict to "
                         "(e.g. --only H01,H01p,H04)")
    ap.add_argument("--perf", default="",
                    help="path to perf_stat_lgn.txt for hardware telemetry banner "
                         "(default: auto-detect alongside the first CSV)")
    return ap.parse_args()


def _auto_detect_perf(csv_paths: list[str]) -> str:
    """Look for perf_stat_lgn.txt next to the first CSV."""
    for c in csv_paths:
        candidate = Path(c).parent / "perf_stat_lgn.txt"
        if candidate.is_file():
            return str(candidate)
    return ""


def main():
    args = parse_args()

    # Sanity-check inputs
    missing = [p for p in args.csv if not Path(p).is_file()]
    if missing:
        print(f"ERROR: file(s) not found: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(args.csv)} file(s)...")
    data = load_csv(args.csv)
    if not data:
        print("ERROR: no usable rows found in input.", file=sys.stderr)
        sys.exit(1)

    total_samples = sum(len(v) for v in data.values())
    print(f"Loaded {total_samples:,} samples across {len(data)} (solver, bench, n) groups")
    print()

    only_set = {h.strip() for h in args.only.split(",") if h.strip()} or None
    analyse(data, only_set)

    print(f"\nalpha = {ALPHA}  (one-tailed Welch t-test, H1: lgn < rival)")

    # Perf telemetry banner
    perf_path = args.perf or _auto_detect_perf(args.csv)
    if perf_path:
        if Path(perf_path).is_file():
            metrics = parse_perf(perf_path)
            print_perf_banner(metrics)
            print(f"\n  source: {perf_path}")
        else:
            print(f"\n!! --perf {perf_path}: file not found")
    # If neither --perf nor auto-detect found anything, just stay silent.

    if args.plots:
        print(f"\nPlots ->  {args.plot_dir}/")
        make_plots(data, Path(args.plot_dir))


if __name__ == "__main__":
    main()
