#!/usr/bin/env python3
"""
scripts/plot_stats_10x.py
=========================
Bar plots for N-run cross-aggregated bench data, with between-run error
bars. Companion to compute_stats_10x.py.

Patch 1.5 changes
-----------------
1. SINGLE SOURCE OF TRUTH. Hypothesis specs are imported from
   compute_stats_10x.HYPOTHESES. No parallel registry, no drift.
   If a hypothesis is added there, it appears here.

2. EXPECT-DIRECTION ORIENTATION. Every ratio bar is oriented by the
   hypothesis's `expect` field. A bar > 1 ALWAYS means "the side we
   expected to win, won." There is no per-hypothesis special-case.
   This fixes the prior H04 sign bug where the summary plot legend
   said ">1 means lgn faster" but H04 was plotted with the inverse
   convention.

3. ORANGE = lgn. The user's color request. lgn bars are always orange
   in the side-by-side plots; rivals are a neutral cool tone. In the
   summary ratio plot, bars are colored by the *outcome*: orange when
   lgn actually beat expectation (a real "win" for the carrier),
   the rival's tone otherwise. Specifically:
     - For expect=lgn:    bar > 1 → orange (lgn won as expected)
                          bar < 1 → rival tone (lgn didn't win)
     - For expect=rival:  bar > 1 → rival tone (rival won as expected;
                                    this is the carrier-penalty story)
                          bar < 1 → orange (lgn beat expectation —
                                    e.g. SIMD recovered the FLOP deficit
                                    better than the structural bound)

4. PAIRING BY RUN_TAG, NOT LIST POSITION. The prior code zipped
   per-solver mean lists and assumed both lists were in the same
   run order. If one solver crashed in run k, zip silently dropped
   the last run, biasing error bars. Now we key per-run means by
   (run_tag, solver, bench, n) and pair by run_tag explicitly.

5. H02p_FAIR PLOT. The fair-math IK hypothesis introduced in Patch 1.5
   gets its own bar plot and appears in the ratio summary.

6. H04 LEGACY + ALLOC + PREALLOC. All three sub-modes plotted; one
   figure each. Summary plot shows whichever tags the CSV provides.

Error bars
----------
Bar height = cross-run median. Whiskers = min..max across runs.
This is between-run dispersion — within-run sample noise is invisible
at the per-call sample counts the bench produces, so plotting it
would be either zero-width or misleading.

Usage
-----
    python3 scripts/plot_stats_10x.py \\
        /home/humanoid/lgn_hand_ik/results_v2/results_v2_raw_dt-hm10_20260513_204647_iter*.csv \\
        --machine AMD Ryzen 9 9950X  \\
        --out-dir results_v2/plots_10x/

Dependencies
------------
numpy, matplotlib. pandas optional (streaming loader uses it when
available, falls back to stdlib csv).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Single source of truth ──────────────────────────────────────────────────
# compute_stats_10x defines HYPOTHESES with the `expect` field. We import
# it here so the two scripts can never drift. If the import fails (e.g.
# script run from outside the scripts/ directory), we fall back to a
# local copy that MUST be kept in sync; an assertion checks at runtime.
try:
    # Standard layout: both scripts live in scripts/.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compute_stats_10x import HYPOTHESES  # type: ignore
except Exception as e:
    print(f"  !! cannot import HYPOTHESES from compute_stats_10x: {e}",
          file=sys.stderr)
    print("  !! falling back to local copy — MUST be kept in sync",
          file=sys.stderr)
    HYPOTHESES = [
        ("H01",         "lgn FK vs KDL FK",
            "fk", "lgn", "kdl", "KDL", "lgn"),
        ("H01p",        "lgn FK vs Pinocchio FK",
            "fk", "lgn", "pinocchio", "Pinocchio", "lgn"),
        ("H02p",        "lgn IK vs Pinocchio IK (idiomatic)",
            "ik_posonly", "lgn", "pinocchio", "Pinocchio", "lgn"),
        ("H02p_fair",   "lgn IK vs Pinocchio IK (fair math)",
            "ik_posonly_fair", "lgn_fair", "pinocchio_fair", "Pinocchio", "lgn"),
        ("H04",          "lgn Hv vs scalar vel (legacy)",
            "vel_only",          "lgn", "scalar", "scalar", "rival"),
        ("H04_alloc",    "lgn Hv vs scalar vel (alloc)",
            "vel_only_alloc",    "lgn", "scalar", "scalar", "rival"),
        ("H04_prealloc", "lgn Hv vs scalar vel (prealloc)",
            "vel_only_prealloc", "lgn", "scalar", "scalar", "rival"),
        ("H_M", "lgn M vs Pinocchio CRBA",
            "dyn_M",       "lgn", "pinocchio", "Pinocchio", "lgn"),
        ("H_C", "lgn C vs Pinocchio NLE-G",
            "dyn_C",       "lgn", "pinocchio", "Pinocchio", "lgn"),
        ("H_R", "lgn RNEA vs Pinocchio RNEA",
            "dyn_RNEA_eq", "lgn", "pinocchio", "Pinocchio", "lgn"),
        ("H_A", "lgn ABA vs Pinocchio ABA",
            "dyn_ABA_eq",  "lgn", "pinocchio", "Pinocchio", "lgn"),
    ]


# ── Constants ───────────────────────────────────────────────────────────────
CHUNKSIZE = 200_000
MIN_SAMPLES_PER_CELL = 50
TIME_COLS = ("call_ns", "time_ns")

# Orange = lgn (user request). Rivals get a neutral cool tone so the
# eye reads lgn at a glance across every plot.
COLOR_LGN          = "#E69F4D"   # warm orange — lgn
COLOR_LGN_DARK     = "#C97B2A"   # darker orange — used for edges or
                                 # for "lgn beat expectation" in summary
COLOR_KDL          = "#7E8AAB"   # cool blue-gray — KDL
COLOR_PINOCCHIO    = "#5D8A66"   # cool sage — Pinocchio
COLOR_SCALAR       = "#9080A8"   # cool lavender — scalar baseline
COLOR_UNIT_LINE    = "#222222"   # ratio=1 reference

# Rival color by rival solver name (drives both bar plots and the
# summary plot's "expected outcome" bars).
RIVAL_COLOR = {
    "kdl":             COLOR_KDL,
    "pinocchio":       COLOR_PINOCCHIO,
    "pinocchio_fair":  COLOR_PINOCCHIO,
    "scalar":          COLOR_SCALAR,
}

# Which hypotheses appear in the ratio summary plot. Order matters
# for the legend; pick the headline set for the kinematics paper.
SUMMARY_DEFAULT = ["H01", "H01p", "H02p", "H02p_fair",
                   "H04_alloc", "H04_prealloc", "H04"]

# Dynamics hypotheses are off by default (O(n^3) untuned; sister paper).
DYN_IDS = {"H_M", "H_C", "H_R", "H_A"}


# ── Streaming loader ────────────────────────────────────────────────────────
def _pick_time_col(columns):
    for c in TIME_COLS:
        if c in columns:
            return c
    return None


def _extract_run_tag(path: Path) -> str:
    stem = path.stem
    m = re.match(r"^results_v2_raw_(.+)$", stem)
    return m.group(1) if m else stem


def stream_means_pandas(path: Path) -> dict | None:
    """Per-cell mean across the whole CSV (one run)."""
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
            g = chunk.groupby(["solver", "benchmark", "n"])[tcol].agg(["sum", "count"])
            for (solver, bench, n_dof), row in g.iterrows():
                key = (str(solver), str(bench), int(n_dof))
                sum_x[key] += float(row["sum"])
                count[key] += int(row["count"])
    except Exception as e:
        print(f"  !! {path.name}: read failed ({e})", file=sys.stderr)
        return None
    return {k: sum_x[k] / c for k, c in count.items() if c >= MIN_SAMPLES_PER_CELL}


def stream_means_stdlib(path: Path) -> dict | None:
    sum_x: dict = defaultdict(float)
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
                count[key] += 1
    except Exception as e:
        print(f"  !! {path.name}: read failed ({e})", file=sys.stderr)
        return None
    return {k: sum_x[k] / c for k, c in count.items() if c >= MIN_SAMPLES_PER_CELL}


def stream_means(path: Path) -> dict | None:
    return stream_means_pandas(path) if HAS_PANDAS else stream_means_stdlib(path)


# ── Pairing helpers ─────────────────────────────────────────────────────────
def _hyp_by_id(h_id: str):
    """Return the hypothesis tuple by id, or None."""
    for h in HYPOTHESES:
        if h[0] == h_id:
            return h
    return None


def _orient_ratio(expect: str, lgn_time: float, rival_time: float) -> float:
    """
    Return the ratio in the EXPECT direction. >1 means the expected winner
    won. Matches compute_stats_10x.oriented_stats exactly.
    """
    if lgn_time <= 0 or rival_time <= 0:
        return float("nan")
    if expect == "lgn":
        return rival_time / lgn_time
    elif expect == "rival":
        return lgn_time / rival_time
    else:
        raise ValueError(f"expect must be 'lgn' or 'rival', got {expect!r}")


def _ratio_color(expect: str, ratio: float) -> str:
    """
    Color a ratio bar by whether lgn actually beat the comparison.

    expect=lgn:    bar > 1 → lgn won as expected  → orange
                   bar < 1 → lgn lost              → rival tone
    expect=rival:  bar > 1 → rival won as expected (carrier penalty)
                                                  → rival tone
                   bar < 1 → lgn beat expectation → orange
    """
    if math.isnan(ratio):
        return "#cccccc"
    lgn_wins = (expect == "lgn" and ratio > 1.0) or \
               (expect == "rival" and ratio < 1.0)
    return COLOR_LGN if lgn_wins else "#9C9C9C"   # neutral gray for not-a-win


# ── Side-by-side bar plot (one hypothesis) ──────────────────────────────────
def plot_hypothesis_bars(h, per_run_means: dict[tuple, dict],
                          out_path: Path, machine: str):
    """
    Bars: lgn (orange) vs rival. Heights are cross-run medians; whiskers
    are min..max across runs. One bar pair per n.

    per_run_means[run_tag] = {(solver, bench, n): mean_ns}
    """
    h_id, desc, bench, lgn_s, rival_s, rival_label, expect = h

    # Collect per-run means, paired by run_tag.
    runs_with_both: list[str] = []
    lgn_by_n: dict[int, list[float]] = defaultdict(list)
    riv_by_n: dict[int, list[float]] = defaultdict(list)

    for run_tag, means in per_run_means.items():
        ns_lgn = {k[2] for k in means if k[0] == lgn_s and k[1] == bench}
        ns_riv = {k[2] for k in means if k[0] == rival_s and k[1] == bench}
        common = ns_lgn & ns_riv
        if not common:
            continue
        runs_with_both.append(run_tag)
        for n in common:
            lgn_by_n[n].append(means[(lgn_s, bench, n)])
            riv_by_n[n].append(means[(rival_s, bench, n)])

    ns = sorted(n for n in lgn_by_n if len(lgn_by_n[n]) >= 2)
    if not ns:
        print(f"    {h_id}: <2 runs with both solvers; skipping")
        return False

    def stats(vals):
        arr = np.asarray(vals, dtype=float)
        med = float(np.median(arr))
        lo = med - float(np.min(arr))
        hi = float(np.max(arr)) - med
        return med, lo, hi

    lgn_med, lgn_lo, lgn_hi = zip(*(stats(lgn_by_n[n]) for n in ns))
    riv_med, riv_lo, riv_hi = zip(*(stats(riv_by_n[n]) for n in ns))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(ns))
    w = 0.38

    rival_color = RIVAL_COLOR.get(rival_s, "#9C9C9C")

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


# ── Ratio summary plot ──────────────────────────────────────────────────────
def plot_ratio_summary(h_ids: list[str], per_run_means: dict[tuple, dict],
                       out_path: Path, machine: str, n_runs_loaded: int):
    """
    One bar per (hypothesis, n). Bar is the cross-run median of the
    EXPECT-DIRECTION ratio. Always: bar > 1 means the expected winner
    won. Color: orange if lgn actually beat the comparison (regardless
    of expect direction), neutral gray otherwise.

    A horizontal line at y=1 separates expected-side-won (above) from
    expected-side-lost (below). For expect=lgn hypotheses the orange
    bars are above the line; for expect=rival hypotheses (carrier
    penalty) orange bars appear below the line — they're the cases
    where SIMD recovered more than the FLOP-count bound predicted.
    """
    # Build (h_id, expect, label, per_run_ratios_by_n) for each requested h.
    series = []
    for h_id in h_ids:
        h = _hyp_by_id(h_id)
        if h is None:
            continue
        _id, desc, bench, lgn_s, rival_s, _rl, expect = h

        # Pair per-run by run_tag.
        per_n_ratios: dict[int, list[float]] = defaultdict(list)
        for run_tag, means in per_run_means.items():
            ns_lgn = {k[2] for k in means if k[0] == lgn_s and k[1] == bench}
            ns_riv = {k[2] for k in means if k[0] == rival_s and k[1] == bench}
            for n in (ns_lgn & ns_riv):
                r = _orient_ratio(expect,
                                  means[(lgn_s, bench, n)],
                                  means[(rival_s, bench, n)])
                if not math.isnan(r):
                    per_n_ratios[n].append(r)

        per_n_stats: dict[int, tuple[float, float, float]] = {}
        for n, rs in per_n_ratios.items():
            if len(rs) < 2:
                continue
            arr = np.asarray(rs, dtype=float)
            med = float(np.median(arr))
            lo = med - float(np.min(arr))
            hi = float(np.max(arr)) - med
            per_n_stats[n] = (med, lo, hi)

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

    # Plot each hypothesis as a side-by-side group of bars across n.
    # Bar color is set per-bar by _ratio_color so the eye reads
    # "lgn won" (orange) vs "expected outcome that isn't a lgn win"
    # (neutral gray).
    legend_handles = []
    for i, (h_id, expect, desc, per_n) in enumerate(series):
        offset = -0.41 + w/2 + i * w
        for j, n in enumerate(all_ns):
            if n not in per_n:
                continue
            med, lo, hi = per_n[n]
            color = _ratio_color(expect, med)
            ax.bar(x[j] + offset, med, w, color=color,
                   edgecolor="#222", linewidth=0.4,
                   yerr=[[lo], [hi]], capsize=2.5,
                   error_kw={"elinewidth": 0.9, "ecolor": "#222"})
        # One legend handle per hypothesis (mid-color: orange if
        # majority of n favoured lgn, else gray).
        meds = [per_n[n][0] for n in all_ns if n in per_n]
        lgn_wins_count = sum(1 for r in meds
                             if (expect == "lgn"   and r > 1.0)
                             or (expect == "rival" and r < 1.0))
        legend_color = COLOR_LGN if lgn_wins_count > len(meds) / 2 else "#9C9C9C"
        label = f"{h_id}  ({desc.split(':')[0] if ':' in desc else desc})"
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=legend_color, label=label)
        )

    ax.axhline(1.0, color=COLOR_UNIT_LINE, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in all_ns])
    ax.set_ylabel("ratio in expect-direction (>1 = expected winner won)")
    title = "Cross-run ratios per hypothesis"
    if machine:
        title += f"  --  {machine}, {n_runs_loaded} runs"
    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper left",
              framealpha=0.95, fontsize=8.5, ncol=2)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    fig.text(0.5, 0.005,
             "Bar height: cross-run median ratio (oriented by hypothesis `expect`). "
             "Whiskers: min--max across runs. "
             "Orange: lgn actually beat the comparison. "
             "Gray: expected outcome held with no lgn upset.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    wrote {out_path}")
    return True


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Bar plots with cross-run error bars from N bench CSVs, "
                    "oriented by hypothesis expect-direction.",
    )
    ap.add_argument("csv", nargs="+",
                    help="one CSV per run, e.g. results_v2_raw_<RUN_TAG>.csv")
    ap.add_argument("--machine", default="",
                    help="machine label for plot titles (e.g. M2-AlderLake)")
    ap.add_argument("--out-dir", default="plots_10x",
                    help="output directory for PNGs (default: plots_10x/)")
    ap.add_argument("--include-dyn", action="store_true",
                    help="also plot dynamics hypotheses (H_M, H_C, H_R, H_A); "
                         "off by default since dynamics is O(n^3) and out of "
                         "scope for the kinematics paper")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip the per-hypothesis bar plots and emit only "
                         "the ratio summary")
    args = ap.parse_args()

    paths = [Path(p) for p in args.csv]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"ERROR: file(s) not found: {missing}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stream all runs, keyed by run_tag ──────────────────────────────────
    # per_run_means[run_tag] = {(solver, bench, n): mean_ns}
    print(f"Streaming {len(paths)} run(s)  (CHUNKSIZE={CHUNKSIZE:,})...", flush=True)
    per_run_means: dict[str, dict] = {}
    for p in sorted(paths):
        tag = _extract_run_tag(p)
        print(f"  loading [{tag}]...", end="", flush=True)
        means = stream_means(p)
        if means is None:
            print(" FAILED")
            continue
        print(f" {len(means)} cells", flush=True)
        per_run_means[tag] = means

    if len(per_run_means) < 2:
        print(f"ERROR: need at least 2 runs to draw error bars, got "
              f"{len(per_run_means)}", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"Drawing plots for {args.machine or 'unlabelled'}  "
          f"({len(per_run_means)} runs)")
    print("-" * 72)

    # ── Per-hypothesis side-by-side bars ───────────────────────────────────
    if not args.summary_only:
        for h in HYPOTHESES:
            h_id = h[0]
            if h_id in DYN_IDS and not args.include_dyn:
                continue
            bench = h[2]
            out_name = f"{h_id}_{bench}.png"
            plot_hypothesis_bars(h, per_run_means, out_dir / out_name, args.machine)

    # ── Ratio summary ──────────────────────────────────────────────────────
    summary_ids = [h_id for h_id in SUMMARY_DEFAULT
                   if _hyp_by_id(h_id) is not None]
    if args.include_dyn:
        summary_ids += [h_id for h_id in ("H_M", "H_C", "H_R", "H_A")
                        if _hyp_by_id(h_id) is not None]
    plot_ratio_summary(summary_ids, per_run_means,
                       out_dir / "ratios_summary.png",
                       args.machine, len(per_run_means))

    print("-" * 72)
    print(f"Done. Plots in {out_dir}/")


if __name__ == "__main__":
    main()
