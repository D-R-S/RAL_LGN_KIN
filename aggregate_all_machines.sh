#!/bin/bash
# =============================================================================
#  aggregate_all_machines.sh
#
#  Walks a root directory of per-machine subfolders, each holding 10 (or N)
#  iteration CSVs from `run_lgn_bench.sh`, and:
#
#    (1) For each machine folder, runs compute_stats_10x.py + plot_stats_10x.py
#        to produce per-machine stats tables and plots.
#    (2) Runs compute_stats_cross_machine.py with every iter CSV under the root
#        to produce the pooled cross-machine stats and plots.
#
#  Expected layout:
#
#    <ROOT>/
#      pc_1_latitude/
#        results_v2_raw_*_iter*.csv          (10 files)
#      pc_2_zen5/
#        results_v2_raw_*_iter*.csv
#      pc_3_i9_13gen/
#        results_v2_raw_*_iter*.csv
#      ...
#
#  Each subfolder name becomes the machine label. Override per folder by
#  placing a `.label` file inside containing the desired label string.
#
#  Output layout:
#
#    <ROOT>/
#      pc_1_latitude/
#        out/cross_run_stats.log
#        out/paper_block.tex
#        out/plots/*.png
#      ...
#      cross_machine/
#        cross_machine_stats.log
#        paper_block_cross_machine.tex
#        plots/*.png
#
#  Usage:
#    ./aggregate_all_machines.sh /path/to/root_of_pc_folders
#    REPO=/path/to/lgn_hand_ik ./aggregate_all_machines.sh ./all_results
# =============================================================================
set -euo pipefail

ROOT="${1:-}"
if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
    echo "Usage: $0 <root_dir_of_pc_folders>" >&2
    exit 1
fi
ROOT="$(cd "$ROOT" && pwd)"

REPO="${REPO:-$HOME/lgn_hand_ik}"
SCRIPTS="$REPO/scripts"
PY="${PYTHON:-python3}"

STATS_PY="$SCRIPTS/compute_stats_10x.py"
PLOT_PY="$SCRIPTS/plot_stats_10x.py"
CROSS_PY="$SCRIPTS/compute_stats_cross_machine.py"

for f in "$STATS_PY" "$PLOT_PY" "$CROSS_PY"; do
    [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

CROSS_DIR="$ROOT/cross_machine"
mkdir -p "$CROSS_DIR/plots"

# Collect every iter CSV under every pc_*/ folder for the cross-machine pass.
ALL_CSVS=()

echo
echo "=================================================================="
echo "  PER-MACHINE AGGREGATION"
echo "=================================================================="

for pc_dir in "$ROOT"/*/; do
    [ -d "$pc_dir" ] || continue
    pc_name="$(basename "$pc_dir")"
    # Skip the cross_machine output dir if we re-run on the same root
    [ "$pc_name" = "cross_machine" ] && continue

    # Determine the label: .label file if present, else folder name
    if [ -f "$pc_dir/.label" ]; then
        label="$(cat "$pc_dir/.label" | head -1 | tr -d ' \t\r\n')"
    else
        label="$pc_name"
    fi

    # Find iter CSVs in the folder
    mapfile -t iters < <(ls -1 "$pc_dir"/results_v2_raw_*_iter*.csv 2>/dev/null | sort)
    n_iters="${#iters[@]}"

    if [ "$n_iters" -lt 2 ]; then
        echo
        echo "  [skip] $pc_name -- only $n_iters iter CSVs found (need >= 2)"
        continue
    fi

    out_dir="$pc_dir/out"
    mkdir -p "$out_dir/plots"

    echo
    echo "  [run]  $pc_name  (label=$label, $n_iters iters)"
    echo "         -> $out_dir/"

    # (a) stats
    "$PY" -u "$STATS_PY" \
        "${iters[@]}" \
        --machine "$label" \
        --markdown "$out_dir/paper_block.tex" \
        > "$out_dir/cross_run_stats.log" 2>&1 \
        || { echo "    !! stats failed; see $out_dir/cross_run_stats.log"; continue; }

    # (b) plots
    "$PY" -u "$PLOT_PY" \
        "${iters[@]}" \
        --machine "$label" \
        --out-dir "$out_dir/plots" \
        > "$out_dir/plot_stats.log" 2>&1 \
        || echo "    !! plots failed; see $out_dir/plot_stats.log"

    # Accumulate for the cross-machine pass with a label prefix.
    # We pass --label-map entries to compute_stats_cross_machine.py so it
    # knows which files belong to which machine.
    for f in "${iters[@]}"; do
        ALL_CSVS+=("${label}::${f}")
    done

    echo "    done"
done

# ----------------------------------------------------------------------------
# Cross-machine pass
# ----------------------------------------------------------------------------
echo
echo "=================================================================="
echo "  CROSS-MACHINE AGGREGATION"
echo "=================================================================="
echo
echo "  ${#ALL_CSVS[@]} total iter CSVs across all machines"

if [ "${#ALL_CSVS[@]}" -lt 2 ]; then
    echo "  not enough data for cross-machine aggregation; exiting"
    exit 0
fi

"$PY" -u "$CROSS_PY" \
    --label-csv "${ALL_CSVS[@]}" \
    --out-dir "$CROSS_DIR" \
    --markdown "$CROSS_DIR/paper_block_cross_machine.tex" \
    2>&1 | tee "$CROSS_DIR/cross_machine_stats.log"

echo
echo "=================================================================="
echo "  DONE"
echo "=================================================================="
echo "  Per-machine outputs: <ROOT>/pc_*/out/"
echo "  Cross-machine outputs: $CROSS_DIR/"
