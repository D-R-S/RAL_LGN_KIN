#!/bin/bash
# =============================================================================
#  run_lgn_bench.sh   - Phase 0..6 bench pipeline for lgn_hand_ik
#                       10x cross-run aggregation built in.
#
#  Differences from the prior single-run version:
#    - Phase 4 (bench) runs N_RUNS times (default 10). Each iteration's three
#      CSVs are concatenated into results_v2_raw_<RUN_TAG>_iter<NN>.csv .
#    - Phases 0-3 (CPU prep + build) run ONCE per invocation.
#    - Phase 5 (perf telemetry) runs ONCE, on the last iteration.
#    - Phase 6 runs the original compute_stats.py on the last iteration's
#      CSV (for backward compat with the existing summary), then runs
#      compute_stats_10x.py to cross-aggregate all N iterations.
#
#  Setup (one-time per machine):
#    - Drop scripts/compute_stats_10x.py into the repo.
#    - Save THIS file over the existing run_lgn_bench.sh.
#
#  Invocation:
#    N_RUNS=10 MACHINE_LABEL=M2-Zen5 ./run_lgn_bench.sh
#
#  Override knobs (env vars):
#    N_RUNS                 (default 10)     iterations of Phase 4
#    SLEEP_BETWEEN_RUNS     (default 2)      seconds between iterations
#    MACHINE_LABEL          (default hostname)  used as row label in paper output
#    RUN_DYNAMICS_COMPARISON (default 0)     full pin-dyn crosscheck (will abort if H_05 fails)
#    PIN_CORE               (default 2)
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Manual toggles
# -----------------------------------------------------------------------------
RUN_DYNAMICS_COMPARISON=${RUN_DYNAMICS_COMPARISON:-0}
N_RUNS="${N_RUNS:-10}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-2}"
MACHINE_LABEL="${MACHINE_LABEL:-$(hostname -s)}"

# Paths (env-overridable)
REPO="${LGN_REPO:-$HOME/lgn_hand_ik}"
IMGUI_DIR="${IMGUI_DIR:-$HOME/imgui}"
PIN_CORE="${PIN_CORE:-2}"
DEMO_URDF="${DEMO_URDF:-$REPO/URDFs/two_segment.urdf}"
RESULTS="$REPO/results_v2"
RUN_TAG="${RUN_TAG:-$(hostname -s)_$(date +%Y%m%d_%H%M%S)}"

PIN_CMAKE="-DCMAKE_PREFIX_PATH=/opt/ros/jazzy;/opt/ros/jazzy/lib/x86_64-linux-gnu/cmake"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { echo ""; echo "=== $* ==="; }
warn() { echo "!! $*"; }
die()  { echo "FATAL: $*"; exit 1; }

[ -d "$REPO" ]      || die "Repo not found at $REPO"
[ -d "$IMGUI_DIR" ] || die "ImGui not found at $IMGUI_DIR"

if [ ! -f "$DEMO_URDF" ] && [ -f "$REPO/demo/two_segment.urdf" ]; then
    DEMO_URDF="$REPO/demo/two_segment.urdf"
fi

cd "$REPO"

# -----------------------------------------------------------------------------
# PHASE 0 - system prep
# -----------------------------------------------------------------------------
log "PHASE 0  system info and CPU prep"
echo "REPO          : $REPO"
echo "IMGUI_DIR     : $IMGUI_DIR"
echo "PIN_CORE      : $PIN_CORE"
echo "RUN_TAG       : $RUN_TAG"
echo "MACHINE_LABEL : $MACHINE_LABEL"
echo "N_RUNS        : $N_RUNS"
echo "RUN_DYNAMICS_COMPARISON : $RUN_DYNAMICS_COMPARISON"
echo

lscpu | grep -E 'Model name|Vendor|^CPU\(s\)|MHz' | head -5 || true
uname -r
g++ --version | head -1

echo
log "CPU governor -> performance, paranoid -> 1, drop caches"

set_governor () {
    sudo bash -c '
        if command -v cpupower >/dev/null 2>&1; then
            cpupower frequency-set -g performance 2>&1 | tail -3
            exit 0
        fi
        cp_bin=$(ls /usr/lib/linux-tools/*/cpupower 2>/dev/null | tail -1)
        if [ -n "$cp_bin" ] && [ -x "$cp_bin" ]; then
            "$cp_bin" frequency-set -g performance 2>&1 | tail -3
            exit 0
        fi
        wrote=0
        for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            [ -w "$f" ] || continue
            echo performance > "$f" && wrote=1
        done
        if [ "$wrote" = "1" ]; then
            echo "governor set via sysfs"
            exit 0
        fi
        echo "no cpufreq driver - governor not changed"
        exit 1
    '
}
set_governor || warn "could not pin governor; results may drift"

echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid >/dev/null
echo "perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid)"
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

echo
log "CPU topology - verify pin core $PIN_CORE"
lscpu -e=CPU,CORE,SOCKET,MAXMHZ | head -20 || true
if [ -f "/sys/devices/system/cpu/cpu${PIN_CORE}/topology/thread_siblings_list" ]; then
    echo "core $PIN_CORE SMT siblings: $(cat /sys/devices/system/cpu/cpu${PIN_CORE}/topology/thread_siblings_list)"
else
    warn "cpu${PIN_CORE} not present"
fi

echo
log "Clean build dirs and results"
rm -rf build_test build_demo build_bench build_bench_asm
mkdir -p "$RESULTS"

# -----------------------------------------------------------------------------
# PHASE 1 - correctness gates
# -----------------------------------------------------------------------------
log "PHASE 1  correctness gates (Debug)"
mkdir -p build_test
cd build_test

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_ROS=OFF \
    -DBUILD_DEMO=OFF \
    -DBUILD_TESTING=ON \
    -DBUILD_BENCHMARKS=OFF 2>&1 | tail -25

make -j"$(nproc)" test_correctness test_roundtrip 2>&1 | tail -15

echo
log "test_correctness"
./test_correctness 2>&1 | tee "$RESULTS/test_correctness.log" | tail -10

echo
log "test_roundtrip"
./test_roundtrip 2>&1 | tee "$RESULTS/test_roundtrip.log" | tail -15

cd "$REPO"

# -----------------------------------------------------------------------------
# PHASE 2 - demo build, run dynamics gates headlessly
# -----------------------------------------------------------------------------
log "PHASE 2  demo build (Release) and dynamics gates"
mkdir -p build_demo
cd build_demo

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_ROS=OFF \
    -DBUILD_DEMO=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DIMGUI_DIR="$IMGUI_DIR" 2>&1 | tail -15

make -j"$(nproc)" lgn_demo 2>&1 | tail -10
ls -lh lgn_demo

DEMO_GATES="SKIPPED"
if [ -f "$DEMO_URDF" ]; then
    log "Running demo gates (headless) on $DEMO_URDF"
    ( unset DISPLAY; ./lgn_demo "$DEMO_URDF" 2>&1 || true ) > "$RESULTS/demo_gates.log"
    sed -n '/Dynamics Correctness Gates/,/Dynamics gates:/p' "$RESULTS/demo_gates.log"
    if grep -q "Dynamics gates: ALL PASS" "$RESULTS/demo_gates.log"; then
        DEMO_GATES="PASS"
    else
        DEMO_GATES="FAIL"
        warn "Demo dynamics gates FAILED - see $RESULTS/demo_gates.log"
        warn "  (dyn module is known-broken; FK/IK numbers below remain valid)"
    fi
else
    warn "DEMO_URDF not found at $DEMO_URDF - skipping demo gate run"
fi
echo "DEMO_GATES=$DEMO_GATES" > "$RESULTS/.gate_state"

cd "$REPO"

# -----------------------------------------------------------------------------
# PHASE 3 - bench build
# -----------------------------------------------------------------------------
log "PHASE 3  bench build (Release)"
mkdir -p build_bench
cd build_bench

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_ROS=OFF \
    -DBUILD_DEMO=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_BENCHMARKS=ON \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -Wno-return-type" \
    $PIN_CMAKE 2>&1 | tee "$RESULTS/cmake_bench.log" | tail -30

if ! grep -qE 'Pinocchio:[[:space:]]+TRUE' "$RESULTS/cmake_bench.log"; then
    warn "CMake did not report Pinocchio TRUE - see $RESULTS/cmake_bench.log"
fi
if ! grep -qE 'KDL:[[:space:]]+1' "$RESULTS/cmake_bench.log"; then
    warn "CMake did not report KDL: 1"
fi

make -j"$(nproc)" bench_v2_lgn_kdl bench_v2_pin bench_v2_lgn_only 2>&1 | tail -10
ls -lh bench_v2_*

cd "$REPO"

# -----------------------------------------------------------------------------
# PHASE 4 - isolated bench runs (N_RUNS iterations, pinned to PIN_CORE, OMP=1)
# -----------------------------------------------------------------------------
log "PHASE 4  isolated bench runs on core $PIN_CORE  ($N_RUNS iterations)"

run_bench_to () {
    # $1 name  $2 binary  $3 required  $4 skip_dyn  $5 out_dir
    local name="$1"
    local bin="$2"
    local required="${3:-1}"
    local skip_dyn="${4:-0}"
    local out_dir="$5"

    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    echo
    log "$name"
    date

    set +e
    if [ "$skip_dyn" = "1" ]; then
        LGN_SKIP_DYN=1 OMP_NUM_THREADS=1 taskset -c "$PIN_CORE" "./build_bench/$bin" \
            > "$out_dir/results_${name}.csv" 2> "$out_dir/log_${name}.txt"
    else
        OMP_NUM_THREADS=1 taskset -c "$PIN_CORE" "./build_bench/$bin" \
            > "$out_dir/results_${name}.csv" 2> "$out_dir/log_${name}.txt"
    fi
    local rc=$?
    set -e
    date

    if [ "$rc" != "0" ]; then
        if [ "$required" = "1" ]; then
            die "$bin exited rc=$rc - see $out_dir/log_${name}.txt"
        else
            warn "$bin aborted (rc=$rc). Partial CSV: $(wc -l < "$out_dir/results_${name}.csv") lines"
            warn "  tail of log:"
            tail -20 "$out_dir/log_${name}.txt" | sed 's/^/    /'
        fi
    fi
}

SUCCESSFUL_ITERS=0
FAILED_ITERS=()

for ITER in $(seq 1 "$N_RUNS"); do
    ITER_TAG="${RUN_TAG}_iter$(printf '%02d' "$ITER")"
    ITER_DIR="$RESULTS/iter_$(printf '%02d' "$ITER")"
    mkdir -p "$ITER_DIR"

    echo
    echo "######################################################################"
    echo "##  ITERATION $ITER / $N_RUNS    tag = $ITER_TAG"
    echo "######################################################################"

    if [ "$RUN_DYNAMICS_COMPARISON" = "1" ]; then
        run_bench_to lgn_kdl  bench_v2_lgn_kdl  1 0 "$ITER_DIR"
        run_bench_to pin      bench_v2_pin      1 0 "$ITER_DIR"
        run_bench_to lgn_only bench_v2_lgn_only 1 0 "$ITER_DIR"
    else
        run_bench_to lgn_kdl  bench_v2_lgn_kdl  1 1 "$ITER_DIR"
        run_bench_to pin      bench_v2_pin      0 1 "$ITER_DIR"
        run_bench_to lgn_only bench_v2_lgn_only 1 1 "$ITER_DIR"
    fi

    # Concatenate this iteration's three CSVs.
    HEADER_SRC=""
    for f in "$ITER_DIR/results_lgn_kdl.csv" "$ITER_DIR/results_lgn_only.csv" "$ITER_DIR/results_pin.csv"; do
        if [ -s "$f" ]; then HEADER_SRC="$f"; break; fi
    done
    if [ -z "$HEADER_SRC" ]; then
        warn "iter $ITER produced no usable CSV; dropping"
        FAILED_ITERS+=("$ITER")
        continue
    fi

    ITER_OUT="$RESULTS/results_v2_raw_${ITER_TAG}.csv"
    head -1 "$HEADER_SRC" > "$ITER_OUT"
    for f in "$ITER_DIR/results_lgn_kdl.csv" "$ITER_DIR/results_pin.csv" "$ITER_DIR/results_lgn_only.csv"; do
        [ -s "$f" ] && tail -n +2 -q "$f" >> "$ITER_OUT"
    done

    LINES=$(wc -l < "$ITER_OUT")
    echo "iter $ITER -> $ITER_OUT  ($LINES lines)"
    SUCCESSFUL_ITERS=$((SUCCESSFUL_ITERS + 1))

    # Remove per-iter individual CSVs to save disk; keep logs.
    rm -f "$ITER_DIR/results_lgn_kdl.csv" \
          "$ITER_DIR/results_lgn_only.csv" \
          "$ITER_DIR/results_pin.csv" 2>/dev/null || true

    [ "$ITER" -lt "$N_RUNS" ] && sleep "$SLEEP_BETWEEN_RUNS"
done

echo
log "Phase 4 done: $SUCCESSFUL_ITERS / $N_RUNS iterations produced usable CSVs"
if [ "${#FAILED_ITERS[@]}" -gt 0 ]; then
    warn "Failed iterations: ${FAILED_ITERS[*]}"
fi
if [ "$SUCCESSFUL_ITERS" = "0" ]; then
    die "no iterations produced usable CSVs - aborting"
fi

# Backward compat: point the canonical $RESULTS/results_v2_raw.csv at the
# last successful iteration so Phase 5 perf telemetry + the legacy
# compute_stats.py summary still see a "current" run.
LAST_ITER_CSV=$(ls -1t "$RESULTS"/results_v2_raw_${RUN_TAG}_iter*.csv 2>/dev/null | head -1)
if [ -n "$LAST_ITER_CSV" ]; then
    cp -f "$LAST_ITER_CSV" "$RESULTS/results_v2_raw.csv"
fi

# Cross-check gate output from the last iteration (mirrors original behaviour)
echo
log "Cross-check gates (from last iteration)"
LAST_PIN_LOG=$(ls -1t "$RESULTS"/iter_*/log_pin.txt 2>/dev/null | head -1)
if [ "$RUN_DYNAMICS_COMPARISON" = "1" ] && [ -n "$LAST_PIN_LOG" ]; then
    echo "--- H_04 mass_matrix ---"
    grep "crosscheck.*mass_matrix" "$LAST_PIN_LOG" || warn "no mass_matrix lines"
    echo "--- H_05 coriolis_vector ---"
    grep "crosscheck.*coriolis_vector" "$LAST_PIN_LOG" || warn "no coriolis_vector lines"
else
    warn "dyn comparison disabled - H_04/H_05 not exercised"
    warn "  flip RUN_DYNAMICS_COMPARISON to 1 once dyn is fixed"
fi

# -----------------------------------------------------------------------------
# PHASE 5 - perf telemetry (once per invocation, after all iters)
# -----------------------------------------------------------------------------
log "PHASE 5  perf telemetry (single snapshot, last iteration)"
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

PERF_SCRIPT="$REPO/benchmarks/perf_avx2.sh"
if [ -f "$PERF_SCRIPT" ]; then
    sed -i 's/\r$//' "$PERF_SCRIPT" 2>/dev/null || true
    chmod +x "$PERF_SCRIPT" 2>/dev/null || true
    bash "$PERF_SCRIPT" "$RESULTS" 2>&1 | tee "$RESULTS/perf_avx2.log"
else
    warn "$PERF_SCRIPT not found - skipping perf telemetry"
fi

# -----------------------------------------------------------------------------
# PHASE 6 - aggregate and summary
# -----------------------------------------------------------------------------
log "PHASE 6  aggregate and summary"

DEMO_GATES="UNKNOWN"
if [ -f "$RESULTS/.gate_state" ]; then
    . "$RESULTS/.gate_state"
fi

echo
echo "===================================================================="
echo "  PIPELINE STATE"
echo "===================================================================="
echo "  RUN_DYNAMICS_COMPARISON : $RUN_DYNAMICS_COMPARISON  (1=on, 0=off)"
echo "  N_RUNS                  : $N_RUNS  (successful: $SUCCESSFUL_ITERS)"
echo "  Demo dynamics gates     : $DEMO_GATES"
if [ "$DEMO_GATES" = "FAIL" ]; then
    echo "    [!] dyn module is broken. FK and IK numbers below remain valid;"
    echo "        Pinocchio dyn comparison skipped on purpose."
fi

echo
echo "===================================================================="
echo "  HEADLINE NUMBERS - last-iteration block-mean per (solver, bench, n)"
echo "===================================================================="
echo "  (legacy single-run summary; see CROSS-RUN below for the real stats)"
for bench in fk ik_posonly ik_posonly_fair vel_only_alloc vel_only_prealloc dyn_M dyn_C dyn_RNEA_eq dyn_ABA_eq; do
    echo
    echo "--- $bench ---"
    awk -F, -v b="$bench" '
        NR>1 && $2==b { s[$1","$3]+=$4; n[$1","$3]++ }
        END { for (k in s) printf "%-20s %12.1f ns  (samples=%d)\n", k, s[k]/n[k], n[k] }
    ' "$RESULTS/results_v2_raw.csv" | sort -t, -k1,1 -k2,2n
done

echo
log "Capturing box info"
{
    echo "=== hostname ==="; hostname
    echo "=== uname ==="; uname -a
    echo "=== gcc ==="; gcc --version | head -1
    echo "=== cpu ==="; lscpu
    echo "=== mem ==="; free -h
    echo "=== os ==="; cat /etc/os-release
    echo "=== pin ==="; echo "core=$PIN_CORE"
    echo "=== run_tag ==="; echo "$RUN_TAG"
    echo "=== machine_label ==="; echo "$MACHINE_LABEL"
    echo "=== n_runs ==="; echo "$N_RUNS"
    echo "=== successful_iters ==="; echo "$SUCCESSFUL_ITERS"
    echo "=== dyn_comparison ==="; echo "$RUN_DYNAMICS_COMPARISON"
} > "$RESULTS/box_info_${RUN_TAG}.txt"

echo
log "Files in $RESULTS/"
ls -lh "$RESULTS/"

# Legacy single-run summary (last iteration) for backward compat
echo
log "Legacy single-run stats + plots (last iteration only)"
python3 "$REPO/scripts/compute_stats.py" "$RESULTS/results_v2_raw.csv" \
    --plots --plot-dir "$RESULTS/plots" \
    || warn "compute_stats.py failed"

# -----------------------------------------------------------------------------
# Cross-run aggregation: the actual headline stats for the paper
# -----------------------------------------------------------------------------
echo
log "Cross-run stats over $SUCCESSFUL_ITERS iterations  (machine: $MACHINE_LABEL)"

ITER_CSVS=( "$RESULTS"/results_v2_raw_${RUN_TAG}_iter*.csv )
if [ "${#ITER_CSVS[@]}" -lt 2 ] || [ ! -e "${ITER_CSVS[0]}" ]; then
    warn "fewer than 2 iteration CSVs found; skipping cross-run aggregate"
else
    CROSS_LOG="$RESULTS/cross_run_stats_${MACHINE_LABEL}.log"
    PAPER_TEX="$RESULTS/paper_block_${MACHINE_LABEL}.tex"
    python3 "$REPO/scripts/compute_stats_10x.py" \
        "${ITER_CSVS[@]}" \
        --machine "$MACHINE_LABEL" \
        --markdown "$PAPER_TEX" \
        2>&1 | tee "$CROSS_LOG" \
        || warn "compute_stats_10x.py failed"

    echo
    echo "Cross-run summary log :  $CROSS_LOG"
    echo "Paper-ready LaTeX     :  $PAPER_TEX"
fi

log "ALL DONE  (RUN_TAG=$RUN_TAG, $SUCCESSFUL_ITERS/$N_RUNS iterations on $MACHINE_LABEL)"
