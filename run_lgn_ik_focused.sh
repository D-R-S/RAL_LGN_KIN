#!/bin/bash
# =============================================================================
#  run_lgn_ik_focused.sh
#
#  Stripped-down bench pipeline for the IK paper. Skips:
#    - Dynamics benches (known-broken, not part of IK story)
#    - Demo dynamics gates (same reason)
#    - KDL (not the headline rival)
#    - perf telemetry's AVX detection (works on x86 only; here we just IPC)
#
#  Runs:
#    - test_correctness  (sanity, fast)
#    - bench_v2_lgn_only (FK + IK + velocity, no Pinocchio link)
#    - bench_v2_pin      (FK + IK, with LGN_SKIP_DYN=1 to dodge the abort)
#    - compute_stats_v2.py on the resulting CSV, restricted to H01,H01p,H02p,H04
#
#  Works on x86_64 and aarch64. Auto-detects the Pinocchio install location.
#
#  Run from anywhere. Overrides:
#    LGN_REPO=/path PIN_CORE=4 ./run_lgn_ik_focused.sh
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
REPO="${LGN_REPO:-$HOME/lgn_hand_ik}"
PIN_CORE="${PIN_CORE:-2}"
RESULTS="$REPO/results_v2"
RUN_TAG="${RUN_TAG:-$(hostname -s)_$(date +%Y%m%d_%H%M%S)}"

log()  { echo ""; echo "=== $* ==="; }
warn() { echo "!! $*"; }
die()  { echo "FATAL: $*"; exit 1; }

[ -d "$REPO" ] || die "Repo not found at $REPO"
cd "$REPO"

# -----------------------------------------------------------------------------
# Detect Pinocchio install (jazzy vs humble, x86 vs arm path)
# -----------------------------------------------------------------------------
PIN_ROOT=""
for cand in /opt/ros/jazzy /opt/ros/humble /usr; do
    if ls "$cand"/lib/*/libpinocchio_default.so.* >/dev/null 2>&1 || \
       ls "$cand"/lib/libpinocchio_default.so.*     >/dev/null 2>&1; then
        PIN_ROOT="$cand"
        break
    fi
done
if [ -z "$PIN_ROOT" ]; then
    die "Pinocchio not found under /opt/ros/{jazzy,humble} or /usr"
fi
echo "Pinocchio root: $PIN_ROOT"
PIN_CMAKE="-DCMAKE_PREFIX_PATH=$PIN_ROOT;$PIN_ROOT/lib/cmake"

# -----------------------------------------------------------------------------
# PHASE 0 - system prep
# -----------------------------------------------------------------------------
log "PHASE 0  system info and CPU prep"
echo "REPO    : $REPO"
echo "PIN_CORE: $PIN_CORE"
echo "RUN_TAG : $RUN_TAG"
echo "ARCH    : $(uname -m)"
echo
lscpu | grep -E 'Model name|Vendor|Architecture|^CPU\(s\)|MHz' | head -6 || true
uname -r
g++ --version | head -1
echo

log "Governor -> performance, paranoid -> 1, drop caches"
set_governor () {
    sudo bash -c '
        if command -v cpupower >/dev/null 2>&1; then
            cpupower frequency-set -g performance 2>&1 | tail -3; exit 0; fi
        cp_bin=$(ls /usr/lib/linux-tools/*/cpupower 2>/dev/null | tail -1)
        if [ -n "$cp_bin" ] && [ -x "$cp_bin" ]; then
            "$cp_bin" frequency-set -g performance 2>&1 | tail -3; exit 0; fi
        wrote=0
        for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            [ -w "$f" ] || continue
            echo performance > "$f" && wrote=1
        done
        [ "$wrote" = "1" ] && { echo "governor via sysfs"; exit 0; }
        echo "no cpufreq driver"; exit 1
    '
}
set_governor || warn "could not pin governor"

# Jetson-specific: lock clocks if available (no-op on other boxes)
if command -v jetson_clocks >/dev/null 2>&1; then
    log "Jetson: locking clocks (jetson_clocks)"
    sudo jetson_clocks
fi
if command -v nvpmodel >/dev/null 2>&1; then
    sudo nvpmodel -m 0 2>&1 | head -3 || true
    nvpmodel -q 2>&1 | head -3 || true
fi

echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid >/dev/null
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

log "CPU topology"
lscpu -e=CPU,CORE,SOCKET,MAXMHZ 2>/dev/null | head -16 || true
if [ -f "/sys/devices/system/cpu/cpu${PIN_CORE}/topology/thread_siblings_list" ]; then
    echo "core $PIN_CORE siblings: $(cat /sys/devices/system/cpu/cpu${PIN_CORE}/topology/thread_siblings_list)"
fi

log "Clean build dirs"
rm -rf build_test build_bench
mkdir -p "$RESULTS"

# -----------------------------------------------------------------------------
# PHASE 1 - correctness sanity (fast)
# -----------------------------------------------------------------------------
log "PHASE 1  correctness sanity (Debug, test_correctness only)"
mkdir -p build_test
cd build_test
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_ROS=OFF -DBUILD_DEMO=OFF \
    -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=OFF 2>&1 | tail -15
make -j"$(nproc)" test_correctness 2>&1 | tail -8
./test_correctness 2>&1 | tee "$RESULTS/test_correctness.log" | tail -5
cd "$REPO"

# -----------------------------------------------------------------------------
# PHASE 2 - bench build
# -----------------------------------------------------------------------------
log "PHASE 2  bench build (Release)"
mkdir -p build_bench
cd build_bench

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_ROS=OFF -DBUILD_DEMO=OFF \
    -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=ON \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -Wno-return-type" \
    $PIN_CMAKE 2>&1 | tee "$RESULTS/cmake_bench.log" | tail -25

if ! grep -qE 'Pinocchio:[[:space:]]+TRUE' "$RESULTS/cmake_bench.log"; then
    warn "CMake did not report Pinocchio TRUE"
    warn "If this is ARM and CMakeLists.txt globs for x86_64-linux-gnu,"
    warn "you need to patch the glob or symlink the libs. See install script."
fi

# Build only the two binaries we need for the IK story.
# bench_v2_lgn_only: lgn FK/IK/vel, no Pinocchio link.
# bench_v2_pin: Pinocchio FK/IK + dyn (we set LGN_SKIP_DYN=1 to skip dyn abort).
make -j"$(nproc)" bench_v2_lgn_only bench_v2_pin 2>&1 | tail -10
ls -lh bench_v2_lgn_only bench_v2_pin
cd "$REPO"

# -----------------------------------------------------------------------------
# PHASE 3 - isolated bench runs
# -----------------------------------------------------------------------------
log "PHASE 3  bench runs on core $PIN_CORE"

run_bench () {
    local name="$1"; local bin="$2"; local required="${3:-1}"; local skip_dyn="${4:-0}"
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    echo
    log "$name"
    date
    set +e
    if [ "$skip_dyn" = "1" ]; then
        LGN_SKIP_DYN=1 OMP_NUM_THREADS=1 taskset -c "$PIN_CORE" "./build_bench/$bin" \
            > "$RESULTS/results_${name}.csv" 2> "$RESULTS/log_${name}.txt"
    else
        OMP_NUM_THREADS=1 taskset -c "$PIN_CORE" "./build_bench/$bin" \
            > "$RESULTS/results_${name}.csv" 2> "$RESULTS/log_${name}.txt"
    fi
    local rc=$?
    set -e
    date
    if [ "$rc" != "0" ]; then
        if [ "$required" = "1" ]; then
            die "$bin rc=$rc - see $RESULTS/log_${name}.txt"
        else
            warn "$bin aborted (rc=$rc). Partial CSV: $(wc -l < "$RESULTS/results_${name}.csv") lines"
            tail -10 "$RESULTS/log_${name}.txt" | sed 's/^/    /'
        fi
    fi
}

# lgn-only is required (it's the whole left side of the story)
run_bench lgn_only bench_v2_lgn_only 1 0
# Pinocchio bench is the rival; allow abort, will still produce FK/IK rows
run_bench pin      bench_v2_pin      0 1

# Concatenate into one raw CSV
echo
log "Concatenating raw CSV"
HEADER_SRC=""
for f in "$RESULTS/results_lgn_only.csv" "$RESULTS/results_pin.csv"; do
    [ -s "$f" ] && { HEADER_SRC="$f"; break; }
done
[ -n "$HEADER_SRC" ] || die "no CSV output - both binaries aborted"
head -1 "$HEADER_SRC" > "$RESULTS/results_v2_raw.csv"
for f in "$RESULTS/results_lgn_only.csv" "$RESULTS/results_pin.csv"; do
    [ -s "$f" ] && tail -n +2 -q "$f" >> "$RESULTS/results_v2_raw.csv"
done
wc -l "$RESULTS/results_v2_raw.csv"

# -----------------------------------------------------------------------------
# PHASE 4 - basic perf (IPC + cache; AVX events skipped on ARM)
# -----------------------------------------------------------------------------
log "PHASE 4  basic perf telemetry"
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
if command -v perf >/dev/null 2>&1; then
    echo "--- perf stat (basic) ---" > "$RESULTS/perf_stat_lgn.txt"
    sudo perf stat \
        -e instructions,cycles,cache-misses,cache-references \
        --repeat 3 \
        env OMP_NUM_THREADS=1 taskset -c "$PIN_CORE" \
            ./build_bench/bench_v2_lgn_only \
        2>> "$RESULTS/perf_stat_lgn.txt" > /dev/null || \
        warn "perf stat failed - paranoid setting or missing events"
    echo
    grep -E 'instructions|cycles|cache-' "$RESULTS/perf_stat_lgn.txt" | head -10
else
    warn "perf not installed - skipping"
fi

# -----------------------------------------------------------------------------
# PHASE 5 - stats
# -----------------------------------------------------------------------------
log "PHASE 5  stats (H01, H01p, H02p, H04 only)"
if [ -f "$REPO/scripts/compute_stats_v2.py" ]; then
    python3 "$REPO/scripts/compute_stats_v2.py" \
        "$RESULTS/results_v2_raw.csv" \
        --only H01,H01p,H02p,H04 \
        --perf "$RESULTS/perf_stat_lgn.txt" \
        || warn "compute_stats_v2.py failed"
else
    warn "scripts/compute_stats_v2.py not found - skipping stats"
fi

# -----------------------------------------------------------------------------
# Box snapshot
# -----------------------------------------------------------------------------
log "Capturing box info"
{
    echo "=== hostname ==="; hostname
    echo "=== uname ==="; uname -a
    echo "=== gcc ==="; gcc --version | head -1
    echo "=== cpu ==="; lscpu
    echo "=== mem ==="; free -h
    echo "=== os ==="; cat /etc/os-release
    echo "=== pin ==="; echo "core=$PIN_CORE"
    echo "=== pinocchio_root ==="; echo "$PIN_ROOT"
    if command -v nvpmodel >/dev/null 2>&1; then
        echo "=== nvpmodel ==="; nvpmodel -q
    fi
    if command -v jetson_clocks >/dev/null 2>&1; then
        echo "=== jetson_clocks ==="; jetson_clocks --show 2>/dev/null | head -20
    fi
} > "$RESULTS/box_info_${RUN_TAG}.txt"

cp -n "$RESULTS/results_v2_raw.csv" "$RESULTS/results_v2_raw_${RUN_TAG}.csv" 2>/dev/null || true

echo
log "Files in $RESULTS/"
ls -lh "$RESULTS/" | head -20

echo
log "DONE  (RUN_TAG=$RUN_TAG)"
