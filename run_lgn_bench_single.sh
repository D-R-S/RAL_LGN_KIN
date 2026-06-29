#!/bin/bash
# =============================================================================
#  run_lgn_bench.sh   - Phase 0..6 bench pipeline for lgn_hand_ik
#
#  Three-stage build (test / demo / bench), three isolated bench runs on a
#  pinned core, demo dynamics gates, optional Pinocchio dynamics comparison,
#  perf telemetry, aggregation.
#
#  Edit the toggles below, or override at invocation:
#     RUN_DYNAMICS_COMPARISON=1 PIN_CORE=4 ./run_lgn_bench.sh
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Manual toggles
# -----------------------------------------------------------------------------

# Dynamics comparison vs Pinocchio.
#   0 = OFF (default). Demo dyn gates still RUN and report PASS/FAIL, but the
#       Pinocchio dyn crosscheck is allowed to abort without killing the run.
#       Use this while the lgn dyn module is broken.
#   1 = ON. Full Pinocchio dyn comparison. Aborts the whole run if H_05 fails.
RUN_DYNAMICS_COMPARISON=${RUN_DYNAMICS_COMPARISON:-0}

# Paths (env-overridable)
REPO="${LGN_REPO:-$HOME/lgn_hand_ik}"
IMGUI_DIR="${IMGUI_DIR:-$HOME/imgui}"
PIN_CORE="${PIN_CORE:-2}"
DEMO_URDF="${DEMO_URDF:-$REPO/URDFs/two_segment.urdf}"
RESULTS="$REPO/results_v2"
RUN_TAG="${RUN_TAG:-$(hostname -s)_$(date +%Y%m%d_%H%M%S)}"

# CMake hint so Pinocchio's cmake configs are found alongside the lgn globs
PIN_CMAKE="-DCMAKE_PREFIX_PATH=/opt/ros/jazzy;/opt/ros/jazzy/lib/x86_64-linux-gnu/cmake"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { echo ""; echo "=== $* ==="; }
warn() { echo "!! $*"; }
die()  { echo "FATAL: $*"; exit 1; }

[ -d "$REPO" ]      || die "Repo not found at $REPO"
[ -d "$IMGUI_DIR" ] || die "ImGui not found at $IMGUI_DIR"

# Fallback for DEMO_URDF
if [ ! -f "$DEMO_URDF" ] && [ -f "$REPO/demo/two_segment.urdf" ]; then
    DEMO_URDF="$REPO/demo/two_segment.urdf"
fi

cd "$REPO"

# -----------------------------------------------------------------------------
# PHASE 0 - system prep
# -----------------------------------------------------------------------------
log "PHASE 0  system info and CPU prep"
echo "REPO       : $REPO"
echo "IMGUI_DIR  : $IMGUI_DIR"
echo "PIN_CORE   : $PIN_CORE"
echo "RUN_TAG    : $RUN_TAG"
echo "RUN_DYNAMICS_COMPARISON : $RUN_DYNAMICS_COMPARISON"
echo

lscpu | grep -E 'Model name|Vendor|^CPU\(s\)|MHz' | head -5 || true
uname -r
g++ --version | head -1

echo
log "CPU governor -> performance, paranoid -> 1, drop caches"

# Governor with fallback: cpupower on PATH, kernel-tools path, then sysfs.
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
# PHASE 4 - isolated bench runs, pinned to PIN_CORE, OMP=1
# -----------------------------------------------------------------------------
log "PHASE 4  isolated bench runs on core $PIN_CORE"

run_bench () {
    local name="$1"
    local bin="$2"
    local required="${3:-1}"
    local skip_dyn="${4:-0}"

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
            die "$bin exited rc=$rc - see $RESULTS/log_${name}.txt"
        else
            warn "$bin aborted (rc=$rc). Partial CSV: $(wc -l < "$RESULTS/results_${name}.csv") lines"
            warn "  tail of log:"
            tail -20 "$RESULTS/log_${name}.txt" | sed 's/^/    /'
        fi
    fi
}

if [ "$RUN_DYNAMICS_COMPARISON" = "1" ]; then
    log "RUN_DYNAMICS_COMPARISON=1 - full Pinocchio dyn comparison"
    run_bench lgn_kdl  bench_v2_lgn_kdl  1 0
    run_bench pin      bench_v2_pin      1 0
    run_bench lgn_only bench_v2_lgn_only 1 0
else
    log "RUN_DYNAMICS_COMPARISON=0 - Pinocchio dyn-section skipped (LGN_SKIP_DYN=1)"
    log "  lgn dyn benches still run; their numbers are valid even while dyn module is broken"
    run_bench lgn_kdl  bench_v2_lgn_kdl  1 0
    run_bench pin      bench_v2_pin      0 1
    run_bench lgn_only bench_v2_lgn_only 1 0
fi

# Cross-check gate output (Pinocchio bench writes them on stderr)
echo
log "Cross-check gates"
if [ "$RUN_DYNAMICS_COMPARISON" = "1" ]; then
    echo "--- H_04 mass_matrix ---"
    grep "crosscheck.*mass_matrix" "$RESULTS/log_pin.txt" || warn "no mass_matrix lines"
    echo "--- H_05 coriolis_vector ---"
    grep "crosscheck.*coriolis_vector" "$RESULTS/log_pin.txt" || warn "no coriolis_vector lines"
else
    warn "dyn comparison disabled - H_04/H_05 not exercised"
    warn "  flip RUN_DYNAMICS_COMPARISON to 1 once dyn is fixed"
fi

# Concatenate raw CSV, picking the first non-empty file as header source
echo
log "Concatenating raw CSV"
HEADER_SRC=""
for f in "$RESULTS/results_lgn_kdl.csv" "$RESULTS/results_lgn_only.csv" "$RESULTS/results_pin.csv"; do
    if [ -s "$f" ]; then
        HEADER_SRC="$f"
        break
    fi
done
[ -n "$HEADER_SRC" ] || die "no bench produced any CSV output - everything aborted"

head -1 "$HEADER_SRC" > "$RESULTS/results_v2_raw.csv"
for f in "$RESULTS/results_lgn_kdl.csv" "$RESULTS/results_pin.csv" "$RESULTS/results_lgn_only.csv"; do
    [ -s "$f" ] && tail -n +2 -q "$f" >> "$RESULTS/results_v2_raw.csv"
done
wc -l "$RESULTS"/*.csv

# -----------------------------------------------------------------------------
# PHASE 5 - perf telemetry
# -----------------------------------------------------------------------------
log "PHASE 5  perf telemetry"
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
echo "  Demo dynamics gates     : $DEMO_GATES"
if [ "$DEMO_GATES" = "FAIL" ]; then
    echo "    [!] dyn module is broken. FK and IK numbers below remain valid;"
    echo "        Pinocchio dyn comparison skipped on purpose."
fi

echo
echo "===================================================================="
echo "  CORRECTNESS GATES"
echo "===================================================================="
if [ "$RUN_DYNAMICS_COMPARISON" = "1" ]; then
    echo "Mass matrix H_04:"
    grep "crosscheck.*mass_matrix" "$RESULTS/log_pin.txt" || true
    echo
    echo "Coriolis H_05:"
    grep "crosscheck.*coriolis_vector" "$RESULTS/log_pin.txt" || true
else
    echo "  H_04 / H_05 not exercised (RUN_DYNAMICS_COMPARISON=0)."
    echo "  See $RESULTS/demo_gates.log for C1/C2/C3 demo-side gates."
fi

echo
echo "===================================================================="
echo "  HEADLINE NUMBERS - block-mean per (solver, benchmark, n)"
echo "===================================================================="
for bench in fk ik_posonly vel_only dyn_M dyn_C dyn_RNEA_eq dyn_ABA_eq; do
    echo
    echo "--- $bench ---"
    awk -F, -v b="$bench" '
        NR>1 && $2==b { s[$1","$3]+=$4; n[$1","$3]++ }
        END { for (k in s) printf "%-20s %12.1f ns  (samples=%d)\n", k, s[k]/n[k], n[k] }
    ' "$RESULTS/results_v2_raw.csv" | sort -t, -k1,1 -k2,2n
done

echo
echo "===================================================================="
echo "  KINEMATICS RATIOS"
echo "===================================================================="
echo
echo "--- FK lgn/KDL and lgn/Pinocchio ---"
awk -F, '
    NR>1 && $2 == "fk" { sum[$1","$3] += $4; cnt[$1","$3] += 1 }
    END {
        for (k in sum) mean[k] = sum[k] / cnt[k]
        for (k in mean) {
            split(k, a, ",")
            if (a[1] == "lgn") {
                ndof = a[2]
                line = sprintf("  n=%-4s", ndof)
                if (("kdl,"ndof) in mean)
                    line = line sprintf("  lgn/KDL=%.2fx", mean["kdl,"ndof]/mean["lgn,"ndof])
                if (("pinocchio,"ndof) in mean)
                    line = line sprintf("  lgn/Pin=%.2fx", mean["pinocchio,"ndof]/mean["lgn,"ndof])
                print line
            }
        }
    }
' "$RESULTS/results_v2_raw.csv" | sort -V

echo
echo "--- IK lgn/Pinocchio ---"
awk -F, '
    NR>1 && $2 == "ik_posonly" { sum[$1","$3] += $4; cnt[$1","$3] += 1 }
    END {
        for (k in sum) mean[k] = sum[k] / cnt[k]
        for (k in mean) {
            split(k, a, ",")
            if (a[1] == "lgn") {
                ndof = a[2]
                if (("pinocchio,"ndof) in mean)
                    printf "  n=%-4s lgn/Pin=%.2fx\n", ndof, mean["pinocchio,"ndof]/mean["lgn,"ndof]
            }
        }
    }
' "$RESULTS/results_v2_raw.csv" | sort -V

echo
echo "--- Velocity propagation lgn/scalar ---"
awk -F, '
    NR>1 && $2 == "vel_only" { sum[$1","$3] += $4; cnt[$1","$3] += 1 }
    END {
        for (k in sum) mean[k] = sum[k] / cnt[k]
        for (k in mean) {
            split(k, a, ",")
            if (a[1] == "lgn") {
                ndof = a[2]
                if (("scalar,"ndof) in mean)
                    printf "  n=%-4s lgn/scalar=%.2fx\n", ndof, mean["lgn,"ndof]/mean["scalar,"ndof]
            }
        }
    }
' "$RESULTS/results_v2_raw.csv" | sort -V

echo
echo "===================================================================="
echo "  DYNAMICS RATIOS - lgn vs Pinocchio"
echo "===================================================================="
for bench in dyn_M dyn_C dyn_RNEA_eq dyn_ABA_eq; do
    echo
    echo "--- $bench (pin/lgn ratio; <1 means Pinocchio is faster) ---"
    awk -F, -v b="$bench" '
        NR>1 && $2 == b { sum[$1","$3] += $4; cnt[$1","$3] += 1 }
        END {
            for (k in sum) mean[k] = sum[k] / cnt[k]
            for (k in mean) {
                split(k, a, ",")
                if (a[1] == "lgn") {
                    ndof = a[2]
                    if (("pinocchio,"ndof) in mean)
                        printf "  n=%-4s pin/lgn=%.2fx  (lgn=%9.0f ns, pin=%9.0f ns)\n", \
                            ndof, mean["pinocchio,"ndof]/mean["lgn,"ndof], \
                            mean["lgn,"ndof], mean["pinocchio,"ndof]
                }
            }
        }
    ' "$RESULTS/results_v2_raw.csv" | sort -V
done

echo
echo "===================================================================="
echo "  PERF TELEMETRY"
echo "===================================================================="
if [ -f "$RESULTS/perf_stat_lgn.txt" ]; then
    grep -E 'instructions|cycles|cache-references|cache-misses|fp_arith' \
        "$RESULTS/perf_stat_lgn.txt" | head -15
else
    warn "perf_stat_lgn.txt not found"
fi

# -----------------------------------------------------------------------------
# Box snapshot for cross-machine analysis
# -----------------------------------------------------------------------------
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
    echo "=== dyn_comparison ==="; echo "$RUN_DYNAMICS_COMPARISON"
} > "$RESULTS/box_info_${RUN_TAG}.txt"

cp -n "$RESULTS/results_v2_raw.csv" "$RESULTS/results_v2_raw_${RUN_TAG}.csv" 2>/dev/null || true

echo
log "Files in $RESULTS/"
ls -lh "$RESULTS/"




echo
log "Computing stats + plots"
python3 "$REPO/scripts/compute_stats.py" "$RESULTS/results_v2_raw.csv" --plots --plot-dir "$RESULTS/plots" || warn "compute_stats.py failed"
log "DONE  (RUN_TAG=$RUN_TAG)"
