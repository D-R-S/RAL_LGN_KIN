#!/usr/bin/env bash
# ============================================================================
#  perf_avx2.sh  —  patch 1.4
#  AVX2/AVX-512 lane utilisation and SIMD throughput analysis
#
#  Patch 1.4: extractor moved to a standalone Python file (no inline heredoc)
#  to eliminate bash-vs-Python quoting ambiguity on the regex line.
#
#  Two tools, run in sequence:
#    1. llvm-mca   — static analysis of the hot FK loop assembly.
#    2. perf stat  — hardware counter measurement of the live FK binary.
#
#  Prerequisites:
#    sudo apt install llvm clang linux-tools-$(uname -r) linux-tools-common
#
#  Usage:
#    cd ~/lgn_hand_ik
#    bash benchmarks/perf_avx2.sh 2>&1 | tee results_v2/perf_avx2.log
# ============================================================================
set -euo pipefail

RESULTS="${1:-results_v2}"
mkdir -p "$RESULTS"

BENCH_BIN="build_bench/bench_v2_lgn_only"
BENCH_SRC="benchmarks/bench_v2.cpp"
BUILD_DIR="build_bench_asm"

# ── Vendor check ──────────────────────────────────────────────────────────────
VENDOR=$(grep -m1 vendor_id /proc/cpuinfo | awk '{print $3}')
if [ "$VENDOR" != "GenuineIntel" ]; then
  echo "Note: CPU vendor is '$VENDOR'."
  echo "  Intel-named fp_arith_inst_retired.* events may not exist on AMD;"
  echo "  the script automatically falls back to basic counters in that case."
  echo ""
fi

# ── Kernel paranoid precheck ──────────────────────────────────────────────────
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "?")
if [ "$PARANOID" != "?" ] && [ "$PARANOID" -gt 1 ]; then
  echo "Note: perf_event_paranoid=$PARANOID; perf needs sudo or a lower setting."
  echo "  echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
  echo ""
fi

# ── Tool prerequisites ────────────────────────────────────────────────────────
for cmd in llvm-mca perf objdump c++filt; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: $cmd not found."
    echo "  Install: sudo apt install llvm clang linux-tools-\$(uname -r) linux-tools-common"
    exit 1
  fi
done

mkdir -p "$BUILD_DIR"

# ── Eigen include path ────────────────────────────────────────────────────────
EIGEN_INC=""
if command -v pkg-config &>/dev/null && pkg-config --exists eigen3; then
  EIGEN_INC=$(pkg-config --cflags-only-I eigen3 | tr -d ' ')
fi
if [ -z "$EIGEN_INC" ] && [ -d /usr/include/eigen3 ]; then
  EIGEN_INC="-I/usr/include/eigen3"
fi
if [ -z "$EIGEN_INC" ]; then
  echo "ERROR: Eigen3 not found via pkg-config or in /usr/include/eigen3"
  echo "  Install: sudo apt install libeigen3-dev"
  exit 1
fi

echo "=== Step 1: Compile FK hot loop to assembly ==="
echo "Eigen include: $EIGEN_INC"

if clang++ -O3 -march=native -ffast-math -std=c++17 \
     -DBENCH_LGN=1 -DBENCH_VEL_BASELINE=1 \
     $EIGEN_INC -I include -I third_party/tinyxml2 \
     -S -mllvm -x86-asm-syntax=intel \
     -o "$BUILD_DIR/bench_lgn_fk.s" \
     "$BENCH_SRC" 2>"$BUILD_DIR/clang_err.log"; then
  ASM_SYNTAX="intel"
  COMPILER="clang++"
else
  echo "clang++ -S failed (see $BUILD_DIR/clang_err.log) — falling back to g++ AT&T syntax"
  g++ -O3 -march=native -ffast-math -std=c++17 \
    -DBENCH_LGN=1 -DBENCH_VEL_BASELINE=1 \
    $EIGEN_INC -I include -I third_party/tinyxml2 \
    -S -o "$BUILD_DIR/bench_lgn_fk.s" \
    "$BENCH_SRC"
  ASM_SYNTAX="att"
  COMPILER="g++"
fi
echo "Assembly written: $BUILD_DIR/bench_lgn_fk.s ($COMPILER, $ASM_SYNTAX syntax)"

echo ""
echo "=== Step 2: Extract FK hot loop ==="

# Write the extractor as a standalone Python script — no shell quoting risk.
EXTRACTOR="$BUILD_DIR/extract_fk.py"
{
  echo "import re, subprocess, sys"
  echo ""
  echo "asm_path = 'build_bench_asm/bench_lgn_fk.s'"
  echo "out_path = 'build_bench_asm/fk_hot_loop.s'"
  echo ""
  echo "with open(asm_path) as f:"
  echo "    asm_lines = f.readlines()"
  echo ""
  echo "LABEL_RE = re.compile(r'^([_A-Za-z][_A-Za-z0-9]*):\\s*(#.*)?\$')"
  echo ""
  echo "candidate_symbols = []"
  echo "for line in asm_lines:"
  echo "    m = LABEL_RE.match(line)"
  echo "    if m:"
  echo "        sym = m.group(1)"
  echo "        try:"
  echo "            demangled = subprocess.check_output(['c++filt', sym], text=True).strip()"
  echo "            if 'fk' in demangled.lower() or 'forwardkin' in demangled.lower():"
  echo "                candidate_symbols.append((sym, demangled))"
  echo "        except subprocess.CalledProcessError:"
  echo "            pass"
  echo ""
  echo "if not candidate_symbols:"
  echo "    print('WARNING: No FK-related symbols found in assembly.', file=sys.stderr)"
  echo "    print('  Falling back to first 500 lines of assembly.', file=sys.stderr)"
  echo ""
  echo "candidate_symbols.sort(key=lambda x: 0 if 'fk_call_for_perf' not in x[1] else 1)"
  echo ""
  echo "hot_lines = []"
  echo "if candidate_symbols:"
  echo "    target_sym = candidate_symbols[0][0]"
  echo "    print(f'Targeting symbol: {target_sym}')"
  echo "    print(f'  ({candidate_symbols[0][1]})')"
  echo "    in_func = False"
  echo "    for line in asm_lines:"
  echo "        if line.strip().startswith(target_sym + ':'):"
  echo "            in_func = True"
  echo "        if in_func:"
  echo "            hot_lines.append(line)"
  echo "            if '.cfi_endproc' in line or line.strip().startswith('.size'):"
  echo "                break"
  echo ""
  echo "if not hot_lines:"
  echo "    hot_lines = asm_lines[:500]"
  echo ""
  echo "with open(out_path, 'w') as f:"
  echo "    f.writelines(hot_lines[:500])"
  echo ""
  echo "print(f'Extracted {len(hot_lines[:500])} lines to {out_path}')"
} > "$EXTRACTOR"

python3 "$EXTRACTOR"

echo ""
echo "=== Step 3: llvm-mca analysis of FK hot loop ==="
echo "--- lgn FK (4x4 matmul path) ---" | tee "$RESULTS/llvm_mca_lgn.txt"
llvm-mca -march=x86-64 -mcpu=native --timeline --bottleneck-analysis --resource-pressure \
  "$BUILD_DIR/fk_hot_loop.s" \
  >> "$RESULTS/llvm_mca_lgn.txt" 2>&1 \
  || echo "llvm-mca completed with warnings (normal for full function bodies)"

echo ""
echo "=== Step 4: perf stat — live hardware counters ==="

if [ ! -f "$BENCH_BIN" ]; then
  echo "ERROR: $BENCH_BIN not found. Run: make -C build_bench bench_v2_lgn_only"
  exit 1
fi

echo "--- perf stat: lgn FK ($BENCH_BIN) ---" | tee "$RESULTS/perf_stat_lgn.txt"

# Try Intel events first; fall back to basic counters.
if ! sudo perf stat \
       -e instructions,cycles,cache-misses,cache-references \
       -e fp_arith_inst_retired.256b_packed_double \
       -e fp_arith_inst_retired.scalar_double \
       -e fp_arith_inst_retired.128b_packed_double \
       --repeat 5 \
       env OMP_NUM_THREADS=1 taskset -c 2 "$BENCH_BIN" \
       2>> "$RESULTS/perf_stat_lgn.txt" \
       > /dev/null; then
  echo "Note: Intel-named PMU events unavailable; using basic counters."
  sudo perf stat \
    -e instructions,cycles,cache-misses,cache-references \
    --repeat 5 \
    env OMP_NUM_THREADS=1 taskset -c 2 "$BENCH_BIN" \
    2>> "$RESULTS/perf_stat_lgn.txt" \
    > /dev/null
fi

echo ""
echo "=== Step 5: Parse results ==="
PARSER="$BUILD_DIR/parse_perf.py"
{
  echo "import re"
  echo ""
  echo "def parse_perf(path):"
  echo "    try:"
  echo "        with open(path) as f:"
  echo "            text = f.read()"
  echo "    except FileNotFoundError:"
  echo "        print(f'  {path} not found'); return"
  echo ""
  echo "    metrics = {}"
  echo "    patterns = ["
  echo "        (r'([\\d,]+)\\s+instructions', 'instructions'),"
  echo "        (r'([\\d,]+)\\s+cycles', 'cycles'),"
  echo "        (r'([\\d,]+)\\s+cache-misses', 'cache_misses'),"
  echo "        (r'([\\d,]+)\\s+cache-references', 'cache_refs'),"
  echo "        (r'([\\d,]+)\\s+fp_arith_inst_retired\\.256b_packed_double', 'avx2_double'),"
  echo "        (r'([\\d,]+)\\s+fp_arith_inst_retired\\.scalar_double', 'scalar_double'),"
  echo "    ]"
  echo "    for line in text.splitlines():"
  echo "        for pat, key in patterns:"
  echo "            m = re.search(pat, line)"
  echo "            if m and key not in metrics:"
  echo "                metrics[key] = int(m.group(1).replace(',',''))"
  echo ""
  echo "    if 'instructions' in metrics and 'cycles' in metrics:"
  echo "        ipc = metrics['instructions'] / metrics['cycles']"
  echo "        print(f'  IPC:              {ipc:.2f}')"
  echo "    if 'cache_refs' in metrics and 'cache_misses' in metrics:"
  echo "        rate = 100 * metrics['cache_misses'] / metrics['cache_refs']"
  echo "        print(f'  Cache miss rate:  {rate:.3f}% of refs')"
  echo "    if 'avx2_double' in metrics and 'scalar_double' in metrics:"
  echo "        total = metrics['avx2_double'] + metrics['scalar_double']"
  echo "        if total > 0:"
  echo "            avx2_pct = 100 * metrics['avx2_double'] / total"
  echo "            print(f'  AVX2 fraction:    {avx2_pct:.1f}% of FP ops')"
  echo ""
  echo "print('lgn FK perf stat:')"
  echo "parse_perf('$RESULTS/perf_stat_lgn.txt')"
} > "$PARSER"

python3 "$PARSER"

echo ""
echo "=== Done ==="
echo "Results in $RESULTS/:"
echo "  llvm_mca_lgn.txt    — static pipeline analysis"
echo "  perf_stat_lgn.txt   — hardware counter measurements"
