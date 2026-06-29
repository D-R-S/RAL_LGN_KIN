humanoid@dt-hm10:~$ cd lgn_hand_ik/
humanoid@dt-hm10:~/lgn_hand_ik$ # ─────────────────────────────────────────────────────────────────────────────
#  PHASE 0 — system prep (run once per session)
# ─────────────────────────────────────────────────────────────────────────────
cd ~/lgn_hand_ik

echo "=== System info ==="
lscpu | grep -E 'Model name|Vendor|^CPU\(s\)|MHz' | head -5
uname -r
gcc --version | head -1

echo ""
echo "=== CPU governor → performance ==="
sudo cpupower frequency-set -g performance 2>&1 | tail -3

echo ""
echo "=== perf paranoid → 1 ==="
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null
cat /proc/sys/kernel/perf_event_paranoid

echo ""
echo "=== Drop caches ==="
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

cat /sys/devices/system/cpu/cpu2/topology/thread_siblings_list
=== System info ===
CPU(s):                                  32
Vendor ID:                               AuthenticAMD
Model name:                              AMD Ryzen 9 9950X 16-Core Processor
CPU(s) scaling MHz:                      44%
CPU max MHz:                             5756.4521
6.17.0-23-generic
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0

=== CPU governor → performance ===
[sudo] password for humanoid: 
Setting cpu: 29
Setting cpu: 30
Setting cpu: 31

=== perf paranoid → 1 ===
1

=== Drop caches ===

=== Clean build dirs ===

=== CPU topology — verify core 2 ===
CPU CORE SOCKET    MAXMHZ
  0    0      0 5756.4521
  1    1      0 5756.4521
  2    2      0 5756.4521
  3    3      0 5756.4521
  4    4      0 5756.4521
  5    5      0 5756.4521
  6    6      0 5756.4521
  7    7      0 5756.4521
  8    8      0 5756.4521
  9    9      0 5756.4521
 10   10      0 5756.4521
 11   11      0 5756.4521
 12   12      0 5756.4521
 13   13      0 5756.4521
 14   14      0 5756.4521
 15   15      0 5756.4521
 16    0      0 5756.4521
 17    1      0 5756.4521
 18    2      0 5756.4521
core 2 SMT siblings:
2,18
humanoid@dt-hm10:~/lgn_hand_ik$ # ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — correctness gates (Debug build with sanitizer)
# ─────────────────────────────────────────────────────────────────────────────
cd ~/lgn_hand_ik
mkdir -p build_test && cd build_test

echo "=== cmake (Debug, tests only) ==="
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_ROS=OFF \
  -DBUILD_DEMO=OFF \
  -DBUILD_TESTING=ON \
  -DBUILD_BENCHMARKS=OFF 2>&1 | tail -25

echo ""
echo "=== make ==="
make -j$(nproc) test_correctness test_roundtrip 2>&1 | tail -15

echo ""
echo "=== test_correctness ==="
./test_correctness 2>&1 | tail -10

echo ""
cd ..t_roundtrip 2>&1 | tail -15
=== cmake (Debug, tests only) ===
-- The CXX compiler identification is GNU 13.3.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found OpenMP_CXX: -fopenmp (found version "4.5") 
-- Found OpenMP: TRUE (found version "4.5")  
-- OpenMP found — parallel finger IK enabled (HandIKSolver only)
-- Found GTest: /usr/lib/x86_64-linux-gnu/cmake/GTest/GTestConfig.cmake (found version "1.14.0")  
-- 
-- lgn_hand_ik 0.1.0 — build summary
--   Build type:   Debug
--   ROS2 nodes:   OFF
--   Demo (GL):    OFF
--   Tests:        ON
--   Benchmarks:   OFF
--   OpenMP:       TRUE
-- 
-- Configuring done (0.3s)
-- Generating done (0.0s)
-- Build files have been written to: /home/humanoid/lgn_hand_ik/build_test

=== make ===
[ 50%] Building CXX object CMakeFiles/test_correctness.dir/test/test_correctness.cpp.o
[100%] Linking CXX executable test_correctness
[100%] Built target test_correctness
[ 50%] Building CXX object CMakeFiles/test_roundtrip.dir/test/test_roundtrip.cpp.o
[100%] Linking CXX executable test_roundtrip
[100%] Built target test_roundtrip

=== test_correctness ===
[lgn] Loaded: 4 links, 1 DOFs, 1 tip(s)
[       OK ] Exclusions.ExplicitExclusionWorks (0 ms)
[ RUN      ] Exclusions.ExclusionPreventsContactDetection
[lgn] Loaded: 4 links, 1 DOFs, 1 tip(s)
[       OK ] Exclusions.ExclusionPreventsContactDetection (0 ms)
[----------] 4 tests from Exclusions (0 ms total)

[----------] Global test environment tear-down
[==========] 40 tests from 14 test suites ran. (32 ms total)
[  PASSED  ] 40 tests.

=== test_roundtrip ===
[ RUN      ] RoundTrip.ExternalFiles
/home/humanoid/lgn_hand_ik/test/test_roundtrip.cpp:240: Skipped
Set LGN_TEST_URDF and LGN_TEST_MJCF env vars to test external files.
  export LGN_TEST_URDF=/path/to/robot.urdf
  export LGN_TEST_MJCF=/path/to/robot.mjcf
  ./test_roundtrip

[  SKIPPED ] RoundTrip.ExternalFiles (0 ms)
[----------] 3 tests from RoundTrip (27 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (27 ms total)
[  PASSED  ] 2 tests.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] RoundTrip.ExternalFiles
humanoid@dt-hm10:~/lgn_hand_ik$ # ──────────────────────────────────────────────humanoid@dt-hm10:~/lgn_hand_ik$ # ─────────────────────────────────────────────────────────────────────────────e, dynamics restitution gate)
#  PHASE 2 — demo build (Release, dynamics restitution gate)───────────────────
# ─────────────────────────────────────────────────────────────────────────────
cd ~/lgn_hand_ikemo && cd build_demo
mkdir -p build_demo && cd build_demo
echo "=== cmake (Release, demo only) ==="
echo "=== cmake (Release, demo only) ==="
cmake .. \BUILD_TYPE=Release \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_ROS=OFF \
  -DBUILD_DEMO=ON \FF \
  -DBUILD_TESTING=OFF \F \
  -DBUILD_BENCHMARKS=OFF \id/imgui 2>&1 | tail -15
  -DIMGUI_DIR=/home/humanoid/imgui 2>&1 | tail -15
echo ""
echo ""== make ==="
echo "=== make ==="_demo 2>&1 | tail -10
make -j$(nproc) lgn_demo 2>&1 | tail -10
echo ""
echo ""== demo binary ==="
echo "=== demo binary ==="
ls -lh lgn_demo
cd ..
cd ..
=== cmake (Release, demo only) ===
-- Found OpenMP: TRUE (found version "4.5")  
-- OpenMP found — parallel finger IK enabled (HandIKSolver only)
-- Found OpenGL: /usr/lib/x86_64-linux-gnu/libOpenGL.so   
-- 
-- lgn_hand_ik 0.1.0 — build summary
--   Build type:   Release
--   ROS2 nodes:   OFF
--   Demo (GL):    ON
--   Tests:        OFF
--   Benchmarks:   OFF
--   OpenMP:       TRUE
-- 
-- Configuring done (0.2s)
-- Generating done (0.0s)
-- Build files have been written to: /home/humanoid/lgn_hand_ik/build_demo

=== make ===
[ 33%] Building CXX object CMakeFiles/imgui.dir/home/humanoid/imgui/imgui_tables.cpp.o
[ 33%] Building CXX object CMakeFiles/imgui.dir/home/humanoid/imgui/imgui_draw.cpp.o
[ 44%] Building CXX object CMakeFiles/imgui.dir/home/humanoid/imgui/imgui_widgets.cpp.o
[ 55%] Building CXX object CMakeFiles/imgui.dir/home/humanoid/imgui/backends/imgui_impl_opengl3.cpp.o
[ 66%] Building CXX object CMakeFiles/imgui.dir/home/humanoid/imgui/backends/imgui_impl_glfw.cpp.o
[ 77%] Linking CXX static library libimgui.a
[ 77%] Built target imgui
[ 88%] Building CXX object CMakeFiles/lgn_demo.dir/demo/main.cpp.o
[100%] Linking CXX executable lgn_demo
[100%] Built target lgn_demo

=== demo binary ===
-rwxrwxr-x 1 humanoid humanoid 1.3M May 11 00:07 lgn_demo
humanoid@dt-hm10:~/lgn_hand_ik$ # ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — benchmark build (Release, all three v2 binaries)
# ─────────────────────────────────────────────────────────────────────────────
cd ~/lgn_hand_ik
mkdir -p build_bench && cd build_bench

echo "=== cmake (Release, benchmarks) ==="
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_ROS=OFF \
  -DBUILD_DEMO=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_BENCHMARKS=ON 2>&1 | tail -25

echo ""
echo "=== make all three v2 binaries ==="
make -j$(nproc) bench_v2_lgn_kdl bench_v2_pin bench_v2_lgn_only 2>&1 | tail -10

echo ""
echo "=== bench binaries ==="
ls -lh bench_v2_*

cd ..
=== cmake (Release, benchmarks) ===
-- NumPy include dir: /home/humanoid/.local/lib/python3.12/site-packages/numpy/_core/include
-- Found Boost: /usr/include (found version "1.83.0") found components: python312 
-- Default C++ standard: 201703
-- Minimal C++ standard upgraded to 11
-- C++ standard sufficient: Minimal required 11, currently defined: 17
-- hpp-fcl not FOUND.
-- Minimal C++ standard upgraded to 14
-- C++ standard sufficient: Minimal required 14, currently defined: 17
-- C++ standard sufficient: Minimal required 14, currently defined: 17
-- ROS Jazzy Pinocchio detected: /opt/ros/jazzy/lib/x86_64-linux-gnu/libpinocchio_default.so.3.9.0
-- KDL found — KDL legacy bench ENABLED
-- 
-- lgn_hand_ik 0.1.0 — build summary
--   Build type:   Release
--   ROS2 nodes:   OFF
--   Demo (GL):    OFF
--   Tests:        OFF
--   Benchmarks:   ON
--   OpenMP:       TRUE
--   KDL:          1
--   Pinocchio:    TRUE (ROS Jazzy isolated)
-- 
-- Configuring done (0.7s)
-- Generating done (0.0s)
-- Build files have been written to: /home/humanoid/lgn_hand_ik/build_bench

=== make all three v2 binaries ===
                 from /home/humanoid/lgn_hand_ik/benchmarks/bench_v2.cpp:101:
/opt/ros/jazzy/include/pinocchio/algorithm/aba.hxx: In function ‘const typename pinocchio::DataTpl<_Scalar, _Options, JointCollectionTpl>::TangentVectorType& pinocchio::aba(const ModelTpl<_Scalar, _Options, JointCollectionTpl>&, DataTpl<Scalar, Options, JointCollectionTpl>&, const Eigen::MatrixBase<Matrix3Like>&, const Eigen::MatrixBase<MatRet>&, const Eigen::MatrixBase<TangentVectorType>&, Convention) [with Scalar = double; int Options = 0; JointCollectionTpl = JointCollectionDefaultTpl; ConfigVectorType = Eigen::Matrix<double, -1, 1>; TangentVectorType1 = Eigen::Matrix<double, -1, 1>; TangentVectorType2 = Eigen::Matrix<double, -1, 1>]’:
/opt/ros/jazzy/include/pinocchio/algorithm/aba.hxx:948:3: warning: control reaches end of non-void function [-Wreturn-type]
  948 |   }
      |   ^
[100%] Linking CXX executable bench_v2_pin
[100%] Built target bench_v2_pin
[ 50%] Building CXX object CMakeFiles/bench_v2_lgn_only.dir/benchmarks/bench_v2.cpp.o
[100%] Linking CXX executable bench_v2_lgn_only
[100%] Built target bench_v2_lgn_only

=== bench binaries ===
-rwxrwxr-x 1 humanoid humanoid 178K May 11 00:08 bench_v2_lgn_kdl
-rwxrwxr-x 1 humanoid humanoid 177K May 11 00:08 bench_v2_lgn_only
-rwxrwxr-x 1 humanoid humanoid 1.3M May 11 00:08 bench_v2_pin
humanoid@dt-hm10:~/lgn_hand_ik$ # ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4 — full bench runs, three isolated processes on pinned core 2
# ─────────────────────────────────────────────────────────────────────────────
cd ~/lgn_hand_ik
mkdir -p results_v2

sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo "=== bench_v2_lgn_kdl ==="
date
OMP_NUM_THREADS=1 taskset -c 2 ./build_bench/bench_v2_lgn_kdl \
    > results_v2/results_lgn_kdl.csv 2> results_v2/log_lgn_kdl.txt
date

sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo ""
echo "=== bench_v2_pin (FK + IK + dynamics + H₀₄ gate) ==="
date
OMP_NUM_THREADS=1 taskset -c 2 ./build_bench/bench_v2_pin \
    > results_v2/results_pin.csv 2> results_v2/log_pin.txt
date

sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo ""
wc -l results_v2/*.csvlts_v2_raw.csv\v > results_v2/results_v2_raw.csv
=== bench_v2_lgn_kdl ===
Mon May 11 12:08:46 AM IDT 2026
Mon May 11 12:08:52 AM IDT 2026

=== bench_v2_pin (FK + IK + dynamics + H₀₄ gate) ===
Mon May 11 12:08:52 AM IDT 2026
Mon May 11 12:08:58 AM IDT 2026

=== bench_v2_lgn_only (FK + IK + velocity + dynamics) ===
Mon May 11 12:08:58 AM IDT 2026
Mon May 11 12:09:03 AM IDT 2026

=== H₀₄ correctness gate ===
[crosscheck] mass_matrix n=2 max_abs_err=6.071532e-18 PASS (gate 1.000000e-10)
[crosscheck] mass_matrix n=4 max_abs_err=1.110223e-16 PASS (gate 1.000000e-10)
[crosscheck] mass_matrix n=8 max_abs_err=8.881784e-16 PASS (gate 1.000000e-10)
[crosscheck] mass_matrix n=16 max_abs_err=1.065814e-14 PASS (gate 1.000000e-10)
[crosscheck] mass_matrix n=32 max_abs_err=5.684342e-14 PASS (gate 1.000000e-10)

=== concatenate raw CSV ===

  1927505 results_v2/results_lgn_kdl.csv
  2759732 results_v2/results_lgn_only.csv
  2039410 results_v2/results_pin.csv
  6726645 results_v2/results_v2_raw.csv
 13453292 total
humanoid@dt-hm10:~/lgn_hand_ik$ # ─────────────────────────────────────────────────────────────────────────────
#  PHASE 5 — perf telemetry (separate from bench; perf interferes with timing)
# ─────────────────────────────────────────────────────────────────────────────
cd ~/lgn_hand_ik
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

bash benchmarks/perf_avx2.sh 2>&1 | tee results_v2/perf_avx2.log
Note: CPU vendor is 'AuthenticAMD'.
  Intel-named fp_arith_inst_retired.* events may not exist on AMD;
  the script automatically falls back to basic counters in that case.

=== Step 1: Compile FK hot loop to assembly ===
Eigen include: -I/usr/include/eigen3
Assembly written: build_bench_asm/bench_lgn_fk.s (clang++, intel syntax)

=== Step 2: Extract FK hot loop ===
Targeting symbol: _ZN3lgn13KinematicTree2fkERKN5Eigen6MatrixIdLin1ELi1ELi0ELin1ELi1EEE
  (lgn::KinematicTree::fk(Eigen::Matrix<double, -1, 1, 0, -1, 1> const&))
Extracted 398 lines to build_bench_asm/fk_hot_loop.s

=== Step 3: llvm-mca analysis of FK hot loop ===
--- lgn FK (4x4 matmul path) ---

=== Step 4: perf stat — live hardware counters ===
--- perf stat: lgn FK (build_bench/bench_v2_lgn_only) ---
Note: Intel-named PMU events unavailable; using basic counters.

=== Step 5: Parse results ===
lgn FK perf stat:
  IPC:              3.13
  Cache miss rate:  3.900% of refs

=== Done ===
Results in results_v2/:
  llvm_mca_lgn.txt    — static pipeline analysis
  perf_stat_lgn.txt   — hardware counter measurements
humanoid@dt-hm10:~/lgn_hand_ik$ # ─────────────────────────────────────────────────────────────────────────────
#  PHASE 6 — aggregate + summary
# ─────────────────────────────────────────────────────────────────────────────
cd ~/lgn_hand_ik

echo "===================================================================="
echo "  CORRECTNESS GATES"
echo "===================================================================="
echo "Mass matrix H₀₄:"
grep crosscheck results_v2/log_pin.txt

echo ""
echo "===================================================================="
echo "  HEADLINE NUMBERS — block-mean per (solver, benchmark, n)"
echo "===================================================================="
for bench in fk ik_posonly vel_only dyn_M dyn_C dyn_RNEA_eq dyn_ABA_eq; do
    echo ""
    echo "--- $bench ---"
    awk -F, -v b="$bench" 'NR>1 && $2==b {s[$1","$3]+=$4; n[$1","$3]++} \
             END {for (k in s) printf "%-20s %12.1f ns  (samples=%d)\n", k, s[k]/n[k], n[k]}' \
        results_v2/results_v2_raw.csv | sort -t, -k1,1 -k2,2n
ls -lh results_v2/========================================================"f ns)
====================================================================
  CORRECTNESS GATES
====================================================================
Mass matrix H₀₄:
[crosscheck] mass_matrix n=2 max_abs_err=6.071532e-18 PASS (gate 1.000000e-10)
[crosscheck] mass_matrix n=4 max_abs_err=1.110223e-16 PASS (gate 1.000000e-10)
[crosscheck] mass_matrix n=8 max_abs_err=8.881784e-16 PASS (gate 1.000000e-10)
[crosscheck] mass_matrix n=16 max_abs_err=1.065814e-14 PASS (gate 1.000000e-10)
[crosscheck] mass_matrix n=32 max_abs_err=5.684342e-14 PASS (gate 1.000000e-10)

====================================================================
  HEADLINE NUMBERS — block-mean per (solver, benchmark, n)
====================================================================

--- fk ---
kdl,2                       104.2 ns  (samples=200000)
kdl,4                       182.3 ns  (samples=200000)
kdl,8                       334.4 ns  (samples=200000)
kdl,16                      651.2 ns  (samples=100000)
kdl,32                     1280.8 ns  (samples=100000)
kdl,64                     2547.4 ns  (samples=50000)
kdl,128                    5093.1 ns  (samples=20000)
kdl,256                   10137.8 ns  (samples=10000)
lgn,2                        83.7 ns  (samples=600000)
lgn,4                       145.2 ns  (samples=600000)
lgn,8                       259.8 ns  (samples=600000)
lgn,16                      491.1 ns  (samples=300000)
lgn,32                      965.8 ns  (samples=300000)
lgn,64                     1933.6 ns  (samples=150000)
lgn,128                    3831.6 ns  (samples=60000)
lgn,256                    7618.5 ns  (samples=30000)
pinocchio,2                  49.4 ns  (samples=200000)
pinocchio,4                 107.7 ns  (samples=200000)
pinocchio,8                 221.1 ns  (samples=200000)
pinocchio,16                468.1 ns  (samples=100000)
pinocchio,32                976.8 ns  (samples=100000)
pinocchio,64               2004.1 ns  (samples=50000)
pinocchio,128              4053.2 ns  (samples=20000)
pinocchio,256              8136.8 ns  (samples=10000)

--- ik_posonly ---
kdl,2                      2191.1 ns  (samples=10000)
kdl,4                      9122.2 ns  (samples=10000)
kdl,8                     12534.4 ns  (samples=3000)
kdl,16                    25041.2 ns  (samples=3000)
kdl,32                    68584.7 ns  (samples=1000)
kdl,64                   220647.7 ns  (samples=500)
kdl,128                  767241.0 ns  (samples=200)
kdl,256                 2835685.1 ns  (samples=80)
lgn,2                       786.1 ns  (samples=30000)
lgn,4                      4175.4 ns  (samples=30000)
lgn,8                      5852.5 ns  (samples=9000)
lgn,16                     8423.0 ns  (samples=9000)
lgn,32                    13986.8 ns  (samples=3000)
lgn,64                    27545.5 ns  (samples=1500)
lgn,128                   50988.1 ns  (samples=600)
lgn,256                   96724.1 ns  (samples=240)
pinocchio,2                  44.5 ns  (samples=10000)
pinocchio,4                2025.7 ns  (samples=10000)
pinocchio,8                8450.2 ns  (samples=3000)
pinocchio,16              11524.3 ns  (samples=3000)
pinocchio,32              20082.3 ns  (samples=1000)
pinocchio,64              39619.0 ns  (samples=500)
pinocchio,128             72182.7 ns  (samples=200)
pinocchio,256            131302.0 ns  (samples=80)

--- vel_only ---
lgn,2                        80.1 ns  (samples=200000)
lgn,4                       117.4 ns  (samples=200000)
lgn,8                       193.5 ns  (samples=200000)
lgn,16                      340.6 ns  (samples=100000)
lgn,32                      639.6 ns  (samples=100000)
lgn,64                     1240.1 ns  (samples=50000)
lgn,128                    2436.8 ns  (samples=20000)
scalar,2                     41.8 ns  (samples=200000)
scalar,4                     53.2 ns  (samples=200000)
scalar,8                     78.3 ns  (samples=200000)
scalar,16                   118.8 ns  (samples=100000)
scalar,32                   206.7 ns  (samples=100000)
scalar,64                   471.0 ns  (samples=50000)
scalar,128                  993.3 ns  (samples=20000)

--- dyn_M ---
lgn,2                       174.8 ns  (samples=30000)
lgn,4                       372.7 ns  (samples=30000)
lgn,8                      1233.4 ns  (samples=30000)
lgn,16                     4745.2 ns  (samples=9000)
lgn,32                    25017.5 ns  (samples=9000)
lgn,64                   170213.9 ns  (samples=3000)
lgn,128                 1477850.6 ns  (samples=900)
pinocchio,2                  57.2 ns  (samples=10000)
pinocchio,4                 163.6 ns  (samples=10000)
pinocchio,8                 416.0 ns  (samples=10000)
pinocchio,16               1175.3 ns  (samples=3000)
pinocchio,32               3597.5 ns  (samples=3000)
pinocchio,64              11951.7 ns  (samples=1000)
pinocchio,128             43571.4 ns  (samples=300)

--- dyn_C ---
lgn,2                       262.1 ns  (samples=30000)
lgn,4                       471.6 ns  (samples=30000)
lgn,8                      1357.6 ns  (samples=30000)
lgn,16                     3209.7 ns  (samples=9000)
lgn,32                     8021.0 ns  (samples=9000)
lgn,64                    21394.1 ns  (samples=3000)
lgn,128                  254377.0 ns  (samples=900)
pinocchio,2                  93.5 ns  (samples=10000)
pinocchio,4                 259.7 ns  (samples=10000)
pinocchio,8                 589.5 ns  (samples=10000)
pinocchio,16               1249.7 ns  (samples=3000)
pinocchio,32               2567.4 ns  (samples=3000)
pinocchio,64               5226.5 ns  (samples=1000)
pinocchio,128             10480.7 ns  (samples=300)

--- dyn_RNEA_eq ---
lgn,2                       583.0 ns  (samples=15000)
lgn,4                      1108.3 ns  (samples=15000)
lgn,8                      3119.6 ns  (samples=15000)
lgn,16                     9090.3 ns  (samples=4500)
lgn,32                    35574.9 ns  (samples=4500)
lgn,64                   198497.0 ns  (samples=1500)
lgn,128                 1758089.4 ns  (samples=450)
pinocchio,2                 100.7 ns  (samples=5000)
pinocchio,4                 280.9 ns  (samples=5000)
pinocchio,8                 636.6 ns  (samples=5000)
pinocchio,16               1353.9 ns  (samples=1500)
pinocchio,32               2758.8 ns  (samples=1500)
pinocchio,64               5606.4 ns  (samples=500)
pinocchio,128             11213.9 ns  (samples=150)

--- dyn_ABA_eq ---
lgn,2                       643.1 ns  (samples=15000)
lgn,4                      1208.5 ns  (samples=15000)
lgn,8                      3388.2 ns  (samples=15000)
lgn,16                     9784.3 ns  (samples=4500)
lgn,32                    37500.7 ns  (samples=4500)
lgn,64                   206593.4 ns  (samples=1500)
lgn,128                 1772726.8 ns  (samples=450)
pinocchio,2                 158.6 ns  (samples=5000)
pinocchio,4                 527.2 ns  (samples=5000)
pinocchio,8                1254.8 ns  (samples=5000)
pinocchio,16               2934.3 ns  (samples=1500)
pinocchio,32               5708.3 ns  (samples=1500)
pinocchio,64              11568.6 ns  (samples=500)
pinocchio,128             23368.6 ns  (samples=150)

====================================================================
  KINEMATICS RATIOS
====================================================================

--- FK lgn/KDL and lgn/Pinocchio ---
  n=2     lgn/KDL=1.25x  lgn/Pin=0.59x
  n=4     lgn/KDL=1.26x  lgn/Pin=0.74x
  n=8     lgn/KDL=1.29x  lgn/Pin=0.85x
  n=16    lgn/KDL=1.33x  lgn/Pin=0.95x
  n=32    lgn/KDL=1.33x  lgn/Pin=1.01x
  n=64    lgn/KDL=1.32x  lgn/Pin=1.04x
  n=128   lgn/KDL=1.33x  lgn/Pin=1.06x
  n=256   lgn/KDL=1.33x  lgn/Pin=1.07x

--- IK lgn/Pinocchio ---
  n=2    lgn/Pin=0.06x
  n=4    lgn/Pin=0.49x
  n=8    lgn/Pin=1.44x
  n=16   lgn/Pin=1.37x
  n=32   lgn/Pin=1.44x
  n=64   lgn/Pin=1.44x
  n=128  lgn/Pin=1.42x
  n=256  lgn/Pin=1.36x

--- Velocity propagation lgn/scalar (FLOP prediction: 4.6x) ---
  n=2    lgn/scalar=1.92x
  n=4    lgn/scalar=2.21x
  n=8    lgn/scalar=2.47x
  n=16   lgn/scalar=2.87x
  n=32   lgn/scalar=3.09x
  n=64   lgn/scalar=2.63x
  n=128  lgn/scalar=2.45x

====================================================================
  DYNAMICS RATIOS — lgn vs Pinocchio
====================================================================

--- dyn_M (Pinocchio / lgn = how much faster Pinocchio is; >1 = lgn wins) ---
  n=2    pin/lgn=0.33x  (lgn=      175 ns, pin=       57 ns)
  n=4    pin/lgn=0.44x  (lgn=      373 ns, pin=      164 ns)
  n=8    pin/lgn=0.34x  (lgn=     1233 ns, pin=      416 ns)
  n=16   pin/lgn=0.25x  (lgn=     4745 ns, pin=     1175 ns)
  n=32   pin/lgn=0.14x  (lgn=    25017 ns, pin=     3597 ns)
  n=64   pin/lgn=0.07x  (lgn=   170214 ns, pin=    11952 ns)
  n=128  pin/lgn=0.03x  (lgn=  1477851 ns, pin=    43571 ns)

--- dyn_C (Pinocchio / lgn = how much faster Pinocchio is; >1 = lgn wins) ---
  n=2    pin/lgn=0.36x  (lgn=      262 ns, pin=       93 ns)
  n=4    pin/lgn=0.55x  (lgn=      472 ns, pin=      260 ns)
  n=8    pin/lgn=0.43x  (lgn=     1358 ns, pin=      589 ns)
  n=16   pin/lgn=0.39x  (lgn=     3210 ns, pin=     1250 ns)
  n=32   pin/lgn=0.32x  (lgn=     8021 ns, pin=     2567 ns)
  n=64   pin/lgn=0.24x  (lgn=    21394 ns, pin=     5227 ns)
  n=128  pin/lgn=0.04x  (lgn=   254377 ns, pin=    10481 ns)

--- dyn_RNEA_eq (Pinocchio / lgn = how much faster Pinocchio is; >1 = lgn wins) ---
  n=2    pin/lgn=0.17x  (lgn=      583 ns, pin=      101 ns)
  n=4    pin/lgn=0.25x  (lgn=     1108 ns, pin=      281 ns)
  n=8    pin/lgn=0.20x  (lgn=     3120 ns, pin=      637 ns)
  n=16   pin/lgn=0.15x  (lgn=     9090 ns, pin=     1354 ns)
  n=32   pin/lgn=0.08x  (lgn=    35575 ns, pin=     2759 ns)
  n=64   pin/lgn=0.03x  (lgn=   198497 ns, pin=     5606 ns)
  n=128  pin/lgn=0.01x  (lgn=  1758089 ns, pin=    11214 ns)

--- dyn_ABA_eq (Pinocchio / lgn = how much faster Pinocchio is; >1 = lgn wins) ---
  n=2    pin/lgn=0.25x  (lgn=      643 ns, pin=      159 ns)
  n=4    pin/lgn=0.44x  (lgn=     1208 ns, pin=      527 ns)
  n=8    pin/lgn=0.37x  (lgn=     3388 ns, pin=     1255 ns)
  n=16   pin/lgn=0.30x  (lgn=     9784 ns, pin=     2934 ns)
  n=32   pin/lgn=0.15x  (lgn=    37501 ns, pin=     5708 ns)
  n=64   pin/lgn=0.06x  (lgn=   206593 ns, pin=    11569 ns)
  n=128  pin/lgn=0.01x  (lgn=  1772727 ns, pin=    23369 ns)

====================================================================
  PERF TELEMETRY
====================================================================
event syntax error: 'fp_arith_inst_retired.256b_packed_double'
Unable to find event on a PMU of 'fp_arith_inst_retired.256b_packed_double'
    93,431,833,573      instructions                     #    3.13  insn per cycle              ( +-  0.00% )
    29,875,653,103      cycles                                                                  ( +-  0.06% )
        92,682,497      cache-misses                     #    3.90% of all cache refs           ( +-  0.32% )
     2,376,260,230      cache-references                                                        ( +-  0.24% )

====================================================================
  Files in results_v2/
====================================================================
total 239M
-rw-rw-r-- 1 humanoid humanoid  83K May 11 00:09 llvm_mca_lgn.txt
-rw-rw-r-- 1 humanoid humanoid  721 May 11 00:08 log_lgn_kdl.txt
-rw-rw-r-- 1 humanoid humanoid  621 May 11 00:09 log_lgn_only.txt
-rw-rw-r-- 1 humanoid humanoid 1.5K May 11 00:08 log_pin.txt
-rw-rw-r-- 1 humanoid humanoid 1.1K May 11 00:09 perf_avx2.log
-rw-rw-r-- 1 humanoid humanoid 4.1K May 11 00:09 perf_stat_lgn.txt
-rw-rw-r-- 1 humanoid humanoid  30M May 11 00:08 results_lgn_kdl.csv
-rw-rw-r-- 1 humanoid humanoid  53M May 11 00:09 results_lgn_only.csv
-rw-rw-r-- 1 humanoid humanoid  38M May 11 00:08 results_pin.csv
-rw-rw-r-- 1 humanoid humanoid 120M May 11 00:09 results_v2_raw.csv

