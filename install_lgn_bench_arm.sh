#!/bin/bash
# =============================================================================
#  install_lgn_bench_arm.sh
#  JetPack 5.x / 6.x on Ubuntu 22.04 (Jammy), aarch64.
#
#  Targets Jetson Orin NX / AGX Orin / Xavier NX or any other
#  ARM64 box running Ubuntu 22.04.
#
#  Run as the normal user (NOT root). Re-runnable.
# =============================================================================
set -euo pipefail

log()  { echo ""; echo "=== $* ==="; }
warn() { echo "!! $*"; }
die()  { echo "FATAL: $*"; exit 1; }

if [ "$EUID" -eq 0 ]; then
    die "Run as a normal user, not root. The script will sudo when needed."
fi

. /etc/os-release
ARCH="$(uname -m)"
echo "Detected: $PRETTY_NAME on $ARCH"
if [ "${VERSION_ID:-}" != "22.04" ]; then
    warn "This script targets Ubuntu 22.04 (Jammy). Detected $VERSION_ID."
    warn "Continuing, but ROS Humble apt source assumes Jammy."
fi
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    warn "Not on ARM64 ($ARCH detected). The script will still work but the"
    warn "Jetson-specific power-mode commands at the end will be skipped."
fi

IS_JETSON=0
if [ -f /etc/nv_tegra_release ] || [ -d /sys/firmware/devicetree/base/nvidia,boardids ]; then
    IS_JETSON=1
    echo "Jetson platform detected."
fi

# -----------------------------------------------------------------------------
# 1. Base system
# -----------------------------------------------------------------------------
log "apt update"
sudo apt update

log "Build essentials & compilers"
sudo apt install -y \
    build-essential cmake git pkg-config ninja-build ccache \
    gcc-11 g++-11 clang \
    software-properties-common curl ca-certificates gnupg lsb-release \
    python3 python3-dev

log "Math, XML, KDL, urdfdom"
sudo apt install -y \
    libeigen3-dev libboost-all-dev \
    libtinyxml2-dev \
    liborocos-kdl-dev \
    liburdfdom-dev liburdfdom-headers-dev

log "GTest / GMock"
sudo apt install -y libgtest-dev libgmock-dev

log "Python stack for compute_stats_v2.py (apt route, sanest on 22.04)"
sudo apt install -y \
    python3-numpy python3-scipy python3-pandas python3-matplotlib

log "Perf / telemetry (best effort - aarch64 perf events differ from x86)"
sudo apt install -y linux-tools-common cpufrequtils || \
    warn "cpufrequtils unavailable; sysfs governor fallback will be used"
# On Jetson the kernel is custom (NVIDIA); generic linux-tools-$(uname -r)
# is usually NOT packaged. perf binary may still exist via linux-tools-common.

log "Demo GL stack (skip if Orin headless — but installing is harmless)"
sudo apt install -y libgl1-mesa-dev libglfw3-dev libglew-dev || \
    warn "GL stack not fully installed; demo phase may fail (not needed for IK paper)"

# -----------------------------------------------------------------------------
# 2. ROS Humble repo (for ros-humble-pinocchio)
#    Humble's Pinocchio is 3.2-ish, which is the same API as 3.9.
# -----------------------------------------------------------------------------
log "Adding ROS 2 Humble apt repository"
sudo add-apt-repository -y universe
sudo install -d -m 0755 /usr/share/keyrings
if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
    sudo curl -sSL \
        https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu jammy main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
sudo apt update

log "Installing ros-humble-pinocchio"
sudo apt install -y ros-humble-pinocchio

log "Verifying Pinocchio install"
PIN_LIB_DIR=/opt/ros/humble/lib/aarch64-linux-gnu
# Some ROS arm64 builds put libs under /opt/ros/humble/lib (no triplet)
ALT_LIB_DIR=/opt/ros/humble/lib
PIN_FOUND=""
for d in "$PIN_LIB_DIR" "$ALT_LIB_DIR"; do
    if ls "$d/libpinocchio_default.so."* >/dev/null 2>&1; then
        PIN_FOUND="$d"
        break
    fi
done
if [ -z "$PIN_FOUND" ]; then
    die "Pinocchio libraries not found under /opt/ros/humble/lib*"
fi
echo "Pinocchio libraries at: $PIN_FOUND"
ls "$PIN_FOUND/libpinocchio_default.so."*
ls "$PIN_FOUND/libpinocchio_parsers.so."* 2>/dev/null || \
    warn "libpinocchio_parsers not found - URDF loading via Pinocchio may fail"

# -----------------------------------------------------------------------------
# 3. ImGui (only if you want the demo on ARM; not needed for IK paper)
# -----------------------------------------------------------------------------
log "ImGui source tree at \$HOME/imgui (optional for IK paper)"
if [ ! -d "$HOME/imgui/.git" ]; then
    rm -rf "$HOME/imgui"
    git clone --depth=1 https://github.com/ocornut/imgui "$HOME/imgui"
fi

# -----------------------------------------------------------------------------
# 4. Bench-prep helper (governor + caches + Jetson power mode)
# -----------------------------------------------------------------------------
log "Installing setup_bench_env_arm.sh helper"
cat > "$HOME/setup_bench_env_arm.sh" <<'EOF'
#!/bin/bash
# Run as: sudo ~/setup_bench_env_arm.sh

set_governor () {
    if command -v cpupower >/dev/null 2>&1; then
        cpupower frequency-set -g performance 2>&1 | tail -3
        return 0
    fi
    cp_bin=$(ls /usr/lib/linux-tools/*/cpupower 2>/dev/null | tail -1)
    if [ -n "$cp_bin" ] && [ -x "$cp_bin" ]; then
        "$cp_bin" frequency-set -g performance 2>&1 | tail -3
        return 0
    fi
    wrote=0
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -w "$f" ] || continue
        echo performance > "$f" && wrote=1
    done
    if [ "$wrote" = "1" ]; then
        echo "governor set via sysfs"
        return 0
    fi
    echo "no cpufreq driver"
    return 1
}

echo "=== CPU governor -> performance ==="
set_governor || true

echo "=== Jetson power mode (if Jetson) ==="
# nvpmodel 0 == MAXN on Orin NX/AGX (all cores, full clocks).
# Pick a different mode if you want a constrained-power data point.
if command -v nvpmodel >/dev/null 2>&1; then
    nvpmodel -m 0 || echo "nvpmodel -m 0 failed"
    nvpmodel -q
fi

echo "=== Lock clocks (Jetson DVFS off) ==="
if command -v jetson_clocks >/dev/null 2>&1; then
    jetson_clocks
    jetson_clocks --show 2>/dev/null | head -10 || true
fi

echo "=== perf paranoid -> 1 ==="
echo 1 > /proc/sys/kernel/perf_event_paranoid 2>/dev/null && \
    cat /proc/sys/kernel/perf_event_paranoid

echo "=== drop caches ==="
sync && echo 3 > /proc/sys/vm/drop_caches
echo done
EOF
chmod +x "$HOME/setup_bench_env_arm.sh"

# -----------------------------------------------------------------------------
# 5. Summary
# -----------------------------------------------------------------------------
log "Versions"
g++ --version | head -1
cmake --version | head -1
pkg-config --modversion eigen3 || true

log "DONE"
cat <<EOM

Next steps:
  1. Put repo at:        \$HOME/lgn_hand_ik
  2. Pin Jetson clocks:  sudo ~/setup_bench_env_arm.sh
  3. Run bench:          ./run_lgn_ik_focused.sh

CMake notes for ARM:
  Your CMakeLists.txt almost certainly globs Pinocchio at:
      /opt/ros/jazzy/lib/x86_64-linux-gnu/libpinocchio_*.so*
  On Jetson (aarch64) + Humble it lives at:
      $PIN_FOUND/libpinocchio_*.so*
  You must either:
    (a) edit the CMake to accept /opt/ros/{jazzy,humble}/lib/* paths, OR
    (b) symlink the libs into a path your CMake already accepts, OR
    (c) replace the glob with proper find_package(pinocchio CONFIG REQUIRED)
        after setting CMAKE_PREFIX_PATH=/opt/ros/humble
  Option (c) is cleanest. Option (b) gets you running in 30 seconds:
      sudo mkdir -p /opt/ros/jazzy/lib/aarch64-linux-gnu
      sudo ln -s $PIN_FOUND/libpinocchio_default.so.* \
                 /opt/ros/jazzy/lib/aarch64-linux-gnu/
      (etc. for libpinocchio_parsers, liburdfdom_model)
  The run script passes CMAKE_PREFIX_PATH=/opt/ros/humble so find_package
  will work once the CMakeLists tries it.

EOM
