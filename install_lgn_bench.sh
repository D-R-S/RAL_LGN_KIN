#!/bin/bash
# =============================================================================
#  install_lgn_bench.sh
#  Fresh Ubuntu 24.04 → fully ready for the lgn_hand_ik v2 bench pipeline.
#
#  Target box layout (this script assumes):
#    ~/imgui           ImGui source tree
#    ~/lgn_hand_ik     repo (with URDFs/ already inside)
#
#  Run as the normal user (NOT root). It will sudo when needed.
#  Re-runnable: every step is idempotent.
# =============================================================================
set -euo pipefail

log()  { echo -e "\n\033[1;36m=== $* ===\033[0m"; }
warn() { echo -e "\033[1;33m!! $*\033[0m"; }

# -----------------------------------------------------------------------------
# 0. Sanity
# -----------------------------------------------------------------------------
if [[ $EUID -eq 0 ]]; then
    echo "Run as a normal user, not root. The script will sudo when needed."
    exit 1
fi
. /etc/os-release
if [[ "${VERSION_ID:-}" != "24.04" ]]; then
    warn "This script targets Ubuntu 24.04. Detected: ${PRETTY_NAME:-unknown}. Continuing anyway."
fi

# -----------------------------------------------------------------------------
# 1. Base system
# -----------------------------------------------------------------------------
log "apt update & upgrade"
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y

log "Build essentials & compilers"
sudo apt install -y \
    build-essential cmake git pkg-config ninja-build ccache \
    gcc-13 g++-13 clang llvm \
    software-properties-common curl ca-certificates gnupg lsb-release

log "Perf / telemetry stack"
# linux-cpupower no longer ships on 24.04; cpupower comes with kernel-matched linux-tools.
sudo apt install -y linux-tools-common linux-tools-generic cpufrequtils
# Kernel-matched perf + cpupower (the real source of /usr/lib/linux-tools/$ver/cpupower)
sudo apt install -y "linux-tools-$(uname -r)" 2>/dev/null || \
    warn "No linux-tools for kernel $(uname -r); falling back to sysfs governor write"
# linux-cpupower used to provide a thin wrapper — gone on 24.04. Not fatal.
sudo apt install -y linux-cpupower 2>/dev/null || \
    warn "linux-cpupower not available on 24.04 (expected) — run script handles fallback"

log "Math, XML, KDL, urdfdom"
sudo apt install -y \
    libeigen3-dev libboost-all-dev \
    libtinyxml2-dev \
    liborocos-kdl-dev \
    liburdfdom-dev liburdfdom-headers-dev

log "GTest / GMock"
sudo apt install -y libgtest-dev libgmock-dev

log "Demo GL stack (ImGui needs GLFW + GLEW)"
sudo apt install -y libgl1-mesa-dev libglfw3-dev libglew-dev

# -----------------------------------------------------------------------------
# 2. ROS Jazzy repo (only for ros-jazzy-pinocchio)
#    The lgn CMake detects Pinocchio via globs under /opt/ros/jazzy/lib/...
#    so this is the supported path. We do NOT source ROS for anything else.
# -----------------------------------------------------------------------------
log "Adding ROS 2 apt repository"
sudo add-apt-repository -y universe
sudo install -d -m 0755 /usr/share/keyrings
if [[ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]]; then
    sudo curl -sSL \
        https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu noble main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
sudo apt update

log "Installing ros-jazzy-pinocchio"
sudo apt install -y ros-jazzy-pinocchio

# Verify Pinocchio artefacts that lgn's CMake globs for
log "Verifying Pinocchio install (the exact paths lgn's CMake globs for)"
PIN_LIB_DIR=/opt/ros/jazzy/lib/x86_64-linux-gnu
missing=0
for pat in \
    "libpinocchio_default.so.3."* \
    "libpinocchio_parsers.so.3."* \
    "liburdfdom_model.so."*; do
    if ! ls $PIN_LIB_DIR/$pat >/dev/null 2>&1; then
        warn "MISSING: $PIN_LIB_DIR/$pat"
        missing=1
    fi
done
if (( missing )); then
    echo "Pinocchio install is incomplete — aborting."
    exit 1
fi
ls $PIN_LIB_DIR/libpinocchio_default.so.3.*
ls $PIN_LIB_DIR/libpinocchio_parsers.so.3.*

# -----------------------------------------------------------------------------
# 3. ImGui source (the demo links a static lib built from it)
#    The user's existing path was /home/lg-clean-bench/imigui — that's a typo;
#    we standardise on ~/imgui and symlink the typo path if it exists.
# -----------------------------------------------------------------------------
log "ImGui source tree at \$HOME/imgui"
if [[ ! -d "$HOME/imgui/.git" ]]; then
    rm -rf "$HOME/imgui"
    git clone --depth=1 https://github.com/ocornut/imgui "$HOME/imgui"
fi
# If a misspelled directory exists, leave it but also expose the correct name
if [[ -d "$HOME/imigui" && ! -e "$HOME/imgui" ]]; then
    ln -s "$HOME/imigui" "$HOME/imgui"
fi
test -f "$HOME/imgui/imgui.cpp"
test -f "$HOME/imgui/backends/imgui_impl_glfw.cpp"

# -----------------------------------------------------------------------------
# 4. CPU performance prep helper (installed for the run script to call)
# -----------------------------------------------------------------------------
log "Installing setup_bench_env.sh helper"
cat > "$HOME/setup_bench_env.sh" <<'EOF'
#!/bin/bash
# Run as: sudo ~/setup_bench_env.sh
# Deliberately NOT using `set -euo pipefail` — this is a best-effort helper;
# individual steps should warn and continue, not abort the whole run.

set_governor () {
    local cp=""
    # 1) cpupower on PATH
    if command -v cpupower >/dev/null 2>&1; then
        cpupower frequency-set -g performance 2>&1 | tail -3
        return 0
    fi
    # 2) cpupower in kernel-tools dir (24.04 default location)
    cp="$(ls /usr/lib/linux-tools/*/cpupower 2>/dev/null | tail -1)"
    if [[ -n "$cp" && -x "$cp" ]]; then
        "$cp" frequency-set -g performance 2>&1 | tail -3
        return 0
    fi
    # 3) raw sysfs write — works without any package
    local wrote=0
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -w "$f" ]] || continue
        echo performance > "$f" && wrote=1
    done
    if (( wrote )); then
        echo "governor set via sysfs"
        return 0
    fi
    echo "WARN: no way to set governor (no cpufreq driver?)" >&2
    return 1
}

echo "=== CPU governor -> performance ==="
set_governor || true

echo "=== perf paranoid -> 1 ==="
echo 1 > /proc/sys/kernel/perf_event_paranoid
cat /proc/sys/kernel/perf_event_paranoid

echo "=== drop caches ==="
sync && echo 3 > /proc/sys/vm/drop_caches
echo done
EOF
chmod +x "$HOME/setup_bench_env.sh"

# -----------------------------------------------------------------------------
# 5. Final summary
# -----------------------------------------------------------------------------
log "Versions"
g++ --version | head -1
cmake --version | head -1
pkg-config --modversion eigen3 || true

log "DONE"
cat <<EOM

Next steps:
  1. Make sure the repo is at:  \$HOME/lgn_hand_ik
     (containing URDFs/, benchmarks/, demo/, test/, CMakeLists.txt)
  2. Run the bench script:      ./run_lgn_bench.sh
     (it expects \$HOME/imgui and \$HOME/lgn_hand_ik)

If you keep the existing path \$HOME/imigui (typo), the install step above
created a symlink \$HOME/imgui -> \$HOME/imigui so both work.

EOM
