# ============================================================================
#  CMakeLists.txt  —  patch 1.2
#
#  Patch 1.2 changes vs 1.1:
#    - test_roundtrip is registered as a GTest target (was dead code).
#    - bench_all (Google Benchmark) target removed (migrated to direct
#      steady_clock timers in bench_v2 and bench_lgn_kdl).
#    - Pinocchio path uses file(GLOB ...) for libpinocchio_default.so.3.*
#      so a future apt minor-version bump (3.10) does not break the build.
#    - Residual OpenMP linkage on bench targets removed; bench binaries are
#      single-threaded by design (paper §4.1).
#    - bench_v2 target structure unchanged: three isolated binaries,
#      lgn+KDL, lgn+Pinocchio, lgn-only-with-velocity-baseline.
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(lgn_hand_ik VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-march=native>
  $<$<CONFIG:Release>:-ffast-math>
  $<$<CONFIG:Debug>:-O0>
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Debug>:-fsanitize=address>
)
add_link_options(
  $<$<CONFIG:Debug>:-fsanitize=address>
)

option(BUILD_ROS        "Build ROS2 nodes"              ON)
option(BUILD_DEMO       "Build standalone OpenGL demo"  ON)
option(BUILD_TESTING    "Build GTest correctness suite" ON)
option(BUILD_BENCHMARKS "Build benchmark suite"         OFF)
set(HAND_URDF "" CACHE PATH "Optional: path to Amazing Hand URDF for full-hand bench")

find_package(Eigen3 REQUIRED NO_MODULE)

find_package(tinyxml2 QUIET)
if(NOT tinyxml2_FOUND)
  message(STATUS "tinyxml2 not found via find_package — using vendored copy")
  add_library(tinyxml2 STATIC third_party/tinyxml2/tinyxml2.cpp)
  target_include_directories(tinyxml2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyxml2>
    $<INSTALL_INTERFACE:include/tinyxml2>)
  set_target_properties(tinyxml2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

find_package(OpenMP)

add_library(lgn_core INTERFACE)
add_library(lgn::core ALIAS lgn_core)
target_include_directories(lgn_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(lgn_core INTERFACE Eigen3::Eigen tinyxml2)
if(OpenMP_CXX_FOUND)
  target_link_libraries(lgn_core INTERFACE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP found — parallel finger IK enabled (HandIKSolver only)")
else()
  message(STATUS "OpenMP not found — single-thread IK only")
endif()

if(BUILD_ROS)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp      REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(std_msgs    REQUIRED)

  add_executable(hand_ik_node src/hand_ik_node.cpp)
  target_link_libraries(hand_ik_node lgn::core)
  ament_target_dependencies(hand_ik_node rclcpp sensor_msgs geometry_msgs std_msgs)

  add_executable(virtual_sensor_node src/virtual_sensor_node.cpp)
  target_link_libraries(virtual_sensor_node lgn::core)
  ament_target_dependencies(virtual_sensor_node rclcpp sensor_msgs geometry_msgs std_msgs)

  install(TARGETS hand_ik_node virtual_sensor_node DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch OPTIONAL)
  install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config OPTIONAL)
endif()

install(DIRECTORY include/ DESTINATION include)

if(BUILD_DEMO)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 QUIET)
  if(NOT glfw3_FOUND)
    message(FATAL_ERROR "BUILD_DEMO=ON but GLFW3 not found.\n  Ubuntu: sudo apt install libglfw3-dev")
  endif()
  find_package(GLEW QUIET)
  if(NOT GLEW_FOUND)
    message(FATAL_ERROR "BUILD_DEMO=ON but GLEW not found.\n  Ubuntu: sudo apt install libglew-dev")
  endif()
  set(IMGUI_DIR "" CACHE PATH "Path to imgui source directory (required for BUILD_DEMO)")
  if(NOT IMGUI_DIR OR NOT EXISTS "${IMGUI_DIR}/imgui.cpp")
    message(FATAL_ERROR
      "BUILD_DEMO=ON but IMGUI_DIR is not set or invalid.\n"
      "  Clone ImGui:  git clone https://github.com/ocornut/imgui --depth=1\n"
      "  Then:         cmake .. -DIMGUI_DIR=/path/to/imgui")
  endif()
  add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
  target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
  target_link_libraries(imgui PUBLIC OpenGL::GL glfw GLEW::GLEW)

  add_executable(lgn_demo demo/main.cpp)
  target_link_libraries(lgn_demo PRIVATE lgn::core imgui OpenGL::GL glfw GLEW::GLEW)
  install(TARGETS lgn_demo DESTINATION bin OPTIONAL)
  install(FILES demo/two_segment.urdf DESTINATION share/${PROJECT_NAME}/demo OPTIONAL)
endif()

if(BUILD_TESTING)
  if(BUILD_ROS)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_correctness test/test_correctness.cpp)
    target_link_libraries(test_correctness lgn::core)

    # Patch 1.2: register URDF↔MJCF round-trip test
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_roundtrip.cpp")
      ament_add_gtest(test_roundtrip test/test_roundtrip.cpp)
      target_link_libraries(test_roundtrip lgn::core)
    endif()
  else()
    find_package(GTest REQUIRED)
    enable_testing()
    add_executable(test_correctness test/test_correctness.cpp)
    target_link_libraries(test_correctness PRIVATE lgn::core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(test_correctness)

    # Patch 1.2: register URDF↔MJCF round-trip test
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_roundtrip.cpp")
      add_executable(test_roundtrip test/test_roundtrip.cpp)
      target_link_libraries(test_roundtrip PRIVATE lgn::core GTest::gtest_main)
      gtest_discover_tests(test_roundtrip)
    endif()
  endif()
endif()

if(BUILD_BENCHMARKS)

  # ── Dependency detection ────────────────────────────────────────────────────
  find_package(orocos_kdl QUIET)
  find_package(pinocchio QUIET)

  # Detect ROS Jazzy Pinocchio even when find_package(pinocchio) fails
  # (happens when KDL is also present — ABI conflict prevents joint linkage).
  # Patch 1.2: glob for any 3.x version so 3.10.0 won't break the build.
  set(PIN_JAZZY_LIB_DIR "/opt/ros/jazzy/lib/x86_64-linux-gnu")
  set(PIN_JAZZY_INC     "/opt/ros/jazzy/include")
  file(GLOB PIN_JAZZY_LIBS "${PIN_JAZZY_LIB_DIR}/libpinocchio_default.so.3.*")
  list(LENGTH PIN_JAZZY_LIBS PIN_LIB_COUNT)
  if(PIN_LIB_COUNT GREATER 0)
    list(GET PIN_JAZZY_LIBS 0 PIN_DEFAULT_LIB)
    set(HAVE_PIN_JAZZY TRUE)
    message(STATUS "ROS Jazzy Pinocchio detected: ${PIN_DEFAULT_LIB}")
  else()
    set(HAVE_PIN_JAZZY FALSE)
    message(STATUS "ROS Jazzy Pinocchio not found at ${PIN_JAZZY_LIB_DIR}")
  endif()

  # Glob the parsers / urdfdom libraries similarly
  if(HAVE_PIN_JAZZY)
    file(GLOB PIN_PARSERS_LIB    "${PIN_JAZZY_LIB_DIR}/libpinocchio_parsers.so.3.*")
    file(GLOB URDFDOM_MODEL_LIB  "${PIN_JAZZY_LIB_DIR}/liburdfdom_model.so.*")
    file(GLOB URDFDOM_WORLD_LIB  "${PIN_JAZZY_LIB_DIR}/liburdfdom_world.so.*")
    file(GLOB URDFDOM_SENSOR_LIB "${PIN_JAZZY_LIB_DIR}/liburdfdom_sensor.so.*")
    list(GET PIN_PARSERS_LIB    0 PIN_PARSERS_LIB)
    list(GET URDFDOM_MODEL_LIB  0 URDFDOM_MODEL_LIB)
    list(GET URDFDOM_WORLD_LIB  0 URDFDOM_WORLD_LIB)
    list(GET URDFDOM_SENSOR_LIB 0 URDFDOM_SENSOR_LIB)
  endif()

  # ── Helper macro: common bench settings (no OpenMP — single-threaded) ─────
  macro(lgn_bench_base target src)
    add_executable(${target} ${src})
    target_include_directories(${target} PRIVATE
      ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(${target} PRIVATE -O3 -march=native -ffast-math)
    target_link_libraries(${target} PRIVATE Eigen3::Eigen tinyxml2)
    if(HAND_URDF AND EXISTS "${HAND_URDF}")
      target_compile_definitions(${target} PRIVATE HAND_URDF="${HAND_URDF}")
    endif()
  endmacro()

  # ── bench_lgn_kdl (legacy steady_clock harness, kept for reference) ───────
  lgn_bench_base(bench_lgn_kdl benchmarks/bench_lgn_kdl.cpp)
  if(orocos_kdl_FOUND)
    target_compile_definitions(bench_lgn_kdl PRIVATE HAVE_KDL)
    target_link_libraries(bench_lgn_kdl PRIVATE ${orocos_kdl_LIBRARIES})
    target_include_directories(bench_lgn_kdl PRIVATE ${orocos_kdl_INCLUDE_DIRS})
    message(STATUS "KDL found — KDL legacy bench ENABLED")
  else()
    message(STATUS "KDL not found — bench_lgn_kdl will run lgn-only")
    message(STATUS "  Install: sudo apt install liborocos-kdl-dev")
  endif()

  # ── bench_pinocchio_direct (legacy standalone, kept for reference) ────────
  if(HAVE_PIN_JAZZY)
    add_executable(bench_pinocchio_direct benchmarks/bench_pinocchio_direct.cpp)
    target_include_directories(bench_pinocchio_direct PRIVATE
      ${CMAKE_SOURCE_DIR}/include
      ${PIN_JAZZY_INC})
    target_compile_options(bench_pinocchio_direct PRIVATE -O3 -march=native -ffast-math)
    target_link_options(bench_pinocchio_direct PRIVATE
      -Wl,-rpath,${PIN_JAZZY_LIB_DIR})
    target_link_libraries(bench_pinocchio_direct PRIVATE
      Eigen3::Eigen
      ${PIN_PARSERS_LIB}
      ${PIN_DEFAULT_LIB}
      ${URDFDOM_MODEL_LIB}
      ${URDFDOM_WORLD_LIB}
      ${URDFDOM_SENSOR_LIB})
  endif()

  # ── bench_v2: three isolated binaries (apples-to-apples v2 suite) ─────────
  #
  #  Why three binaries instead of one:
  #  Pinocchio (ROS Jazzy) and KDL (system apt) link against different builds
  #  of Eigen and Boost. In the same process they cause SIGFAULT (139). Three
  #  separate processes also give stronger isolation between solver runs (no
  #  shared cache / branch-predictor state). This is documented in §4.1.
  #
  #    bench_v2_lgn_kdl  — lgn + KDL    (position-only DLS IK, fair compare)
  #    bench_v2_pin      — lgn + Pin    (Pinocchio FK/IK + mass-matrix gate)
  #    bench_v2_lgn_only — lgn alone    (velocity baseline, no third-party)

  # Binary 1: lgn + KDL
  lgn_bench_base(bench_v2_lgn_kdl benchmarks/bench_v2.cpp)
  target_compile_definitions(bench_v2_lgn_kdl PRIVATE BENCH_LGN=1)
  if(orocos_kdl_FOUND)
    target_compile_definitions(bench_v2_lgn_kdl PRIVATE BENCH_KDL=1)
    target_link_libraries(bench_v2_lgn_kdl PRIVATE ${orocos_kdl_LIBRARIES})
    target_include_directories(bench_v2_lgn_kdl PRIVATE ${orocos_kdl_INCLUDE_DIRS})
  endif()

  # Binary 2: lgn + Pinocchio (ROS Jazzy, isolated — NO KDL linkage)
  if(HAVE_PIN_JAZZY)
    lgn_bench_base(bench_v2_pin benchmarks/bench_v2.cpp)
    target_compile_definitions(bench_v2_pin PRIVATE
      BENCH_LGN=1 BENCH_PINOCCHIO=1)
    target_include_directories(bench_v2_pin PRIVATE ${PIN_JAZZY_INC})
    target_link_options(bench_v2_pin PRIVATE
      -Wl,-rpath,${PIN_JAZZY_LIB_DIR})
    target_link_libraries(bench_v2_pin PRIVATE
      ${PIN_PARSERS_LIB}
      ${PIN_DEFAULT_LIB}
      ${URDFDOM_MODEL_LIB}
      ${URDFDOM_WORLD_LIB}
      ${URDFDOM_SENSOR_LIB})
  else()
    message(STATUS "bench_v2_pin skipped — ROS Jazzy Pinocchio not found")
  endif()

  # Binary 3: lgn only — velocity baseline (no third-party dependencies)
  lgn_bench_base(bench_v2_lgn_only benchmarks/bench_v2.cpp)
  target_compile_definitions(bench_v2_lgn_only PRIVATE
    BENCH_LGN=1 BENCH_VEL_BASELINE=1)

  # ── twin_sim (visual demo: lgn vs pinocchio side-by-side) ─────────────────
  #
  #  Standalone visual sanity check for co-authors and the paper video.
  #  Same URDF, same initial condition, same integrator, same gravity:
  #  two engines integrating a free-fall pendulum in parallel. gnuplot
  #  renders both stick figures live with the residual
  #  ||q_lgn − q_pin||_inf in the title bar.
  #
  #  Built alongside the benchmarks because it has the same linkage
  #  requirements (lgn + ROS Jazzy Pinocchio, no KDL). Runs standalone.
  if(HAVE_PIN_JAZZY)
    add_executable(twin_sim sim/twin_sim.cpp)
    target_include_directories(twin_sim PRIVATE
      ${CMAKE_SOURCE_DIR}/include
      ${PIN_JAZZY_INC})
    target_compile_options(twin_sim PRIVATE -O3 -march=native -ffast-math)
    target_link_options(twin_sim PRIVATE
      -Wl,-rpath,${PIN_JAZZY_LIB_DIR})
    target_link_libraries(twin_sim PRIVATE
      Eigen3::Eigen
      tinyxml2
      ${PIN_PARSERS_LIB}
      ${PIN_DEFAULT_LIB}
      ${URDFDOM_MODEL_LIB}
      ${URDFDOM_WORLD_LIB}
      ${URDFDOM_SENSOR_LIB})
  else()
    message(STATUS "twin_sim skipped — ROS Jazzy Pinocchio not found")
  endif()
endif() # BUILD_BENCHMARKS

if(BUILD_ROS)
  ament_package()
endif()

message(STATUS "")
message(STATUS "lgn_hand_ik ${PROJECT_VERSION} — build summary")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  ROS2 nodes:   ${BUILD_ROS}")
message(STATUS "  Demo (GL):    ${BUILD_DEMO}")
message(STATUS "  Tests:        ${BUILD_TESTING}")
message(STATUS "  Benchmarks:   ${BUILD_BENCHMARKS}")
message(STATUS "  OpenMP:       ${OpenMP_CXX_FOUND}")
if(BUILD_BENCHMARKS)
  message(STATUS "  KDL:          ${orocos_kdl_FOUND}")
  message(STATUS "  Pinocchio:    ${HAVE_PIN_JAZZY} (ROS Jazzy isolated)")
  if(HAND_URDF)
    message(STATUS "  Hand URDF:    ${HAND_URDF}")
  endif()
endif()
message(STATUS "")
