# =============================================================================
# Neural-Network-Cpp
#
# This project is a small, PyTorch-like neural network *library*: the real
# deliverable is everything under src/. main.cpp is only a usage example that
# happens to live at the repo root, and tests/test.cpp is a standalone test
# binary. The build below reflects that: one library target, two thin
# executables that link against it.
# =============================================================================

# 3.15 is the floor for reliably using generator expressions in
# target_compile_options() (used below for per-compiler warning flags) plus
# solid Ninja/Visual Studio support on Windows.
cmake_minimum_required(VERSION 3.15)

project(NeuralNetworkCpp
    VERSION 0.1.0
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Global language/tooling settings
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)          # the codebase relies on C++17 (structured code, <memory>, etc.)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # fail the configure step instead of silently downgrading the standard
set(CMAKE_CXX_EXTENSIONS OFF)       # prefer strict -std=c++17 over GNU extensions (-std=gnu++17) for portability

# Emits compile_commands.json, which clangd / VS Code's C++ extension can read
# for accurate IntelliSense - replaces the manual files.associations list in
# .vscode/settings.json, which only fakes syntax highlighting, not real parsing.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Single-config generators (Makefiles, Ninja) need CMAKE_BUILD_TYPE set explicitly
# or they default to an unoptimized, non-debug "no type" build. Multi-config
# generators (Visual Studio, Xcode) ignore this and use --config instead, so we
# only set it when CMAKE_CONFIGURATION_TYPES (the multi-config marker) is absent.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# Matches the -O3 that notes.txt / README.md previously told you to pass to g++ by hand.
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# -----------------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------------
# Lets you (or CI) configure e.g. `cmake -B build -DNNCPP_BUILD_EXAMPLE=OFF` to
# build only the library, without editing this file.

option(NNCPP_BUILD_EXAMPLE      "Build main.cpp, the usage example (not part of the library itself)" ON)
option(NNCPP_BUILD_TESTS        "Build tests/test.cpp and register it with CTest" ON)
option(NNCPP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors (-Werror / /WX)" OFF)

# -----------------------------------------------------------------------------
# Core library (everything the framework itself is made of)
# -----------------------------------------------------------------------------
# Deliberately an explicit source list rather than file(GLOB ...): CMake only
# re-scans globbed directories when you re-run `cmake`, so a plain glob can
# silently fail to pick up a newly-added .cpp on an incremental build. This is
# the CMake-documented best practice for regular source additions; add each
# new file here as the library grows (tensors, optimizers, batchnorm, ... per
# notes.txt).
add_library(nn_cpp STATIC
    src/core/neural_network.cpp
    src/math/matrix.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/utils/mnist_loader.cpp
    src/utils/matrix_utils.cpp
)

# PUBLIC (not PRIVATE): the existing headers already include each other with
# paths rooted at the project directory (e.g. main.cpp's "./src/core/..." and
# tests/test.cpp's "../src/..."). Making this PUBLIC means anything that links
# against nn_cpp - main.cpp, tests, or a future external consumer of the
# library - automatically inherits this include path too, without needing to
# restate it.
target_include_directories(nn_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Reasonable default warnings: PUBLIC so the example/tests get them too, since
# they're first-party code in this repo, not third-party code we want to stay
# quiet about. Expressed as generator expressions so the same CMakeLists.txt
# works whether you build with g++/clang++ (MinGW on Windows, or Linux/macOS)
# or MSVC.
target_compile_options(nn_cpp PUBLIC
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${NNCPP_WARNINGS_AS_ERRORS}>>:-Werror>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<BOOL:${NNCPP_WARNINGS_AS_ERRORS}>>:/WX>
)

# -----------------------------------------------------------------------------
# Example executable - demonstrates using the library; it is NOT the library
# -----------------------------------------------------------------------------
if(NNCPP_BUILD_EXAMPLE)
    add_executable(nn_example main.cpp)
    target_link_libraries(nn_example PRIVATE nn_cpp)

    # Keep the historical binary name so existing habits (`./main`) still work.
    set_target_properties(nn_example PROPERTIES OUTPUT_NAME main)
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(NNCPP_BUILD_TESTS)
    enable_testing() # activates the `ctest` command

    add_executable(nn_tests tests/test.cpp)
    target_link_libraries(nn_tests PRIVATE nn_cpp)
    set_target_properties(nn_tests PROPERTIES OUTPUT_NAME test)

    # NOTE: tests/test.cpp's main() always `return 0`, regardless of how many
    # individual checks fail (see TestRunner::printSummary in test_runner.hpp),
    # so CTest can't rely on the exit code the way it normally would. As a
    # workaround, scan stdout for the "FAILED" string TestRunner prints per
    # test so at least one real failure still fails the `ctest` run. The real
    # fix is to make test.cpp exit non-zero on failure; consider that a
    # follow-up, not something this CMake change silently patches over.
    add_test(NAME nn_tests COMMAND nn_tests)
    set_tests_properties(nn_tests PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
endif()
