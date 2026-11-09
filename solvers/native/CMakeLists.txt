cmake_minimum_required(VERSION 3.18)

project(bbp_kernel_cpp LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" -c
    "import glob, os, numpy; libs=glob.glob(os.path.join(os.path.dirname(numpy.__file__), '..', 'numpy.libs', 'libscipy_openblas64_*.so')); print(libs[0] if libs else '')"
  OUTPUT_VARIABLE PYIOLS_NUMPY_OPENBLAS64_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

if(NOT PYIOLS_NUMPY_OPENBLAS64_LIBRARY)
  find_package(LAPACK REQUIRED)
endif()

option(
  PYIOLS_ENABLE_ASAN
  "Build pyiols._kernel_cpp with AddressSanitizer instrumentation"
  OFF
)

find_path(
  EIGEN3_INCLUDE_DIR
  NAMES Eigen/Core
  HINTS
    "$ENV{EIGEN3_INCLUDE_DIR}"
    /usr/include/eigen3
    /usr/local/include/eigen3
)

if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen 3.4 headers are required; pass -DEIGEN3_INCLUDE_DIR=/path/to/eigen3")
endif()

pybind11_add_module(_kernel_cpp MODULE bind.cpp)
target_compile_features(_kernel_cpp PRIVATE cxx_std_17)
target_include_directories(_kernel_cpp PRIVATE "${EIGEN3_INCLUDE_DIR}")
if(PYIOLS_NUMPY_OPENBLAS64_LIBRARY)
  target_compile_definitions(_kernel_cpp PRIVATE PYIOLS_HAVE_SCIPY_OPENBLAS64=1)
  target_link_libraries(_kernel_cpp PRIVATE "${PYIOLS_NUMPY_OPENBLAS64_LIBRARY}")
  if(NOT MSVC AND NOT APPLE)
    set_target_properties(
      _kernel_cpp
      PROPERTIES
        BUILD_RPATH "\$ORIGIN/../numpy.libs"
        INSTALL_RPATH "\$ORIGIN/../numpy.libs"
    )
  endif()
else()
  target_compile_definitions(_kernel_cpp PRIVATE PYIOLS_HAVE_LAPACK=1)
  target_link_libraries(_kernel_cpp PRIVATE ${LAPACK_LIBRARIES})
endif()

if(MSVC)
  target_compile_options(_kernel_cpp PRIVATE /fp:precise)
else()
  target_compile_options(
    _kernel_cpp
    PRIVATE
      -fno-fast-math
      -fno-finite-math-only
      -ffp-contract=off
  )
endif()

if(PYIOLS_ENABLE_ASAN)
  if(MSVC)
    message(FATAL_ERROR "PYIOLS_ENABLE_ASAN is not wired for MSVC builds")
  endif()
  target_compile_options(_kernel_cpp PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(_kernel_cpp PRIVATE -fsanitize=address)
endif()

install(TARGETS _kernel_cpp LIBRARY DESTINATION historical_iols)
