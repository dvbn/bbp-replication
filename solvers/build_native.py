"""Build the historical C++ iOLS kernel for the current Python interpreter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eigen-include', type=Path,
                        default=os.environ.get('EIGEN3_INCLUDE_DIR'),
                        help='Directory containing Eigen/Core (Eigen 3.4 headers)')
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    build = (args.build_dir or root / 'build').resolve()
    eigen = args.eigen_include
    if eigen is not None and not (eigen / 'Eigen/Core').is_file():
        parser.error('--eigen-include must contain Eigen/Core')
    try:
        import pybind11
    except ImportError as exc:
        raise SystemExit('Install the build requirements first (pybind11 and CMake).') from exc
    command = ['cmake', '-S', str(root / 'native'), '-B', str(build),
               '-DPYTHON_EXECUTABLE=' + sys.executable,
               '-DPython_EXECUTABLE=' + sys.executable,
               '-DPYBIND11_FINDPYTHON=ON', '-DCMAKE_BUILD_TYPE=Release',
               '-Dpybind11_DIR=' + pybind11.get_cmake_dir(),
               '-DFETCHCONTENT_FULLY_DISCONNECTED=ON', '-DPYIOLS_ENABLE_ASAN=OFF',
               '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(build)]
    if eigen is not None:
        command.append('-DEIGEN3_INCLUDE_DIR=' + str(eigen.resolve()))
    subprocess.run(command, check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', '_kernel_cpp', '-j', '1'], check=True)
    files = sorted(build.glob('_kernel_cpp*.so')) + sorted(build.glob('_kernel_cpp*.pyd'))
    if len(files) != 1:
        raise SystemExit('Build completed, but exactly one native module was not found in ' + str(build))
    print('Built ' + str(files[0]))


if __name__ == '__main__':
    main()
