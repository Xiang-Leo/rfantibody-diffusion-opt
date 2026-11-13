#!/usr/bin/env python3
"""Build the opt-in RFdiffusion CUDA kernels with a pinned cuBLAS runtime."""

import argparse
import ctypes
from importlib.metadata import PackageNotFoundError, distribution
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from long_exact.runtime import LINEAR_LIBRARY, NATIVE_LIBRARIES


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "long_exact"
DEFAULT_OUTPUT = HERE / "native-build"
CUBLAS_PACKAGE = "nvidia-cublas-cu11"
CUBLAS_VERSION = "11.11.3.6"
CUBLASLT_VERSION = 111103


def cublas_file(explicit):
    if explicit:
        path = explicit.expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"expected a libcublasLt.so.11 file: {path}")
        return path
    try:
        package = distribution(CUBLAS_PACKAGE)
    except PackageNotFoundError as exc:
        raise ValueError(f"install {CUBLAS_PACKAGE}=={CUBLAS_VERSION} or supply --cublas-lib") from exc
    if package.version != CUBLAS_VERSION:
        raise ValueError(f"{CUBLAS_PACKAGE} must be {CUBLAS_VERSION}, found {package.version}")
    return package.locate_file("nvidia/cublas/lib/libcublasLt.so.11").resolve(strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")),
                        help="CUDA toolkit with nvcc and SM89 support")
    parser.add_argument("--cuda-include", type=Path, help="separate cuBLAS development header directory")
    parser.add_argument("--cublas-lib", type=Path, help="explicit pinned libcublasLt.so.11 for a staged environment")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"), help="C++17 compiler")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="directory for generated libraries")
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        parser.error("activate the pinned Linux Python 3.10 environment")
    cuda = args.cuda_home.expanduser().resolve()
    nvcc = cuda / "bin/nvcc"
    include = args.cuda_include.expanduser().resolve() if args.cuda_include else cuda / "include"
    compiler = shutil.which(args.cxx)
    if not nvcc.is_file() or not (include / "cublasLt.h").is_file() or not compiler:
        parser.error("set --cuda-home and --cuda-include for nvcc and cublasLt.h, and choose a C++17 compiler")
    try:
        runtime = cublas_file(args.cublas_lib)
        api = ctypes.CDLL(str(runtime))
        api.cublasLtGetVersion.argtypes = []
        api.cublasLtGetVersion.restype = ctypes.c_size_t
        if api.cublasLtGetVersion() != CUBLASLT_VERSION:
            raise ValueError(f"cuBLASLt runtime version must be {CUBLASLT_VERSION}")
        reported = subprocess.check_output([nvcc, "--version"], text=True)
        match = re.search(r"release (\d+)\.(\d+)", reported)
        if match is None or tuple(map(int, match.groups())) < (11, 8):
            raise ValueError("nvcc 11.8 or newer is required to build SM89 kernels")
    except (AttributeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))

    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=output) as temporary:
        scratch = Path(temporary)
        for name in NATIVE_LIBRARIES.values():
            source = SOURCE / (name.removesuffix("_kernel.so") + ".cu")
            subprocess.run([nvcc, "-O3", "-arch=sm_89", "--shared", "-Xcompiler=-fPIC", "-cudart=static",
                            "--fmad=false", "-o", scratch / name, source], check=True)
        # The wheel's SONAME is used at runtime. Avoid baking an absolute RPATH
        # into artifacts so the active environment can locate its own wheel.
        subprocess.run([compiler, "-O3", "-std=c++17", "-shared", "-fPIC", "-I", include,
                        SOURCE / "linear_cublaslt.cpp", runtime, "-o", scratch / LINEAR_LIBRARY], check=True)
        for name in (*NATIVE_LIBRARIES.values(), LINEAR_LIBRARY):
            (scratch / name).replace(output / name)
    print(f"Built RFdiffusion long-exact libraries in {output}")


if __name__ == "__main__":
    main()
