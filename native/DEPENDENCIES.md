# Native Dependency Inventory

The manylinux build image relies on the following native toolchain and
libraries:

| Component | Purpose | Notes |
|-----------|---------|-------|
| `gcc`/`g++` 11+ | Compile C++ kernels | Provided by the manylinux base image |
| `cmake` 3.20+ | Configure native builds | Installed in the Docker image |
| `ninja` | Parallel build backend | Optional but enabled in the toolchain |
| `openmp` (`libgomp`) | Parallel execution support | Bundled with GCC |
| `python3-devel` | Python headers for extension modules | Provided by manylinux |

The `native/cpp` directory retains the existing `CMakeLists.txt` entry point
for building extension modules. Use `native/cpp/README.md` to document target
specifics and update this manifest when new dependencies are introduced.
