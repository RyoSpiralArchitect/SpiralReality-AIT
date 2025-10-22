# spiral_boundary_cpp

This directory contains reference implementations of the boundary student in C++
using [pybind11](https://github.com/pybind/pybind11).  The default extension
exports a ``CppBoundaryStudent`` class that mirrors the Python implementation
and can be discovered automatically by ``integrated.boundary_cpp``.  An
accelerator-aware companion module (``spiral_boundary_gpu``) mirrors the
interface so that CUDA, ROCm/HIP, or Metal builds can be surfaced to Python
once their kernels are implemented.

## Building

```bash
python -m venv .venv
source .venv/bin/activate
pip install pybind11 cmake
cmake -S native/cpp -B build
cmake --build build
```

Enable the accelerator stub explicitly if you plan to surface native devices
to Python:

```bash
cmake -S native/cpp -B build -DSPIRAL_BUILD_GPU=ON
cmake --build build
```

Optional toggles are provided for CUDA, ROCm/HIP, and Metal MPS discovery.  By
default only CUDA is probed via ``find_package(CUDAToolkit)``; pass
``-DSPIRAL_ENABLE_ROCM=ON`` or ``-DSPIRAL_ENABLE_METAL=ON`` to advertise those
targets when compiling on systems that provide them.  The module currently
exposes device inventories and configuration hooks but still executes on the
CPU until accelerator kernels are implemented.

The compiled modules can then be installed with ``cmake --install build`` or
copied into your Python path.  Once available, the high-level ``OnePassAIT``
pipeline will load the compiled backend transparently.

## Interface

The extensions expose:

* ``configure(cfg_dict)`` — accept the serialized ``StudentTrainingConfig``.
* ``train(texts, segments, cfg_dict)`` — collect character transition
  statistics and return training metadata.
* ``boundary_probs(text)`` — return per-character boundary probabilities.
* ``decode(text)`` — greedy segmentation using the learned transition model.
* ``export_state()`` / ``load_state(state)`` — serialize model parameters.
* ``available_devices()``, ``preferred_device()``, ``to_device(device)`` —
  lightweight device reporting helpers (common to both modules).

The accelerator-oriented module reports ``available_devices``,
``selected_device``, and ``accelerator_available`` fields in its training
summary so downstream telemetry can see which hardware targets were compiled
in, even though computation still runs on the CPU today.
