# spiral_boundary_cpp

This directory contains reference implementations of the boundary student in C++
using [pybind11](https://github.com/pybind/pybind11).  The default extension
exports a ``CppBoundaryStudent`` class that mirrors the Python implementation
and can be discovered automatically by ``integrated.boundary_cpp``.  A
GPU-aware backend (``spiral_boundary_gpu``) mirrors the interface and keeps the
pipeline ready for CUDA kernels.

## Building

```bash
python -m venv .venv
source .venv/bin/activate
pip install pybind11 cmake
cmake -S native/cpp -B build
cmake --build build
```

Enable the GPU backend explicitly if you have the CUDA toolkit available:

```bash
cmake -S native/cpp -B build -DSPIRAL_BUILD_GPU=ON
cmake --build build
```

When CUDA is detected the ``spiral_boundary_gpu`` module is compiled with the
``SPIRAL_HAS_CUDA`` definition and linked against ``cudart``.  Without CUDA the
module is still generated in CPU emulation mode so the Python loader can use a
single discovery path while the GPU kernels are implemented incrementally.

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
  lightweight device reporting helpers.

The GPU-oriented module adds ``accelerated`` and ``accelerator_available`` keys
to its training summary so downstream telemetry can distinguish between CPU
emulation and true CUDA execution once kernels are wired in.
