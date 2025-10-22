# spiral_boundary_cpp

This directory contains a reference implementation of the boundary student in C++
using [pybind11](https://github.com/pybind/pybind11).  The resulting extension
exports a ``CppBoundaryStudent`` class that mirrors the Python implementation
and can be discovered automatically by ``integrated.boundary_cpp``.

## Building

```bash
python -m venv .venv
source .venv/bin/activate
pip install pybind11 cmake
cmake -S native/cpp -B build
cmake --build build
```

The compiled module (``spiral_boundary_cpp``) can then be installed with
``cmake --install build`` or copied into your Python path.  Once available, the
high-level ``OnePassAIT`` pipeline will load the compiled backend
transparently.

## Interface

The extension exposes:

* ``configure(cfg_dict)`` — accept the serialized ``StudentTrainingConfig``.
* ``train(texts, segments, cfg_dict)`` — collect character transition
  statistics and return training metadata.
* ``boundary_probs(text)`` — return per-character boundary probabilities.
* ``decode(text)`` — greedy segmentation using the learned transition model.
* ``export_state()`` / ``load_state(state)`` — serialize model parameters.
* ``available_devices()``, ``preferred_device()``, ``to_device(device)`` —
  lightweight device reporting helpers.

The implementation is intentionally lightweight: it focuses on clean data flow
between Python and C++ rather than raw performance.  It provides a solid base
for downstream optimization (vectorization, GPU kernels, CRF decoding, etc.).
