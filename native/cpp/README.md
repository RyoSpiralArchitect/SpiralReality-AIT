# spiral_boundary_cpp

This directory contains reference implementations of the boundary student and
transformer encoder in C++ using [pybind11](https://github.com/pybind/pybind11).
The default extensions export ``CppBoundaryStudent`` and
``CppTransformerAdapter`` classes that mirror their Python counterparts and can
be discovered automatically by ``integrated.boundary_cpp`` and
``integrated.encoder_backends``.  An accelerator-aware companion module
(``spiral_boundary_gpu``) mirrors the boundary student interface so that CUDA,
ROCm/HIP, or Metal builds can be surfaced to Python once their kernels are
implemented.

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

Optional toggles are provided for CUDA, ROCm/HIP, and Metal MPS discovery.
Passing ``-DSPIRAL_ENABLE_CUDA=ON`` (default), ``-DSPIRAL_ENABLE_ROCM=ON`` or
``-DSPIRAL_ENABLE_METAL=ON`` defines the corresponding build macros for **both**
the boundary student and transformer modules so they can advertise additional
devices to Python once kernels land.  The modules currently expose device
inventories and configuration hooks but still execute on the CPU until the
accelerated paths are implemented.

At runtime the transformer adapter honours ``SPIRAL_TRANSFORMER_DEVICE``,
``SPIRAL_DEVICE``, or ``SPIRAL_DEFAULT_DEVICE`` environment variables.  Setting
any of these to ``cuda``, ``rocm``, ``mps``, ``gpu``, or a specific device name
will bias the default selection to that accelerator when it was compiled in,
while ``auto``/``default`` fall back to the first advertised non-CPU target.

The compiled modules can then be installed with ``cmake --install build`` or
copied into your Python path.  Installation drops
``spiral_boundary_cpp``/``spiral_boundary_gpu`` into
``spiralreality_AIT_onepass_aifcore_integrated/integrated`` and
``_spiral_transformer_cpp`` alongside ``spiral_transformer_cpp/__init__.py`` so
``import spiral_transformer_cpp`` discovers the native adapter automatically.

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
* ``set_device(device)`` — select one of the advertised targets for the
  transformer adapter (falls back to CPU today).

The transformer adapter additionally exposes ``forward(X, gate_pos, gate_mask)``
for the attention pass and ``tune_from_boundary(base_gate, targets, lr)`` to
stay in sync with the boundary-derived gating heuristics.

The accelerator-oriented module reports ``available_devices``,
``selected_device``, and ``accelerator_available`` fields in its training
summary so downstream telemetry can see which hardware targets were compiled
in, even though computation still runs on the CPU today.
