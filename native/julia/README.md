# SpiralBoundaryJulia

This directory contains a pure Julia boundary student that mirrors the Python
implementation.  The module exposes a ``JuliaBoundaryStudent`` type whose
methods are provided via ``getproperty`` so that ``PyCall``/``PyJulia`` can
surface the instance as a Python-friendly object.

## Usage

```julia
using Pkg
Pkg.activate("native/julia")  # optional but recommended
include("native/julia/SpiralBoundaryJulia.jl")
student = SpiralBoundaryJulia.create_student()
summary = student.train([
    "hello world",
], [["hello ", "world"]], Dict())
```

When ``PyJulia`` loads the module, the Python bridge will find the
``JuliaBoundaryStudent`` type automatically (``boundary_julia`` looks for a
``JuliaBoundaryStudent`` constructor or a ``create_student`` factory).

The implementation keeps the training logic intentionally lightweight so it can
serve as a starting point for more sophisticated native backends.  When
``CUDA.jl`` is available the module advertises a ``cuda`` device and performs
its pair reductions on GPU memory, allowing the Python loader to target the same
accelerators as the new C++ backend while keeping the training loop free from
PyTorch dependencies.
