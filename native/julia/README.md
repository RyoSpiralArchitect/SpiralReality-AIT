# Spiral Julia Backends

This directory contains the Julia reference implementations used by the
SpiralReality project:

* `SpiralBoundaryJulia.jl` – the boundary student that mirrors the Python
  `BoundaryStudent` API.
* `SpiralTransformerJulia.jl` – a multi-head transformer encoder compatible
  with `SpectralTransformerAdapter`.

Both modules are designed to be driven from Python through PyJulia or
`juliacall`, keeping the runtime free of PyTorch while still enabling optional
acceleration through CUDA.jl, AMDGPU.jl, or Metal.jl when those packages are
available.

When loading either adapter the default device now respects
``SPIRAL_TRANSFORMER_DEVICE`` (falling back to ``SPIRAL_DEVICE`` or
``SPIRAL_DEFAULT_DEVICE``).  Values such as ``gpu``, ``cuda``, or ``mps`` pick
the first matching accelerator advertised by the module, while ``auto``/``default``
select the first non-CPU option or remain on the CPU if no GPU stack is
available.

## Getting started

1. Install Julia 1.9 or newer.
2. Install `juliacall` or PyJulia if you plan to load the modules from Python.
3. Instantiate the Julia project dependencies:

   ```sh
   julia --project=native/julia -e 'using Pkg; Pkg.instantiate()'
   ```

To exercise the boundary student directly:

```julia
include("native/julia/SpiralBoundaryJulia.jl")
student = SpiralBoundaryJulia.create_student()
println(SpiralBoundaryJulia.AVAILABLE_DEVICES)
```

To spin up the transformer adapter:

```julia
include("native/julia/SpiralTransformerJulia.jl")
encoder = SpiralTransformerJulia.create_adapter()
println(encoder.device)
```

Both implementations expose a dictionary-based `export_state` and corresponding
`load_state!` helper so Python callers can snapshot and restore parameters
without needing to understand Julia serialisation formats.
