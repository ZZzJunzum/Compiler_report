# FusionScope — Lab 01: Inspecting IR and Fusion Decisions

## Objective
Observe how a simple elementwise computation appears in Jaxpr, StableHLO and optimized HLO, and compare the IR under different XLA compiler flags. This is a **CPU-only compiler-IR exercise**, not a GPU benchmark.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install jax numpy
```

## Run experiment A: default compiler settings
```powershell
Remove-Item Env:\XLA_FLAGS -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe .\fusionscope_ir_lab.py --size 1024 --depth 1 --output results\default
```

## Run experiment B: disable named HLO fusion pass
```powershell
$env:XLA_FLAGS = '--xla_disable_hlo_passes=fusion'
.\.venv\Scripts\python.exe .\fusionscope_ir_lab.py --size 1024 --depth 1 --output results\no_fusion
Remove-Item Env:\XLA_FLAGS -ErrorAction SilentlyContinue
```

## Files produced by each run
- `jaxpr.txt`: traced JAX IR
- `stablehlo.mlir`: MLIR-based high-level IR
- `optimized_hlo.txt`: final optimized HLO on your local CPU backend
- `summary.json`: metadata, lines with fusion instructions in the HLO `ENTRY`, simple correctness check
- `output.npy`: output values, which can be compared across runs

## What to inspect
1. Find `mul`, `add`, and `max` in Jaxpr.
2. Find `stablehlo.multiply`, `stablehlo.add`, and `stablehlo.maximum` in StableHLO.
3. In optimized HLO, search for `%fused_computation`, `fusion(`, and `ENTRY`.
4. Compare which instructions are *inside* a fused computation in the two runs. **Do not equate the word `fusion` with a physical GPU kernel on CPU**, and do not infer the absence of all fusion just because the flag disabled one HLO pass.
5. Compare output arrays with NumPy, and note floating-point differences caused by rounding and code generation.

## Tested example
Test environment: JAX 0.9.0.1, CPU backend, size=1024, depth=1. The default compiled HLO has one fusion instruction in ENTRY that covers the computation. With `--xla_disable_hlo_passes=fusion`, the ENTRY has six `fusion`-named wrappers around individual broadcasts / primitive computations (not one fused chain). The results depend on XLA version and backend. Do not generalize these CPU HLO patterns into measured GPU performance.

## References
- JAX AOT API: https://docs.jax.dev/en/latest/aot.html
- OpenXLA HLO dump guide: https://openxla.org/xla/hlo_dumps
- OpenXLA XLA:GPU architecture: https://openxla.org/xla/gpu_architecture
