"""FusionScope Lab 01: inspect Jaxpr, StableHLO and optimized XLA HLO.

Run in separate Python processes with default flags and with
XLA_FLAGS=--xla_disable_hlo_passes=fusion to compare the HLO results.
CPU-only execution is supported. This is an IR inspection lab,
NOT a GPU kernel benchmark.
"""
import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def make_workload(depth):
    def workload(x):
        for i in range(depth):
            x = jnp.maximum(
                x * jnp.float32(1.7 + 0.01 * i)
                + jnp.float32(0.2 + 0.01 * i),
                jnp.float32(0.0),
            )
        return x
    return workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('results/default'))
    args = parser.parse_args()
    if args.size <= 0 or args.depth <= 0:
        parser.error('--size and --depth must be positive')
    args.output.mkdir(parents=True, exist_ok=True)

    fn = make_workload(args.depth)
    x_np = np.linspace(-2.0, 2.0, args.size, dtype=np.float32)
    x = jnp.asarray(x_np)
    traced = jax.make_jaxpr(fn)(x)
    lowered = jax.jit(fn).lower(x)
    stablehlo = lowered.as_text()
    compiled = lowered.compile()
    optimized = compiled.as_text()
    output = np.asarray(compiled(x))

    numpy_output = x_np.copy()
    for i in range(args.depth):
        numpy_output = np.maximum(
            numpy_output * np.float32(1.7 + 0.01 * i)
            + np.float32(0.2 + 0.01 * i),
            np.float32(0.0),
        )
    max_abs_error = float(np.max(np.abs(output - numpy_output)))

    np.save(args.output / 'output.npy', output)
    (args.output / 'jaxpr.txt').write_text(str(traced), encoding='utf-8')
    (args.output / 'stablehlo.mlir').write_text(stablehlo, encoding='utf-8')
    (args.output / 'optimized_hlo.txt').write_text(optimized, encoding='utf-8')

    entry = optimized.split('\nENTRY ', 1)[-1] if '\nENTRY ' in optimized else optimized
    summary = {
        'jax_version': jax.__version__,
        'jax_devices': [str(d) for d in jax.devices()],
        'xla_flags': os.environ.get('XLA_FLAGS', ''),
        'tensor_size': args.size,
        'chain_depth': args.depth,
        'entry_fusion_call_lines': [
            line.strip() for line in entry.splitlines()
            if ' fusion(' in line
        ],
        'max_absolute_error_vs_numpy': max_abs_error,
        'notes': (
            'Fusion instructions in CPU HLO do NOT prove the number of GPU kernels. '
            'Inspect the called fusion computations and the backend-specific output.'
        )
    }
    (args.output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('JAX:', jax.__version__, '| Devices:', jax.devices())
    print('XLA_FLAGS:', repr(summary['xla_flags']))
    print('Saved IR to:', args.output.resolve())
    print('Fusion instructions in HLO ENTRY:', len(summary['entry_fusion_call_lines']))
    print('Maximum absolute difference vs NumPy:', max_abs_error)
    print('IMPORTANT: a fusion instruction on CPU is not a measured GPU kernel.')


if __name__ == '__main__':
    main()
