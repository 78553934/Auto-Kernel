# AutoKernel ↔ SOLAR bridge (MVP)

This directory is an intentionally small adapter. It does not fork or modify
either upstream project:

- SOLAR supplies architecture-aware latency predictions when its
  `perf_*.yaml` outputs are available.
- Otherwise `solar_bridge.py` computes a transparent Roofline lower bound from
  explicit `flops`/`bytes` or a small set of standard kernel templates.
- `theory_advisor.py` compares that floor with AutoKernel's measured latency
  and emits `CONTINUE`, `SWITCH_STRATEGY`, or `MOVE_ON`.

## Try the loop

```bash
python integration/solar_bridge.py init \
  --manifest integration/example_manifest.json \
  --arch H100_PCIe \
  --output workspace/theory_state.json

# Or consume AutoKernel's native profiler output directly (preferred):
python integration/solar_bridge.py init \
  --profile workspace/profile_report.json \
  --output workspace/theory_state.json

# Or consume the plan produced by `extract.py` (uses exact model_shape fields):
python integration/solar_bridge.py init \
  --plan workspace/optimization_plan.json \
  --profile workspace/profile_report.json \
  --output workspace/theory_state.json

python integration/theory_advisor.py report
python integration/theory_advisor.py suggest --kernel kernel_matmul_1

# After `uv run bench.py > run.log 2>&1`:
python integration/theory_advisor.py update \
  --kernel kernel_matmul_1 \
  --run-log run.log
```

`autokernel/orchestrate.py` automatically reads the same
`workspace/theory_state.json`. You can also pass the measured latency directly
when recording an experiment:

```bash
cd autokernel
uv run orchestrate.py record workspace/kernel_matmul_1.py \
  120.5 kept "tile 128x64" --latency-us 15.2
```

For real SOLAR output, pass `--solar-perf-dir` and include the kernel id in the
prediction filename. For exact hardware numbers, pass `--arch-config` with the
same architecture YAML used by SOLAR; the built-in H100/A100/RTX4090 values are
only approximate defaults.

To run SOLAR's native five-stage pipeline from this checkout, use
`run_solar.py` with a SOLAR-compatible model file (a file defining `Model` and
`get_inputs()`):

```bash
python integration/run_solar.py \
  --model-file ../solar/examples/Attention/Attention.py \
  --solar-root ../solar \
  --output-dir workspace/solar \
  --arch-config H100_PCIe
```

The resulting `workspace/solar/perf/` files can be supplied to
`solar_bridge.py` or inspected alongside AutoKernel's measured benchmark.

## Manifest format

The bridge can consume AutoKernel's `workspace/profile_report.json` directly;
it maps `top_kernels[*]` to bridge records and uses the profiler's exact GPU
peak numbers when present. For custom or extracted kernels, a manifest item
should contain `kernel_id`, `kernel_type`,
`shape`, `dtype`, and `profile_fraction`. Supplying `flops` and `bytes` is
recommended for attention, fused MLP, and custom kernels because their traffic
depends on the implementation.
