# AutoKernel + SOLAR

This workspace contains the two upstream checkouts and the integrated
AutoKernel bridge:

- `autokernel/` — copied from [RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel)
- `solar/` — copied from [NVlabs/SOLAR](https://github.com/NVlabs/SOLAR)
- `autokernel/integration/` — theory-state adapter and search advisor

Start in `autokernel/`:

```bash
cd autokernel
uv sync
uv run prepare.py
uv run profile.py --model models/llama_7b.py --class-name LlamaModel \
  --input-shape 1,512 --dtype float16
uv run extract.py --top 5
```

`extract.py` now initializes `workspace/theory_state.json` automatically.
`orchestrate.py next` uses the SOLAR/Roofline headroom, and
`orchestrate.py record ... --run-log run.log` feeds measured latency back into
the advisor. To invoke SOLAR's native graph/einsum/performance pipeline, use
`python integration/run_solar.py` from inside `autokernel/`. See
`autokernel/integration/README.md` for the data format and manual commands.
