#!/usr/bin/env python3
"""Run SOLAR's five-stage pipeline from inside an AutoKernel checkout.

The input model must follow SOLAR's standalone convention: define a ``Model``
or ``ReferenceModel`` class and a ``get_inputs()`` function. AutoKernel model
files can be adapted to this convention with a small wrapper if necessary.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(module: str, args: list[str], solar_root: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(solar_root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", module, *args]
    print("+", " ".join(command))
    subprocess.run(command, cwd=solar_root, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", required=True, help="SOLAR-compatible model.py")
    parser.add_argument("--solar-root", default="../solar", help="path to the SOLAR checkout")
    parser.add_argument("--output-dir", default="workspace/solar", help="pipeline output directory")
    parser.add_argument("--arch-config", default="H100_PCIe")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--save-graph", action="store_true")
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--skip-timeloop", action="store_true")
    args = parser.parse_args()

    solar_root = Path(args.solar_root).resolve()
    model_file = Path(args.model_file).resolve()
    output = Path(args.output_dir).resolve()
    if not (solar_root / "solar").is_dir():
        raise SystemExit(f"SOLAR package not found under {solar_root}")
    if not model_file.exists():
        raise SystemExit(f"model file not found: {model_file}")

    graph = output / "graph"
    einsum = output / "einsum"
    analysis = output / "analysis"
    perf = output / "perf"
    timeloop = output / "timeloop"

    process_args = ["--model-file", str(model_file), "--output-dir", str(graph)]
    if args.save_graph:
        process_args.append("--save-graph")
    if args.safe_mode:
        process_args.append("--safe-mode")
    run_step("solar.cli.process_model", process_args, solar_root)
    run_step(
        "solar.cli.toeinsum_model",
        [
            "--graph-path", str(graph / "pytorch_graph.yaml"),
            "--output-dir", str(einsum),
            "--no-copy-graph", "--enable-rename",
            *( ["--save-graph"] if args.save_graph else [] ),
        ],
        solar_root,
    )
    run_step(
        "solar.cli.analyze_model",
        [
            "--einsum-graph-path", str(einsum / "einsum_graph_renamed.yaml"),
            "--output-dir", str(analysis), "--precision", args.precision,
        ],
        solar_root,
    )
    run_step(
        "solar.cli.predict_perf_model",
        [
            "--analysis-path", str(analysis / "analysis.yaml"),
            "--output-dir", str(perf), "--arch-config", args.arch_config,
            "--precision", args.precision,
        ],
        solar_root,
    )
    if not args.skip_timeloop:
        run_step(
            "solar.cli.totimeloop",
            [
                "--einsum-graph-path", str(einsum / "einsum_graph_renamed.yaml"),
                "--output-dir", str(timeloop),
            ],
            solar_root,
        )
    print(f"SOLAR pipeline complete: {output}")


if __name__ == "__main__":
    main()
