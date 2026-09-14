#!/usr/bin/env python3
"""Use SOLAR theory state to steer AutoKernel's real benchmark loop."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRATEGIES = {
    "compute": ["block_size_sweep", "tensor_core_or_precision", "fusion_or_instruction_tuning", "architecture_specific"],
    "memory": ["block_size_sweep", "coalescing_and_tiling", "prefetch_or_cache_swizzle", "fusion_or_layout"],
    "unknown": ["block_size_sweep", "layout_and_fusion", "backend_switch", "architecture_specific"],
}


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _gap(item: dict[str, Any]) -> float | None:
    best = item.get("best_latency_us")
    floor = item.get("calibrated_floor_us") or item.get("theory_latency_us")
    if best is None or floor is None or best <= 0:
        return None
    return max(0.0, (float(best) - float(floor)) / float(best))


def _suggest(item: dict[str, Any]) -> dict[str, Any]:
    gap = _gap(item)
    bottleneck = str(item.get("bottleneck", "unknown"))
    strategies = STRATEGIES.get(bottleneck, STRATEGIES["unknown"])
    index = min(int(item.get("strategy_index", 0)), len(strategies) - 1)
    plateau = int(item.get("plateau_count", 0))

    if gap is not None and gap <= 0.10:
        action = "MOVE_ON"
        reason = "measured latency is within 10% of the calibrated theory floor"
    elif plateau >= 5:
        action = "SWITCH_STRATEGY"
        # Switch once per block of five failed experiments, rather than
        # changing strategy on every subsequent plateau iteration.
        index = min(max(index, plateau // 5), len(strategies) - 1)
        reason = f"{plateau} consecutive non-improving experiments"
    else:
        action = "CONTINUE"
        reason = "theory gap remains exploitable"

    return {
        "kernel_id": item["kernel_id"],
        "action": action,
        "reason": reason,
        "bottleneck": bottleneck,
        "strategy": strategies[index],
        "strategy_index": index,
        "theory_latency_us": item.get("theory_latency_us"),
        "calibrated_floor_us": item.get("calibrated_floor_us"),
        "best_latency_us": item.get("best_latency_us"),
        "gap_ratio": gap,
        "plateau_count": plateau,
    }


def _pick_kernel(state: dict[str, Any]) -> dict[str, Any]:
    scored = []
    for item in state.get("kernels", {}).values():
        gap = _gap(item) or 0.0
        score = float(item.get("profile_fraction", 0.0)) * gap
        scored.append((score, item))
    if not scored:
        raise SystemExit("theory state contains no kernels")
    return max(scored, key=lambda pair: pair[0])[1]


def _parse_metrics(text: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    patterns = {
        "latency_us": r"latency_us\s*[:=]\s*([0-9.eE+-]+)",
        "throughput_tflops": r"throughput_tflops\s*[:=]\s*([0-9.eE+-]+)",
        "speedup_vs_pytorch": r"speedup_vs_pytorch\s*[:=]\s*([0-9.eE+-]+)",
        "correctness": r"correctness\s*[:=]\s*(PASS|FAIL|TIMEOUT|CRASH)",
    }
    for key, pattern in patterns.items():
        found = re.findall(pattern, text, flags=re.IGNORECASE)
        if found:
            value = found[-1]
            result[key] = value if key == "correctness" else float(value)
    return result


def _metrics_from_tsv(path: Path, kernel_type: str) -> dict[str, float | str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in reversed(rows):
        if row.get("kernel_type") == kernel_type:
            result: dict[str, float | str] = {}
            for key in ("latency_us", "throughput_tflops", "speedup_vs_pytorch"):
                if row.get(key):
                    result[key] = float(row[key])
            if row.get("correctness"):
                result["correctness"] = row["correctness"]
            return result
    return {}


def update_item(item: dict[str, Any], metrics: dict[str, float | str], keep: bool | None) -> dict[str, Any]:
    latency = metrics.get("latency_us")
    if not isinstance(latency, (int, float)):
        raise SystemExit("no latency_us found; pass --latency-us or a run log containing latency_us")
    item["experiments"] = int(item.get("experiments", 0)) + 1
    correctness = str(metrics.get("correctness", "PASS")).upper()
    accepted = correctness == "PASS" and (keep is not False)
    if accepted and (item.get("best_latency_us") is None or latency < float(item["best_latency_us"])):
        item["best_latency_us"] = float(latency)
        item["improvements"] = int(item.get("improvements", 0)) + 1
        item["plateau_count"] = 0
        if item.get("baseline_latency_us") is None:
            item["baseline_latency_us"] = float(latency)
            theory = item.get("theory_latency_us")
            if theory:
                # Keep the raw SOLAR/Roofline floor as the search target.  The
                # measured baseline is useful for reporting, but using it as a
                # floor would eliminate all apparent search headroom.  A
                # calibration factor can be supplied explicitly in the
                # manifest once it has been learned from reference kernels.
                item["observed_calibration_factor"] = float(latency) / float(theory)
    else:
        item["plateau_count"] = int(item.get("plateau_count", 0)) + 1
    item["last_latency_us"] = float(latency)
    item["last_correctness"] = correctness
    suggestion = _suggest(item)
    item["strategy_index"] = suggestion["strategy_index"]
    item["last_action"] = suggestion["action"]
    return suggestion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    suggest = sub.add_parser("suggest")
    suggest.add_argument("--state", default="workspace/theory_state.json")
    suggest.add_argument("--kernel")

    update = sub.add_parser("update")
    update.add_argument("--state", default="workspace/theory_state.json")
    update.add_argument("--kernel", required=True)
    update.add_argument("--run-log")
    update.add_argument("--results-tsv")
    update.add_argument("--latency-us", type=float)
    update.add_argument("--throughput-tflops", type=float)
    update.add_argument("--correctness", choices=["PASS", "FAIL", "TIMEOUT", "CRASH"], default="PASS")
    update.add_argument("--keep", action=argparse.BooleanOptionalAction, default=None)

    report = sub.add_parser("report")
    report.add_argument("--state", default="workspace/theory_state.json")
    args = parser.parse_args()
    state_path = Path(args.state)
    state = load_state(state_path)

    if args.command == "suggest":
        item = state["kernels"].get(args.kernel) if args.kernel else _pick_kernel(state)
        if item is None:
            raise SystemExit(f"unknown kernel: {args.kernel}")
        print(json.dumps(_suggest(item), indent=2))
        return

    if args.command == "report":
        print(f"architecture: {state.get('architecture', {}).get('name', 'unknown')}")
        for item in state.get("kernels", {}).values():
            suggestion = _suggest(item)
            print(
                f"{item['kernel_id']:<28} impact={float(item.get('profile_fraction', 0)):.3f} "
                f"best={item.get('best_latency_us', 'n/a')}us "
                f"floor={item.get('calibrated_floor_us', 'n/a')}us "
                f"gap={suggestion['gap_ratio'] if suggestion['gap_ratio'] is not None else 'n/a'} "
                f"action={suggestion['action']} strategy={suggestion['strategy']}"
            )
        return

    item = state["kernels"].get(args.kernel)
    if item is None:
        raise SystemExit(f"unknown kernel: {args.kernel}")
    metrics: dict[str, float | str] = {"correctness": args.correctness}
    if args.latency_us is not None:
        metrics["latency_us"] = args.latency_us
    if args.throughput_tflops is not None:
        metrics["throughput_tflops"] = args.throughput_tflops
    if args.run_log:
        metrics.update(_parse_metrics(Path(args.run_log).read_text(encoding="utf-8")))
    if args.results_tsv:
        metrics.update(_metrics_from_tsv(Path(args.results_tsv), str(item["kernel_type"])))
    suggestion = update_item(item, metrics, args.keep)
    save_state(state_path, state)
    print(json.dumps(suggestion, indent=2))


if __name__ == "__main__":
    main()
