#!/usr/bin/env python3
"""Create a theory-state file for AutoKernel experiments.

This is the first integration layer, not a replacement for SOLAR. It accepts
AutoKernel's profile report, optimization plan, or a small kernel manifest,
computes a Roofline lower bound, and optionally merges predictions exported by
SOLAR. Keeping this adapter independent of the SOLAR runtime makes it usable
even when only the copied source trees are present.

Example:

    python integration/solar_bridge.py init \
        --manifest integration/example_manifest.json \
        --arch H100_PCIe \
        --output workspace/theory_state.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DTYPE_BYTES = {
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
    "float64": 8,
    "fp64": 8,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
}


@dataclass(frozen=True)
class ArchSpec:
    name: str
    peak_tflops: float
    bandwidth_gbps: float
    launch_overhead_us: float = 5.0
    source: str = "approximate bridge default; use SOLAR's arch config for calibration"


# These are intentionally approximate starting points.  For a serious run,
# pass --arch-config with the exact SOLAR architecture YAML used for prediction.
DEFAULT_ARCHES = {
    "H100_PCIe": ArchSpec("H100_PCIe", 756.0, 2000.0),
    "A100": ArchSpec("A100", 312.0, 1555.0),
    "RTX4090": ArchSpec("RTX4090", 330.0, 1008.0),
    "B200": ArchSpec("B200", 2250.0, 8000.0),
}


def _read_structured(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise SystemExit(f"{path} is not JSON; install PyYAML to read YAML") from exc
        return yaml.safe_load(text)


def _first_number(obj: Any, keys: Iterable[str]) -> float | None:
    wanted = {key.lower() for key in keys}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in wanted and isinstance(value, (int, float)):
                return float(value)
        for value in obj.values():
            found = _first_number(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _first_number(value, keys)
            if found is not None:
                return found
    return None


def _load_arch(args: argparse.Namespace) -> ArchSpec:
    if args.arch_config:
        raw = _read_structured(Path(args.arch_config))
        peak = _first_number(raw, ("peak_tflops", "peak_flops_tflops", "fp16_tflops"))
        bandwidth = _first_number(raw, ("bandwidth_gbps", "memory_bandwidth_gbps", "peak_bandwidth_gbps"))
        launch = _first_number(raw, ("launch_overhead_us", "kernel_launch_us")) or args.launch_overhead_us
        if peak is None or bandwidth is None:
            raise SystemExit("architecture config must contain peak_tflops and bandwidth_gbps")
        return ArchSpec(args.arch, peak, bandwidth, launch, str(args.arch_config))

    if args.arch not in DEFAULT_ARCHES:
        choices = ", ".join(sorted(DEFAULT_ARCHES))
        raise SystemExit(f"unknown --arch {args.arch!r}; choose one of {choices} or use --arch-config")
    base = DEFAULT_ARCHES[args.arch]
    return ArchSpec(base.name, base.peak_tflops, base.bandwidth_gbps, args.launch_overhead_us, base.source)


def _shape_values(shape: Any) -> list[int]:
    if isinstance(shape, str):
        return [int(x) for x in re.findall(r"\d+", shape)]
    if isinstance(shape, (list, tuple)):
        return [int(x) for x in shape]
    if isinstance(shape, dict):
        preferred = ("batch", "heads", "seq", "seq_len", "hidden", "dim", "M", "N", "K")
        values = [int(shape[key]) for key in preferred if key in shape and isinstance(shape[key], (int, float))]
        if values:
            return values
        return [int(value) for value in shape.values() if isinstance(value, (int, float))]
    return []


def _matmul_dims(shape: Any) -> tuple[int, int, int] | None:
    """Recover M,K,N from common AutoKernel shape_info strings."""
    if isinstance(shape, dict):
        for keys in (("m", "k", "n"), ("M", "K", "N")):
            if all(key in shape for key in keys):
                return int(shape[keys[0]]), int(shape[keys[1]]), int(shape[keys[2]])
    if isinstance(shape, str):
        groups = re.findall(r"\[([^\[\]]+)\]", shape)
        parsed = []
        for group in groups:
            values = [int(x) for x in re.findall(r"\d+", group)]
            if values:
                parsed.append(values)
        if len(parsed) >= 2 and len(parsed[0]) >= 2 and len(parsed[1]) >= 2:
            return parsed[0][-2], parsed[0][-1], parsed[1][-1]
    return None


def _estimate_work(kernel: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """Return (FLOPs, bytes, reason), preferring explicit manifest values."""
    if kernel.get("flops") is not None and kernel.get("bytes") is not None:
        return float(kernel["flops"]), float(kernel["bytes"]), "manifest"

    kind = str(kernel.get("kernel_type", kernel.get("type", "unknown"))).lower()
    shape = _shape_values(kernel.get("shape", []))
    dtype_bytes = DTYPE_BYTES.get(str(kernel.get("dtype", "float16")).lower(), 2)
    elements = math.prod(shape) if shape else 0

    if kind == "matmul":
        dims = _matmul_dims(kernel.get("shape"))
        if dims is None and len(shape) >= 3:
            dims = tuple(shape[-3:])  # type: ignore[assignment]
        if dims is not None:
            m, k, n = dims
            return 2.0 * m * k * n, (m * k + k * n + m * n) * dtype_bytes, "matmul template"
    if kind in {"softmax", "layernorm", "rmsnorm", "cross_entropy", "rotary_embedding", "reduce"} and elements:
        # Conservative traffic estimates for bandwidth-bound kernels.  Users
        # can override these with explicit flops/bytes in the manifest.
        multipliers = {
            "softmax": (5.0, 3.0),
            "layernorm": (8.0, 4.0),
            "rmsnorm": (5.0, 3.0),
            "cross_entropy": (8.0, 3.0),
            "rotary_embedding": (8.0, 3.0),
            "reduce": (1.0, 2.0),
        }
        flop_multiplier, byte_multiplier = multipliers[kind]
        return elements * flop_multiplier, elements * dtype_bytes * byte_multiplier, f"{kind} template"
    if kind in {"flash_attention", "attention", "fused_mlp"} and len(shape) >= 2:
        # These operations vary substantially by implementation.  Return a
        # deliberately low-confidence estimate unless explicit work is given.
        return None, None, "explicit flops/bytes required for this kernel type"
    return None, None, "explicit flops/bytes required"


def _find_solar_prediction(kernel: dict[str, Any], solar_perf_dir: Path | None) -> float | None:
    if kernel.get("solar_latency_us") is not None:
        return float(kernel["solar_latency_us"])
    if not solar_perf_dir:
        return None
    explicit = kernel.get("solar_perf_file")
    if explicit:
        candidate = Path(str(explicit))
        if not candidate.is_absolute():
            candidate = solar_perf_dir / candidate
        if candidate.exists():
            raw = _read_structured(candidate)
            return _first_number(raw, ("predicted_latency_us", "theoretical_latency_us", "latency_us"))
    kernel_id = str(kernel.get("kernel_id", kernel.get("id", "kernel")))
    candidates = list(solar_perf_dir.glob(f"*{kernel_id}*.yaml")) + list(solar_perf_dir.glob(f"*{kernel_id}*.json"))
    for candidate in candidates:
        raw = _read_structured(candidate)
        value = _first_number(raw, ("predicted_latency_us", "theoretical_latency_us", "latency_us"))
        if value is not None:
            return value
    return None


def _normalise_manifest(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("kernels", raw.get("items", []))
    if not isinstance(raw, list):
        raise SystemExit("manifest must be a JSON list or {\"kernels\": [...]} object")
    result = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"manifest item {index} is not an object")
        item = dict(item)
        item.setdefault("kernel_id", item.get("id", f"kernel_{index}"))
        item.setdefault("kernel_type", item.get("type", "unknown"))
        result.append(item)
    return result


def _manifest_from_profile(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate AutoKernel's profile_report.json into bridge records."""
    kernels = raw.get("top_kernels", [])
    if not isinstance(kernels, list):
        raise SystemExit("profile report has no top_kernels list")
    manifest: list[dict[str, Any]] = []
    for index, entry in enumerate(kernels, start=1):
        if not isinstance(entry, dict):
            continue
        op_type = str(entry.get("op_type", "unknown"))
        # AutoKernel's profile report gives avg_time_us per profiler event.
        # Keep it as a diagnostic only: profiler event time follows a
        # different timing convention than bench.py's standalone latency,
        # so it must not seed baseline_latency_us/best_latency_us.  Seeding
        # it would let the advisor declare MOVE_ON (gap <= 10%) before a
        # single real benchmark has run.
        profile_avg_us = entry.get("avg_time_us")
        if profile_avg_us is None and entry.get("gpu_time_ms") is not None:
            profile_avg_us = float(entry["gpu_time_ms"]) * 1000.0
        roofline = str(entry.get("roofline", "")).lower()
        bottleneck = "compute" if "compute" in roofline else "memory" if "memory" in roofline else "unknown"
        manifest.append(
            {
                "kernel_id": f"kernel_{op_type}_{entry.get('rank', index)}",
                "kernel_type": op_type,
                "shape": entry.get("shape_info"),
                "profile_fraction": float(entry.get("pct_total", 0.0)) / 100.0,
                "profile_avg_time_us": profile_avg_us,
                "bottleneck": bottleneck,
            }
        )
    return manifest


def _manifest_from_plan(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate AutoKernel's optimization_plan.json into bridge records."""
    kernels = raw.get("kernels_to_optimize", raw.get("kernels", []))
    if not isinstance(kernels, list):
        raise SystemExit("optimization plan has no kernels_to_optimize list")
    manifest: list[dict[str, Any]] = []
    for index, entry in enumerate(kernels, start=1):
        if not isinstance(entry, dict):
            continue
        op_type = str(entry.get("op_type", "unknown"))
        # `gpu_time_ms` in optimization_plan.json is aggregated profiler time,
        # not the standalone latency that bench.py measures.  Do not use it as
        # a baseline unless the producer explicitly supplied
        # `baseline_latency_us`.
        baseline = entry.get("baseline_latency_us")
        manifest.append(
            {
                "kernel_id": f"kernel_{op_type}_{entry.get('rank', index)}",
                "kernel_type": op_type,
                "shape": entry.get("model_shape", entry.get("shape")),
                "profile_fraction": float(entry.get("pct_total", 0.0)) / 100.0,
                "baseline_latency_us": baseline,
            }
        )
    return manifest


def build_state(args: argparse.Namespace) -> dict[str, Any]:
    profile_raw: dict[str, Any] | None = None
    if args.profile:
        loaded = _read_structured(Path(args.profile))
        if not isinstance(loaded, dict):
            raise SystemExit("AutoKernel profile report must be a JSON object")
        profile_raw = loaded
    if args.manifest:
        manifest = _normalise_manifest(_read_structured(Path(args.manifest)))
    elif args.plan:
        loaded = _read_structured(Path(args.plan))
        if not isinstance(loaded, dict):
            raise SystemExit("AutoKernel optimization plan must be a JSON object")
        manifest = _manifest_from_plan(loaded)
        # The plan has exact parsed model shapes; the profile has exact GPU
        # peak numbers and profiler bottleneck labels. Merge both when given.
        if profile_raw:
            prof_by_key = {
                (str(item.get("kernel_type")), str(item.get("kernel_id", "")).rsplit("_", 1)[-1]): item
                for item in _manifest_from_profile(profile_raw)
            }
            for item in manifest:
                key = (str(item["kernel_type"]), str(item["kernel_id"]).rsplit("_", 1)[-1])
                profile_item = prof_by_key.get(key)
                if profile_item:
                    for field in ("profile_fraction", "bottleneck"):
                        if profile_item.get(field) is not None:
                            item[field] = profile_item[field]
    elif profile_raw:
        manifest = _manifest_from_profile(profile_raw)
    else:
        raise SystemExit("provide --manifest, --profile, or --plan")

    # Prefer the exact GPU peaks captured by AutoKernel's profiler when no
    # explicit SOLAR architecture config was supplied.
    if profile_raw and not args.arch_config:
        peak = profile_raw.get("gpu_peak_tflops_fp16")
        bandwidth = profile_raw.get("gpu_peak_bandwidth_gb_s")
        if isinstance(peak, (int, float)) and isinstance(bandwidth, (int, float)) and peak > 0 and bandwidth > 0:
            arch = ArchSpec(
                str(profile_raw.get("gpu_name", args.arch)),
                float(peak),
                float(bandwidth),
                args.launch_overhead_us,
                f"AutoKernel profile: {args.profile}",
            )
        else:
            arch = _load_arch(args)
    else:
        arch = _load_arch(args)
    solar_dir = Path(args.solar_perf_dir) if args.solar_perf_dir else None
    kernels: dict[str, Any] = {}

    for kernel in manifest:
        kernel_id = str(kernel["kernel_id"])
        flops, bytes_, estimate_reason = _estimate_work(kernel)
        solar_latency = _find_solar_prediction(kernel, solar_dir)
        if solar_latency is not None:
            theory_latency = solar_latency
            theory_source = "SOLAR prediction"
            bottleneck = str(kernel.get("bottleneck", "unknown"))
        elif flops is not None and bytes_ is not None:
            compute_us = flops / (arch.peak_tflops * 1e12) * 1e6
            memory_us = bytes_ / (arch.bandwidth_gbps * 1e9) * 1e6
            theory_latency = max(compute_us, memory_us) + arch.launch_overhead_us
            bottleneck = "compute" if compute_us >= memory_us else "memory"
            theory_source = f"Roofline ({estimate_reason})"
        else:
            theory_latency = None
            bottleneck = "unknown"
            theory_source = estimate_reason

        # Preserve AutoKernel's profiler classification when it is available;
        # the template estimate is only a fallback for missing metadata.
        provided_bottleneck = str(kernel.get("bottleneck", "")).lower()
        if provided_bottleneck in {"compute", "memory"}:
            bottleneck = provided_bottleneck

        baseline = kernel.get("baseline_latency_us", kernel.get("measured_latency_us"))
        baseline = float(baseline) if baseline is not None else None
        # A baseline is recorded for comparison, but it must not immediately
        # become the floor: doing so would tell the advisor that the very first
        # implementation is already optimal.  A calibrated floor is used only
        # when the manifest explicitly supplies a calibration factor learned
        # from one or more known kernels.
        calibration_factor = kernel.get("calibration_factor")
        calibration_factor = float(calibration_factor) if calibration_factor is not None else None
        calibrated_floor = theory_latency * calibration_factor if theory_latency and calibration_factor else theory_latency
        if baseline is not None and calibrated_floor is not None:
            # A modelled lower bound cannot be slower than an observed
            # implementation.  Clamp inconsistent architecture estimates and
            # retain the raw theory value above for diagnostics.
            calibrated_floor = min(float(calibrated_floor), baseline)

        kernels[kernel_id] = {
            "kernel_id": kernel_id,
            "kernel_type": kernel["kernel_type"],
            "shape": kernel.get("shape"),
            "dtype": kernel.get("dtype", "float16"),
            "profile_fraction": float(kernel.get("profile_fraction", 0.0)),
            "flops": flops,
            "bytes": bytes_,
            "bottleneck": bottleneck,
            "theory_latency_us": theory_latency,
            "calibrated_floor_us": calibrated_floor,
            "theory_source": theory_source,
            "baseline_latency_us": baseline,
            "best_latency_us": baseline,
            "calibration_factor": calibration_factor,
            "experiments": 0,
            "improvements": 0,
            "plateau_count": 0,
            "strategy_index": 0,
            "last_action": "INIT",
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": asdict(arch),
        "kernels": kernels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="build theory_state.json from a manifest or AutoKernel profile")
    init.add_argument("--manifest", help="JSON/YAML kernel manifest")
    init.add_argument("--profile", help="AutoKernel workspace/profile_report.json")
    init.add_argument("--plan", help="AutoKernel workspace/optimization_plan.json")
    init.add_argument("--output", default="workspace/theory_state.json")
    init.add_argument("--arch", default="H100_PCIe")
    init.add_argument("--arch-config", help="SOLAR-compatible YAML/JSON architecture config")
    init.add_argument("--solar-perf-dir", help="directory containing SOLAR perf_*.yaml files")
    init.add_argument("--launch-overhead-us", type=float, default=5.0)
    args = parser.parse_args()

    state = build_state(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    for item in state["kernels"].values():
        theory = item["theory_latency_us"]
        theory_text = f"{theory:.3f} us" if theory is not None else "n/a"
        print(f"{item['kernel_id']:<28} {item['bottleneck']:<7} theory={theory_text}")


if __name__ == "__main__":
    main()
