#!/usr/bin/env python3
"""CPU-only ONNX Runtime benchmark script for Jetson Nano 2GB."""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import psutil
from tabulate import tabulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX models on CPU-only ONNX Runtime (Jetson Nano focus)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing .onnx files.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of timed inference runs per model (default: 50).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs before timing starts (default: 10).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to save benchmark results as CSV.",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    return args


def ensure_cpu_provider() -> None:
    available = ort.get_available_providers()
    if "CPUExecutionProvider" not in available:
        raise RuntimeError(
            "CPUExecutionProvider is not available in ONNX Runtime. "
            f"Available providers: {available}"
        )


def sanitize_dim(dim: onnx.TensorShapeProto.Dimension) -> int:
    if dim.HasField("dim_value") and dim.dim_value > 0:
        return int(dim.dim_value)
    return 1


def extract_input_shape_and_name(model_path: Path) -> Tuple[str, List[int], str]:
    model = onnx.load(str(model_path))
    graph = model.graph

    if not graph.input:
        raise RuntimeError(f"Model {model_path.name} has no graph inputs.")

    input_value_info = graph.input[0]
    input_name = input_value_info.name
    tensor_type = input_value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise RuntimeError(f"Input shape is unavailable for model {model_path.name}.")

    dims = [sanitize_dim(d) for d in tensor_type.shape.dim]
    if not dims:
        raise RuntimeError(f"Input shape has no dimensions for model {model_path.name}.")

    resolution = "x".join(str(d) for d in dims)
    return input_name, dims, resolution


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def benchmark_model(
    model_path: Path, runs: int, warmup: int, process: psutil.Process
) -> Dict[str, float]:
    baseline_rss_mb = process.memory_info().rss / (1024 * 1024)

    input_name, input_shape, resolution = extract_input_shape_and_name(model_path)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(
            f"Model {model_path.name}: expected only CPUExecutionProvider, got {session.get_providers()}"
        )

    post_load_rss_mb = process.memory_info().rss / (1024 * 1024)

    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    feed = {input_name: dummy_input}

    for _ in range(warmup):
        session.run(None, feed)

    latencies_ms: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feed)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    post_inference_rss_mb = process.memory_info().rss / (1024 * 1024)

    lat = np.array(latencies_ms, dtype=np.float64)
    return {
        "Archivo": model_path.name,
        "Resolución": resolution,
        "Runs": runs,
        "Mean": float(np.mean(lat)),
        "Std": float(np.std(lat, ddof=0)),
        "Min": float(np.min(lat)),
        "Max": float(np.max(lat)),
        "P50": percentile(lat, 50),
        "P95": percentile(lat, 95),
        "RAM baseline": baseline_rss_mb,
        "RAM post-load": post_load_rss_mb,
        "RAM post-inference": post_inference_rss_mb,
        "Delta load": post_load_rss_mb - baseline_rss_mb,
        "Delta inference": post_inference_rss_mb - baseline_rss_mb,
    }


def format_float(value: float) -> float:
    return round(value, 3)


def print_results(rows: Sequence[Dict[str, float]]) -> None:
    latency_table = []
    memory_table = []

    for row in rows:
        latency_table.append(
            [
                row["Archivo"],
                row["Resolución"],
                int(row["Runs"]),
                format_float(row["Mean"]),
                format_float(row["Std"]),
                format_float(row["Min"]),
                format_float(row["Max"]),
                format_float(row["P50"]),
                format_float(row["P95"]),
            ]
        )
        memory_table.append(
            [
                row["Archivo"],
                row["Resolución"],
                format_float(row["RAM baseline"]),
                format_float(row["RAM post-load"]),
                format_float(row["RAM post-inference"]),
                format_float(row["Delta load"]),
                format_float(row["Delta inference"]),
            ]
        )

    print("\n=== Table 1 — Latency (ms) ===")
    print(
        tabulate(
            latency_table,
            headers=["Archivo", "Resolución", "Runs", "Mean", "Std", "Min", "Max", "P50", "P95"],
            tablefmt="github",
        )
    )

    print("\n=== Table 2 — Memory (MB RSS) ===")
    print(
        tabulate(
            memory_table,
            headers=[
                "Archivo",
                "Resolución",
                "RAM baseline",
                "RAM post-load",
                "RAM post-inference",
                "Delta load",
                "Delta inference",
            ],
            tablefmt="github",
        )
    )

    print("\n=== Requirements Check ===")
    for row in rows:
        latency_ok = row["Mean"] <= 10000.0
        ram_ok = row["RAM post-inference"] <= 2048.0
        print(f"\nModel: {row['Archivo']} ({row['Resolución']})")
        print(
            f"SYS-REQ-001: latency mean ≤ 10000ms -> "
            f"{'PASS' if latency_ok else 'FAIL'} (actual: {format_float(row['Mean'])} ms)"
        )
        print(
            f"SW-REQ-006: RAM total ≤ 2048MB -> "
            f"{'PASS' if ram_ok else 'FAIL'} (actual: {format_float(row['RAM post-inference'])} MB)"
        )

    print(
        "\nWARNING: These measurements were collected with Jetson Nano CPU-only "
        "(CPUExecutionProvider) and constitute the formal benchmark for the thesis."
    )


def save_csv(rows: Sequence[Dict[str, float]], csv_path: Path) -> None:
    import csv

    fieldnames = [
        "Archivo",
        "Resolución",
        "Runs",
        "Mean",
        "Std",
        "Min",
        "Max",
        "P50",
        "P95",
        "RAM baseline",
        "RAM post-load",
        "RAM post-inference",
        "Delta load",
        "Delta inference",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    ensure_cpu_provider()

    model_dir = Path(args.input)
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found or not a directory: {model_dir}")

    model_files = sorted(model_dir.glob("*.onnx"))
    if not model_files:
        raise FileNotFoundError(f"No .onnx files found in: {model_dir}")

    process = psutil.Process(os.getpid())
    all_results = []

    for model_path in model_files:
        result = benchmark_model(model_path, args.runs, args.warmup, process)
        all_results.append(result)

    print_results(all_results)

    if args.csv:
        save_csv(all_results, Path(args.csv))
        print(f"\nSaved CSV results to: {args.csv}")


if __name__ == "__main__":
    main()
