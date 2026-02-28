"""Benchmark original ONNX weights vs quantized models for NDLOCR-Lite.

Primary use in this repository:
- Measure CPU latency of original ONNX weights.
- Generate INT8 ONNX weights (dynamic quantization) and compare speed/size.
- Optionally benchmark existing Android TFLite models if they are present.
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import tempfile
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

try:
    import tensorflow as tf
    HAS_TFLITE = True
except Exception:
    tf = None
    HAS_TFLITE = False


MODEL_DIR = os.path.dirname(__file__)


@dataclass
class BenchResult:
    name: str
    path: str
    size_mb: float
    mean_ms: float
    p95_ms: float


def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def _fix_dim(dim: object) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    return 1


def _random_input_for_ort(session: ort.InferenceSession) -> Dict[str, np.ndarray]:
    inputs: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = tuple(_fix_dim(d) for d in inp.shape)
        typ = inp.type
        if "int64" in typ:
            arr = np.random.randint(0, 100, shape, dtype=np.int64)
        elif "int32" in typ:
            arr = np.random.randint(0, 100, shape, dtype=np.int32)
        elif "float16" in typ:
            arr = np.random.random(shape).astype(np.float16)
        else:
            arr = np.random.random(shape).astype(np.float32)
        inputs[inp.name] = arr
    return inputs


def benchmark_onnx(path: str, runs: int, warmup: int, threads: int) -> BenchResult:
    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    feed = _random_input_for_ort(session)

    for _ in range(max(1, warmup)):
        session.run(None, feed)

    latencies_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return BenchResult(
        name=os.path.basename(path),
        path=path,
        size_mb=model_size_mb(path),
        mean_ms=statistics.mean(latencies_ms),
        p95_ms=float(np.percentile(latencies_ms, 95)),
    )


def quantize_int8_onnx(fp32_path: str, out_path: str) -> str:
    # Work around non-ASCII TEMP path issues in onnxruntime shape-inference temp files.
    safe_tmp = os.path.join(MODEL_DIR, ".tmp_ort_quant")
    os.makedirs(safe_tmp, exist_ok=True)
    prev_temp = tempfile.gettempdir()
    prev_tmp_env = os.environ.get("TMP")
    prev_temp_env = os.environ.get("TEMP")
    os.environ["TMP"] = safe_tmp
    os.environ["TEMP"] = safe_tmp
    tempfile.tempdir = safe_tmp
    try:
        quantize_dynamic(
            model_input=fp32_path,
            model_output=out_path,
            op_types_to_quantize=["MatMul", "Gemm"],
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    finally:
        tempfile.tempdir = None
        if prev_tmp_env is None:
            os.environ.pop("TMP", None)
        else:
            os.environ["TMP"] = prev_tmp_env
        if prev_temp_env is None:
            os.environ.pop("TEMP", None)
        else:
            os.environ["TEMP"] = prev_temp_env
        # Restore process-level cached tmp dir if possible.
        if os.path.isdir(prev_temp):
            tempfile.tempdir = prev_temp
    return out_path


def find_matching_tflite(onnx_path: str, tflite_dir: str) -> str | None:
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    candidates = [
        f"{stem}.tflite",
        f"{stem}_int8.tflite",
        f"{stem}_android_int8.tflite",
        f"{stem}_fp16.tflite",
    ]
    for name in candidates:
        path = os.path.join(tflite_dir, name)
        if os.path.exists(path):
            return path
    return None


def benchmark_tflite(path: str, runs: int, warmup: int, threads: int) -> BenchResult:
    if not HAS_TFLITE:
        raise RuntimeError("TensorFlow/TFLite runtime unavailable in this environment.")

    interpreter = tf.lite.Interpreter(model_path=path, num_threads=max(1, threads))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    tensors = {}
    for inp in input_details:
        shape = tuple(int(max(1, d)) for d in inp["shape"])
        dtype = inp["dtype"]
        if dtype == np.int64:
            arr = np.random.randint(0, 100, shape, dtype=np.int64)
        elif dtype == np.int32:
            arr = np.random.randint(0, 100, shape, dtype=np.int32)
        elif dtype == np.int8:
            arr = np.random.randint(-128, 127, shape, dtype=np.int8)
        elif dtype == np.uint8:
            arr = np.random.randint(0, 255, shape, dtype=np.uint8)
        elif dtype == np.float16:
            arr = np.random.random(shape).astype(np.float16)
        else:
            arr = np.random.random(shape).astype(np.float32)
        tensors[inp["index"]] = arr

    for _ in range(max(1, warmup)):
        for idx, arr in tensors.items():
            interpreter.set_tensor(idx, arr)
        interpreter.invoke()

    latencies_ms = []
    for _ in range(runs):
        for idx, arr in tensors.items():
            interpreter.set_tensor(idx, arr)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return BenchResult(
        name=os.path.basename(path),
        path=path,
        size_mb=model_size_mb(path),
        mean_ms=statistics.mean(latencies_ms),
        p95_ms=float(np.percentile(latencies_ms, 95)),
    )


def list_fp32_onnx(model_dir: str, pattern: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(model_dir, pattern)))
    return [p for p in files if not p.endswith("_int8.onnx")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NDLOCR-Lite ONNX weights vs quantized variants."
    )
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory containing model files.")
    parser.add_argument("--onnx-pattern", default="*.onnx", help="Glob for ONNX models.")
    parser.add_argument("--runs", type=int, default=50, help="Measured inference runs.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs before timing.")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads (applied to ONNX and TFLite runtime).",
    )
    parser.add_argument(
        "--make-int8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate and benchmark dynamic INT8 ONNX models.",
    )
    parser.add_argument(
        "--rebuild-int8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate *_int8.onnx even if file already exists.",
    )
    parser.add_argument(
        "--benchmark-tflite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark matching .tflite files if found.",
    )
    parser.add_argument(
        "--tflite-dir",
        default=None,
        help="Directory to search for .tflite models (default: --model-dir).",
    )
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 88)
    print(text)
    print("=" * 88)


def main() -> None:
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    tflite_dir = os.path.abspath(args.tflite_dir or model_dir)
    fp32_models = list_fp32_onnx(model_dir, args.onnx_pattern)

    if not fp32_models:
        print(f"No ONNX models found in {model_dir} using pattern {args.onnx_pattern}")
        return

    print_header("NDLOCR-Lite Model Benchmark (CPU)")
    print(f"Model dir:   {model_dir}")
    print(f"TFLite dir:  {tflite_dir}")
    print(f"Runs:        {args.runs} (warmup={args.warmup})")
    print(f"Threads:     {args.threads}")
    print(f"TensorFlow:  {'available' if HAS_TFLITE else 'unavailable'}")

    summary_rows = []

    for fp32_path in fp32_models:
        name = os.path.basename(fp32_path)
        print_header(f"Model: {name}")

        fp32_res = benchmark_onnx(fp32_path, runs=args.runs, warmup=args.warmup, threads=args.threads)
        print(
            f"[FP32 ONNX] size={fp32_res.size_mb:.2f} MB  "
            f"mean={fp32_res.mean_ms:.2f} ms  p95={fp32_res.p95_ms:.2f} ms"
        )

        int8_res = None
        if args.make_int8:
            int8_path = os.path.splitext(fp32_path)[0] + "_int8.onnx"
            if args.rebuild_int8 or not os.path.exists(int8_path):
                print(f"Generating INT8 ONNX: {os.path.basename(int8_path)}")
                quantize_int8_onnx(fp32_path, int8_path)
            int8_res = benchmark_onnx(int8_path, runs=args.runs, warmup=args.warmup, threads=args.threads)
            print(
                f"[INT8 ONNX] size={int8_res.size_mb:.2f} MB  "
                f"mean={int8_res.mean_ms:.2f} ms  p95={int8_res.p95_ms:.2f} ms"
            )

        tflite_res = None
        if args.benchmark_tflite:
            tflite_path = find_matching_tflite(fp32_path, tflite_dir)
            if tflite_path and HAS_TFLITE:
                tflite_res = benchmark_tflite(tflite_path, runs=args.runs, warmup=args.warmup, threads=args.threads)
                print(
                    f"[TFLITE   ] size={tflite_res.size_mb:.2f} MB  "
                    f"mean={tflite_res.mean_ms:.2f} ms  p95={tflite_res.p95_ms:.2f} ms"
                )
            elif tflite_path and not HAS_TFLITE:
                print(f"[TFLITE   ] found but TensorFlow runtime unavailable: {tflite_path}")
            else:
                print("[TFLITE   ] matching file not found (skip)")

        summary_rows.append((name, fp32_res, int8_res, tflite_res))

    print_header("Summary")
    print(
        f"{'Model':<48} {'Type':<12} {'Size(MB)':>10} {'Mean(ms)':>10} {'P95(ms)':>10} {'vs FP32':>10}"
    )
    print("-" * 88)
    for name, fp32_res, int8_res, tflite_res in summary_rows:
        print(
            f"{name:<48} {'FP32 ONNX':<12} "
            f"{fp32_res.size_mb:>10.2f} {fp32_res.mean_ms:>10.2f} {fp32_res.p95_ms:>10.2f} {'1.00x':>10}"
        )
        if int8_res:
            speedup = fp32_res.mean_ms / max(1e-9, int8_res.mean_ms)
            print(
                f"{'':<48} {'INT8 ONNX':<12} "
                f"{int8_res.size_mb:>10.2f} {int8_res.mean_ms:>10.2f} {int8_res.p95_ms:>10.2f} {speedup:>10.2f}x"
            )
        if tflite_res:
            speedup = fp32_res.mean_ms / max(1e-9, tflite_res.mean_ms)
            print(
                f"{'':<48} {'TFLite':<12} "
                f"{tflite_res.size_mb:>10.2f} {tflite_res.mean_ms:>10.2f} {tflite_res.p95_ms:>10.2f} {speedup:>10.2f}x"
            )


if __name__ == "__main__":
    main()
