"""Bench adapter for mizorewww/laya-mlx (see bench/PLAN.md, "Adapter interface").

Variants set dtype, TF32 and multi-question handling. The environment switch for TF32 is
read once by MLX (a static in mlx/utils.h), so it is applied before mlx is imported.
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
CONTENDER = "laya-mlx"
MODELS = ["typed-decisions", "multilingual"]

# name -> Agent kwargs and environment. "tf32": None leaves MLX's default (TF32 on for fp32
# matmuls on M5 with macOS >= 26.2), "0" forces full fp32.
VARIANTS = {
    "mlx-fp16": dict(
        kw=dict(dtype="float16"),
        tf32=None,
        notes="Library default path: fp16 weights and compute, questions batched in one "
        "forward (batch_size=16), padded to the longest row, no mx.compile.",
    ),
    "mlx-fp16-opt": dict(
        kw=dict(dtype="float16", compile=True, pad_to_multiple=16, cache_prompts=True),
        tf32=None,
        notes="Opt-in fast path (laya-snake --optimize): mx.compile, rows padded to a multiple "
        "of 16, bounded question-prefix token cache (128 entries; helps when questions repeat).",
    ),
    "mlx-fp16-c512": dict(
        kw=dict(dtype="float16"),
        tf32=None,
        cache_mb=512,
        notes="mlx-fp16 with mx.set_cache_limit(512 MiB). Added for the sys1rust spike.",
    ),
    "mlx-fp16-opt-c512": dict(
        kw=dict(dtype="float16", compile=True, pad_to_multiple=16, cache_prompts=True),
        tf32=None,
        cache_mb=512,
        notes="mlx-fp16-opt with mx.set_cache_limit(512 MiB). Added for the sys1rust spike: "
        "MLX's default cache limit lets freed buffers grow to about the size of RAM.",
    ),
    "mlx-fp16-perq": dict(
        kw=dict(dtype="float16", batch_size=1),
        tf32=None,
        notes="fp16, one forward per question (batch_size=1), no padding across questions.",
    ),
    "mlx-bf16": dict(
        kw=dict(dtype="bfloat16"),
        tf32=None,
        notes="bf16 weights and compute, batched. Not in laya-mlx's published validation matrix.",
    ),
    "mlx-fp32": dict(
        kw=dict(dtype="float32"),
        tf32=None,
        notes="fp32 weights, batched, MLX default on M5: fp32 matmuls run as TF32 on the "
        "GPU neural accelerators.",
    ),
    "mlx-fp32-notf32": dict(
        kw=dict(dtype="float32"),
        tf32="0",
        notes="fp32, batched, MLX_ENABLE_TF32=0 (full fp32 matmuls).",
    ),
    "mlx-fp32-notf32-perq": dict(
        kw=dict(dtype="float32", batch_size=1),
        tf32="0",
        notes="fp32, MLX_ENABLE_TF32=0, one forward per question (batch_size=1).",
    ),
}


def process_start_time():
    """Unix start time of this process from sysctl(KERN_PROC_PID); `run` execs python, so
    this is when `run` was launched. Falls back to now."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        mib = (ctypes.c_int * 4)(1, 14, 1, os.getpid())  # CTL_KERN, KERN_PROC, KERN_PROC_PID
        buf = ctypes.create_string_buffer(1024)
        size = ctypes.c_size_t(len(buf))
        if libc.sysctl(mib, 4, buf, ctypes.byref(size), None, 0) != 0 or size.value < 16:
            raise OSError
        sec = int.from_bytes(buf.raw[0:8], "little", signed=True)
        usec = int.from_bytes(buf.raw[8:12], "little", signed=True)
        return sec + usec / 1e6
    except Exception:
        return time.time()


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def list_variants():
    out = []
    for name, v in VARIANTS.items():
        env = "MLX_ENABLE_TF32=0" if v["tf32"] == "0" else "MLX default env"
        out.append(
            {
                "variant": name,
                "models": MODELS,
                "mode": "inproc",
                "max_state_tokens": 1024,
                "notes": v["notes"] + f" [{env}; model max_len 1024 incl. question, longer input "
                "is truncated like upstream]",
            }
        )
    print(json.dumps(out, indent=1))


def read_workload(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    t_process_start = process_start_time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-variants", action="store_true")
    ap.add_argument("--variant")
    ap.add_argument("--model")
    ap.add_argument("--workload")
    ap.add_argument("--out")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()
    if args.list_variants:
        list_variants()
        return 0
    for need in ("variant", "model", "workload", "out"):
        if getattr(args, need) is None:
            ap.error(f"--{need} is required")
    if args.variant not in VARIANTS:
        ap.error(f"unknown variant {args.variant}; see --list-variants")
    if args.model not in MODELS:
        ap.error(f"model {args.model} not supported by {CONTENDER}; supported: {MODELS}")
    if args.concurrency != 1:
        ap.error("--concurrency applies to http mode only; all laya-mlx variants are inproc")
    v = VARIANTS[args.variant]

    if v["tf32"] is None:
        os.environ.pop("MLX_ENABLE_TF32", None)
    else:
        os.environ["MLX_ENABLE_TF32"] = v["tf32"]

    lock = json.loads((BENCH / "models.lock.json").read_text())[args.model]
    requests = read_workload(args.workload)
    out = open(args.out, "w")

    t0 = time.perf_counter()
    import mlx.core as mx  # noqa: E402  (after the TF32 switch)

    import laya_mlx  # noqa: E402

    if v.get("cache_mb"):
        mx.set_cache_limit(v["cache_mb"] << 20)
    agent = laya_mlx.load(lock["repo"], revision=lock["sha"], **v["kw"])
    load_ms = (time.perf_counter() - t0) * 1000.0

    meta = {
        "type": "meta",
        "contender": CONTENDER,
        "variant": args.variant,
        "model": args.model,
        "model_sha": lock["sha"],
        "code_version": git_sha(HERE / "src"),
        "backend": "mlx",
        "mode": "inproc",
        "load_ms": round(load_ms, 3),
        "pid": os.getpid(),
        "t_process_start": t_process_start,
        "device": str(agent.device),
        "dtype": v["kw"]["dtype"],
        "batch_size": agent.batch_size,
        "mlx_version": mx.__version__,
        "mlx_enable_tf32_env": os.environ.get("MLX_ENABLE_TF32"),
        "mlx_cache_limit_mb": v.get("cache_mb"),
        "weights_source": f"{lock['repo']}@{lock['sha']} original safetensors, converted to MLX in memory at load (no export step)",
    }
    out.write(json.dumps(meta) + "\n")
    out.flush()

    def run_one(req):
        body = req["body"]
        t_start = time.time()
        p0 = time.perf_counter()
        res = agent.predict(body.get("state", ""), body["questions"])
        latency_ms = (time.perf_counter() - p0) * 1000.0
        return t_start, latency_ms, res["answers"]

    for req in requests[: max(0, args.warmup)]:
        try:
            run_one(req)
        except Exception:
            pass

    def emit(req, repeat):
        try:
            t_start, latency_ms, answers = run_one(req)
            rec = {"type": "result", "id": req["id"], "repeat": repeat, "t_start": t_start,
                   "latency_ms": round(latency_ms, 4), "answers": answers,
                   "mlx_mb": {"active": mx.get_active_memory() >> 20, "cache": mx.get_cache_memory() >> 20,
                              "peak": mx.get_peak_memory() >> 20}}
        except Exception as e:  # noqa: BLE001
            rec = {"type": "result", "id": req["id"], "repeat": repeat, "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.duration:
        deadline = time.monotonic() + args.duration
        repeat = 0
        while time.monotonic() < deadline:
            for req in requests:
                if time.monotonic() >= deadline:
                    break
                emit(req, repeat)
            repeat += 1
    else:
        for repeat in range(args.repeats):
            for req in requests:
                emit(req, repeat)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
