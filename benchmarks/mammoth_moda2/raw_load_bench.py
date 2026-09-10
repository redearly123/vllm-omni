# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Raw safetensors -> GPU micro-benchmark, without vLLM's loader.

Separates storage read, dtype conversion and host-to-device copy from any loader
overhead, and compares where the fp32 -> bf16 cast runs:

  --mode cpu_cast         read -> cast to bf16 on the CPU -> copy to GPU   (what vLLM's loader does)
  --mode gpu_cast         read -> copy fp32 to GPU -> cast on the GPU
  --mode pinned_gpu_cast  read -> pin -> async copy fp32 to GPU -> cast on the GPU

  python benchmarks/mammoth_moda2/raw_load_bench.py --model /path/MammothModa2-Preview --shards 6,7,8
  OMP_NUM_THREADS=4 python benchmarks/mammoth_moda2/raw_load_bench.py --model /path --shards 6,7,8 --mode cpu_cast
"""

import argparse
import json
import os
import re
import time

import torch
from safetensors import safe_open

SHARD_NO = re.compile(r"-(\d+)-of-\d+\.safetensors$")


def shard_files(model: str, shards: str) -> list[str]:
    """Resolve ``--shards`` to files, refusing empty or partial matches so a typo cannot pass as a fast load."""
    files = sorted(f for f in os.listdir(model) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit(f"no .safetensors files under {model}")
    if shards != "all":
        want = {int(x) for x in shards.split(",")}
        found = {int(m.group(1)): f for f in files if (m := SHARD_NO.search(f))}
        missing = sorted(want - found.keys())
        if missing:
            raise SystemExit(f"shard(s) {missing} not found under {model}; available: {sorted(found)}")
        files = [found[n] for n in sorted(want)]
    return [os.path.join(model, f) for f in files]


def sync() -> None:
    if torch.cuda.is_available():
        torch.accelerator.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shards", default="all", help="comma list of 1-based shard numbers, or 'all'")
    ap.add_argument("--mode", choices=("cpu_cast", "gpu_cast", "pinned_gpu_cast"), default="cpu_cast")
    ap.add_argument("--label", default="raw")
    ap.add_argument("--no-cuda", action="store_true", help="read + CPU cast only (mode must be cpu_cast)")
    a = ap.parse_args()
    if a.no_cuda and a.mode != "cpu_cast":
        ap.error("--no-cuda only makes sense with --mode cpu_cast")

    paths = shard_files(a.model, a.shards)
    total_bytes = sum(os.path.getsize(p) for p in paths)

    t0 = time.perf_counter()
    n_tensors, cpu_bytes = 0, 0
    dtypes: dict[str, int] = {}
    t_read = t_cast = t_h2d = 0.0
    for p in paths:
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                r0 = time.perf_counter()
                t = f.get_tensor(k)  # mmap -> materialize on CPU
                t_read += time.perf_counter() - r0
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
                cpu_bytes += t.numel() * t.element_size()
                needs_cast = t.is_floating_point() and t.dtype != torch.bfloat16
                if a.mode == "cpu_cast":
                    c0 = time.perf_counter()
                    if needs_cast:
                        t = t.to(torch.bfloat16)
                    t_cast += time.perf_counter() - c0
                    if not a.no_cuda:
                        h0 = time.perf_counter()
                        g = t.cuda()
                        sync()
                        t_h2d += time.perf_counter() - h0
                        del g
                else:
                    h0 = time.perf_counter()
                    if a.mode == "pinned_gpu_cast":
                        g = t.pin_memory().cuda(non_blocking=True)
                    else:
                        g = t.cuda()
                    sync()
                    t_h2d += time.perf_counter() - h0
                    c0 = time.perf_counter()
                    if needs_cast:
                        g = g.to(torch.bfloat16)
                    sync()
                    t_cast += time.perf_counter() - c0
                    del g
                n_tensors += 1
                del t
    total = time.perf_counter() - t0
    res = {
        "label": a.label,
        "mode": a.mode,
        "files": [os.path.basename(p) for p in paths],
        "file_bytes_gib": round(total_bytes / 2**30, 2),
        "tensor_bytes_gib": round(cpu_bytes / 2**30, 2),
        "n_tensors": n_tensors,
        "dtypes": dtypes,
        "torch_threads": torch.get_num_threads(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "read_s": round(t_read, 2),
        "cast_s": round(t_cast, 2),
        "h2d_s": round(t_h2d, 2),
        "total_s": round(total, 2),
        "read_gbps": round(total_bytes / 1e9 / max(t_read, 1e-9), 2),
        "total_gbps": round(total_bytes / 1e9 / total, 2),
    }
    print("RAWLOAD_JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
