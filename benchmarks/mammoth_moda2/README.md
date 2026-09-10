# MammothModa2 startup / loading benchmark

Measures how long a MammothModa2 (AR → DiT) deployment takes to go from process
start to the first image, broken down by stage and phase. Tracks the
"Startup and loading" item of
[#7075](https://github.com/vllm-project/vllm-omni/issues/7075).

## What is measured

`bench_startup.py` runs a single fresh process and records the wall-clock phases
it can see; the stage engine cores are subprocesses, so their internal phases are
recovered by `parse_startup_log.py` from log lines the engine already emits
(1 s timestamp resolution). Boundaries:

| metric | from -> to | contains |
| --- | --- | --- |
| imports | process start -> `vllm_omni` imported | Python imports in the benchmark process |
| time to ready | `Omni()` call -> return | every stage spawned, loaded, profiled and warmed up (`AsyncOmniEngine initialized`) |
| spawn + import (per stage) | `Stage-N set runtime devices` -> first stage log line (`world_size=`) | subprocess spawn, imports, config |
| device init (per stage) | -> `Starting to load model` | device / distributed init, model runner setup |
| model init (per stage) | `Model loading took` minus `Loading weights took` | model construction, parameter allocation |
| weight load (per stage) | `Starting to load model` -> `Loading weights took` | shard read, dtype conversion, copy to device; not separable from logs, see `raw_load_bench.py` |
| profile + KV + warmup (per stage) | -> `init engine (profile, create kv cache, warmup model) took` | one merged interval |
| torch.compile (per stage) | `torch.compile takes N s in total` | only when the stage runs without `enforce_eager` and the model supports compilation; the parser flags `torch.compile is turned on, but the model ... does not support it` |
| CUDA graph capture (per stage) | `Graph capturing finished in N secs, took M GiB` | only without `enforce_eager`; part of the profile + KV + warmup interval |
| first request / subsequent | `generate()` wall time, plus vllm-omni `stage_metrics` | request 1 vs requests 2..N of the same process |
| host RSS peak, GPU used peak | sampled once a second by the benchmark | process tree RSS; `nvidia-smi` memory in use |

With `parallel_stage_init` the stages overlap, so per-stage intervals are
reported side by side and never summed.

`raw_load_bench.py` reads shards with `safetensors`, materializes every tensor
on the CPU (so `read` touches all pages), converts to bf16 on the CPU or on the
GPU after an fp32 copy, and synchronizes the device after each GPU step. It
isolates the loading mechanism; its ratios are not the service-level speedup.

`bench_storage_scenarios.sh` runs the benchmark warm and after requesting
page-cache eviction (`posix_fadvise(DONTNEED)` on every checkpoint file;
`drop_caches` when available). Eviction is a request, not a guarantee.

## Usage

```bash
# single run (keep the log; the per-stage breakdown is parsed from it)
python benchmarks/mammoth_moda2/bench_startup.py \
    --model /path/MammothModa2-Preview \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
    --height 1024 --width 1024 --seed 42 \
    --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
    --repeat 2 --label my-run --output-json my-run.json 2>&1 | tee my-run.log
python benchmarks/mammoth_moda2/parse_startup_log.py my-run.log --markdown

# storage scenarios
MODEL_NFS=/nfs/MammothModa2-Preview MODEL_LOCAL=/local/MammothModa2-Preview OUT_DIR=./startup-bench \
    bash benchmarks/mammoth_moda2/bench_storage_scenarios.sh

# raw safetensors -> GPU micro-benchmark, no vLLM (--mode cpu_cast | gpu_cast | pinned_gpu_cast)
python benchmarks/mammoth_moda2/raw_load_bench.py --model /path/MammothModa2-Preview --shards 6,7,8 --mode cpu_cast
```

`--parallel-stage-init` passes `parallel_stage_init=True` to `Omni()` so the
stages that share a GPU initialize concurrently (see the A/B below).

`--save-image` writes the first request's image; with a fixed `--seed` it is
bit-identical across runs and can be used as a cheap correctness check when
comparing loading changes (`md5sum`).

## Reference results: 1× A800-SXM4-80GB, 16-CPU cgroup quota

vLLM 0.28.0+cu129, torch 2.13.0+cu129, vllm-omni `34aa1d2a`, driver 580.126,
Ubuntu 22.04 container with a cgroup quota of 16 CPUs on a 128-core host
(`torch.get_num_threads()` = 64 by default), checkpoint on local NVMe (34.5 GiB
in 8 shards, 1010 of 1391 tensors stored in float32). Committed deploy config
(eager, `gpu_memory_utilization` 0.5 / 0.3). Preview text-to-image, seed 42.

Design: 2×2 of torch thread count (default vs `OMP_NUM_THREADS=16` = quota) ×
stage initialization (serial vs `parallel_stage_init`), 3 rounds with the four
configurations rotated inside each round; then 3 runs each after requested
page-cache eviction and after removing the FlashInfer JIT cache. Startup runs
send one 512×512 / 8-step request after startup as a smoke check; memory peaks
come from one extra run per cell with sampling enabled.

### Time to ready

`Omni()` call → return, mean ± std over 3 independent processes. Add ~10 s of
Python imports before `Omni()` for the process-start-to-ready figure.

| configuration | time to ready | host RSS peak | GPU used peak |
| --- | --- | --- | --- |
| default threads (64), serial | 61.0 ± 0.4 s | 12.8 GiB | 45.7 GiB |
| 16 threads, serial | 57.3 ± 0.3 s | 12.4 GiB | 45.7 GiB |
| default threads, `parallel_stage_init` | 42.1 ± 1.0 s | 15.7 GiB | 44.8 GiB |
| **16 threads, `parallel_stage_init`** | **36.2 ± 0.3 s** | 15.3 GiB | 44.8 GiB |
| default, serial, page cache evicted | 80.8 ± 0.3 s | | |
| default, serial, FlashInfer JIT cache removed | 129.1 ± 0.2 s | | |
| default, serial, first-ever start on the machine (n=1) | 150.6 s | | |

### Where startup goes, per stage (seconds, mean of 3)

| phase | AR, control | DiT, control | AR, 16 threads + parallel | DiT, 16 threads + parallel |
| --- | --- | --- | --- | --- |
| spawn + import | 11.0 | 11.0 | 11.0 | 11.0 |
| device init | 6.3 | 6.3 | 6.0 | 6.0 |
| model init | 0.4 | 0.2 | 0.5 | 0.2 |
| weight load | 6.1 | 3.7 | 5.2 | 1.5 |
| profile + KV + warmup (merged) | 5.0 | 2.0 | 5.3 | 3.0 |
| stage total | 29.3 | 24.3 | 29.0 | 22.0 |
| orchestrator wiring after the last stage | 6.0 | | 6.0 | |

Serial: the stage totals add up (29.3 + 24.3 + 6.0 ≈ 61 s). Parallel: both
stages log `set runtime devices` in the same second, DiT reports ready first,
and time to ready is the AR stage plus wiring (29.0 + 6.0 ≈ 36 s). Weight
loads that run concurrently cost each other ~1 s. Page-cache eviction shows up
only in weight load (AR 18.9 s, DiT 10.1 s). Removing the FlashInfer cache shows
up only in the AR stage's profile/warmup interval (74.3 s instead of 5.0 s):
the sampling kernels are JIT-built with ninja on first use and cached under
`~/.cache/flashinfer`.

### Where the weight-load time goes (`raw_load_bench.py`, DiT shards 6–8, 13.3 GiB fp32, n=3)

| path | read | cast | H2D | total |
| --- | --- | --- | --- | --- |
| CPU cast, 64 threads | 0.12 s | 2.01 ± 0.22 s | 2.06 ± 0.19 s | 5.36 ± 0.09 s |
| CPU cast, 16 threads | 0.03 s | 0.53 ± 0.01 s | 1.19 ± 0.03 s | 2.64 ± 0.04 s |
| fp32 to GPU, cast on GPU | 0.02 s | 0.04 s | 2.84 ± 0.02 s | 3.34 ± 0.02 s |

### Requests (1024×1024, 50 steps, guidance 4.0; one process, 16 threads)

| request | latency | AR stage | DiT stage |
| --- | --- | --- | --- |
| 1 | 95.2 s | 77.9 s (4,161 visual tokens, TPOT 18.7 ms) | 17.1 s |
| 2 | 94.9 s | 77.6 s | 17.2 s |
| 3 | 94.6 s | 77.3 s | 17.1 s |

### Correctness

All 23 startup runs produce the same 512×512 image (one md5); the 1024×1024
request produces one image for its three requests.

### AR stage without `enforce_eager` (CUDA graphs; the change proposed in #7319)

Same machine, `OMP_NUM_THREADS=16`, serial stage init, a copy of the deploy
config with `enforce_eager: false` for stage 0 only. vLLM logs
`torch.compile is turned on, but the model ... does not support it`, so this
enables CUDA-graph capture (28 piecewise + 15 decode shapes) and no
compilation. "Cold" removes `~/.cache/vllm/torch_compile_cache` and
`/tmp/torchinductor_root` before the run.

| configuration | n | time to ready | AR profile + capture + warmup | graph capture | AR KV cache |
| --- | --- | --- | --- | --- | --- |
| eager (control, same session) | 1 | 57.6 s | 6.0 s | - | 15.7 GiB, 147,376 tokens |
| AR graphs, caches warm | 3 | 62.5 ± 0.1 s | 10.0 ± 0.0 s | 4.0 s, 0.20 GiB | 2.0 GiB, 19,088 tokens |
| AR graphs, caches cold | 3 | 66.8 ± 0.9 s | 14.3 ± 0.5 s | 4.0 s, 0.20 GiB | 2.0 GiB |
| AR graphs + `parallel_stage_init` | 1 | fails: `No available memory for the cache blocks` | | | |

Requests (1024×1024, 50 steps, AR graphs, warm): 77.3 / 77.1 / 77.1 s, AR
60.0 s (TPOT 14.4 ms instead of 18.7 ms), DiT 17.2 s; eager was 95.2 / 94.9 /
94.6 s. The AR stage's non-KV memory as measured by vLLM's profiling grows from
21.9 GiB to 35.4 GiB with graphs, which is what shrinks the KV cache at
`gpu_memory_utilization` 0.5 and leaves no room for the DiT stage to initialize
concurrently. Output is deterministic within graph mode (6 runs, one md5) but
not bit-identical to eager: 512×512 images differ in 83 % of pixels (mean
|diff| 3.5/255, PSNR 31.3 dB), 1024×1024 in 87 % (mean 4.3/255, PSNR 25.0 dB).

## Observations

1. **`parallel_stage_init` is the largest and cheapest win**: 61.0 → 42.1 s
   (−31 %). It raises the host RSS peak by ~3 GiB and leaves the GPU peak
   unchanged, well inside the deploy config's 0.8 budget. Output is identical.
2. **Matching torch's thread count to the cgroup quota** saves 6 % here
   (61.0 → 57.3 s) and stacks with the parallel init (36.2 s, −41 % in total).
   The cost is the fp32→bf16 cast of the checkpoint: 2.0 s → 0.5 s in the
   micro-benchmark, DiT weight load 3.7 → 1.2 s. The penalty grows with the
   oversubscription ratio (64 threads on 16 CPUs here); on a worker with a
   4-CPU quota the same fix cut the AR weight load from 38 s to 5 s and time
   to ready from 160 s to 82 s. vLLM's `startup_omp_num_threads()` handles this
   on the `MultiprocExecutor` path; the single-GPU stages here run through
   `UniProcExecutor`, where nothing sets it.
3. **Page cache**: reading the 34.5 GiB checkpoint from local NVMe costs ~20 s
   when evicted (80.8 vs 61.0 s).
4. **Compile warmup**: under the committed eager config the only one is
   FlashInfer's JIT, ~68 s more in the AR stage's warmup on the first start on
   a machine (129.1 vs 61.0 s). With `enforce_eager: false` on the AR stage
   (#7319) CUDA-graph capture adds 4 s (plus ~4 s the first time after the
   inductor/Triton caches are cleared); torch.compile itself is not supported
   by the model yet, so there is no compilation warmup to measure.
5. **No first-request overhead**: 95.2 s vs 94.9 / 94.6 s for the next two.
   The AR stage is 82 % of a request.
6. **What is left after both fixes** (36 s): 11 s spawn + import and 6 s
   device init per stage (overlapped), 5 s AR profile/warmup, 6 s orchestrator
   wiring, plus ~10 s of imports in the client process. Imports are the next
   target.
7. Restricting the DiT stage to the three shards it needs did not help on the
   earlier worker (−1 %, n=3): the loader converts only the tensors it keeps.
8. **CUDA graphs on the AR stage trade memory for speed**: AR 77.9 → 60.0 s per
   request, but the stage's non-KV memory grows by 13.5 GiB, the KV cache
   shrinks from 15.7 to 2.0 GiB at the committed 0.5 budget, and
   `parallel_stage_init` no longer fits. Output is deterministic but differs
   from eager (PSNR 25–31 dB). Raising the AR budget or keeping serial init is
   needed if #7319 lands with this deploy config.
