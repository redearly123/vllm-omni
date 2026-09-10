# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Merge per-stage startup numbers from a bench_startup.py log with its BENCH_JSON line.

The stage engine cores run in subprocesses and report weight-loading / init timings
only through log lines such as:

  (StageEngineCoreProc_stage0_replica0 pid=1) INFO ... Loading weights took 39.66 seconds
  (StageEngineCoreProc_stage0_replica0 pid=1) INFO ... Model loading took 21.45 GiB memory and 40.59 seconds
  (StageEngineCoreProc_stage0_replica0 pid=1) INFO ... init engine (profile, create kv cache, warmup model) took 7.00 s
  (StageEngineCoreProc_stage0_replica0 pid=1) INFO ... Filesystem type for checkpoints: NFS4.
      Checkpoint size: 34.52 GiB. Available RAM: 22.76 GiB.
  INFO ... [Omni] AsyncOmniEngine initialized in 159.74 seconds

Usage:
  python benchmarks/mammoth_moda2/parse_startup_log.py run.log [more.log ...] [--markdown] [--json-out summary.json]
"""

import argparse
import json
import re
import sys
from typing import Any

STAGE = r"\(StageEngineCoreProc_stage(?P<stage>\d+)_replica\d+ pid=\d+\)"
PATTERNS = {
    "weights_load_s": re.compile(STAGE + r".*Loading weights took (?P<v>[\d.]+) seconds"),
    "model_load_s": re.compile(STAGE + r".*Model loading took (?P<mem>[\d.]+) GiB memory and (?P<v>[\d.]+) seconds"),
    "engine_init_s": re.compile(
        STAGE + r".*init engine \(profile, create kv cache, warmup model\) took (?P<v>[\d.]+) s"
    ),
    # Only present when the stage does not run with enforce_eager.
    "graph_capture_s": re.compile(
        STAGE + r".*Graph capturing finished in (?P<v>[\d.]+) secs, took (?P<mem>[\d.]+) GiB"
    ),
    "compile_s": re.compile(STAGE + r".*torch\.compile takes (?P<v>[\d.]+) s in total"),
    "ckpt_fs": re.compile(
        STAGE
        + r".*Filesystem type for checkpoints: (?P<fs>\w+)\. Checkpoint size: (?P<size>[\d.]+) GiB\."
        + r" Available RAM: (?P<ram>[\d.]+) GiB"
    ),
}
ENGINE_READY = re.compile(r"AsyncOmniEngine initialized in (?P<v>[\d.]+) seconds")
PREFETCH_SKIP = re.compile(STAGE + r".*exceeds 90% of available RAM")
COMPILE_UNSUPPORTED = re.compile(STAGE + r".*`torch\.compile` is turned on, but the model .* does not support it")

# Startup timeline: (milestone key, regex). Timestamps in vllm logs are "MM-DD HH:MM:SS" (1 s resolution).
TS = re.compile(r"(?:INFO|WARNING|ERROR) (?P<ts>\d\d-\d\d \d\d:\d\d:\d\d)")
MILESTONES = [
    ("omni_init_start", re.compile(r"\[Omni\] Initializing with model")),
    ("stage{stage}_launch", re.compile(r"Stage-(?P<stage>\d+) set runtime devices")),
    ("stage{stage}_proc_up", re.compile(STAGE + r".*world_size=\d+ rank=\d+")),
    ("stage{stage}_load_start", re.compile(STAGE + r".*Starting to load model")),
    ("stage{stage}_load_done", re.compile(STAGE + r".*Loading weights took")),
    ("stage{stage}_init_done", re.compile(STAGE + r".*init engine \(profile, create kv cache, warmup model\) took")),
    ("stage{stage}_ready", re.compile(r"\[StageRuntime\] Stage (?P<stage>\d+) initialized")),
    ("engine_ready", re.compile(r"AsyncOmniEngine initialized in")),
]


def _ts_seconds(ts: str) -> int:
    _, hms = ts.split(" ")
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def timeline(lines: list[str]) -> dict:
    """Return milestone timestamps (s, relative to omni_init_start) and derived per-stage phase durations."""
    marks: dict[str, int] = {}
    for line in lines:
        tsm = TS.search(line)
        if not tsm:
            continue
        for key, pat in MILESTONES:
            m = pat.search(line)
            if m:
                k = key.format(stage=m.group("stage")) if "{stage}" in key else key
                marks.setdefault(k, _ts_seconds(tsm.group("ts")))
                break
    if "omni_init_start" not in marks:
        return {}
    t0 = marks["omni_init_start"]
    rel = {k: v - t0 for k, v in marks.items()}
    phases: dict[str, Any] = {}
    names = ("launch", "proc_up", "load_start", "load_done", "init_done", "ready")
    for s in sorted({k.split("_")[0] for k in rel if k.startswith("stage")}):
        found = {name: rel.get(f"{s}_{name}") for name in names}
        if any(v is None for v in found.values()):
            continue
        ts = {name: int(v) for name, v in found.items() if v is not None}
        phases[s] = {
            "spawn_import_config_s": ts["proc_up"] - ts["launch"],
            "device_dist_init_s": ts["load_start"] - ts["proc_up"],
            "weights_load_s": ts["load_done"] - ts["load_start"],
            "profile_kv_warmup_s": ts["init_done"] - ts["load_done"],
            "ready_handoff_s": ts["ready"] - ts["init_done"],
            "total_s": ts["ready"] - ts["launch"],
        }
    if "engine_ready" in rel:
        last_ready = max((v for k, v in rel.items() if k.endswith("_ready") and k != "engine_ready"), default=0)
        phases["post_stages_wiring_s"] = rel["engine_ready"] - last_ready
        phases["engine_ready_s"] = rel["engine_ready"]
    return {"marks_rel_s": rel, "phases": phases}


def parse(path: str) -> dict:
    stages: dict[str, dict] = {}
    rec: dict[str, Any] = {"log": path, "stages": stages}
    with open(path, errors="replace") as f:
        lines = f.readlines()
    rec["timeline"] = timeline(lines)
    for line in lines:
        if line.startswith("BENCH_JSON "):
            rec["bench"] = json.loads(line[len("BENCH_JSON ") :])
            continue
        m = ENGINE_READY.search(line)
        if m:
            rec["engine_ready_s_reported"] = float(m.group("v"))
            continue
        m = PREFETCH_SKIP.search(line)
        if m:
            stages.setdefault(m.group("stage"), {})["page_cache_prefetch_skipped"] = True
            continue
        m = COMPILE_UNSUPPORTED.search(line)
        if m:
            stages.setdefault(m.group("stage"), {})["torch_compile_unsupported"] = True
            continue
        for key, pat in PATTERNS.items():
            m = pat.search(line)
            if not m:
                continue
            st = stages.setdefault(m.group("stage"), {})
            if key == "ckpt_fs":
                st["checkpoint_fs"] = m.group("fs")
                st["checkpoint_size_gib"] = float(m.group("size"))
                st["available_ram_gib"] = float(m.group("ram"))
            elif key == "graph_capture_s":
                st["graph_capture_s"] = float(m.group("v"))
                st["graph_capture_mem_gib"] = float(m.group("mem"))
            elif key == "model_load_s":
                st["model_load_s"] = float(m.group("v"))
                st["model_load_mem_gib"] = float(m.group("mem"))
                # "Model loading took" covers model construction + weight loading; "Loading weights took" only
                # the weights, so their difference is the model-construction (model_init) time.
                if "weights_load_s" in st:
                    st["model_init_s"] = round(st["model_load_s"] - st["weights_load_s"], 2)
            else:
                st[key] = float(m.group("v"))
            break
    return rec


def to_markdown(recs: list[dict]) -> str:
    rows = []
    hdr = [
        "label",
        "model fs",
        "stage",
        "ckpt read (GiB)",
        "model init (s)",
        "weights load (s)",
        "load mem (GiB)",
        "compile (s)",
        "graph capture (s)",
        "engine init (s)",
    ]
    rows.append("| " + " | ".join(hdr) + " |")
    rows.append("|" + "---|" * len(hdr))
    for r in recs:
        b = r.get("bench", {})
        label = b.get("label", r["log"])
        fs = b.get("model_fs", "?")
        for sid, st in sorted(r["stages"].items()):
            rows.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    label,
                    fs,
                    sid,
                    st.get("checkpoint_size_gib", ""),
                    st.get("model_init_s", ""),
                    st.get("weights_load_s", ""),
                    st.get("model_load_mem_gib", ""),
                    "unsupported" if st.get("torch_compile_unsupported") else st.get("compile_s", ""),
                    st.get("graph_capture_s", ""),
                    st.get("engine_init_s", ""),
                )
            )
    rows.append("")
    hdr2 = [
        "label",
        "imports (s)",
        "engine ready (s)",
        "first request (s)",
        "steady request (s)",
        "process→first image (s)",
        "host RSS peak (GiB)",
        "GPU used peak (GiB)",
    ]
    rows.append("| " + " | ".join(hdr2) + " |")
    rows.append("|" + "---|" * len(hdr2))
    for r in recs:
        b = r.get("bench", {})
        s = b.get("summary_s", {})
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                b.get("label", r["log"]),
                s.get("imports", ""),
                s.get("engine_init", ""),
                s.get("first_request", ""),
                s.get("steady_request_avg", ""),
                s.get("time_to_first_image_from_process_start", ""),
                s.get("host_rss_peak_gib", ""),
                s.get("gpu_used_peak_gib", ""),
            )
        )
    rows.append("")
    hdr3 = [
        "label",
        "stage",
        "spawn+import (s)",
        "device/dist init (s)",
        "weights load (s)",
        "profile/kv/warmup (s)",
        "stage total (s)",
    ]
    rows.append("| " + " | ".join(hdr3) + " |")
    rows.append("|" + "---|" * len(hdr3))
    for r in recs:
        b = r.get("bench", {})
        ph = (r.get("timeline") or {}).get("phases", {})
        for sid, p in sorted((k, v) for k, v in ph.items() if k.startswith("stage")):
            rows.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    b.get("label", r["log"]),
                    sid,
                    p["spawn_import_config_s"],
                    p["device_dist_init_s"],
                    p["weights_load_s"],
                    p["profile_kv_warmup_s"],
                    p["total_s"],
                )
            )
        if "post_stages_wiring_s" in ph:
            rows.append(
                "| {} | wiring after last stage | | | | | {} |".format(
                    b.get("label", r["log"]), ph["post_stages_wiring_s"]
                )
            )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    recs = [parse(p) for p in args.logs]
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(recs, f, indent=2)
    if args.markdown:
        print(to_markdown(recs))
    else:
        json.dump(recs, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
