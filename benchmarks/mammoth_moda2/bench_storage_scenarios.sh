#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Run bench_startup.py under controlled storage / page-cache scenarios and summarize.
#
#   MODEL_NFS=/nfs/MammothModa2-Preview MODEL_LOCAL=/local/MammothModa2-Preview \
#   OUT_DIR=./startup-bench bash benchmarks/mammoth_moda2/bench_storage_scenarios.sh
#
# Scenarios (SCENARIOS env, space separated; default: all that have a model path):
#   local-cold  local-warm  nfs-cold  nfs-warm
# "cold" evicts the model files from the page cache first. /proc/sys/vm/drop_caches is
# usually read-only inside containers, so we fall back to posix_fadvise(DONTNEED) per file,
# which needs no privileges.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=${OUT_DIR:-./startup-bench}
DEPLOY_CONFIG=${DEPLOY_CONFIG:-vllm_omni/deploy/mammoth_moda2.yaml}
STEPS=${STEPS:-50}
SIZE=${SIZE:-1024}
REPEAT=${REPEAT:-2}
SEED=${SEED:-42}
EXTRA=${EXTRA:-'{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": '$STEPS'}'}
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/json" "$OUT_DIR/images"

evict_page_cache() {  # $1 = model dir
  sync
  if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    echo "[bench] page cache dropped (drop_caches)"; return
  fi
  python - "$1" <<'EOF'
import os, sys
root, n = sys.argv[1], 0
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            n += 1
        except OSError as e:
            print("[bench] fadvise failed", p, e)
print(f"[bench] page cache evicted via posix_fadvise(DONTNEED) for {n} files under {root}")
EOF
}

run_one() {  # $1 = label, $2 = model dir
  local label=$1 model=$2
  echo "[bench] === $label  model=$model  $(date +%T)"
  python "$HERE/bench_startup.py" --model "$model" --deploy-config "$DEPLOY_CONFIG" \
    --height "$SIZE" --width "$SIZE" --seed "$SEED" --extra-body "$EXTRA" --repeat "$REPEAT" \
    --label "$label" --output-json "$OUT_DIR/json/$label.json" --save-image "$OUT_DIR/images/$label.png" \
    > "$OUT_DIR/logs/$label.log" 2>&1
  echo "[bench] === $label exit=$? $(date +%T)"
  grep -E "^BENCH " "$OUT_DIR/logs/$label.log"
}

default_scenarios=""
[ -n "${MODEL_LOCAL:-}" ] && default_scenarios="$default_scenarios local-cold local-warm"
[ -n "${MODEL_NFS:-}" ] && default_scenarios="$default_scenarios nfs-cold nfs-warm"
for s in ${SCENARIOS:-$default_scenarios}; do
  case $s in
    local-cold) evict_page_cache "$MODEL_LOCAL"; run_one local-cold "$MODEL_LOCAL" ;;
    local-warm) run_one local-warm "$MODEL_LOCAL" ;;
    nfs-cold)   evict_page_cache "$MODEL_NFS";   run_one nfs-cold "$MODEL_NFS" ;;
    nfs-warm)   run_one nfs-warm "$MODEL_NFS" ;;
    *) echo "[bench] unknown scenario: $s" ;;
  esac
done
python "$HERE/parse_startup_log.py" "$OUT_DIR"/logs/*.log --markdown --json-out "$OUT_DIR/summary.json" | tee "$OUT_DIR/summary.md"
echo BENCH_ALL_DONE
