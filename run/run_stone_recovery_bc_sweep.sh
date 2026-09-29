#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
RUN_ID=${MINESTUDIO_BC_RUN_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
OUTPUT_DIRECTORY=${MINESTUDIO_BC_OUTPUT_DIRECTORY:-$PROJECT_DIRECTORY/output/stone_recovery_bc/runs/$RUN_ID}
RUN_LOG=$OUTPUT_DIRECTORY/sweep.log
COMPLETE_MARKER=$OUTPUT_DIRECTORY/sweep.complete
RUN_STATE=$OUTPUT_DIRECTORY/run_state.json

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
mkdir -p "$OUTPUT_DIRECTORY"
ln -sfn "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/output/stone_recovery_bc/latest_run"
rm -f "$COMPLETE_MARKER"
export MINESTUDIO_BC_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY"
export MINESTUDIO_BC_TELEGRAM=1

python -m pip freeze > "$OUTPUT_DIRECTORY/packages.txt"
nvidia-smi > "$OUTPUT_DIRECTORY/nvidia-smi.txt"
python - <<'PY' > "$OUTPUT_DIRECTORY/environment.json"
import json
import platform

import torch


value = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
}
print(json.dumps(value, indent=2))
PY

write_sweep_state() {
    local status=$1
    local variant=${2:-null}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"%s","variant":"%s","pid":%s,"updated_at":"%s"}\n' "$status" "$RUN_ID" "$variant" "$$" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_sweep_state failed "${variant:-unknown}"
    fi
}

on_signal() {
    write_sweep_state interrupted "${variant:-unknown}"
    exit 143
}

trap on_exit EXIT
trap on_signal TERM INT
write_sweep_state running none

for variant in action_head action_head_recurrent upper_lora; do
    write_sweep_state running "$variant"
    echo "{\"status\":\"variant_start\",\"variant\":\"$variant\",\"timestamp\":\"$(date --iso-8601=seconds)\"}" | tee -a "$RUN_LOG"
    MINESTUDIO_BC_VARIANT="$variant" python run/train_stone_recovery_bc.py 2>&1 | tee -a "$RUN_LOG"
done

date --iso-8601=seconds > "$COMPLETE_MARKER"
write_sweep_state completed none
echo "{\"status\":\"sweep_complete\",\"timestamp\":\"$(cat "$COMPLETE_MARKER")\"}" | tee -a "$RUN_LOG"
