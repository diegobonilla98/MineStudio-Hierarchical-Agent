#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
TRAINING_DIRECTORY=$PROJECT_DIRECTORY/output/stone_recovery_bc/architecture_sweeps/20260917T213751Z
TASK_BANK=$TRAINING_DIRECTORY/gemini_task_bank.json
RUN_ID=${MINESTUDIO_SCREEN_RUN_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
SCREEN_DIRECTORY=${MINESTUDIO_SCREEN_DIRECTORY:-$PROJECT_DIRECTORY/output/stone_acquisition/behavioral_screens/$RUN_ID}
RUN_STATE=$SCREEN_DIRECTORY/run_state.json
POLICIES=(vanilla upper_lora_r8_step800 upper_lora_r8_step1600 upper_lora_r8_step3200 upper_lora_r8_step4800 upper_lora_r8_step6400 upper_lora_r32_step6400 recurrent_upper_lora_step6400)
WORKERS=2

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
mkdir -p "$SCREEN_DIRECTORY"
ln -sfn "$SCREEN_DIRECTORY" "$PROJECT_DIRECTORY/output/stone_acquisition/latest_behavioral_screen"
export MINESTUDIO_SCREEN_DIRECTORY="$SCREEN_DIRECTORY"

notify() {
    TELEGRAM_MESSAGE="$1" python - <<'PY' || true
import os

from telegram_training_helper import TelegramBot


TelegramBot(poll_updates=False, enable_terminal_commands=False, drop_pending_updates=False).send_message(text=os.environ["TELEGRAM_MESSAGE"])
PY
}

write_state() {
    local status=$1
    local phase=${2:-none}
    local split=${3:-none}
    local policy=${4:-none}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"%s","phase":"%s","split":"%s","policy":"%s","pid":%s,"screen_directory":"%s","updated_at":"%s"}\n' "$status" "$RUN_ID" "$phase" "$split" "$policy" "$$" "$SCREEN_DIRECTORY" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed "${phase:-unknown}" "${split:-unknown}" "${policy:-unknown}"
        notify "MineStudio behavioral screen failed
phase=${phase:-unknown}
split=${split:-unknown}
policy=${policy:-unknown}
output=$SCREEN_DIRECTORY"
    fi
}

on_signal() {
    write_state interrupted "${phase:-unknown}" "${split:-unknown}" "${policy:-unknown}"
    exit 143
}

trap on_exit EXIT
trap on_signal TERM INT

checkpoint_for() {
    case "$1" in
        vanilla) echo "$PROJECT_DIRECTORY/checkpoints/steve_one_official" ;;
        upper_lora_r8_step800) echo "$TRAINING_DIRECTORY/upper_lora/snapshots/step_800_model" ;;
        upper_lora_r8_step1600) echo "$TRAINING_DIRECTORY/upper_lora/snapshots/step_1600_model" ;;
        upper_lora_r8_step3200) echo "$TRAINING_DIRECTORY/upper_lora/snapshots/step_3200_model" ;;
        upper_lora_r8_step4800) echo "$TRAINING_DIRECTORY/upper_lora/snapshots/step_4800_model" ;;
        upper_lora_r8_step6400) echo "$TRAINING_DIRECTORY/upper_lora/snapshots/step_6400_model" ;;
        upper_lora_r32_step6400) echo "$TRAINING_DIRECTORY/upper_lora_rank32/best_model" ;;
        recurrent_upper_lora_step6400) echo "$TRAINING_DIRECTORY/recurrent_upper_lora/best_model" ;;
    esac
}

run_evaluation() {
    split=$1
    policy=$2
    episodes=$3
    seed=$4
    manifest=$5
    phase=evaluation
    write_state running "$phase" "$split" "$policy"
    experiment=screen_${split}_${policy}_v1
    checkpoint=$(checkpoint_for "$policy")
    MINESTUDIO_STONE_CHECKPOINT="$checkpoint" \
    MINESTUDIO_STONE_EXPERIMENT="$experiment" \
    MINESTUDIO_STONE_EPISODES="$episodes" \
    MINESTUDIO_STONE_WORKERS="$WORKERS" \
    MINESTUDIO_STONE_BASE_SEED="$seed" \
    MINESTUDIO_STONE_MANIFEST_KIND="$manifest" \
    MINESTUDIO_STONE_STAGE="Paired behavioral screening" \
    MINESTUDIO_STONE_EVENT_TASKS=1 \
    MINESTUDIO_STONE_TASK_BANK="$TASK_BANK" \
    python run/benchmark_stone_acquisition.py 2>&1 | tee -a "$SCREEN_DIRECTORY/${split}_${policy}.log"
}

phase=materialize
split=none
policy=upper_lora
write_state running "$phase" "$split" "$policy"
MINESTUDIO_BC_OUTPUT_DIRECTORY="$TRAINING_DIRECTORY" MINESTUDIO_BC_VARIANT=upper_lora MINESTUDIO_BC_MATERIALIZE_STEPS=800,1600,3200,4800,6400 python run/materialize_stone_recovery_snapshots.py 2>&1 | tee "$SCREEN_DIRECTORY/materialize.log"

python -m pip freeze > "$SCREEN_DIRECTORY/packages.txt"
nvidia-smi > "$SCREEN_DIRECTORY/nvidia-smi.txt"
cp "$TASK_BANK" "$SCREEN_DIRECTORY/gemini_task_bank.json"
notify "MineStudio paired behavioral screen started
policies=${#POLICIES[@]}
normal=40 each
hazard=60 each
output=$SCREEN_DIRECTORY"

for split in normal hazard; do
    if test "$split" = normal; then
        episodes=40
        seed=2026092301
        manifest=general_natural
    else
        episodes=60
        seed=2026092401
        manifest=hazard_stress
    fi
    for policy in "${POLICIES[@]}"; do
        notify "MineStudio behavioral evaluation started
split=$split
policy=$policy
episodes=$episodes"
        run_evaluation "$split" "$policy" "$episodes" "$seed" "$manifest"
    done
done

phase=summarize
split=none
policy=none
write_state running "$phase" "$split" "$policy"
python run/summarize_stone_behavioral_screen.py > "$SCREEN_DIRECTORY/behavioral_screen_report.stdout.json"
phase=gemini_critic
write_state running "$phase" "$split" "$policy"
python run/criticize_stone_behavioral_screen.py 2>&1 | tee "$SCREEN_DIRECTORY/gemini_critic.log"
critic_status=$(jq -r '.status' "$SCREEN_DIRECTORY/gemini_trajectory_critic_summary.json")
if test "$critic_status" = blocked_billing; then
    write_state evaluation_complete_critic_blocked gemini_critic none none
    notify "MineStudio behavioral evaluation completed, but Gemini critique is paused because API credits are depleted.
report=$SCREEN_DIRECTORY/behavioral_screen_report.json
partial_critic=$SCREEN_DIRECTORY/gemini_trajectory_critic_summary.json"
    exit 0
fi
write_state completed none none none
notify "MineStudio behavioral screen and Gemini critique completed
report=$SCREEN_DIRECTORY/behavioral_screen_report.json
critic=$SCREEN_DIRECTORY/gemini_trajectory_critic_summary.json"
