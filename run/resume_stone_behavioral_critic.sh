#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
SCREEN_DIRECTORY=$PROJECT_DIRECTORY/output/stone_acquisition/behavioral_screens/20260919T111059Z
RUN_STATE=$SCREEN_DIRECTORY/run_state.json

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
export MINESTUDIO_SCREEN_DIRECTORY="$SCREEN_DIRECTORY"

write_state() {
    local status=$1
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"20260919T111059Z","phase":"gemini_critic","split":"none","policy":"none","pid":%s,"screen_directory":"%s","updated_at":"%s"}\n' "$status" "$$" "$SCREEN_DIRECTORY" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

write_state running
python run/criticize_stone_behavioral_screen.py 2>&1 | tee -a "$SCREEN_DIRECTORY/gemini_critic.log"
critic_status=$(jq -r '.status' "$SCREEN_DIRECTORY/gemini_trajectory_critic_summary.json")
if test "$critic_status" = completed; then
    write_state completed
else
    write_state evaluation_complete_critic_blocked
fi
