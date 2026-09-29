#!/usr/bin/env bash
set -euo pipefail

project_directory=/home/boni/projects/MineStudio
output_directory="$project_directory/output/stone_acquisition/stage0_baseline_v1"
state_path="$output_directory/runner_state.json"
log_path="$output_directory/runner.log"
lock_path="$output_directory/runner.lock"
matrix_pattern='^[^ ]*python[^ ]* run/benchmark_steve_skill_matrix.py$'

mkdir -p "$output_directory"
exec 9>"$lock_path"
if ! flock -n 9; then
    printf '{"state":"already_active","timestamp":"%s"}\n' "$(date --iso-8601=seconds)"
    exit 0
fi

write_state() {
    state_value=$1
    matrix_pid_value=${2:-null}
    printf '{"state":"%s","runner_pid":%d,"matrix_pid":%s,"timestamp":"%s","log":"%s"}\n' \
        "$state_value" "$$" "$matrix_pid_value" "$(date --iso-8601=seconds)" "$log_path" > "$state_path.tmp"
    mv "$state_path.tmp" "$state_path"
}

trap 'write_state interrupted null; exit 130' INT TERM

while true; do
    matrix_pid=$(pgrep -f "$matrix_pattern" | head -n 1 || true)
    if [[ -z "$matrix_pid" ]]; then
        break
    fi
    write_state waiting_for_skill_matrix "$matrix_pid"
    sleep 60
done

write_state matrix_exit_grace_period null
sleep 60
matrix_pid=$(pgrep -f "$matrix_pattern" | head -n 1 || true)
if [[ -n "$matrix_pid" ]]; then
    while [[ -n "$matrix_pid" ]]; do
        write_state waiting_for_skill_matrix "$matrix_pid"
        sleep 60
        matrix_pid=$(pgrep -f "$matrix_pattern" | head -n 1 || true)
    done
fi

cd "$project_directory"
source /home/boni/ai/envs/dgx-dl/bin/activate
source run/dgx_env.sh
write_state running null
set +e
python run/benchmark_stone_acquisition.py >> "$log_path" 2>&1
exit_code=$?
set -e
if [[ $exit_code -eq 0 ]]; then
    write_state completed null
else
    write_state failed null
fi
exit "$exit_code"
