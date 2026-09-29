#!/usr/bin/env bash
set -euo pipefail

project_directory=/home/boni/projects/MineStudio
runtime_directory=/home/boni/.cache/minestudio/run
pid_file="$runtime_directory/gemini-coach.pid"
source "$project_directory/run/dgx_env.sh"
export ALSOFT_DRIVERS=null
export PYTHONUNBUFFERED=1
mkdir -p "$runtime_directory"
if [ -f "$pid_file" ]; then
    existing_pid=$(cat "$pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "A MineStudio coach server is already running with PID $existing_pid." >&2
        exit 1
    fi
fi
echo $$ > "$pid_file"
cd "$project_directory"
exec python minestudio/tutorials/simulator/gemini_coach.py
