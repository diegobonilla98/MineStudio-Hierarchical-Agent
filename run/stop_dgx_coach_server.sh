#!/usr/bin/env bash
set -euo pipefail

pid_file=/home/boni/.cache/minestudio/run/gemini-coach.pid

if [ ! -f "$pid_file" ]; then
    exit 0
fi

server_pid=$(cat "$pid_file")
if ! [[ "$server_pid" =~ ^[0-9]+$ ]]; then
    rm -f "$pid_file"
    exit 0
fi
if ! kill -0 "$server_pid" 2>/dev/null; then
    rm -f "$pid_file"
    exit 0
fi

command_line=$(tr '\0' ' ' < "/proc/$server_pid/cmdline")
if [[ "$command_line" != *"gemini_coach.py"* ]]; then
    echo "Refusing to stop PID $server_pid because it is not the recorded coach server." >&2
    exit 1
fi

collect_descendants() {
    local parent_pid=$1
    local child_pid
    while read -r child_pid; do
        if [ -n "$child_pid" ]; then
            collect_descendants "$child_pid"
            echo "$child_pid"
        fi
    done < <(pgrep -P "$parent_pid" || true)
}

descendant_pids=$(collect_descendants "$server_pid")
if [ -n "$descendant_pids" ]; then
    kill -TERM $descendant_pids 2>/dev/null || true
fi
kill -TERM "$server_pid" 2>/dev/null || true

for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        break
    fi
    sleep 1
done
if kill -0 "$server_pid" 2>/dev/null; then
    descendant_pids=$(collect_descendants "$server_pid")
    if [ -n "$descendant_pids" ]; then
        kill -KILL $descendant_pids 2>/dev/null || true
    fi
    kill -KILL "$server_pid" 2>/dev/null || true
fi
rm -f "$pid_file"
