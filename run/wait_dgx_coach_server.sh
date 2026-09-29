#!/usr/bin/env bash
set -euo pipefail

pid_file=/home/boni/.cache/minestudio/run/gemini-coach.pid
server_port=18765
maximum_attempts=240

for ((attempt = 1; attempt <= maximum_attempts; attempt++)); do
    if [ -f "$pid_file" ]; then
        server_pid=$(cat "$pid_file")
        if [[ "$server_pid" =~ ^[0-9]+$ ]]; then
            if ! kill -0 "$server_pid" 2>/dev/null; then
                echo "The recorded MineStudio coach process exited before becoming ready." >&2
                exit 1
            fi
            if ss -ltnp "sport = :$server_port" | grep -q "pid=$server_pid,"; then
                echo "MineStudio coach server is ready with PID $server_pid."
                exit 0
            fi
        fi
    fi
    sleep 0.5
done

echo "Timed out waiting for the MineStudio coach server to bind port $server_port." >&2
exit 1
