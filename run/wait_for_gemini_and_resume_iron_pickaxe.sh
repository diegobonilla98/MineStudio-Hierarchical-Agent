#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
RUN_ID=20260921T134140Z
OUTPUT_DIRECTORY=$PROJECT_DIRECTORY/output/iron_pickaxe_viability/$RUN_ID
STATE_PATH=$OUTPUT_DIRECTORY/run_state.json

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh

write_state() {
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"waiting","run_id":"%s","phase":"gemini_credit","detail":"automatic_resume_pending","pid":%s,"output_directory":"%s","updated_at":"%s"}\n' "$RUN_ID" "$$" "$OUTPUT_DIRECTORY" "$timestamp" > "$STATE_PATH.tmp"
    mv "$STATE_PATH.tmp" "$STATE_PATH"
}

write_state
while true; do
    if python - <<'PY'
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv("/home/boni/projects/MineStudio/.env")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Reply READY.",
    config=types.GenerateContentConfig(max_output_tokens=20, temperature=0),
)
if "READY" not in response.text.upper():
    raise RuntimeError(response.text)
PY
    then
        exec bash run/resume_iron_pickaxe_viability.sh
    fi
    write_state
    sleep 120
done
