$remote = @'
cd /home/boni/projects/MineStudio
directory=$(readlink -f output/stone_acquisition/latest_behavioral_screen 2>/dev/null || true)
if test -z "$directory" || ! test -d "$directory"; then
    echo state=not_started
    exit 0
fi
echo screen_directory=$directory
jq -r '"state=" + .status + " phase=" + .phase + " active=" + .split + "/" + .policy' "$directory/run_state.json"
for split in normal hazard; do
    for policy in vanilla upper_lora_r8_step800 upper_lora_r8_step1600 upper_lora_r8_step3200 upper_lora_r8_step4800 upper_lora_r8_step6400 upper_lora_r32_step6400 recurrent_upper_lora_step6400; do
        summary="output/stone_acquisition/screen_${split}_${policy}_v1/summary.json"
        target=40
        if test "$split" = hazard; then target=60; fi
        if test -f "$summary"; then
            jq -r '"'"$split/$policy"'=" + (.episodes_complete|tostring) + "/'"$target"' success_rate=" + (.overall.success_rate|tostring)' "$summary"
        else
            echo "$split/$policy=0/$target"
        fi
    done
done
if test -f "$directory/gemini_trajectory_critic_summary.json"; then
    critic_status=$(jq -r '.status // "legacy"' "$directory/gemini_trajectory_critic_summary.json")
    if test "$(jq -r '.status' "$directory/run_state.json")" = running && test "$(jq -r '.phase' "$directory/run_state.json")" = gemini_critic; then
        critic_status=running
    fi
    critic_target=$(jq -r '.target // .critiques' "$directory/gemini_trajectory_critic_summary.json")
    critic_saved=$(wc -l < "$directory/gemini_trajectory_critiques.jsonl")
    echo "critic_status=$critic_status critic_saved=$critic_saved/$critic_target remaining=$((critic_target - critic_saved))"
fi
echo recent_log:
split=$(jq -r '.split // empty' "$directory/run_state.json")
policy=$(jq -r '.policy // empty' "$directory/run_state.json")
if test -f "$directory/${split}_${policy}.log"; then tail -n 8 "$directory/${split}_${policy}.log"; elif test -f "$directory/gemini_critic.log"; then tail -n 8 "$directory/gemini_critic.log"; fi
'@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
ssh dgx-spark "echo '$encoded' | base64 --decode | bash"
