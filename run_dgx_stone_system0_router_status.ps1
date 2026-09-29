$remote = @'
cd /home/boni/projects/MineStudio
directory=$(readlink -f output/stone_acquisition/latest_system0_router_ablation 2>/dev/null || true)
if test -z "$directory" || ! test -d "$directory"; then
    echo state=not_started
    exit 0
fi
echo output_directory=$directory
jq -r '"state=" + .status + " phase=" + .phase + " split=" + .split' "$directory/run_state.json"
for split in normal hazard; do
    summary="output/stone_acquisition/system0_ablation_${split}_repaired_v1/summary.json"
    target=40
    if test "$split" = hazard; then target=60; fi
    if test -f "$summary"; then
        jq -r '"'"$split"'=" + (.episodes_complete|tostring) + "/'"$target"' success_rate=" + (.overall.success_rate|tostring)' "$summary"
    else
        echo "$split=0/$target"
    fi
done
if test -f "$directory/system0_router_ablation_report.json"; then
    jq -r '"old_combined=" + (.combined.old_system0_router.success_rate|tostring) + " repaired_combined=" + (.combined.repaired_system0_router.success_rate|tostring) + " delta=" + (.combined.success_rate_difference|tostring)' "$directory/system0_router_ablation_report.json"
fi
if test -f "$directory/gemini_trajectory_critic_summary.json"; then
    jq -r '"critic_status=" + .status + " critic_saved=" + (.critiques|tostring) + "/" + (.target|tostring)' "$directory/gemini_trajectory_critic_summary.json"
fi
echo recent_log:
split=$(jq -r '.split // empty' "$directory/run_state.json")
if test -f "$directory/${split}_repaired.log"; then tail -n 8 "$directory/${split}_repaired.log"; elif test -f "$directory/gemini_critic.log"; then tail -n 8 "$directory/gemini_critic.log"; fi
'@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
ssh dgx-spark "echo '$encoded' | base64 --decode | bash"
