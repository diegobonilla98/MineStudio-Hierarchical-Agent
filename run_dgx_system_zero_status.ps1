$remote = @'
cd /home/boni/projects/MineStudio
state=output/system_zero/run_state.json
result=output/system_zero/priority_one_smoke.json
if test -f "$state"; then
    jq -r '"state=" + .status + " phase=" + .phase + " pid=" + (.pid|tostring) + " reason=" + .reason' "$state"
else
    echo state=not_started
fi
if test -f "$result"; then
    jq -r '.summary | to_entries[] | .key + "=" + (.value.success|tostring) + "/" + (.value.attempts|tostring) + " success_rate=" + (.value.success_rate|tostring)' "$result"
fi
echo recent_log:
tail -n 12 output/system_zero/priority_one_smoke.log 2>/dev/null || true
'@
ssh dgx-spark $remote
