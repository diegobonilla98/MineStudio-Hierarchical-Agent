$RemoteCommand = @'
set -e
project_directory=/home/boni/projects/MineStudio
output_root=$project_directory/output/stone_acquisition
suite_directory=$output_root/latest_evaluation_suite

if ! test -e "$suite_directory"; then
    echo state=not_started
    exit 0
fi

suite_directory=$(readlink -f "$suite_directory")
echo "suite_directory=$suite_directory"
if test -f "$suite_directory/run_state.json"; then
    suite_status=$(jq -r '.status' "$suite_directory/run_state.json")
    active_split=$(jq -r '.split' "$suite_directory/run_state.json")
    active_variant=$(jq -r '.variant' "$suite_directory/run_state.json")
    echo "state=$suite_status active=$active_split/$active_variant"
fi

for split_variant in frozen/action_head frozen/action_head_recurrent frozen/upper_lora heldout/vanilla heldout/action_head heldout/action_head_recurrent heldout/upper_lora; do
    split=${split_variant%/*}
    variant=${split_variant#*/}
    experiment=$output_root/stage1_${split}_${variant}_v1
    summary=$experiment/summary.json
    episodes=$experiment/episodes.jsonl
    complete=0
    if test -f "$episodes"; then
        complete=$(wc -l < "$episodes")
    fi
    if test -f "$summary"; then
        success_rate=$(jq -r '.overall.success_rate' "$summary")
        echo "$split/$variant=$complete/1000 success_rate=$success_rate"
    else
        echo "$split/$variant=$complete/1000"
    fi
done

active_log=$suite_directory/${active_split}_${active_variant}.log
if test -f "$active_log"; then
    echo recent_log:
    tail -n 8 "$active_log"
fi

if test -f "$suite_directory/evaluation_report.json"; then
    echo report=$suite_directory/evaluation_report.json
fi
'@

ssh dgx-spark $RemoteCommand
