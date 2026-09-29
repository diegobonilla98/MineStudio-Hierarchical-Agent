$RemoteCommand = @'
set -e
project_directory=/home/boni/projects/MineStudio
output_root=$project_directory/output/stone_acquisition
suite_directory=$output_root/latest_confirmation_suite

if ! test -e "$suite_directory"; then
    echo state=not_started
    exit 0
fi

suite_directory=$(readlink -f "$suite_directory")
echo "suite_directory=$suite_directory"
state_file=$suite_directory/run_state.json
if test -f "$state_file"; then
    status=$(jq -r '.status' "$state_file")
    phase=$(jq -r '.phase' "$state_file")
    active_split=$(jq -r '.split' "$state_file")
    active_policy=$(jq -r '.policy' "$state_file")
    training_directory=$(jq -r '.training_directory' "$state_file")
    echo "state=$status phase=$phase active=$active_split/$active_policy"
fi

metrics=$training_directory/upper_lora/metrics.jsonl
if test -s "$metrics"; then
    training_step=$(tail -n 1 "$metrics" | jq -r '.step')
    echo "training_step=$training_step/800"
fi

for step in 400 500 600 700 800; do
    trainable=$training_directory/upper_lora/snapshots/step_${step}_trainable.pt
    model=$training_directory/upper_lora/snapshots/step_${step}_model/model.safetensors
    if test -f "$model"; then
        echo "checkpoint_$step=materialized"
    elif test -f "$trainable"; then
        echo "checkpoint_$step=frozen_trainable"
    else
        echo "checkpoint_$step=pending"
    fi
done

for split_policy in general/vanilla general/lora400 general/lora800 stress/vanilla stress/lora400 stress/lora800; do
    split=${split_policy%/*}
    policy=${split_policy#*/}
    target=1000
    if test "$split" = stress; then target=600; fi
    experiment=$output_root/stage1b_${split}_${policy}_v1
    episodes=$experiment/episodes.jsonl
    summary=$experiment/summary.json
    complete=0
    if test -f "$episodes"; then complete=$(wc -l < "$episodes"); fi
    if test -f "$summary"; then
        success_rate=$(jq -r '.overall.success_rate' "$summary")
        echo "$split/$policy=$complete/$target success_rate=$success_rate"
    else
        echo "$split/$policy=$complete/$target"
    fi
done

if test -f "$suite_directory/confirmation_report.json"; then
    echo report=$suite_directory/confirmation_report.json
fi

active_log=$suite_directory/${active_split}_${active_policy}.log
if test "$phase" = train_lora800; then active_log=$suite_directory/training.log; fi
if test "$phase" = materialize_snapshots; then active_log=$suite_directory/materialize.log; fi
if test -f "$active_log"; then
    echo recent_log:
    tail -n 10 "$active_log"
fi
'@

ssh dgx-spark $RemoteCommand
