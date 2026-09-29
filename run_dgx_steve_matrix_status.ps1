$RemoteProject = "/home/boni/projects/MineStudio"
$OutputDirectory = "output/steve_skill_matrix/controlled_15_skill_100"
$RunnerLog = "output/steve_skill_matrix/controlled_15_skill_100_runner.log"
$RemoteCommand = @'
cd /home/boni/projects/MineStudio
count=0
if test -f output/steve_skill_matrix/controlled_15_skill_100/episodes.jsonl; then
    count=$(wc -l < output/steve_skill_matrix/controlled_15_skill_100/episodes.jsonl)
fi
percent=$(awk -v count="$count" 'BEGIN { printf "%.2f", 100 * count / 1500 }')
if pgrep -af '[b]enchmark_steve_skill_matrix.py' >/dev/null; then
    running=true
else
    running=false
fi
echo "running=$running episodes=$count/1500 percent=$percent%"
if test -f output/steve_skill_matrix/controlled_15_skill_100/summary.json; then
    echo "summary=/home/boni/projects/MineStudio/output/steve_skill_matrix/controlled_15_skill_100/summary.json"
fi
echo "recent results:"
grep -E '\{"complete"|restarting_failed_workers|Traceback' output/steve_skill_matrix/controlled_15_skill_100_runner.log | tail -12
'@

ssh dgx-spark $RemoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the DGX STEVE-1 matrix status."
}
