max_proc=30
cmdfile=$(python launchfiles/setup_params.py)
echo $cmdfile

num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"
xargs -L 1 -P $max_proc srun --mem=16G -p cpu < $cmdfile
