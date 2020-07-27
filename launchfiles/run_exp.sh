max_proc=100
expdir=$(python launchfiles/setup_params.py 0)
cmdfile="$expdir/cmdfile.sh"
echo $cmdfile

#Save code for reproducibility
python reproducibility.py $(pwd) $expdir

num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"
xargs -L 1 -P $max_proc srun --mem=16G -p cpu < $cmdfile
