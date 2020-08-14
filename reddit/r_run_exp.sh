max_proc=250
labelgen=1
hyp=1
expdir=$(python reddit/r_setup_params.py $labelgen $hyp)
cmdfile="$expdir/cmdfile.sh"
echo $cmdfile
echo $expdir
#Save code for reproducibility
#python reproducibility.py $(pwd) $expdir

num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"
xargs -L 1 -P $max_proc srun --mem=24G -p cpu < $cmdfile
