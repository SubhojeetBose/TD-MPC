if [ $# -ne 2 ]; then 
    echo "missing experiment id parameter usage\n ./run.sh task_name=<task_name> exp_id=<experiment_id>"
else
    python train_tdmpc.py reward_type=dense agent=tdmpc experiment=tdmpc $1 $2
fi
