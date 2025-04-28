if [ $# -ne 1 ]; then 
    echo "missing experiment id parameter usage\n ./run.sh exp_id=<experiment_id>"
else
    python train_tdmpc.py reward_type=dense agent=tdmpc experiment=tdmpc task_name=LunarLanderContinuous-v3 $1
fi
