#!/bin/bash -l
#SBATCH --output=evaluate_pabbo_%A.out
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --constraint=milan
##SBATCH --gres=gpu:1

srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP1D experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1 eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0