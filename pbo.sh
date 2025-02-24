#!/bin/bash -l
#SBATCH --output=evaluate_pbo_1d_%A.out
#SBATCH --mem=700M
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --array=0-5
#SBATCH --constraint=milan

CHUNKSIZE=5
n=$SLURM_ARRAY_TASK_ID
indexes=`seq $((n*CHUNKSIZE)) $(((n + 1)*CHUNKSIZE - 1))`

for i in $indexes
do
   srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$i experiment.wandb=false experiment.model=qTS experiment.expid=qTS_GP1D experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=1 eval.seed_id=$i experiment.model=mpes experiment.expid=qTS_GP1D experiment.device=cpu data.name=forrester1D data.d_x=1 data.x_range="[[0, 1]]" data.Xopt="[[0.75724876]]" data.yopt="[[-6.020740]]"
done