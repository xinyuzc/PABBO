#!/bin/bash -l
#SBATCH --output=evaluate_pbo_%A.out
#SBATCH --mem=3G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --array=0-29
#SBATCH --constraint=milan

CHUNKSIZE=1
n=$SLURM_ARRAY_TASK_ID
seeds=`seq $((n*CHUNKSIZE)) $(((n + 1)*CHUNKSIZE - 1))`

# seeds=`seq $((19)) $((19))`
datasets=`seq $((0)) $((0))`
for j in $seeds
do
   for i in $datasets
   do 
   # GP1D   
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=rs experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qEI experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qEUBO experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qTS experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qNEI experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=mpes experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1

   # # GP2D
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=rs experiment.device=cpu data.name=GP2D data.d_x=2 data.x_range="[[-1,1],[-1,1]]" data.max_num_ctx=100 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qTS experiment.device=cpu data.name=GP2D data.d_x=2 data.x_range="[[-1,1],[-1,1]]" data.max_num_ctx=100 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qEUBO experiment.device=cpu data.name=GP2D data.d_x=2 data.x_range="[[-1,1],[-1,1]]" data.max_num_ctx=100 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qEI experiment.device=cpu data.name=GP2D data.d_x=2 data.x_range="[[-1,1],[-1,1]]" data.max_num_ctx=100 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qNEI experiment.device=cpu data.name=GP2D data.d_x=2 data.x_range="[[-1,1],[-1,1]]" data.max_num_ctx=100 data.min_num_ctx=1
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=mpes experiment.device=cpu data.name=GP2D data.d_x=2 data.x_range="[[-1,1],[-1,1]]" data.max_num_ctx=100 data.min_num_ctx=1
   
   # candy 
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=rs experiment.device=cpu eval.eval_max_T=100 data.name=candy 
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qTS experiment.device=cpu eval.eval_max_T=100 data.name=candy 
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qEI experiment.device=cpu eval.eval_max_T=100 data.name=candy 
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qEUBO experiment.device=cpu eval.eval_max_T=100 data.name=candy 
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qNEI experiment.device=cpu eval.eval_max_T=100 data.name=candy 
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=mpes experiment.device=cpu eval.eval_max_T=100 data.name=candy 
   
   # sushi 
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=rs experiment.device=cpu eval.eval_max_T=100 data.name=sushi 
   srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qTS experiment.device=cpu eval.eval_max_T=100 data.name=sushi
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qEI experiment.device=cpu eval.eval_max_T=100 data.name=sushi
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qEUBO experiment.device=cpu eval.eval_max_T=100 data.name=sushi
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=qNEI experiment.device=cpu eval.eval_max_T=100 data.name=sushi
   # srun python baseline.py --config-name=evaluate eval.dataset_id=$i eval.seed_id=$j experiment.wandb=false experiment.model=mpes experiment.device=cpu eval.eval_max_T=100 data.name=sushi
   
   done
done