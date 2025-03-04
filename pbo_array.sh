#!/bin/bash -l
#SBATCH --output=evaluate_pbo_%A.out
#SBATCH --mem=3G
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --array=0-29
#SBATCH --constraint=milan

## sushi 
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=rs experiment.device=cpu eval.eval_max_T=100 data.name=sushi 
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=${SLURM_ARRAY_TASK_ID} experiment.wandb=false experiment.model=qEUBO experiment.device=cpu eval.eval_max_T=100 data.name=sushi
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=${SLURM_ARRAY_TASK_ID} experiment.wandb=false experiment.model=qNEI experiment.device=cpu eval.eval_max_T=100 data.name=sushi
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=${SLURM_ARRAY_TASK_ID} experiment.wandb=false experiment.model=mpes experiment.device=cpu eval.eval_max_T=100 data.name=sushi

## sushi 
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=rs experiment.device=cpu eval.eval_max_T=100 data.name=sushi_old
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qEI experiment.device=cpu eval.eval_max_T=100 data.name=sushi_old
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qEUBO experiment.device=cpu eval.eval_max_T=100 data.name=sushi_old
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qNEI experiment.device=cpu eval.eval_max_T=100 data.name=sushi_old
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qTS experiment.device=cpu eval.eval_max_T=100 data.name=sushi_old
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=mpes experiment.device=cpu eval.eval_max_T=100 data.name=sushi_old

## hartmann
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=rs experiment.device=cpu data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qTS experiment.device=cpu data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qEI experiment.device=cpu data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qEUBO experiment.device=cpu data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qNEI experiment.device=cpu data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=mpes experiment.device=cpu data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_max_T=60

# ackley
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=rs experiment.device=cpu data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qTS experiment.device=cpu data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qEI experiment.device=cpu data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qEUBO experiment.device=cpu data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qNEI experiment.device=cpu data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=mpes experiment.device=cpu data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_max_T=60

# rosenbrock
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qEI experiment.device=cpu data.name=rosenbrock6D data.d_x=6 data.x_range="[[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=qNEI experiment.device=cpu data.name=rosenbrock6D data.d_x=6 data.x_range="[[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_max_T=60
# srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$SLURM_ARRAY_TASK_ID experiment.wandb=false experiment.model=mpes experiment.device=cpu data.name=rosenbrock6D data.d_x=6 data.x_range="[[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_max_T=60

# CHUNKSIZE=1
# n=$SLURM_ARRAY_TASK_ID
# SEED=`seq $((n*CHUNKSIZE)) $(((n + 1)*CHUNKSIZE - 1))`

# DATASET=`seq $((0)) $((5))`
# for s in $SEED
# do 
# for d in $DATASET
# do 

## HPOB7609
# DATASET=`seq $((0)) $((6))`
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=rs experiment.device=cpu eval.eval_max_T=100 data.name=HPOB7609 data.standardize=true data.search_space_id="7609" data.d_x=9 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qTS experiment.device=cpu eval.eval_max_T=100 data.name=HPOB7609 data.standardize=true data.search_space_id="7609" data.d_x=9 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=mpes experiment.device=cpu eval.eval_max_T=100 data.name=HPOB7609 data.standardize=true data.search_space_id="7609" data.d_x=9 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 

## HPOB5636
# DATASET=`seq $((3)) $((5))`
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=rs experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5636 data.standardize=true data.search_space_id="5636" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qTS experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5636 data.standardize=true data.search_space_id="5636" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qEI experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5636 data.standardize=true data.search_space_id="5636" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qEUBO experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5636 data.standardize=true data.search_space_id="5636" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qNEI experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5636 data.standardize=true data.search_space_id="5636" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=mpes experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5636 data.standardize=true data.search_space_id="5636" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 

# HPOB5859
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=rs experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5859 data.standardize=true data.search_space_id="5859" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qTS experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5859 data.standardize=true data.search_space_id="5859" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qEI experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5859 data.standardize=true data.search_space_id="5859" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qEUBO experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5859 data.standardize=true data.search_space_id="5859" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qNEI experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5859 data.standardize=true data.search_space_id="5859" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 

## HPOB5971
# DATASET=`seq $((0)) $((5))`
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=rs experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5971 data.standardize=true data.search_space_id="5971" data.d_x=16 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qTS experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5971 data.standardize=true data.search_space_id="5971" data.d_x=16 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qEI experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5971 data.standardize=true data.search_space_id="5971" data.d_x=16 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qEUBO experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5971 data.standardize=true data.search_space_id="5971" data.d_x=16 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python baseline.py --config-name=evaluate_hpob eval.dataset_id=$d eval.seed_id=$s experiment.wandb=false experiment.model=qNEI experiment.device=cpu eval.eval_max_T=100 data.name=HPOB5971 data.standardize=true data.search_space_id="5971" data.d_x=16 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# done
# done
