#!/bin/bash -l
#SBATCH --output=train_GP_%A.out
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=25:00:00
#SBATCH --partition=gpu-a100-80g
##SBATCH --partition=gpu-debug

# srun python train.py --config-name=train1D experiment.expid=PABBO_GP1D_240227 
# srun python train.py --config-name=train2D experiment.expid=PABBO_GP2D_T_48 experiment.resume=True train.n_burnin=3002 train.max_T=48
srun python train.py --config-name=train6D experiment.expid=PABBO_GP6D_T_64 train.max_T=64 experiment.resume=true