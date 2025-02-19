#!/bin/bash -l
#SBATCH --output=train_1D_%A.out
#SBATCH --mem=3G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-debug

srun python train.py --config-name=train1D experiment.expid=PABBO_GP1D_DEBUG_COMBINED_LOSS_2 train.n_burnin=50

# srun python evaluate.py --model_name PABBO_Mixture