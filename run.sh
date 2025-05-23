#!/bin/bash -l
#SBATCH --output=outputs/train_%A.out
#SBATCH --mem=6G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=25:00:00
#SBATCH --partition=gpu-v100-32g

python train.py --config-name=train  \
experiment.expid=PABBO_GP2D  \
data.name=GP2D  \
data.d_x=2 \
data.x_range="[[-1,1], [-1, 1]]"  \
data.min_num_ctx=1  \
data.max_num_ctx=100 \
train.num_query_points=200 \
train.num_prediction_points=200 \
train.n_random_pairs=200 
