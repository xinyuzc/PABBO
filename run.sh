#!/bin/bash -l
#SBATCH --output=train_%A.out
#SBATCH --mem=3G
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-a100-80g
##SBATCH --partition=gpu-debug

# python train.py --config-name=train \
# experiment.expid=PABBO_GP1D \
# data.name=GP1D \
# data.d_x=1 \
# data.x_range="[[-1,1]]" \
# data.min_num_ctx=1 \
# data.max_num_ctx=50 \
# train.num_query_points=100 \
# train.num_prediction_points=100 \
# train.n_random_pairs=100 \

# python train.py --config-name=train  \
# experiment.expid=PABBO_GP4D \
# data.name=GP4D \
# data.d_x=4 \
# data.x_range="[[-1, 1],[-1, 1],[-1, 1],[-1, 1]]" \
# data.min_num_ctx=1 \
# data.max_num_ctx=100 \
# train.num_query_points=300 \
# train.num_prediction_points=300 \
# train.n_random_pairs=300 

# python train.py --config-name=train  \
# experiment.expid=PABBO_GP6D_st \
# data.name=GP6D \
# data.d_x=6 \
# data.x_range="[[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1]]" \
# data.min_num_ctx=1 \
# data.max_num_ctx=200 \
# train.num_query_points=300 \
# train.num_prediction_points=300 \
# train.n_random_pairs=300 

# python train.py --config-name=train \
# experiment.expid=PABBO_HPOB5859_st \
# data.name=HPOB5859 \
# data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]"  \
# data.standardize=true \
# data.min_num_ctx=1 \
# data.max_num_ctx=200 \
# data.search_space_id=5859 \
# data.d_x=6 \
# train.num_query_points=300 \
# train.num_prediction_points=300 \
# train.n_random_pairs=300 
