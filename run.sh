#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=5G
##SBATCH --nodes=1
#SBATCH --output=ablation_meta_datasets_%A_%a.outs

srun python evaluate.py --model_name PABBO_Mixture