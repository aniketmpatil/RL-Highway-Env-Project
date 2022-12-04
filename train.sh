#!/bin/bash

#SBATCH --mail-user=pbhamare@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J train_rl
#SBATCH --output=slurm_outputs/train_%j.out
#SBATCH --error=slurm_outputs/train_%j.err

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -p long
#SBATCH -t 168:00:00

python3 ./main.py 