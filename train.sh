#!/bin/bash

#SBATCH --mail-user=rrbonde@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J train_prj
#SBATCH --output=slurm_outputs/train_ppo_%j.out
#SBATCH --error=slurm_outputs/train_ppo_%j.err

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -p long
#SBATCH -t 168:00:00

python3 ./main.py --agent PPO \
                --exp_id ppo_train_10_Cars \
                --num_episodes 5000 --batch_size 256 \
                # --epsilon 0.6 --min_epsilon 0 \
                --lr 0.00005 --lr_decay \
                --arch Identity --fc_layers 3 \
                --spawn_vehicles 10

