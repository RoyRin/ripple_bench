#!/bin/bash
#SBATCH --job-name="oracle"
#SBATCH -c 10
#SBATCH -a 0-499
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=high-priority
##SBATCH --exclude=deep-chungus-[1-6]
#SBATCH --output=/mnt/xfs/home/spark/logs/unlearn-l17-oracles-%a-%j

MID=$SLURM_ARRAY_TASK_ID

python3 train_oracles.py $MID