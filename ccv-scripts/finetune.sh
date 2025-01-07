#!/bin/bash

#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH -J finetune-openvla

#SBATCH -o finetune-openvla-%j.out
#SBATCH -e finetune-openvla-%j.out

# Run a command
cd ..
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py --config_path=vla-scripts/config/finetune_robosuite.yaml