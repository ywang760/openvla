#!/bin/bash

#SBATCH --constraint=ampere
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH --time=36:00:00
#SBATCH --mem=96G
#SBATCH -J finetune-openvla

#SBATCH -o log/finetune-openvla-%j.out
#SBATCH -e log/finetune-openvla-%j.err

# Run a command
bash prelaunch.sh

cd ..
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py --config_path=vla-scripts/config/finetune_robomimic_lift.yaml