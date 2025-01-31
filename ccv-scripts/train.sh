#!/bin/bash

#SBATCH --constraint=ampere
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -J train-openvla

#SBATCH -o log/train-openvla-%j.out
#SBATCH -e log/train-openvla-%j.err

# Run a command
bash prelaunch.sh

cd ..

# TODO: add the ommand
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py --config_path=vla-scripts/config/train_robosuite.yaml