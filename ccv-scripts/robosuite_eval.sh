#!/bin/bash

#SBATCH --constraint=ampere
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH -J robosuite-eval

#SBATCH -o log/robosuite-eval-%j.out
#SBATCH -e log/robosuite-eval-%j.err

# Run a command
cd ..

# WARNING: this doesn't currently work -> run from vscode terminal
python experiments/robot/robosuite/run_robosuite_eval.py