#!/bin/bash

#SBATCH --constraint=ampere
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -J robosuite-eval

#SBATCH -o log/robosuite-eval-%j.out
#SBATCH -e log/robosuite-eval-%j.err

# Run a command
bash prelaunch.sh

cd ..
# Change the pretrained_checkpoint path to evaluate on a different lora-finetuned model
python experiments/robot/robosuite/run_robosuite_eval.py \
    --env.type Lift \
    --lora_adapter True \
    --pretrained_checkpoint /users/ywang760/scratch/openvla/runs/openvla-7b+robosuite_dataset+b8+lr-0.0005+lora-r32+dropout-0.0--clip+mae--image_aug \
    --max_episodes 4 \
    --max_steps 120 \
    --run_id_note clip+mae \
    --debug False

# --pretrained_checkpoint /users/ywang760/scratch/openvla/runs/openvla-7b+robosuite_dataset+b8+lr-0.0005+lora-r32+dropout-0.0+q-4bit--clip+mae--image_aug \
# --pretrained_checkpoint /users/ywang760/scratch/openvla/runs/openvla-7b+robosuite_dataset+b8+lr-0.0005+lora-r32+dropout-0.0+q-4bit--None--image_aug \