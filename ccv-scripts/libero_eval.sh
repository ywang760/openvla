#!/bin/bash

#SBATCH --constraint=ampere
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 2
#SBATCH --time=2:00:00
#SBATCH --mem=24G
#SBATCH -J libero-eval

#SBATCH -o log/libero-eval-%j.out
#SBATCH -e log/libero-eval-%j.err

# Run a command
cd ..

# print the hostname and the current conda environment
echo "Running on $(hostname)"
nvidia-smi
echo "Current conda environment: $CONDA_DEFAULT_ENV"

# LoRA finetuned myself
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b \
  --lora_adapter True \
  --lora_exp_id openvla-7b+libero_spatial+b8+lr-0.0005+lora-r32+dropout-0.0+q-4bit--None--image_aug \
  --load_in_4bit True \
  --center_crop True \
  --task_suite_name libero_spatial \
  --num_trials_per_task 20

# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 3 \
  --load_in_4bit True

# # Launch LIBERO-Object evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
#   --task_suite_name libero_object \
#   --center_crop True \
#   --use_wandb True

# # Launch LIBERO-Goal evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
#   --task_suite_name libero_goal \
#   --center_crop True \
#   --use_wandb True

# # Launch LIBERO-10 (LIBERO-Long) evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
#   --task_suite_name libero_10 \
#   --center_crop True \
#   --use_wandb True