# print basic info before running all slurm scripts

echo "Running on $(hostname)"
echo "Current conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi