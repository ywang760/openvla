# interact -q gpu -g 1 -f ampere -m 20g -n 4 -t 04:00:00
module purge
unset LD_LIBRARY_PATH
unset LD_PRELOAD
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate openvla