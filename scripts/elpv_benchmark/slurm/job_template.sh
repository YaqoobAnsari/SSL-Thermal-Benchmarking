#!/bin/bash
#SBATCH --job-name=elpv_ssl
#SBATCH --output=logs/elpv_%A_%a.out
#SBATCH --error=logs/elpv_%A_%a.err
#SBATCH --partition=gpu2
#SBATCH --nodelist=deepnet2
#SBATCH --mcs-label=TUBITAK
# Note: --time will be set at submission time based on backbone
# Note: --gres will be set at submission time based on backbone

# ============================================================================
# ELPV SSL Benchmark - SLURM Job Template for CMUQ Deepnet
# ============================================================================
#
# This template is used for running experiments via SLURM job arrays.
#
# Usage:
#   sbatch --array=1-210%4 --time=04:00:00 --gres=gpu:nvidia_h200_1g.18gb:1 \
#          --export=BACKBONE=wrn_28_2 job_template.sh
#
# The BACKBONE environment variable determines which config list to use.
# ============================================================================

echo "=============================================================="
echo "ELPV SSL Benchmark - Job Started"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Backbone: $BACKBONE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=============================================================="

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ssl_bench

# Verify environment
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set working directory
cd /data1/yansari/SSL-Benchmarking/Semi-supervised-learning

# Get config file from array
CONFIG_LIST="configs/elpv_benchmark/config_list_${BACKBONE}.txt"

if [ ! -f "$CONFIG_LIST" ]; then
    echo "ERROR: Config list not found: $CONFIG_LIST"
    exit 1
fi

CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CONFIG_LIST)

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: No config file at line $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Config file: $CONFIG_FILE"
echo "=============================================================="

# Run experiment
python train.py --c "$CONFIG_FILE"

EXIT_CODE=$?

echo "=============================================================="
echo "Job Finished"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=============================================================="

exit $EXIT_CODE
