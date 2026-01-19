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
#   sbatch --array=1-108%4 --time=02:00:00 --gres=gpu:nvidia_h200_1g.18gb:1 \
#          --export=BACKBONE=wrn_28_2 job_template.sh
#
# The BACKBONE environment variable determines which config list to use.
# ============================================================================

# ============================================================================
# JOB INFO
# ============================================================================
echo "=============================================================="
echo "ELPV SSL Benchmark - Job Started"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Backbone: $BACKBONE"
echo "Start Time: $(date)"
echo "=============================================================="

# ============================================================================
# CONDA ACTIVATION (CMUQ Deepnet specific)
# ============================================================================
# Use full paths to avoid shell initialization issues in SLURM

# Set up conda paths
export CONDA_ROOT="/opt/anaconda3"
export CONDA_ENV="/data1/yansari/.conda/envs/ssl_bench"

# Add conda to PATH
export PATH="${CONDA_ROOT}/bin:${PATH}"

# Activate environment by setting paths directly
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export CONDA_DEFAULT_ENV="ssl_bench"

# Verify activation
echo ""
echo "Environment Setup:"
echo "  CONDA_ROOT: ${CONDA_ROOT}"
echo "  CONDA_ENV: ${CONDA_ENV}"
echo "  PATH: ${PATH}"
echo ""

# ============================================================================
# ENVIRONMENT VERIFICATION
# ============================================================================
echo "Environment Verification:"
echo "  Python: $(which python)"
echo "  Python Version: $(python --version 2>&1)"
echo "  PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo "  CUDA Version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")' 2>&1)"
echo "  GPU Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>&1)"
echo "  GPU Memory: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A")' 2>&1)"
echo ""

# ============================================================================
# WORKING DIRECTORY
# ============================================================================
PROJECT_DIR="/data1/yansari/SSL-Benchmarking/Semi-supervised-learning"
cd "$PROJECT_DIR"
echo "Working Directory: $(pwd)"

# ============================================================================
# CONFIG FILE SELECTION
# ============================================================================
CONFIG_LIST="configs/elpv_benchmark/config_list_${BACKBONE}.txt"

if [ ! -f "$CONFIG_LIST" ]; then
    echo "ERROR: Config list not found: $CONFIG_LIST"
    exit 1
fi

CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_LIST")

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: No config file at line $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Extract experiment name from config path
EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
RESULTS_DIR="results/elpv_benchmark/experiments/${EXP_NAME}"

echo ""
echo "Experiment: $EXP_NAME"
echo "Config: $CONFIG_FILE"
echo "Results: $RESULTS_DIR"
echo "=============================================================="

# ============================================================================
# CREATE STATUS FILE (for tracking)
# ============================================================================
mkdir -p "$RESULTS_DIR"

# Write initial status
cat > "${RESULTS_DIR}/status.json" << EOF
{
    "status": "RUNNING",
    "experiment": "$EXP_NAME",
    "job_id": "$SLURM_JOB_ID",
    "array_task_id": "$SLURM_ARRAY_TASK_ID",
    "node": "$SLURMD_NODENAME",
    "backbone": "$BACKBONE",
    "start_time": "$(date -Iseconds)",
    "config_file": "$CONFIG_FILE"
}
EOF

echo ""
echo "Status file created: ${RESULTS_DIR}/status.json"
echo ""

# ============================================================================
# RUN EXPERIMENT
# ============================================================================
echo "Starting training..."
echo "=============================================================="
echo ""

# Run with unbuffered output for real-time logging
python -u train.py --c "$CONFIG_FILE"

EXIT_CODE=$?

# ============================================================================
# UPDATE STATUS FILE
# ============================================================================
END_TIME=$(date -Iseconds)

if [ $EXIT_CODE -eq 0 ]; then
    STATUS="COMPLETED"
else
    STATUS="FAILED"
fi

cat > "${RESULTS_DIR}/status.json" << EOF
{
    "status": "$STATUS",
    "experiment": "$EXP_NAME",
    "job_id": "$SLURM_JOB_ID",
    "array_task_id": "$SLURM_ARRAY_TASK_ID",
    "node": "$SLURMD_NODENAME",
    "backbone": "$BACKBONE",
    "start_time": "$(date -Iseconds -d @$SLURM_JOB_START_TIME 2>/dev/null || echo 'unknown')",
    "end_time": "$END_TIME",
    "exit_code": $EXIT_CODE,
    "config_file": "$CONFIG_FILE"
}
EOF

# ============================================================================
# JOB SUMMARY
# ============================================================================
echo ""
echo "=============================================================="
echo "ELPV SSL Benchmark - Job Finished"
echo "=============================================================="
echo "Experiment: $EXP_NAME"
echo "Status: $STATUS"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=============================================================="

# Show final metrics if available
if [ -f "${RESULTS_DIR}/metrics.json" ]; then
    echo ""
    echo "Final Metrics:"
    python -c "
import json
with open('${RESULTS_DIR}/metrics.json') as f:
    m = json.load(f)
    print(f\"  Val Best Acc: {m.get('val_best_accuracy', 'N/A')}\")
    print(f\"  Test Acc: {m.get('test_accuracy', 'N/A')}\")
    print(f\"  Test Balanced Acc: {m.get('test_balanced_accuracy', 'N/A')}\")
    print(f\"  Test F1: {m.get('test_f1', 'N/A')}\")
"
fi

exit $EXIT_CODE
