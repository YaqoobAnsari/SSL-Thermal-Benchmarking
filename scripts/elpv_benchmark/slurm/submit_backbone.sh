#!/bin/bash
# ============================================================================
# ELPV SSL Benchmark - Submit Jobs for a Single Backbone
# ============================================================================
#
# LEAN DESIGN: 216 total experiments
# - WRN-28-2: 108 experiments (~27h with 4 parallel)
# - ViT-B/16: 108 experiments (~67h with 4 parallel)
#
# Usage:
#   ./submit_backbone.sh wrn_28_2
#   ./submit_backbone.sh vit_b_16
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname $(dirname $(dirname $SCRIPT_DIR)))"
CONFIG_DIR="$BASE_DIR/configs/elpv_benchmark"

# MCS Label for TUBITAK research
MCS_LABEL="TUBITAK"

# Check argument
if [ -z "$1" ]; then
    echo "Usage: $0 <backbone>"
    echo "  backbone: wrn_28_2, vit_b_16"
    exit 1
fi

BACKBONE=$1

# Validate backbone
if [[ ! "$BACKBONE" =~ ^(wrn_28_2|vit_b_16)$ ]]; then
    echo "ERROR: Invalid backbone: $BACKBONE"
    echo "Valid options: wrn_28_2, vit_b_16"
    exit 1
fi

# Config list file
CONFIG_LIST="$CONFIG_DIR/config_list_${BACKBONE}.txt"

if [ ! -f "$CONFIG_LIST" ]; then
    echo "ERROR: Config list not found: $CONFIG_LIST"
    exit 1
fi

# Count configs
N_CONFIGS=$(wc -l < "$CONFIG_LIST")

# Set GPU and time based on backbone
case $BACKBONE in
    wrn_28_2)
        GPU_SPEC="gpu:nvidia_h200_1g.18gb:1"
        TIME_LIMIT="02:00:00"  # 2 hours per job
        MEM="32G"
        ;;
    vit_b_16)
        GPU_SPEC="gpu:nvidia_h200_2g.35gb:1"
        TIME_LIMIT="04:00:00"  # 4 hours per job
        MEM="48G"
        ;;
esac

echo "=============================================================="
echo "ELPV SSL Benchmark - LEAN DESIGN"
echo "=============================================================="
echo "Backbone: $BACKBONE"
echo "Config list: $CONFIG_LIST"
echo "Number of experiments: $N_CONFIGS"
echo "GPU: $GPU_SPEC"
echo "Time limit: $TIME_LIMIT"
echo "Memory: $MEM"
echo "MCS Label: $MCS_LABEL"
echo "Max concurrent jobs: 4 (CMUQ policy)"
echo ""
echo "Estimated completion time:"
if [ "$BACKBONE" == "wrn_28_2" ]; then
    echo "  ~27 hours (108 jobs × 1h / 4 parallel)"
else
    echo "  ~67 hours (108 jobs × 2.5h / 4 parallel)"
fi
echo "=============================================================="

# Create logs directory
mkdir -p "$BASE_DIR/logs"

# Build sbatch command
# Using %4 to limit to 4 concurrent jobs (CMUQ policy)
CMD="sbatch --array=1-${N_CONFIGS}%4 \
    --mcs-label=$MCS_LABEL \
    --time=$TIME_LIMIT \
    --gres=$GPU_SPEC \
    --mem=$MEM \
    --export=BACKBONE=$BACKBONE \
    $SCRIPT_DIR/job_template.sh"

echo ""
echo "Command:"
echo "  $CMD"
echo ""

read -p "Submit jobs? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    $CMD
    echo ""
    echo "Jobs submitted!"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  python scripts/elpv_benchmark/07_track_experiments.py"
else
    echo "Cancelled."
fi
