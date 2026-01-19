#!/bin/bash
# ============================================================================
# ELPV SSL Benchmark - Submit Jobs for a Single Backbone
# ============================================================================
#
# LEAN DESIGN: 96 total experiments (4 methods × 2 backbones × 4 labels × 3 seeds)
# - WRN-28-2: 48 experiments (~10h each = 120h total with 4 parallel = 5 days)
# - ViT-Tiny: 48 experiments (~2h each = 24h total with 4 parallel = 1 day)
#
# Usage:
#   ./submit_backbone.sh wrn_28_2
#   ./submit_backbone.sh vit_tiny_patch2_32
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
    echo "  backbone: wrn_28_2, vit_tiny_patch2_32"
    exit 1
fi

BACKBONE=$1

# Validate backbone
if [[ ! "$BACKBONE" =~ ^(wrn_28_2|vit_tiny_patch2_32)$ ]]; then
    echo "ERROR: Invalid backbone: $BACKBONE"
    echo "Valid options: wrn_28_2, vit_tiny_patch2_32"
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
        TIME_LIMIT="12:00:00"  # 12 hours per job (safe margin for ~10h)
        MEM="32G"
        EST_TIME="~5 days (48 jobs × 10h / 4 parallel)"
        ;;
    vit_tiny_patch2_32)
        GPU_SPEC="gpu:nvidia_h200_1g.18gb:1"  # Tiny ViT can use smaller GPU
        TIME_LIMIT="04:00:00"  # 4 hours per job (safe margin for ~2h)
        MEM="24G"
        EST_TIME="~1 day (48 jobs × 2h / 4 parallel)"
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
echo "  $EST_TIME"
echo "=============================================================="

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

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
