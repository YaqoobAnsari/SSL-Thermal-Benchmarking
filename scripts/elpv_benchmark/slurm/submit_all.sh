#!/bin/bash
# ============================================================================
# ELPV SSL Benchmark - Submit All Jobs
# ============================================================================
#
# This script submits all experiments as SLURM job arrays.
# Each backbone gets its own job array for better scheduling.
#
# Usage:
#   ./submit_all.sh
#   ./submit_all.sh --dry-run  # Show commands without submitting
#
# ============================================================================

set -e

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "DRY RUN MODE - Commands will be shown but not executed"
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname $(dirname $(dirname $SCRIPT_DIR)))"
CONFIG_DIR="$BASE_DIR/configs/elpv_benchmark"

# Create logs directory
mkdir -p "$BASE_DIR/logs"

echo "=============================================="
echo "ELPV SSL Benchmark - Job Submission"
echo "=============================================="
echo "Base directory: $BASE_DIR"
echo "Config directory: $CONFIG_DIR"
echo ""

# Count configs per backbone
for BACKBONE in resnet18 resnet50 wrn_28_2 vit_b_16; do
    CONFIG_LIST="$CONFIG_DIR/config_list_${BACKBONE}.txt"
    
    if [ ! -f "$CONFIG_LIST" ]; then
        echo "WARNING: Config list not found: $CONFIG_LIST"
        continue
    fi
    
    N_CONFIGS=$(wc -l < "$CONFIG_LIST")
    echo "Backbone: $BACKBONE - $N_CONFIGS experiments"
    
    # Build sbatch command
    CMD="sbatch --array=1-${N_CONFIGS} --export=BACKBONE=${BACKBONE} $SCRIPT_DIR/job_template.sh"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  Would run: $CMD"
    else
        echo "  Submitting..."
        $CMD
    fi
done

echo ""
echo "=============================================="
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No jobs submitted"
else
    echo "All jobs submitted!"
    echo "Monitor with: squeue -u $USER"
fi
echo "=============================================="
