# SSL-Thermal-Benchmarking

**A Comprehensive Benchmark for Semi-Supervised Learning on Solar Cell & PV Module Defect Detection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)

---

## 📋 Overview

This repository provides a rigorous evaluation framework for **Semi-Supervised Learning (SSL)** algorithms on photovoltaic (PV) defect detection using electroluminescence (EL) and thermal imaging. The benchmark evaluates **8 SSL algorithms** across **2 backbone architectures** (CNN + Transformer) with varying amounts of labeled data.

**Target Venue:** IEEE Transactions on Industrial Informatics (TII)

---

## 🎯 Key Features

- 🔬 **8 SSL Algorithms**: FixMatch, FlexMatch, FreeMatch, SoftMatch, MeanTeacher, ABC, DARP, DASO
- 🏗️ **2 Backbones**: WRN-28-2 (CNN) + ViT-B/16 (Transformer)
- 📊 **4 Label Amounts**: 10, 50, 200, 800 labeled samples
- 🎯 **3 Datasets**: ELPV (2-class), THED-PV (5-class), PVEL-AD (12-class)
- ⚖️ **Imbalance Testing**: 1:30 class imbalance scenarios
- 📈 **Statistical Analysis**: Friedman Rank with Nemenyi post-hoc test
- ✅ **Robust**: Incremental checkpointing, graceful shutdown handling

---

## 🔬 Experimental Design

### SSL Algorithms

| Category | Algorithms | Description |
|----------|------------|-------------|
| **SOTA** | FixMatch, FlexMatch, FreeMatch, SoftMatch | State-of-the-art pseudo-labeling methods |
| **Imbalance-Aware** | ABC, DARP, DASO | Designed for class-imbalanced scenarios |
| **Classic** | MeanTeacher | Foundational EMA-based approach |
| **Baseline** | Supervised | Pure supervised learning |

### Backbones

| Backbone | Type | Input Size | Pretrained |
|----------|------|------------|------------|
| **WRN-28-2** | CNN | 96×96 | No |
| **ViT-B/16** | Transformer | 224×224 | ImageNet |

### Datasets

| Dataset | Classes | Task | Status |
|---------|---------|------|--------|
| **ELPV** | 2 | Binary (functional/defective) | ✅ Ready |
| **THED-PV** | 5 | Multi-class thermal defects | 📋 Planned |
| **PVEL-AD** | 12 | Fine-grained EL anomalies | 📋 Planned |

### Experiment Matrix

```
9 methods × 2 backbones × 4 labels × 3 seeds = 216 experiments per dataset
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YaqoobAnsari/SSL-Thermal-Benchmarking.git
cd SSL-Thermal-Benchmarking

# Create conda environment
conda create -n ssl_bench python=3.10 -y
conda activate ssl_bench

# Install PyTorch (with CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm psutil \
    ruamel.yaml tensorboard timm transformers wandb scikit-posthocs \
    scikit-image progress elpv-dataset
```

### Running Experiments

```bash
# 1. Verify environment
python scripts/elpv_benchmark/00_verify_environment.py

# 2. Prepare data splits (run once)
python scripts/elpv_benchmark/01_prepare_data.py

# 3. Generate configs
python scripts/elpv_benchmark/02_generate_configs.py

# 4. Validate setup
python scripts/elpv_benchmark/03_validate_setup.py

# 5. Run single experiment
python scripts/elpv_benchmark/04_run_single.py \
    --config configs/elpv_benchmark/experiments/fixmatch_wrn_28_2_100_seed0.yaml

# 6. Aggregate results (after all experiments)
python scripts/elpv_benchmark/05_aggregate_results.py

# 7. Compute Friedman rankings
python scripts/elpv_benchmark/06_compute_friedman.py

# 8. Track progress
python scripts/elpv_benchmark/07_track_experiments.py
```

### SLURM Cluster (CMUQ Deepnet)

```bash
cd scripts/elpv_benchmark/slurm

# Submit WRN-28-2 experiments (108 jobs, ~27h)
./submit_backbone.sh wrn_28_2

# After completion, submit ViT experiments (108 jobs, ~67h)
./submit_backbone.sh vit_b_16
```

---

## 📁 Project Structure

```
SSL-Thermal-Benchmarking/
├── configs/elpv_benchmark/
│   ├── base/                    # Base configs per backbone
│   ├── experiments/             # 216 experiment configs
│   └── master_config.yaml       # Locked hyperparameters
├── data/elpv/splits/            # Pre-generated data splits
├── results/elpv_benchmark/
│   ├── experiments/             # Per-experiment outputs
│   ├── aggregated/              # CSVs and rankings
│   └── figures/                 # Plots
├── scripts/elpv_benchmark/
│   ├── 00_verify_environment.py
│   ├── 01_prepare_data.py
│   ├── 02_generate_configs.py
│   ├── 03_validate_setup.py
│   ├── 04_run_single.py
│   ├── 05_aggregate_results.py
│   ├── 06_compute_friedman.py
│   ├── 07_track_experiments.py
│   └── slurm/                   # SLURM job scripts
├── semilearn/                   # Core SSL library
├── train.py                     # Main training script
├── VISION.md                    # Project roadmap
└── README.md
```

---

## 📊 Output Files

Each experiment generates:

```
results/elpv_benchmark/experiments/<experiment_name>/
├── metrics.json              # Complete metrics
├── metrics_checkpoint.json   # Incremental checkpoint
├── status.json               # RUNNING/COMPLETED/INTERRUPTED
├── training_curve.csv        # Training dynamics
├── confusion_matrix.npy      # Test confusion matrix
├── test_logits.npy           # Raw model outputs
├── test_predictions.npz      # Predictions & labels
├── model_best.pth            # Best checkpoint
└── summary.txt               # Human-readable summary
```

---

## ⏱️ Time Estimates

| Phase | Dataset | Variant | Experiments | Time (4 parallel) |
|-------|---------|---------|-------------|-------------------|
| 1 | ELPV | Balanced | 216 | ~4 days |
| 2 | ELPV | 1:30 Imbalance | 216 | ~4 days |
| 3 | THED-PV | Balanced | 216 | ~4 days |
| 4 | THED-PV | 1:30 Imbalance | 216 | ~4 days |
| 5 | PVEL-AD | Balanced | 216 | ~4 days |
| 6 | PVEL-AD | 1:30 Imbalance | 216 | ~4 days |
| **Total** | | | **1,296** | **~24 days** |

---

## 📝 Citation

```bibtex
@misc{ansari2024sslthermal,
  author = {Ansari, Mohammed Yaqoob},
  title = {SSL-Thermal-Benchmarking: Semi-Supervised Learning for PV Defect Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YaqoobAnsari/SSL-Thermal-Benchmarking}
}
```

---

## 🙏 Acknowledgments

- **USB Framework**: [Microsoft Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning)
- **ELPV Dataset**: [zae-bayern/elpv-dataset](https://github.com/zae-bayern/elpv-dataset)
- **Funding**: TÜBİTAK Research Grant

---

## 📄 License

MIT License - see [LICENSE.txt](LICENSE.txt)

---

## 👤 Author

**Mohammed Yaqoob Ansari**  
GitHub: [@YaqoobAnsari](https://github.com/YaqoobAnsari)  
Email: ansarimohammedyaqoob01@gmail.com
