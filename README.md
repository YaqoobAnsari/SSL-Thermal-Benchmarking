# SSL-Thermal-Benchmarking

**A Comprehensive Benchmark for Semi-Supervised Learning on Solar Cell Defect Detection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)

---

## 📋 Overview

This repository provides a rigorous evaluation framework for Semi-Supervised Learning (SSL) algorithms on the **ELPV (Electroluminescence Photovoltaic) dataset** for solar cell defect detection. The benchmark evaluates **9 SSL algorithms** across **4 backbone architectures** with varying amounts of labeled data.

### Key Features

- 🔬 **9 SSL Algorithms**: FixMatch, FlexMatch, FreeMatch, SoftMatch, UDA, MeanTeacher, ABC, DARP, DASO
- 🏗️ **4 Backbone Architectures**: ResNet-18, ResNet-50, WideResNet-28-2, ViT-B/16
- 📊 **7 Label Amounts**: 10, 25, 50, 100, 200, 400, 800 labeled samples
- 🎯 **Comprehensive Metrics**: Accuracy, Balanced Accuracy, Precision, Recall, F1, Cross-Entropy Loss
- 📈 **Statistical Analysis**: Friedman Rank with Nemenyi post-hoc test
- ✅ **Reproducible**: Fixed data splits, seed control, checkpointing

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/YaqoobAnsari/SSL-Thermal-Benchmarking.git
cd SSL-Thermal-Benchmarking

# Create conda environment
conda create -n ssl_bench python=3.10 -y
conda activate ssl_bench

# Install PyTorch (with CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
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

# 3. Generate experiment configs
python scripts/elpv_benchmark/02_generate_configs.py

# 4. Validate setup
python scripts/elpv_benchmark/03_validate_setup.py

# 5. Run a single experiment
python scripts/elpv_benchmark/04_run_single.py --config configs/elpv_benchmark/experiments/fixmatch_wrn_28_2_100_seed0.yaml

# 6. After all experiments, aggregate results
python scripts/elpv_benchmark/05_aggregate_results.py

# 7. Compute Friedman rankings
python scripts/elpv_benchmark/06_compute_friedman.py

# 8. Track experiment progress
python scripts/elpv_benchmark/07_track_experiments.py
```

---

## 📁 Project Structure

```
SSL-Thermal-Benchmarking/
├── configs/
│   └── elpv_benchmark/
│       ├── base/                    # Base configs per backbone
│       ├── experiments/             # 840 experiment configs
│       └── master_config.yaml       # Locked hyperparameters
├── data/
│   └── elpv/
│       └── splits/                  # Pre-generated data splits
│           ├── val_indices.npy
│           ├── test_indices.npy
│           └── train/seed{0,1,2}/   # Labeled/unlabeled splits
├── results/
│   └── elpv_benchmark/
│       ├── experiments/             # Individual experiment outputs
│       ├── aggregated/              # Aggregated results & CSVs
│       └── figures/                 # Plots and visualizations
├── scripts/
│   └── elpv_benchmark/
│       ├── 00_verify_environment.py
│       ├── 01_prepare_data.py
│       ├── 02_generate_configs.py
│       ├── 03_validate_setup.py
│       ├── 04_run_single.py
│       ├── 05_aggregate_results.py
│       ├── 06_compute_friedman.py
│       ├── 07_track_experiments.py
│       └── slurm/                   # SLURM job scripts
├── semilearn/                       # Core SSL library (USB framework)
├── train.py                         # Main training script
└── eval.py                          # Evaluation script
```

---

## 🔬 Experiment Design

### Dataset: ELPV

- **Task**: Binary classification (functional vs. defective solar cells)
- **Total Images**: 2,624 grayscale EL images (300×300)
- **Split**: 70% train / 15% validation / 15% test
- **Class Distribution**: ~69% functional, ~31% defective

### SSL Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| FixMatch | Consistency + Pseudo-labeling | Sohn et al., NeurIPS 2020 |
| FlexMatch | Curriculum Pseudo-labeling | Zhang et al., NeurIPS 2021 |
| FreeMatch | Self-adaptive Thresholding | Wang et al., ICLR 2023 |
| SoftMatch | Soft Pseudo-labels | Chen et al., ICLR 2023 |
| UDA | Unsupervised Data Augmentation | Xie et al., NeurIPS 2020 |
| MeanTeacher | Exponential Moving Average | Tarvainen & Valpola, NeurIPS 2017 |
| ABC | Adaptive Bias Correction | Lee et al., CVPR 2021 |
| DARP | Distribution Alignment | Kim et al., NeurIPS 2020 |
| DASO | Distribution-Aware Self-Training | Oh et al., CVPR 2022 |

### Backbone Architectures

| Backbone | Input Size | Parameters | Pretrained |
|----------|------------|------------|------------|
| WideResNet-28-2 | 96×96 | 1.5M | No |
| ResNet-18 | 224×224 | 11.7M | ImageNet |
| ResNet-50 | 224×224 | 25.6M | ImageNet |
| ViT-B/16 | 224×224 | 86M | ImageNet |

### Evaluation Metrics

- **Top-1 Accuracy**: Standard classification accuracy
- **Balanced Accuracy**: Mean of per-class recall (handles imbalance)
- **Precision** (macro): Averaged across classes
- **Recall** (macro): Averaged across classes
- **F1 Score** (macro): Harmonic mean of precision and recall
- **Cross-Entropy Loss**: Model confidence
- **Friedman Rank**: Statistical ranking across all conditions

---

## 📊 Results

Results are saved in multiple formats:

### Per-Experiment Outputs
```
results/elpv_benchmark/experiments/<experiment_name>/
├── metrics.json           # Complete metrics
├── metrics_checkpoint.json # Incremental checkpoint
├── status.json            # Experiment status
├── training_curve.csv     # Training dynamics
├── confusion_matrix.npy   # Test confusion matrix
├── test_logits.npy        # Raw model outputs
├── test_predictions.npz   # Predictions & labels
├── model_best.pth         # Best model checkpoint
└── summary.txt            # Human-readable summary
```

### Aggregated Results
```
results/elpv_benchmark/aggregated/
├── master_results.csv              # All experiments
├── summary_by_algorithm.csv        # Grouped by SSL algorithm
├── summary_by_backbone.csv         # Grouped by backbone
├── summary_by_labels.csv           # Grouped by label count
├── friedman_ranks.csv              # Friedman rankings
└── statistical_tests.json          # Significance tests
```

---

## 🖥️ SLURM Cluster Usage

For running on HPC clusters with SLURM:

```bash
# Submit all experiments for a backbone
cd scripts/elpv_benchmark/slurm
./submit_backbone.sh wrn_28_2
./submit_backbone.sh resnet18
./submit_backbone.sh resnet50
./submit_backbone.sh vit_b_16

# Monitor jobs
squeue -u $USER

# Track progress
python scripts/elpv_benchmark/07_track_experiments.py
```

---

## 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{ansari2024sslelpv,
  author = {Ansari, Mohammed Yaqoob},
  title = {SSL-Thermal-Benchmarking: A Comprehensive Benchmark for Semi-Supervised Learning on Solar Cell Defect Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YaqoobAnsari/SSL-Thermal-Benchmarking}
}
```

---

## 🙏 Acknowledgments

- **USB Framework**: Built on [USB: A Unified Semi-supervised learning Benchmark](https://github.com/microsoft/Semi-supervised-learning)
- **ELPV Dataset**: [A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery](https://github.com/zae-bayern/elpv-dataset)
- **Funding**: TÜBİTAK Research Grant

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

---

## 👤 Author

**Mohammed Yaqoob Ansari**
- GitHub: [@YaqoobAnsari](https://github.com/YaqoobAnsari)
- Email: ansarimohammedyaqoob01@gmail.com
