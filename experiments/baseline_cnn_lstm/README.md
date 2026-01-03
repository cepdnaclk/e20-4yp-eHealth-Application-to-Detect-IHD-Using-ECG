# Baseline CNN-LSTM Model for IHD Detection

## Overview
This folder contains the baseline CNN-LSTM implementation for detecting Ischemic Heart Disease (IHD) from 12-lead ECG signals.

## Performance Metrics
- **Accuracy:** 89.02%
- **Precision:** 77.63%
- **Recall:** 79.05%
- **F1-Score:** 78.33%
- **ROC-AUC:** 94.09%

## Folder Structure
```
baseline_cnn_lstm/
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Python scripts for training/evaluation
├── data/                   # Preprocessed data (gitignored)
├── saved_models/           # Trained model weights
├── results/                # Outputs and visualizations
│   ├── figures/           # All plots and visualizations
│   ├── metrics/           # Performance metrics
│   └── logs/              # Training logs
└── README.md              # This file
```

## Quick Start

### 1. Setup Environment
```bash
cd /scratch1/e20-fyp-ecg-ihd-detection
conda activate ecg-ihd
```

### 2. Train Model
```bash
cd experiments/baseline_cnn_lstm/scripts
python train.py
```

### 3. Evaluate Model
```bash
python evaluate.py
```

### 4. Generate XAI Visualizations
```bash
python gradcam.py
```

## Files Description

### Notebooks
- `1_data_loading_exploration.ipynb` - Load and explore PTB-XL dataset
- `2_data_preprocessing.ipynb` - Preprocessing pipeline
- `3_baseline_model.ipynb` - Original TensorFlow model (deprecated)

### Scripts
- `train.py` - PyTorch training pipeline with GPU support
- `evaluate.py` - Comprehensive evaluation metrics
- `gradcam.py` - XAI activation map visualization

### Results
- **11 visualization figures** in `results/figures/`
- **Test metrics CSV** in `results/metrics/`
- **TensorBoard logs** in `results/logs/`

## Model Architecture
```
Input: (batch, 1000, 12)
↓
CNN Feature Extractor (Conv1D layers)
↓
LSTM Temporal Modeling (128 units)
↓
Dense Classification (128 → 64 → 1)
↓
Output: IHD probability [0, 1]
```

**Parameters:** 367,169  
**Training Time:** ~49 epochs with early stopping

## Requirements
See `/scratch1/e20-fyp-ecg-ihd-detection/requirements.txt`

## Dataset
- **Source:** PTB-XL
- **Total:** 21,799 ECG recordings
- **Split:** 70% train, 15% val, 15% test
- **Preprocessing:** Baseline removal, standardization, stratified split

## Authors
- e20054 (M. L. De Croos Rubin)
- e20276 (S. M. N. N. Padeniya)
- e20342 (C. A. Rupasinghe)

## Date
January 2026
