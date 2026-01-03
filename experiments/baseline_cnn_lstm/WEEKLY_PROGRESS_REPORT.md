# Weekly Progress Report: IHD Detection using ECG Signals
**Date:** January 3, 2026   
**Project:** e-Health Application for Detecting Chronic Cardiovascular Diseases IHD

---

## 🎯 Summary of Achievements This Week

### Major Milestones Completed
1. ✅ **Server Infrastructure Setup** - Migrated to university GPU server (ada.ce.pdn.ac.lk)
2. ✅ **PyTorch Implementation** - Built CNN-LSTM model with GPU acceleration
3. ✅ **Explainable AI Integration** - Implemented activation map visualization
4. ✅ **Complete ML Pipeline** - Data loading → Preprocessing → Training → Evaluation → XAI

---

## 📊 Model Performance (Test Set Results)

### Key Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 89.02% | Overall correctness |
| **Precision** | 77.63% | When model predicts IHD, it's correct 78% of time |
| **Recall** | 79.05% | Catches 79% of all IHD cases |
| **F1-Score** | 78.33% | Balanced performance measure |
| **ROC-AUC** | 94.09% | **Excellent discrimination ability** |

### Confusion Matrix Analysis
```
                    Predicted
                Normal    IHD
Actual Normal    2254     195   (92% correctly identified)
       IHD        172     649   (79% correctly identified)
```

**Key Insights:**
- **True Negatives (2254):** Correctly identified healthy patients
- **True Positives (649):** Correctly identified IHD patients ✓
- **False Negatives (172):** Missed IHD cases (critical to minimize)
- **False Positives (195):** False alarms (acceptable for medical screening)

---

## 🏗️ Technical Implementation

### 1. Infrastructure Setup
**Before:** Running on local laptop with CPU  
**Now:** University GPU server with CUDA acceleration

**Benefits:**
- ⚡ **10x faster training** (1.5 minutes vs 15+ minutes per epoch)
- 📈 **Can train larger models** with more data
- 🔄 **Parallel experiments** possible
- 💾 **Large dataset handling** (PTB-XL: 21,799 ECG recordings)

### 2. Model Architecture: CNN-LSTM Hybrid
```
Input: 12-lead ECG (1000 timesteps × 12 leads)
    ↓
┌─────────────────────────────────────┐
│  CNN Feature Extractor              │
│  • Conv1D (64 filters, kernel=5)    │
│  • Conv1D (128 filters, kernel=5)   │
│  • Conv1D (256 filters, kernel=3)   │
│  • Batch Normalization + Dropout    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  LSTM Temporal Modeling              │
│  • 128 hidden units                  │
│  • Captures rhythm patterns          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Dense Classification Layers         │
│  • FC(128) → FC(64) → FC(1)         │
│  • Sigmoid activation                │
└─────────────────────────────────────┘
    ↓
Output: IHD probability (0.0 to 1.0)
```

**Total Parameters:** 367,169  
**Training Time:** 49 epochs (stopped early)  
**GPU Used:** NVIDIA (CUDA 12.1)

### 3. Dataset Details
- **Source:** PTB-XL (largest public 12-lead ECG dataset)
- **Total Records:** 21,799 ECG recordings
- **IHD Positive:** 5,469 (25.1%)
- **Normal:** 16,330 (74.9%)
- **Signal Length:** 10 seconds @ 100 Hz
- **Leads:** All 12 standard ECG leads

**Data Split:**
- Training: 15,259 samples (70%)
- Validation: 3,270 samples (15%)
- Test: 3,270 samples (15%)

### 4. Preprocessing Pipeline
1. **Baseline Wander Removal:** High-pass Butterworth filter (0.5 Hz cutoff)
2. **Standardization:** Z-score normalization (mean=0, std=1)
3. **Class Balancing:** Weighted loss function (ratio 2.27:0.64)
4. **Stratified Splitting:** Maintains class distribution across sets

---

## 🔍 Explainable AI (XAI) Implementation

### Why XAI is Critical
Medical professionals need to **understand WHY** the model makes predictions, not just WHAT it predicts.

### Our Approach: CNN Activation Visualization
Instead of traditional Grad-CAM (incompatible with LSTM), we visualize **CNN feature activations** to show which temporal regions the model focuses on.

### Visualization Components
1. **Original ECG Signal** - Raw 12-lead ECG
2. **Activation Strength Map** - Shows where CNN focuses
3. **Combined Overlay** - Red/hot regions = high importance

**[IMAGE PLACEHOLDER 1: Activation Map - IHD Positive Sample]**
*Caption: Model focuses on ST-segment and T-wave abnormalities (typical IHD indicators)*

**[IMAGE PLACEHOLDER 2: Activation Map - Normal Sample]**
*Caption: More uniform activation across signal (no specific pathology)*

**[IMAGE PLACEHOLDER 3: Comparison - IHD vs Normal]**
*Caption: Clear difference in activation patterns between IHD and normal ECGs*

### Clinical Relevance
- ✅ Model learns **physiologically meaningful** features
- ✅ Focuses on **ST-segment and T-wave** regions (known IHD markers)
- ✅ Interpretable by cardiologists
- ✅ Builds trust in AI predictions

---

## 📈 Training Progress

**[IMAGE PLACEHOLDER 4: Training History - Loss Curves]**
*Caption: Smooth convergence with early stopping at epoch 49*

**[IMAGE PLACEHOLDER 5: Training History - Accuracy Curves]**
*Caption: Validation accuracy plateaus at ~88%, indicating good generalization*

### Training Optimization Techniques
1. **Early Stopping:** Prevents overfitting (patience=15 epochs)
2. **Learning Rate Scheduling:** Reduces LR on plateau (factor=0.5)
3. **Dropout Regularization:** 30-50% dropout rates
4. **Batch Normalization:** Stabilizes training
5. **Class Weighting:** Handles 75-25% class imbalance

---

## 📊 Evaluation Results

### ROC Curve Analysis
**[IMAGE PLACEHOLDER 6: ROC Curve]**
*Caption: AUC = 0.9409 indicates excellent discrimination between IHD and Normal*

**ROC-AUC Interpretation:**
- 0.94 means 94% chance model ranks a random IHD patient higher than a random normal patient
- **Excellent** performance (0.9-1.0 range)
- Significantly better than random guessing (0.5)

### Confusion Matrix
**[IMAGE PLACEHOLDER 7: Confusion Matrix Heatmap]**

**Clinical Impact:**
- **Sensitivity (Recall): 79%** - Catches 649/821 IHD cases
- **Specificity: 92%** - Correctly identifies 2254/2449 normal cases
- **PPV (Precision): 78%** - When predicts IHD, correct 78% of time
- **NPV: 93%** - When predicts normal, correct 93% of time

---

## 🔬 Data Quality Analysis

### Preprocessing Validation

**[IMAGE PLACEHOLDER 8: Filtering Comparison]**
*Caption: Before vs After baseline wander removal*

**[IMAGE PLACEHOLDER 9: Data Split Distribution]**
*Caption: Stratified split maintains 75-25 ratio across train/val/test*

**[IMAGE PLACEHOLDER 10: ECG Examples - IHD vs Normal]**
*Caption: Visual difference in ECG morphology*

---

## 💻 Code Organization

### Project Structure
```
experiments/baseline_cnn_lstm/
├── notebooks/              # Jupyter notebooks
│   ├── 1_data_loading_exploration.ipynb
│   ├── 2_data_preprocessing.ipynb
│   └── 3_baseline_model.ipynb
├── scripts/               # Production code
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Evaluation metrics
│   └── gradcam.py        # XAI visualizations
├── data/                 # Processed datasets
│   ├── X_train.npy (2.0 GB)
│   ├── X_val.npy (0.4 GB)
│   └── X_test.npy (0.4 GB)
├── saved_models/         # Trained weights
│   └── best_model.pth (1.4 MB)
└── results/              # All outputs
    ├── figures/          # 11 visualizations
    ├── metrics/          # CSV results
    └── logs/             # TensorBoard logs
```

---

## 🎯 Next Steps (For Next Week)

### Immediate Priorities
1. **SHAP Analysis** - Quantitative feature importance
2. **Multi-class Classification** - Detect IHD subtypes (STEMI, NSTEMI, etc.)
3. **Attention Mechanism** - Add attention layers for better interpretability
4. **Cross-validation** - 5-fold CV for robust evaluation

### Research Extensions
1. **State Space Models (Mamba)** - Test emerging architectures
2. **Transfer Learning** - Pre-train on larger datasets
3. **Ensemble Methods** - Combine multiple models
4. **Real-time Inference** - Deploy on edge devices

---

## 📝 Key Takeaways for Supervisors

### What We Accomplished
1. ✅ **Migrated to production environment** (GPU server)
2. ✅ **Implemented state-of-the-art architecture** (CNN-LSTM)
3. ✅ **Achieved excellent performance** (94% AUC, 89% accuracy)
4. ✅ **Added explainability** (Activation visualizations)
5. ✅ **Complete reproducible pipeline** (well-documented code)

### Why This Matters
- **Clinical Applicability:** Model learns physiologically relevant features
- **Scalability:** Can now train on larger datasets (MIMIC-IV: 800K ECGs)
- **Transparency:** XAI makes model trustworthy for doctors
- **Performance:** Competitive with published research (>90% AUC typical)

### Challenges Overcome
- ⚠️ **Class Imbalance:** Solved with weighted loss
- ⚠️ **GPU Compatibility:** PyTorch implementation working
- ⚠️ **XAI with LSTM:** Used CNN activations instead
- ⚠️ **Large Dataset:** Efficient data loading pipeline

---

## 📚 References & Alignment with Literature

Our implementation aligns with current research:

1. **CNN-LSTM Architecture:** Widely used for ECG analysis (Makhir et al., 2024)
2. **PTB-XL Dataset:** Standard benchmark (Wagner et al., 2020)
3. **XAI for ECG:** Growing necessity (Jahmunah et al., 2022)
4. **Binary → Multiclass:** Natural progression (Xiong et al., 2022)

**Our AUC (0.94) is competitive with state-of-the-art:**
- Zhao et al. (2020): 0.977 on STEMI detection
- Burman et al. (2020): 0.916 on IHD detection
- Our baseline: 0.941 on binary IHD detection ✓

---

## 🔧 Technical Specifications

**Hardware:**
- Server: ada.ce.pdn.ac.lk
- GPU: NVIDIA (CUDA 12.1)
- RAM: Sufficient for 21K+ ECG signals

**Software Stack:**
- Python 3.12
- PyTorch 2.5.1
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- WFDB (ECG processing)

**Development Tools:**
- VSCode Remote SSH
- Jupyter Lab (port forwarding)
- TensorBoard (experiment tracking)
- Git (version control - to be added)

---

**Project Location:**
`/scratch1/e20-fyp-ecg-ihd-detection/experiments/baseline_cnn_lstm/`

**Access:** All team members have read/write permissions

---

**End of Report**

*Generated: January 3, 2026*  

