"""
Comprehensive Evaluation Script
Compares new filtered dataset results with old baseline
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from train import CNN_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return all metrics
    """
    model.eval()
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Predictions
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor).cpu().numpy()
    
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_prob)
    }
    
    return metrics, y_pred, y_pred_prob

def plot_confusion_matrix(y_test, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'MI'],
                yticklabels=['Normal', 'MI'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Filtered Dataset', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm[i].sum() * 100
            plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_roc_curve(y_test, y_pred_prob, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Filtered Dataset', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def create_comparison_table(new_metrics):
    """
    Create comparison table: Old baseline vs New filtered dataset
    """
    # Old baseline results (from previous training on full dataset)
    old_metrics = {
        'Dataset': 'Old Baseline (21,799 ECGs)',
        'Accuracy': 0.8902,
        'Precision': 0.7763,
        'Recall': 0.7905,
        'F1-Score': 0.7833,
        'ROC-AUC': 0.9409,
        'Notes': 'All confidence levels, ECG-wise split'
    }
    
    # New filtered dataset results
    new_metrics_row = {
        'Dataset': 'New Filtered (7,593 ECGs)',
        'Accuracy': new_metrics['accuracy'],
        'Precision': new_metrics['precision'],
        'Recall': new_metrics['recall'],
        'F1-Score': new_metrics['f1'],
        'ROC-AUC': new_metrics['roc_auc'],
        'Notes': '100% confidence only, patient-wise split'
    }
    
    # Calculate improvements
    improvements = {
        'Dataset': 'Improvement',
        'Accuracy': new_metrics['accuracy'] - old_metrics['Accuracy'],
        'Precision': new_metrics['precision'] - old_metrics['Precision'],
        'Recall': new_metrics['recall'] - old_metrics['Recall'],
        'F1-Score': new_metrics['f1'] - old_metrics['F1-Score'],
        'ROC-AUC': new_metrics['roc_auc'] - old_metrics['ROC-AUC'],
        'Notes': 'Δ (New - Old)'
    }
    
    df = pd.DataFrame([old_metrics, new_metrics_row, improvements])
    
    return df

def main():
    print("="*70)
    print("MODEL EVALUATION - FILTERED DATASET")
    print("="*70)
    
    # Load test data
    print("\nLoading test data...")
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')
    
    print(f"Test set: {X_test.shape}")
    print(f"Class distribution: Normal={np.sum(y_test==0)}, MI={np.sum(y_test==1)}")
    
    # Load best model
    print("\nLoading best model...")
    model = CNN_LSTM(input_channels=12, sequence_length=1000).to(device)
    model.load_state_dict(torch.load('../saved_models/best_model.pth', 
                                     map_location=device,
                                     weights_only=True))
    
    # Evaluate
    print("\nEvaluating model...")
    metrics, y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    
    # Print results
    print("\n" + "="*70)
    print("TEST SET RESULTS - NEW FILTERED DATASET")
    print("="*70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("="*70)
    
    # Classification report
    print("\nCLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'MI']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nCONFUSION MATRIX")
    print("="*70)
    print(f"                Predicted")
    print(f"              Normal    MI")
    print(f"Actual Normal  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       MI      {cm[1,0]:5d}  {cm[1,1]:5d}")
    print("="*70)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    
    print("\nCLINICAL METRICS")
    print("="*70)
    print(f"Sensitivity (Recall): {sensitivity:.4f} ({tp}/{tp+fn})")
    print(f"Specificity:          {specificity:.4f} ({tn}/{tn+fp})")
    print(f"PPV (Precision):      {ppv:.4f} ({tp}/{tp+fp})")
    print(f"NPV:                  {npv:.4f} ({tn}/{tn+fn})")
    print("="*70)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred, '../results/figures/confusion_matrix.png')
    plot_roc_curve(y_test, y_pred_prob, '../results/figures/roc_curve.png')
    
    # Create comparison table
    print("\nCreating comparison table...")
    comparison_df = create_comparison_table(metrics)
    
    print("\n" + "="*70)
    print("COMPARISON: OLD BASELINE vs NEW FILTERED DATASET")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    # Save comparison
    comparison_df.to_csv('../results/metrics/baseline_comparison.csv', index=False)
    print("\n✅ Saved: baseline_comparison.csv")
    
    # Save detailed metrics
    detailed_metrics = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    pd.DataFrame([detailed_metrics]).to_csv('../results/metrics/test_metrics.csv', index=False)
    print("✅ Saved: test_metrics.csv")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    acc_improvement = metrics['accuracy'] - 0.8902
    auc_improvement = metrics['roc_auc'] - 0.9409
    
    if acc_improvement > 0:
        print(f"✅ Accuracy IMPROVED by {acc_improvement*100:.2f}%")
    else:
        print(f"⚠️  Accuracy decreased by {abs(acc_improvement)*100:.2f}%")
    
    if auc_improvement > 0:
        print(f"✅ ROC-AUC IMPROVED by {auc_improvement:.4f}")
    else:
        print(f"⚠️  ROC-AUC decreased by {abs(auc_improvement):.4f}")
    
    if metrics['recall'] > 0.7905:
        print(f"✅ Recall IMPROVED to {metrics['recall']:.4f}")
    else:
        print(f"⚠️  Recall decreased to {metrics['recall']:.4f}")
    
    print("\nKey Findings:")
    if metrics['roc_auc'] >= 0.95:
        print("  • Excellent discrimination ability (AUC ≥ 0.95)")
    if metrics['recall'] >= 0.85:
        print("  • High sensitivity - catching most MI cases")
    if specificity >= 0.90:
        print("  • High specificity - few false alarms")
    
    print("\nBenefits of filtered dataset:")
    print("  • 100% confidence labels = cleaner training signal")
    print("  • Patient-wise split = no data leakage")
    print("  • Smaller dataset = faster training (1.7 min vs ~5 min)")
    print("  • Better quality control = more reliable results")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
