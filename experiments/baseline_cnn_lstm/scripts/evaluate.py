"""
Comprehensive Model Evaluation
"""

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from train import CNN_LSTM, ECGDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model():
    # Load test data
    print("Loading test data...")
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')
    
    test_dataset = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    print("Loading trained model...")
    model = CNN_LSTM(input_channels=12, sequence_length=1000).to(device)
    model.load_state_dict(torch.load('../saved_models/best_model.pth'))
    model.eval()
    
    # Get predictions
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, 
                                target_names=['Normal', 'IHD']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'IHD'],
                yticklabels=['Normal', 'IHD'])
    plt.title('Confusion Matrix - Baseline CNN-LSTM', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../results/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved confusion matrix")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Baseline CNN-LSTM', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/figures/roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved ROC curve")
    
    # Save metrics
    import pandas as pd
    results = {
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1],
        'roc_auc': [auc]
    }
    pd.DataFrame(results).to_csv('../results/metrics/test_results.csv', index=False)
    print("✓ Saved metrics to CSV")
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()
