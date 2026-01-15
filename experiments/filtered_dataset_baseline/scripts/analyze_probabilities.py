"""
Analyze Model Probability Outputs
Understanding confidence and calibration
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from train import CNN_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_model_probabilities():
    print("="*70)
    print("MODEL PROBABILITY ANALYSIS")
    print("="*70)
    
    # Load model and data
    print("\nLoading model and test data...")
    model = CNN_LSTM(input_channels=12, sequence_length=1000).to(device)
    model.load_state_dict(torch.load('../saved_models/best_model.pth', 
                                     map_location=device,
                                     weights_only=True))
    model.eval()
    
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')
    
    # Get predictions
    print("Getting probability predictions...")
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor).cpu().numpy()
    
    # Separate by true class
    normal_probs = y_pred_prob[y_test == 0]  # True normals
    mi_probs = y_pred_prob[y_test == 1]      # True MI cases
    
    # ============================================================
    # ANALYSIS 1: Probability Distribution by True Class
    # ============================================================
    print("\n" + "="*70)
    print("1. PROBABILITY DISTRIBUTION BY TRUE CLASS")
    print("="*70)
    
    print("\n📊 For TRUE NORMAL cases (y=0):")
    print(f"  Mean probability: {normal_probs.mean():.4f}")
    print(f"  Median probability: {np.median(normal_probs):.4f}")
    print(f"  Std deviation: {normal_probs.std():.4f}")
    print(f"  Min: {normal_probs.min():.4f}, Max: {normal_probs.max():.4f}")
    print(f"  95th percentile: {np.percentile(normal_probs, 95):.4f}")
    
    print("\n📊 For TRUE MI cases (y=1):")
    print(f"  Mean probability: {mi_probs.mean():.4f}")
    print(f"  Median probability: {np.median(mi_probs):.4f}")
    print(f"  Std deviation: {mi_probs.std():.4f}")
    print(f"  Min: {mi_probs.min():.4f}, Max: {mi_probs.max():.4f}")
    print(f"  5th percentile: {np.percentile(mi_probs, 5):.4f}")
    
    # ============================================================
    # ANALYSIS 2: Confidence Analysis
    # ============================================================
    print("\n" + "="*70)
    print("2. MODEL CONFIDENCE ANALYSIS")
    print("="*70)
    
    # High confidence correct predictions
    high_conf_threshold = 0.9
    high_conf_correct_mi = np.sum((mi_probs >= high_conf_threshold))
    high_conf_correct_normal = np.sum((normal_probs <= 0.1))
    
    print(f"\n🎯 High Confidence Predictions (>90% or <10%):")
    print(f"  MI cases with prob ≥ 0.90: {high_conf_correct_mi}/{len(mi_probs)} ({high_conf_correct_mi/len(mi_probs)*100:.1f}%)")
    print(f"  Normal cases with prob ≤ 0.10: {high_conf_correct_normal}/{len(normal_probs)} ({high_conf_correct_normal/len(normal_probs)*100:.1f}%)")
    
    # Uncertain predictions (near threshold)
    uncertain_mi = np.sum((mi_probs >= 0.4) & (mi_probs <= 0.6))
    uncertain_normal = np.sum((normal_probs >= 0.4) & (normal_probs <= 0.6))
    
    print(f"\n⚠️  Uncertain Predictions (0.4-0.6 range):")
    print(f"  MI cases in uncertain range: {uncertain_mi}/{len(mi_probs)} ({uncertain_mi/len(mi_probs)*100:.1f}%)")
    print(f"  Normal cases in uncertain range: {uncertain_normal}/{len(normal_probs)} ({uncertain_normal/len(normal_probs)*100:.1f}%)")
    
    # ============================================================
    # ANALYSIS 3: Calibration Analysis
    # ============================================================
    print("\n" + "="*70)
    print("3. PROBABILITY CALIBRATION")
    print("="*70)
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_pred_prob, n_bins=10, strategy='uniform'
    )
    
    print("\nCalibration Table:")
    print("(Does 'X% probability' really mean X% are MI?)")
    print("-" * 50)
    print("Predicted Prob | Actual MI Rate | Calibration Gap")
    print("-" * 50)
    for pred, actual in zip(mean_predicted_value, fraction_of_positives):
        gap = abs(pred - actual)
        status = "✓" if gap < 0.10 else "⚠" if gap < 0.20 else "✗"
        print(f"    {pred:.2f}       |     {actual:.2f}      |    {gap:.2f} {status}")
    
    # ============================================================
    # ANALYSIS 4: Clinical Interpretation
    # ============================================================
    print("\n" + "="*70)
    print("4. CLINICAL INTERPRETATION")
    print("="*70)
    
    # Threshold analysis
    threshold = 0.5
    y_pred = (y_pred_prob > threshold).astype(int)
    
    # True Positives with probabilities
    tp_indices = (y_test == 1) & (y_pred == 1)
    tp_probs = y_pred_prob[tp_indices]
    
    # False Negatives with probabilities
    fn_indices = (y_test == 1) & (y_pred == 0)
    fn_probs = y_pred_prob[fn_indices]
    
    # False Positives with probabilities
    fp_indices = (y_test == 0) & (y_pred == 1)
    fp_probs = y_pred_prob[fp_indices]
    
    print(f"\n✅ True Positives (Correctly identified MI): {len(tp_probs)} cases")
    print(f"   Average confidence: {tp_probs.mean():.4f}")
    print(f"   Confidence range: [{tp_probs.min():.4f}, {tp_probs.max():.4f}]")
    
    print(f"\n❌ False Negatives (Missed MI): {len(fn_probs)} cases")
    print(f"   Average probability: {fn_probs.mean():.4f}")
    print(f"   Why missed: Probabilities were below threshold (0.5)")
    print(f"   Closest to threshold: {fn_probs.max():.4f}")
    
    print(f"\n⚠️  False Positives (False alarms): {len(fp_probs)} cases")
    print(f"   Average confidence: {fp_probs.mean():.4f}")
    print(f"   Confidence range: [{fp_probs.min():.4f}, {fp_probs.max():.4f}]")
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram of probabilities by true class
    ax1 = axes[0, 0]
    ax1.hist(normal_probs, bins=50, alpha=0.6, label='True Normal', color='blue', density=True)
    ax1.hist(mi_probs, bins=50, alpha=0.6, label='True MI', color='red', density=True)
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Probability Distribution by True Class', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    ax2 = axes[0, 1]
    box_data = [normal_probs, mi_probs]
    bp = ax2.boxplot(box_data, labels=['True Normal', 'True MI'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_ylabel('Predicted Probability', fontsize=12)
    ax2.set_title('Probability Ranges by True Class', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Plot 3: Calibration curve
    ax3 = axes[1, 0]
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax3.plot(mean_predicted_value, fraction_of_positives, 
             'ro-', linewidth=2, markersize=8, label='Model Calibration')
    ax3.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax3.set_ylabel('Fraction of Positives (True MI Rate)', fontsize=12)
    ax3.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Plot 4: Confidence by outcome
    ax4 = axes[1, 1]
    outcomes = ['True Positive\n(Correct MI)', 
                'False Negative\n(Missed MI)', 
                'False Positive\n(False Alarm)']
    means = [tp_probs.mean(), fn_probs.mean(), fp_probs.mean()]
    colors = ['green', 'red', 'orange']
    
    bars = ax4.bar(outcomes, means, color=colors, alpha=0.7)
    ax4.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax4.set_ylabel('Average Probability', fontsize=12)
    ax4.set_title('Model Confidence by Outcome', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/figures/probability_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: probability_analysis.png")
    
    # ============================================================
    # SAVE DETAILED RESULTS
    # ============================================================
    results = {
        'metric': [
            'Normal Mean Prob', 'Normal Median Prob', 'Normal Std',
            'MI Mean Prob', 'MI Median Prob', 'MI Std',
            'High Conf MI (≥0.9)', 'High Conf Normal (≤0.1)',
            'Uncertain MI (0.4-0.6)', 'Uncertain Normal (0.4-0.6)',
            'TP Average Confidence', 'FN Average Prob', 'FP Average Confidence'
        ],
        'value': [
            normal_probs.mean(), np.median(normal_probs), normal_probs.std(),
            mi_probs.mean(), np.median(mi_probs), mi_probs.std(),
            f"{high_conf_correct_mi}/{len(mi_probs)} ({high_conf_correct_mi/len(mi_probs)*100:.1f}%)",
            f"{high_conf_correct_normal}/{len(normal_probs)} ({high_conf_correct_normal/len(normal_probs)*100:.1f}%)",
            f"{uncertain_mi}/{len(mi_probs)} ({uncertain_mi/len(mi_probs)*100:.1f}%)",
            f"{uncertain_normal}/{len(normal_probs)} ({uncertain_normal/len(normal_probs)*100:.1f}%)",
            tp_probs.mean(), fn_probs.mean(), fp_probs.mean()
        ]
    }
    
    pd.DataFrame(results).to_csv('../results/metrics/probability_analysis.csv', index=False)
    print("✅ Saved: probability_analysis.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    analyze_model_probabilities()
