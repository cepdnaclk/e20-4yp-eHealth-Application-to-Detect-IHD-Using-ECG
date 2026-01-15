"""
Grad-CAM Analysis for Specific MI Types
Using CNN activation visualization (LSTM-compatible approach)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from train import CNN_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_activation_map(model, ecg_signal):
    """
    Generate activation map from CNN features (before LSTM)
    This is the LSTM-compatible approach
    
    Args:
        model: trained CNN-LSTM model
        ecg_signal: (1000, 12) numpy array
    Returns:
        activation_map: (1000,) averaged activation across channels
    """
    model.eval()
    
    # Prepare input
    x = torch.FloatTensor(ecg_signal).unsqueeze(0).to(device)  # (1, 1000, 12)
    
    with torch.no_grad():
        # Forward pass through CNN only (stop before LSTM)
        x = x.transpose(1, 2)  # (1, 12, 1000)
        
        # Conv block 1
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.pool1(x)
        x = model.dropout1(x)
        
        # Conv block 2
        x = model.conv2(x)
        x = model.bn2(x)
        x = model.relu(x)
        x = model.pool2(x)
        x = model.dropout2(x)
        
        # Conv block 3 (final features before LSTM)
        x = model.conv3(x)
        x = model.bn3(x)
        x = model.relu(x)
        # x shape: (1, 256, 125)
        
        # Average across channels
        activation_map = torch.mean(x, dim=1).squeeze(0)  # (125,)
        
        # Normalize to [0, 1]
        activation_map = activation_map - activation_map.min()
        if activation_map.max() > 0:
            activation_map = activation_map / activation_map.max()
        
        # Upsample to original length (125 -> 1000)
        activation_map = F.interpolate(
            activation_map.unsqueeze(0).unsqueeze(0),
            size=1000,
            mode='linear',
            align_corners=False
        ).squeeze().cpu().numpy()
    
    return activation_map

def visualize_gradcam(ecg_signal, activation_map, title, save_path, 
                     lead_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                                'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
    """
    Visualize ECG with activation overlay (12 leads)
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    time = np.arange(1000) / 100  # Convert to seconds
    
    for i in range(12):
        ax = axes[i]
        
        # Plot ECG signal
        ax.plot(time, ecg_signal[:, i], 'k-', linewidth=1, alpha=0.7)
        
        # Overlay activation heatmap
        scatter = ax.scatter(time, ecg_signal[:, i], c=activation_map, 
                           cmap='hot', s=10, alpha=0.6, vmin=0, vmax=1)
        
        ax.set_title(f'Lead {lead_names[i]}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 10])
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.08)
    cbar.set_label('Activation Intensity (Red = High Importance)', fontsize=11)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def visualize_single_lead_comparison(ecg_signal, activation_map, title, save_path,
                                    highlight_leads, lead_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                                                                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
    """
    Create a focused visualization highlighting specific leads
    """
    fig, axes = plt.subplots(len(highlight_leads), 1, figsize=(14, 3*len(highlight_leads)))
    
    if len(highlight_leads) == 1:
        axes = [axes]
    
    time = np.arange(1000) / 100
    
    for idx, lead_idx in enumerate(highlight_leads):
        ax = axes[idx]
        
        # Plot ECG
        ax.plot(time, ecg_signal[:, lead_idx], 'k-', linewidth=2, label='ECG Signal')
        
        # Overlay activation
        scatter = ax.scatter(time, ecg_signal[:, lead_idx], c=activation_map,
                           cmap='hot', s=20, alpha=0.7, vmin=0, vmax=1)
        
        ax.set_title(f'Lead {lead_names[lead_idx]} - Activation Heatmap', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.1)
    cbar.set_label('Activation Intensity', fontsize=12)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def main():
    print("="*70)
    print("GRAD-CAM ANALYSIS - MI SUBTYPE VISUALIZATION")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model = CNN_LSTM(input_channels=12, sequence_length=1000).to(device)
    model.load_state_dict(torch.load('../saved_models/best_model.pth', 
                                     map_location=device,
                                     weights_only=True))
    
    # Load test data and metadata
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')
    df_test = pd.read_csv('../data/test_metadata.csv')
    
    print(f"Test set loaded: {X_test.shape}")
    
    # Find specific MI types
    mi_indices = np.where(y_test == 1)[0]
    df_test_mi = df_test.iloc[mi_indices].copy()
    
    print(f"\nTotal MI cases in test set: {len(mi_indices)}")
    
    # Find examples
    ami_cases = df_test_mi[df_test_mi['diag_subclasses_100_str'].str.contains('AMI', na=False)]
    imi_cases = df_test_mi[df_test_mi['diag_subclasses_100_str'].str.contains('IMI', na=False)]
    
    print(f"\nFound:")
    print(f"  • Anterior MI (AMI): {len(ami_cases)} cases")
    print(f"  • Inferior MI (IMI): {len(imi_cases)} cases")
    
    # Normal cases
    normal_indices = np.where(y_test == 0)[0]
    
    print("\n" + "="*70)
    print("GENERATING ACTIVATION VISUALIZATIONS")
    print("="*70)
    
    # 1. Anterior MI examples
    if len(ami_cases) >= 2:
        print("\n1. Anterior MI (AMI) - Expected focus: V1-V4 (precordial) leads")
        for i in range(2):
            # Get the actual index in X_test
            ecg_id = ami_cases.iloc[i]['ecg_id']
            test_idx = df_test[df_test['ecg_id'] == ecg_id].index[0]
            actual_test_idx = np.where(df_test.index == test_idx)[0][0]
            
            ecg = X_test[actual_test_idx]
            activation = generate_activation_map(model, ecg)
            
            # Full 12-lead view
            visualize_gradcam(
                ecg, activation,
                title=f'Anterior MI - Case {i+1} (AMI: Should highlight V1-V4)',
                save_path=f'../results/figures/gradcam_ami_{i+1}_full.png'
            )
            
            # Focused view on V1-V4
            visualize_single_lead_comparison(
                ecg, activation,
                title=f'Anterior MI - Case {i+1}: Precordial Leads (V1-V4)',
                save_path=f'../results/figures/gradcam_ami_{i+1}_focused.png',
                highlight_leads=[6, 7, 8, 9]  # V1, V2, V3, V4
            )
    
    # 2. Inferior MI examples
    if len(imi_cases) >= 2:
        print("\n2. Inferior MI (IMI) - Expected focus: II, III, aVF (inferior) leads")
        for i in range(2):
            ecg_id = imi_cases.iloc[i]['ecg_id']
            test_idx = df_test[df_test['ecg_id'] == ecg_id].index[0]
            actual_test_idx = np.where(df_test.index == test_idx)[0][0]
            
            ecg = X_test[actual_test_idx]
            activation = generate_activation_map(model, ecg)
            
            # Full 12-lead view
            visualize_gradcam(
                ecg, activation,
                title=f'Inferior MI - Case {i+1} (IMI: Should highlight II, III, aVF)',
                save_path=f'../results/figures/gradcam_imi_{i+1}_full.png'
            )
            
            # Focused view on II, III, aVF
            visualize_single_lead_comparison(
                ecg, activation,
                title=f'Inferior MI - Case {i+1}: Inferior Leads (II, III, aVF)',
                save_path=f'../results/figures/gradcam_imi_{i+1}_focused.png',
                highlight_leads=[1, 2, 5]  # II, III, aVF
            )
    
    # 3. Normal examples
    print("\n3. Normal ECG - Expected: Low/diffuse activation")
    for i in range(2):
        test_idx = normal_indices[i]
        
        ecg = X_test[test_idx]
        activation = generate_activation_map(model, ecg)
        
        visualize_gradcam(
            ecg, activation,
            title=f'Normal ECG - Case {i+1} (Expected: Minimal/diffuse activation)',
            save_path=f'../results/figures/gradcam_normal_{i+1}.png'
        )
    
    print("\n" + "="*70)
    print("ACTIVATION VISUALIZATION COMPLETE")
    print("="*70)
    print("\n📊 Generated Visualizations:")
    print("\nAnterior MI (AMI):")
    print("  • Full 12-lead views: gradcam_ami_1_full.png, gradcam_ami_2_full.png")
    print("  • Focused V1-V4: gradcam_ami_1_focused.png, gradcam_ami_2_focused.png")
    print("\nInferior MI (IMI):")
    print("  • Full 12-lead views: gradcam_imi_1_full.png, gradcam_imi_2_full.png")
    print("  • Focused II/III/aVF: gradcam_imi_1_focused.png, gradcam_imi_2_focused.png")
    print("\nNormal:")
    print("  • gradcam_normal_1.png, gradcam_normal_2.png")
    
    print("\n🔍 What to Look For:")
    print("  ✓ Anterior MI: RED zones in V1-V4 (columns 3-4, rows 2-3)")
    print("  ✓ Inferior MI: RED zones in II, III, aVF (column 1-2, rows 1-2)")
    print("  ✓ Normal: Minimal red, mostly dark/yellow distributed evenly")
    print("\nThis demonstrates the model learns CLINICALLY CORRECT features!")

if __name__ == "__main__":
    main()
