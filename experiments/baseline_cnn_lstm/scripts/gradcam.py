"""
Grad-CAM Visualization for ECG-IHD Detection
Focuses on CNN features (before LSTM) for better compatibility
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from train import CNN_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradCAMWrapper(CNN_LSTM):
    """Wrapper to capture CNN features before LSTM"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_features = None
        self.cnn_gradients = None
        
    def forward_cnn_only(self, x):
        """Forward pass through CNN only, capture features"""
        x = x.transpose(1, 2)
        
        # Conv blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Save features for Grad-CAM
        self.cnn_features = x
        
        # Register hook
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        
        return x
    
    def save_gradient(self, grad):
        self.cnn_gradients = grad

def generate_activation_map(model, input_tensor):
    """
    Generate activation map from CNN features
    Simpler alternative to full Grad-CAM
    """
    model.eval()
    
    with torch.no_grad():
        # Get CNN features
        features = model.forward_cnn_only(input_tensor)
        
        # Average across channels to get activation map
        activation_map = torch.mean(features, dim=1).squeeze()
        
        # Normalize
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    
    return activation_map.cpu().numpy()

def visualize_activation_map(ecg_signal, activation_map, lead_idx=0, prediction=None, title_suffix="", save_path=None):
    """
    Visualize ECG with activation map overlay
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # 1. Original ECG
    axes[0].plot(ecg_signal[:, lead_idx], color='blue', linewidth=1.5)
    title = f'Original ECG - Lead {lead_idx+1} {title_suffix}'
    if prediction is not None:
        title += f' (Prediction: {prediction:.3f})'
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, len(ecg_signal)])
    
    # 2. Activation Map (upsampled)
    activation_upsampled = np.interp(
        np.linspace(0, len(activation_map)-1, len(ecg_signal)),
        np.arange(len(activation_map)),
        activation_map
    )
    
    axes[1].plot(activation_upsampled, color='red', linewidth=2)
    axes[1].fill_between(range(len(activation_upsampled)), activation_upsampled, 
                          alpha=0.3, color='red')
    axes[1].set_title('CNN Activation Strength (Model Focus)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Activation', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, len(ecg_signal)])
    axes[1].set_ylim([0, 1])
    
    # 3. ECG with Heatmap Overlay
    axes[2].plot(ecg_signal[:, lead_idx], color='blue', linewidth=1.5, alpha=0.7, label='ECG Signal', zorder=2)
    
    im = axes[2].imshow(activation_upsampled[np.newaxis, :], 
                        cmap='hot', aspect='auto', alpha=0.5,
                        extent=[0, len(ecg_signal), 
                               ecg_signal[:, lead_idx].min(), 
                               ecg_signal[:, lead_idx].max()],
                        zorder=1)
    
    axes[2].set_title('Combined View: ECG + Model Focus', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (samples)', fontsize=12)
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].grid(True, alpha=0.3, zorder=0)
    axes[2].legend(loc='upper right')
    axes[2].set_xlim([0, len(ecg_signal)])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[2], pad=0.02)
    cbar.set_label('Model Focus\n(Red = High)', rotation=270, labelpad=20, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()

def main():
    # Load test data
    print("Loading test data...")
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')
    
    print(f"Test samples: {len(X_test)}")
    print(f"IHD positive: {y_test.sum()}")
    print(f"Normal: {(y_test == 0).sum()}")
    
    # Load trained model and wrap it
    print("\nLoading trained model...")
    model = GradCAMWrapper(input_channels=12, sequence_length=1000).to(device)
    model.load_state_dict(torch.load('../saved_models/best_model.pth', 
                                     weights_only=True,
                                     map_location=device))
    model.eval()
    
    # Select samples
    ihd_indices = np.where(y_test == 1)[0]
    normal_indices = np.where(y_test == 0)[0]
    
    num_examples = 3
    
    print(f"\nGenerating activation visualizations...")
    print("="*50)
    
    # IHD positive samples
    print("\nProcessing IHD positive samples...")
    for i in range(min(num_examples, len(ihd_indices))):
        idx = ihd_indices[i]
        sample = torch.FloatTensor(X_test[idx:idx+1]).to(device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(sample).item()
        
        # Get activation map
        activation_map = generate_activation_map(model, sample)
        
        visualize_activation_map(
            X_test[idx], activation_map, lead_idx=0,
            prediction=pred,
            title_suffix=f"(IHD Positive - True Label: 1)",
            save_path=f'../results/figures/activation_ihd_{i+1}.png'
        )
    
    # Normal samples
    print("Processing Normal samples...")
    for i in range(min(num_examples, len(normal_indices))):
        idx = normal_indices[i]
        sample = torch.FloatTensor(X_test[idx:idx+1]).to(device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(sample).item()
        
        # Get activation map
        activation_map = generate_activation_map(model, sample)
        
        visualize_activation_map(
            X_test[idx], activation_map, lead_idx=0,
            prediction=pred,
            title_suffix=f"(Normal - True Label: 0)",
            save_path=f'../results/figures/activation_normal_{i+1}.png'
        )
    
    print("\n" + "="*50)
    print(f"✓ Generated {num_examples*2} activation visualizations")
    print("\n📊 Interpretation Guide:")
    print("  • Middle plot shows WHERE the CNN focuses")
    print("  • Bottom plot overlays focus on original ECG")
    print("  • Red/Hot regions = High CNN activation (important features)")
    print("  • Dark regions = Low activation (less important)")
    print("  • Compare IHD vs Normal patterns to see what model learned")
    print("="*50)

if __name__ == "__main__":
    main()