"""
CNN-LSTM Training Script for Filtered Dataset
Patient-wise split with 100% confidence labels
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== Dataset ====================
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== Model Architecture ====================
class CNN_LSTM(nn.Module):
    def __init__(self, input_channels=12, sequence_length=1000):
        super(CNN_LSTM, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.4)
        
        self.relu = nn.ReLU()
        
        # LSTM
        self.lstm = nn.LSTM(256, 128, batch_first=True, dropout=0.4)
        
        # Dense layers
        self.fc1 = nn.Linear(128, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 1)
        
        # For Grad-CAM
        self.activations = None
        self.gradients = None
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations
    
    def save_gradient(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        # x shape: (batch, timesteps, channels)
        x = x.transpose(1, 2)  # (batch, channels, timesteps)
        
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
        
        # Save activations for Grad-CAM
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Prepare for LSTM
        x = x.transpose(1, 2)  # (batch, timesteps, features)
        
        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last timestep
        
        # Dense layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x.squeeze()

# ==================== Training Functions ====================
def train_epoch(model, train_loader, criterion, optimizer, device, class_weights):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        
        # Compute weighted loss
        loss = criterion(outputs, batch_y)
        weights = torch.where(batch_y == 1, class_weights[1], class_weights[0])
        loss = (loss * weights).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, class_weights):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            
            # Compute weighted loss
            loss = criterion(outputs, batch_y)
            weights = torch.where(batch_y == 1, class_weights[1], class_weights[0])
            loss = (loss * weights).mean()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# ==================== Main Training ====================
def main():
    print("="*60)
    print("CNN-LSTM TRAINING - FILTERED DATASET")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_val = np.load('../data/X_val.npy')
    y_val = np.load('../data/y_val.npy')
    
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val: {X_val.shape}, {y_val.shape}")
    print(f"Class distribution - Train: Normal={np.sum(y_train==0)}, MI={np.sum(y_train==1)}")
    
    # Compute class weights
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weights = torch.FloatTensor(class_weights_array).to(device)
    print(f"\nClass weights: Normal={class_weights[0]:.3f}, MI={class_weights[1]:.3f}")
    
    # Create datasets and loaders
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = CNN_LSTM(input_channels=12, sequence_length=1000).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Loss and optimizer
    criterion = nn.BCELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training settings
    num_epochs = 100
    patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    
    # History
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, class_weights)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, class_weights)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), '../saved_models/best_model.pth')
            print(f"  → Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    
    # Save final model
    torch.save(model.state_dict(), '../saved_models/final_model.pth')
    
    # Save history
    np.save('../results/metrics/training_history.npy', history)
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved:")
    print(f"  • best_model.pth (lowest val loss)")
    print(f"  • final_model.pth (last epoch)")

def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/figures/training_history.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: training_history.png")

if __name__ == "__main__":
    main()
