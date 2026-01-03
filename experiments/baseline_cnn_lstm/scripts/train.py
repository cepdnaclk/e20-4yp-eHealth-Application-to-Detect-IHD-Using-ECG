"""
Enhanced CNN-LSTM Model with XAI for IHD Detection
- PyTorch implementation
- GPU acceleration
- TensorBoard logging
- Grad-CAM visualization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. CUSTOM DATASET
# ==========================================

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. CNN-LSTM MODEL
# ==========================================

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
        
        # LSTM Layer
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.4)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Store activations for Grad-CAM
        self.feature_maps = None
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # x shape: (batch, time, channels)
        # Transpose for Conv1d: (batch, channels, time)
        x = x.transpose(1, 2)
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Register hook for Grad-CAM
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.feature_maps = x
        
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Transpose back for LSTM: (batch, time, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]  # Take last timestep
        x = self.dropout_lstm(x)
        
        # Fully Connected
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.feature_maps

# ==========================================
# 3. TRAINING FUNCTION
# ==========================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in tqdm(dataloader, desc='Training'):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_preds.extend((outputs > 0.5).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

# ==========================================
# 4. VALIDATION FUNCTION
# ==========================================

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc='Validation'):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item()
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    
    return epoch_loss, epoch_acc, epoch_auc

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================

def main():
    # Load data
    print("Loading data...")
    X_train = np.load('../data/X_train.npy')
    X_val = np.load('../data/X_val.npy')
    y_train = np.load('../data/y_train.npy')
    y_val = np.load('../data/y_val.npy')
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = CNN_LSTM(input_channels=12, sequence_length=1000).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    # TensorBoard
    writer = SummaryWriter('../results/logs/baseline_cnn_lstm')
    
    # Training
    num_epochs = 100
    best_val_auc = 0.0
    patience = 15
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), '../saved_models/best_model.pth')
            print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    print(f"\nTraining complete! Best validation AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()
