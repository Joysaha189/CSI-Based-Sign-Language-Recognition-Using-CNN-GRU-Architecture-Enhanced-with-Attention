import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# =============================
# Configuration
# =============================
SEED = 42

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =============================
# Load and Combine Data
# =============================
def load_and_split(train_path, test_path, train_ratio=0.8, val_ratio=0.2):
    train = np.load(train_path)
    test = np.load(test_path)
    
    # Combine and normalize
    train_csi = train["data"].astype(np.float32)
    test_csi = test["data"].astype(np.float32)
    
    # Normalize using training set statistics
    mean, std = train_csi.mean(), train_csi.std()
    train_csi = (train_csi - mean) / (std + 1e-8)
    test_csi = (test_csi - mean) / (std + 1e-8)
    
    data_all = np.concatenate([train_csi, test_csi], axis=0)
    labels_all = np.concatenate([train['target'], test['target']], axis=0)

    data_all = torch.tensor(data_all, dtype=torch.float32)
    labels_all = torch.tensor(labels_all, dtype=torch.long)

    unique_labels = torch.unique(labels_all)
    label_mapping = {int(old): i for i, old in enumerate(unique_labels.tolist())}
    labels_all = torch.tensor([label_mapping[int(lbl)] for lbl in labels_all])
    num_classes = len(unique_labels)

    total_size = len(data_all)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    train_data, test_data = random_split(TensorDataset(data_all, labels_all), [train_size, test_size])

    # Further split train into train and val
    train_len = int(len(train_data) * (1 - val_ratio))
    val_len = len(train_data) - train_len
    train_split, val_split = random_split(train_data, [train_len, val_len])

    return train_split, val_split, test_data, num_classes

# =============================
# Simple CNN Model
# =============================
class CNN(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 60, 200)):  # Input shape (channels, height, width)
        super(CNN, self).__init__()

        # Define layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.batch_norm = nn.BatchNorm2d(3, momentum=0.9, eps=1e-06)  # Batch Normalization after Conv
        self.relu = nn.ReLU()                # Activation after Batch Normalization
        self.pool = nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        self.dropout = nn.Dropout(p=0.7)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 20 * 67, num_classes)  # Adjusted linear layer input size

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)  # Batch Normalization
        x = self.relu(x)        # ReLU activation
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

# =============================
# Label Smoothing Loss
# =============================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_label = torch.full_like(pred, self.smoothing / (self.num_classes - 1))
        smooth_label.scatter_(1, target.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-smooth_label * F.log_softmax(pred, dim=1), dim=1))

# =============================
# Training Function
# =============================
def train_model(train_split, val_split, test_split, num_classes, config,
                epochs=150, batch_size=256, device='cpu'):
    # Create model
    model = CNN(num_classes=num_classes, input_shape=(3, 60, 200)).to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Label smoothing loss
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=config['lr'] * 0.01
    )

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_split, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_split, batch_size=batch_size, num_workers=0)

    best_val_acc = 0
    patience = 20
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # Permute to match CNN input format (B, 60, 200, 3) -> (B, 3, 60, 200)
            X = X.permute(0, 3, 1, 2)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                X = X.permute(0, 3, 1, 2)
                outputs = model(X)
                loss_v = criterion(outputs, y)
                val_loss += loss_v.item() * y.size(0)
                pred = outputs.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # Compute average train loss and train acc for epoch
        train_loss_avg = train_loss / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_loss_avg = val_loss / val_total if val_total > 0 else 0.0
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping (save best model state)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model for final test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final test evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            X = X.permute(0, 3, 1, 2)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_acc = correct / total if total > 0 else 0.0

    # Count model params
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return test_acc, best_val_acc, param_count

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    datasets = [
        ('Lab_downlink', os.path.join(parent_dir, 'Lab_downlink', 'trainset.npz'), os.path.join(parent_dir, 'Lab_downlink', 'testset.npz')),
        ('Lab_150_down', os.path.join(parent_dir, 'Lab_150_down', 'trainset.npz'), os.path.join(parent_dir, 'Lab_150_down', 'testset.npz')),
        ('Home_downlink', os.path.join(parent_dir, 'Home_downlink', 'trainset.npz'), os.path.join(parent_dir, 'Home_downlink', 'testset.npz'))
    ]

    # Configuration (same as the complex model for fair comparison)
    config = {
        'name': 'Simple CNN',
        'lr': 0.001,
        'weight_decay': 0.0001
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Max epochs: 150 (with early stopping, patience=20)")
    print(f"Model: Simple CNN\n")

    results = []
    
    print(f"{'='*80}")
    print(f"Running: {config['name']}")
    print(f"{'='*80}\n")
    
    for dataset_name, train_path, test_path in datasets:
        print(f"  Dataset: {dataset_name}...", end=' ', flush=True)
        
        # Load and split data
        train_split, val_split, test_split, num_classes = load_and_split(train_path, test_path)

        # Train model
        test_acc, val_acc, param_count = train_model(
            train_split, val_split, test_split, num_classes,
            config, epochs=150, batch_size=256, device=device)
        
        print(f"Val: {val_acc*100:.2f}%, Test: {test_acc*100:.2f}%")

        results.append({
            'config': config['name'],
            'dataset': dataset_name,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'params': param_count
        })

    # Final summary
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY - Simple CNN")
    print("="*100)
    print(f"{'Configuration':<30} {'Dataset':<20} {'Val Acc (%)':<15} {'Test Acc (%)':<15} {'Params':>12}")
    print("-"*100)
    for entry in results:
        cfg = entry['config']
        ds = entry['dataset']
        val_acc = entry['val_acc']
        test_acc = entry['test_acc']
        params = entry['params']
        print(f"{cfg:<30} {ds:<20} {val_acc*100:<15.2f} {test_acc*100:<15.2f} {params:12,d}")
    print("="*100)