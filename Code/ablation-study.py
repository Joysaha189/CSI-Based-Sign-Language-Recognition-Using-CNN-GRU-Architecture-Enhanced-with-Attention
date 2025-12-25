import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import csv

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
# Squeeze-and-Excitation Block
# =============================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# =============================
# Improved Attention Layer
# =============================
class ImprovedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        
        if num_heads == 1:
            # Single head attention
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            # Use PyTorch's built-in multi-head attention
            self.mha = nn.MultiheadAttention(
                hidden_size, 
                num_heads, 
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, gru_output):
        # gru_output shape: (batch, seq_len, hidden_size)
        gru_output = self.layer_norm(gru_output)
        
        if self.num_heads == 1:
            # Single head
            weights = torch.softmax(self.attention(gru_output), dim=1)
            context = torch.sum(weights * gru_output, dim=1)
            return context, weights
        else:
            # Multi-head using built-in
            attn_output, attn_weights = self.mha(gru_output, gru_output, gru_output)
            # Global average pooling over sequence
            context = attn_output.mean(dim=1)
            return context, attn_weights

# =============================
# ABLATION MODELS
# =============================

class CNNOnly(nn.Module):
    """CNN-only baseline: CNN + Global Pooling + FC"""
    def __init__(self, num_classes, input_shape=(60, 200, 3), dropout=0.3):
        super().__init__()
        
        # Same CNN architecture as full model
        self.conv1 = nn.Conv2d(input_shape[2], 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = SEBlock(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.se3 = SEBlock(256)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # Global pooling instead of GRU
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # FC layers
        self.dropout_fc = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        x = self.pool(torch.relu(self.se1(self.bn1(self.conv1(x)))))
        x = self.dropout_cnn(x)
        
        x = self.pool(torch.relu(self.se2(self.bn2(self.conv2(x)))))
        x = self.dropout_cnn(x)
        
        x = self.pool(torch.relu(self.se3(self.bn3(self.conv3(x)))))
        x = self.dropout_cnn(x)
        
        # Global pooling
        x = self.global_pool(x).view(x.size(0), -1)
        
        # FC layers
        x = self.dropout_fc(x)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        return self.fc3(x)


class GRUOnly(nn.Module):
    """GRU-only baseline: Flatten input -> GRU + Global Pooling + FC"""
    def __init__(self, num_classes, input_shape=(60, 200, 3),
                 gru_hidden=512, gru_layers=3, dropout=0.3,
                 bidirectional=False, use_layer_norm=True):
        super().__init__()
        
        # Flatten input: treat each row as a timestep
        self.seq_len = input_shape[0]  # 60
        self.input_size = input_shape[1] * input_shape[2]  # 200 * 3 = 600
        
        self.input_norm = nn.LayerNorm(self.input_size) if use_layer_norm else nn.Identity()
        
        # GRU layer (same config as full model)
        self.gru = nn.GRU(
            self.input_size, 
            gru_hidden, 
            gru_layers,
            batch_first=True, 
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        out_size = gru_hidden * 2 if bidirectional else gru_hidden
        
        # FC layers
        self.dropout_fc = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # x shape: (B, 60, 200, 3)
        b = x.size(0)
        # Reshape to (B, seq_len, features)
        x = x.reshape(b, self.seq_len, -1)
        x = self.input_norm(x)
        
        # GRU processing
        out, _ = self.gru(x)
        
        # Global average pooling over time
        context = out.mean(dim=1)
        
        # FC layers
        x = self.dropout_fc(context)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        return self.fc3(x)


class CNN_GRU(nn.Module):
    """CNN + GRU (without attention)"""
    def __init__(self, num_classes, input_shape=(60, 200, 3),
                 gru_hidden=512, gru_layers=3, dropout=0.3,
                 bidirectional=False, use_layer_norm=True):
        super().__init__()
        
        # Same CNN as full model
        self.conv1 = nn.Conv2d(input_shape[2], 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = SEBlock(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.se3 = SEBlock(256)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        self.gru_input_size = 256 * 7
        self.gru_seq_len = 25
        
        self.input_norm = nn.LayerNorm(self.gru_input_size) if use_layer_norm else nn.Identity()
        
        # GRU layer
        self.gru = nn.GRU(
            self.gru_input_size, 
            gru_hidden, 
            gru_layers,
            batch_first=True, 
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        out_size = gru_hidden * 2 if bidirectional else gru_hidden
        
        # FC layers
        self.dropout_fc = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # CNN processing
        x = x.permute(0, 3, 1, 2)
        
        x = self.pool(torch.relu(self.se1(self.bn1(self.conv1(x)))))
        x = self.dropout_cnn(x)
        
        x = self.pool(torch.relu(self.se2(self.bn2(self.conv2(x)))))
        x = self.dropout_cnn(x)
        
        x = self.pool(torch.relu(self.se3(self.bn3(self.conv3(x)))))
        x = self.dropout_cnn(x)
        
        # Reshape for GRU
        b = x.size(0)
        x = x.permute(0, 3, 1, 2).reshape(b, self.gru_seq_len, -1)
        x = self.input_norm(x)
        
        # GRU processing
        out, _ = self.gru(x)
        
        # Global average pooling over time (instead of attention)
        context = out.mean(dim=1)
        
        # FC layers
        x = self.dropout_fc(context)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        return self.fc3(x)


class CNN_GRU_Attention(nn.Module):
    """Full model: CNN + GRU + Attention"""
    def __init__(self, num_classes, input_shape=(60, 200, 3),
                 gru_hidden=512, gru_layers=3, dropout=0.3,
                 bidirectional=False, use_layer_norm=True, num_heads=1):
        super().__init__()
        
        # Same CNN as full model
        self.conv1 = nn.Conv2d(input_shape[2], 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = SEBlock(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.se3 = SEBlock(256)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        self.gru_input_size = 256 * 7
        self.gru_seq_len = 25
        
        self.input_norm = nn.LayerNorm(self.gru_input_size) if use_layer_norm else nn.Identity()
        
        # GRU layer
        self.gru = nn.GRU(
            self.gru_input_size, 
            gru_hidden, 
            gru_layers,
            batch_first=True, 
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        out_size = gru_hidden * 2 if bidirectional else gru_hidden
        
        # Attention mechanism
        self.attention = ImprovedAttention(out_size, num_heads=num_heads, dropout=dropout)
        
        # FC layers
        self.dropout_fc = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # CNN processing
        x = x.permute(0, 3, 1, 2)
        
        x = self.pool(torch.relu(self.se1(self.bn1(self.conv1(x)))))
        x = self.dropout_cnn(x)
        
        x = self.pool(torch.relu(self.se2(self.bn2(self.conv2(x)))))
        x = self.dropout_cnn(x)
        
        x = self.pool(torch.relu(self.se3(self.bn3(self.conv3(x)))))
        x = self.dropout_cnn(x)
        
        # Reshape for GRU
        b = x.size(0)
        x = x.permute(0, 3, 1, 2).reshape(b, self.gru_seq_len, -1)
        x = self.input_norm(x)
        
        # GRU processing
        out, _ = self.gru(x)
        
        # Attention mechanism
        context, _ = self.attention(out)
        
        # FC layers
        x = self.dropout_fc(context)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        return self.fc3(x)

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
def train_model(train_split, val_split, test_split, num_classes, config, model_type,
                epochs=150, batch_size=256, device='cpu'):
    # Create model based on type
    if model_type == 'CNN-only':
        model = CNNOnly(
            num_classes=num_classes,
            dropout=config['dropout']
        ).to(device)
    elif model_type == 'GRU-only':
        model = GRUOnly(
            num_classes=num_classes,
            gru_hidden=config['gru_hidden'],
            gru_layers=config['gru_layers'],
            bidirectional=config['bidirectional'],
            dropout=config['dropout'],
            use_layer_norm=config['use_layer_norm']
        ).to(device)
    elif model_type == 'CNN-GRU':
        model = CNN_GRU(
            num_classes=num_classes,
            gru_hidden=config['gru_hidden'],
            gru_layers=config['gru_layers'],
            bidirectional=config['bidirectional'],
            dropout=config['dropout'],
            use_layer_norm=config['use_layer_norm']
        ).to(device)
    else:  # 'CNN-GRU-Attention'
        model = CNN_GRU_Attention(
            num_classes=num_classes,
            gru_hidden=config['gru_hidden'],
            gru_layers=config['gru_layers'],
            bidirectional=config['bidirectional'],
            dropout=config['dropout'],
            use_layer_norm=config['use_layer_norm'],
            num_heads=config['heads']
        ).to(device)
    
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

    # Metrics history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

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

        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc*100:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for final test
    model.load_state_dict(best_model_state)
    
    # Final test evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_acc = correct / total if total > 0 else 0.0

    # Count model params
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Sample counts
    sample_counts = {
        'train': len(train_split),
        'val': len(val_split),
        'test': len(test_split)
    }

    return test_acc, best_val_acc, history, param_count, sample_counts

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

    # Configuration (kept constant across all ablations)
    base_config = {
        'gru_hidden': 256,
        'gru_layers': 2,
        'bidirectional': True,
        'dropout': 0.3,
        'use_layer_norm': True,
        'lr': 0.001,
        'optimizer': 'AdamW',
        'weight_decay': 0.0001,
        'heads': 4
    }

    # Ablation study models
    model_types = [
        # 'CNN-only'
         'GRU-only'
        # 'CNN-GRU'
        # 'CNN-GRU-Attention'
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Max epochs: 150 (with early stopping, patience=20)")
    print(f"\nBase Configuration (kept constant):")
    print(f"  GRU Hidden: {base_config['gru_hidden']}")
    print(f"  GRU Layers: {base_config['gru_layers']}")
    print(f"  Bidirectional: {base_config['bidirectional']}")
    print(f"  Dropout: {base_config['dropout']}")
    print(f"  Learning Rate: {base_config['lr']}")
    print(f"  Attention Heads: {base_config['heads']}\n")

    results = []
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY: {model_type}")
        print(f"{'='*80}\n")
        
        for dataset_name, train_path, test_path in datasets:
            print(f"Processing dataset: {dataset_name}...")
            train_split, val_split, test_split, num_classes = load_and_split(train_path, test_path)

            test_acc, val_acc, history, param_count, sample_counts = train_model(
                train_split, val_split, test_split, num_classes,
                base_config, model_type, epochs=150, batch_size=256, device=device)
            
            # Print summary
            print(f"  Samples - train: {sample_counts['train']}, val: {sample_counts['val']}, test: {sample_counts['test']}")
            print(f"  Model params: {param_count:,}")
            print(f"  Best Val Accuracy: {val_acc*100:.2f}%")
            print(f"  Test Accuracy: {test_acc*100:.2f}%\n")

            # Save per-epoch metrics to CSV
            os.makedirs('results_ablation', exist_ok=True)
            safe_model = model_type.replace(' ', '_').replace('-', '_')
            safe_ds = dataset_name.replace(' ', '_')
            csv_path = os.path.join('results_ablation', f"{safe_ds}_{safe_model}_metrics.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
                epochs_range = list(range(1, len(history['train_loss']) + 1))
                for i in epochs_range:
                    writer.writerow([
                        i,
                        f"{history['train_loss'][i-1]:.6f}",
                        f"{history['train_acc'][i-1]:.6f}",
                        f"{history['val_loss'][i-1]:.6f}",
                        f"{history['val_acc'][i-1]:.6f}"
                    ])

            # Append summary row to final summary CSV
            summary_path = os.path.join('results_ablation', 'ablation_summary.csv')
            write_header = not os.path.exists(summary_path)
            with open(summary_path, 'a', newline='') as sumfile:
                writer = csv.writer(sumfile)
                if write_header:
                    writer.writerow(['Model Type', 'Dataset', 'Val Acc', 'Test Acc', 'Params', 'Train Samples', 'Val Samples', 'Test Samples', 'Metric CSV'])
                writer.writerow([model_type, dataset_name, f"{val_acc:.6f}", f"{test_acc:.6f}", param_count, sample_counts['train'], sample_counts['val'], sample_counts['test'], csv_path])

            results.append({
                'model_type': model_type,
                'dataset': dataset_name,
                'test_acc': test_acc,
                'val_acc': val_acc,
                'params': param_count,
                'samples': sample_counts
            })

    # Print ablation analysis
    print("\n" + "="*120)
    print("ABLATION ANALYSIS (Average across datasets)")
    print("="*120)
    model_avg = {}
    for model_type in model_types:
        model_results = [r for r in results if r['model_type'] == model_type]
        avg_val = sum(r['val_acc'] for r in model_results) / len(model_results)
        avg_test = sum(r['test_acc'] for r in model_results) / len(model_results)
        avg_params = sum(r['params'] for r in model_results) / len(model_results)
        model_avg[model_type] = {'val': avg_val, 'test': avg_test, 'params': avg_params}
        print(f"{model_type:<25} Val: {avg_val*100:.2f}%  Test: {avg_test*100:.2f}%  Params: {int(avg_params):,}")
    print("="*120)