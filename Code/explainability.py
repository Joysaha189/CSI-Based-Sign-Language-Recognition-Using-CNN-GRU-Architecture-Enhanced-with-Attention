import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import csv

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError as e:
    HAS_MATPLOTLIB = False
    print(f"Warning: matplotlib import failed ({e}). Grad-CAM visualizations will be saved as numpy arrays only.")
except Exception as e:
    HAS_MATPLOTLIB = False
    print(f"Warning: matplotlib error ({e}). Grad-CAM visualizations will be saved as numpy arrays only.")

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
# IMPROVED MODEL with Grad-CAM Support
# =============================
class ImprovedCNN_GRU_Attention(nn.Module):
    def __init__(self, num_classes, input_shape=(60, 200, 3),
                 gru_hidden=512, gru_layers=3, dropout=0.3,
                 bidirectional=False, use_layer_norm=True, num_heads=1):
        super().__init__()
        
        # Store for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Deeper CNN with residual connections and SE blocks
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
        
        # Calculate dimensions after pooling
        self.gru_input_size = 256 * 7
        self.gru_seq_len = 25
        
        # Input normalization
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
        
        # Improved attention mechanism
        self.attention = ImprovedAttention(out_size, num_heads=num_heads, dropout=dropout)
        
        # Fully connected layers with residual connection
        self.dropout_fc = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x, return_cam=False):
        # CNN processing with SE blocks
        x = x.permute(0, 3, 1, 2)  # (B, 60, 200, 3) -> (B, 3, 60, 200)
        
        x = self.pool(torch.relu(self.se1(self.bn1(self.conv1(x)))))
        x = self.dropout_cnn(x)
        
        x = self.pool(torch.relu(self.se2(self.bn2(self.conv2(x)))))
        x = self.dropout_cnn(x)
        
        # Target layer for Grad-CAM (last conv layer)
        x = self.pool(torch.relu(self.se3(self.bn3(self.conv3(x)))))
        
        # Register hook for Grad-CAM
        if return_cam and x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x
        
        x = self.dropout_cnn(x)
        
        # Reshape for GRU
        b = x.size(0)
        x = x.permute(0, 3, 1, 2).reshape(b, self.gru_seq_len, -1)
        x = self.input_norm(x)
        
        # GRU processing
        out, _ = self.gru(x)
        
        # Attention mechanism
        context, _ = self.attention(out)
        
        # Fully connected layers with batch norm
        x = self.dropout_fc(context)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        return self.fc3(x)
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations

# =============================
# Grad-CAM Implementation
# =============================
class GradCAM:
    def __init__(self, model):
        self.model = model
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        Args:
            input_tensor: Input CSI data (batch_size, 60, 200, 3)
            target_class: Target class index (if None, uses predicted class)
        Returns:
            cam: Grad-CAM heatmap (60, 200)
            pred_class: Predicted class
        """
        # Ensure gradient computation is enabled and prepare module modes.
        # For cuDNN RNNs the forward must run with the module in training mode
        # so that the backward can run. To keep BatchNorm/Dropout behavior stable
        # we set only RNN modules to train() and BatchNorm/Dropout to eval().
        for param in self.model.parameters():
            param.requires_grad = True

        # Save original training/eval states for modules we will toggle
        module_states = {}
        for name, m in self.model.named_modules():
            module_states[name] = m.training

        # Put BatchNorm/LayerNorm/Dropout into eval to preserve inference stats
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm,
                              nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()

        # Put RNN modules into training mode so cuDNN uses the training kernels
        for m in self.model.modules():
            if isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
                m.train()
        
        # Ensure input requires grad
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        input_tensor = input_tensor.detach().clone()
        input_tensor.requires_grad = True
        
        # Forward pass with gradient tracking
        with torch.set_grad_enabled(True):
            output = self.model(input_tensor, return_cam=True)
            pred_class = output.argmax(dim=1).item()
            
            if target_class is None:
                target_class = pred_class
            
            # Zero gradients
            self.model.zero_grad()

            # Backward pass for target class
            class_loss = output[0, target_class]
            class_loss.backward()

            # Restore original module modes
            for name, m in self.model.named_modules():
                orig_state = module_states.get(name, True)
                if orig_state:
                    m.train()
                else:
                    m.eval()
        
        # Get gradients and activations
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()
        
        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations not captured properly")
        
        # Pool gradients across spatial dimensions
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        activations_copy = activations.clone()
        for i in range(activations_copy.shape[1]):
            activations_copy[:, i, :, :] *= pooled_gradients[i]
        
        # Average across channels and apply ReLU
        heatmap = torch.mean(activations_copy, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-8)
        
        # Resize to original input size (60, 200)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(60, 200),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        return heatmap.detach().cpu().numpy(), pred_class

def visualize_gradcam(input_data, heatmap, pred_class, true_class, save_path=None):
    """
    Visualize Grad-CAM heatmap overlaid on input CSI data
    """
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Original CSI (average across channels)
        csi_avg = np.mean(input_data.cpu().numpy(), axis=2)
        axes[0, 0].imshow(csi_avg, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Original CSI Data (Averaged)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Subcarrier')
        
        # Grad-CAM heatmap
        axes[0, 1].imshow(heatmap, cmap='jet', aspect='auto')
        axes[0, 1].set_title('Grad-CAM Heatmap')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Subcarrier')
        plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
        
        # Overlay
        axes[1, 0].imshow(csi_avg, cmap='gray', aspect='auto')
        axes[1, 0].imshow(heatmap, cmap='jet', alpha=0.5, aspect='auto')
        axes[1, 0].set_title(f'Overlay (Pred: {pred_class}, True: {true_class})')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Subcarrier')
        
        # Subcarrier importance (average across time)
        subcarrier_importance = np.mean(heatmap, axis=1)
        axes[1, 1].barh(range(60), subcarrier_importance)
        axes[1, 1].set_title('Subcarrier Importance')
        axes[1, 1].set_xlabel('Average Activation')
        axes[1, 1].set_ylabel('Subcarrier Index')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # Save as numpy array if matplotlib not available
        if save_path:
            base_path = save_path.replace('.png', '')
            np.save(f"{base_path}_heatmap.npy", heatmap)
            np.save(f"{base_path}_input.npy", input_data.cpu().numpy())
            
            # Save text summary
            with open(f"{base_path}_summary.txt", 'w') as f:
                f.write(f"Grad-CAM Analysis\n")
                f.write(f"="*50 + "\n")
                f.write(f"Predicted Class: {pred_class}\n")
                f.write(f"True Class: {true_class}\n")
                f.write(f"Correct: {pred_class == true_class}\n\n")
                
                # Subcarrier importance
                subcarrier_importance = np.mean(heatmap, axis=1)
                top_k = 10
                top_indices = np.argsort(subcarrier_importance)[-top_k:][::-1]
                
                f.write(f"Top {top_k} Most Important Subcarriers:\n")
                f.write("-"*50 + "\n")
                for rank, idx in enumerate(top_indices, 1):
                    f.write(f"{rank}. Subcarrier {idx}: {subcarrier_importance[idx]:.4f}\n")
                
                # Time importance
                time_importance = np.mean(heatmap, axis=0)
                top_time_indices = np.argsort(time_importance)[-top_k:][::-1]
                
                f.write(f"\nTop {top_k} Most Important Time Steps:\n")
                f.write("-"*50 + "\n")
                for rank, idx in enumerate(top_time_indices, 1):
                    f.write(f"{rank}. Time Step {idx}: {time_importance[idx]:.4f}\n")
                
                # Overall statistics
                f.write(f"\nHeatmap Statistics:\n")
                f.write("-"*50 + "\n")
                f.write(f"Mean Activation: {np.mean(heatmap):.4f}\n")
                f.write(f"Max Activation: {np.max(heatmap):.4f}\n")
                f.write(f"Min Activation: {np.min(heatmap):.4f}\n")
                f.write(f"Std Activation: {np.std(heatmap):.4f}\n")

def generate_gradcam_samples(model, test_loader, device, num_samples=5, save_dir='gradcam_results'):
    """
    Generate Grad-CAM visualizations for sample test cases
    """
    os.makedirs(save_dir, exist_ok=True)
    gradcam = GradCAM(model)
    
    model.eval()
    samples_generated = 0
    
    # Aggregate statistics
    all_subcarrier_importance = []
    all_predictions = []
    all_true_labels = []
    
    # Don't use no_grad context for Grad-CAM
    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        
        for i in range(X.shape[0]):
            if samples_generated >= num_samples:
                # Save aggregate statistics
                save_aggregate_stats(all_subcarrier_importance, all_predictions, 
                                   all_true_labels, save_dir)
                return
            
            input_sample = X[i]
            true_label = y[i].item()
            
            # Generate Grad-CAM (this needs gradients)
            heatmap, pred_class = gradcam.generate_cam(input_sample)
            
            # Store for aggregate analysis
            all_subcarrier_importance.append(np.mean(heatmap, axis=1))
            all_predictions.append(pred_class)
            all_true_labels.append(true_label)
            
            # Visualize
            file_ext = '.png' if HAS_MATPLOTLIB else ''
            save_path = os.path.join(save_dir, f'gradcam_sample_{samples_generated+1}_pred{pred_class}_true{true_label}{file_ext}')
            visualize_gradcam(input_sample, heatmap, pred_class, true_label, save_path)
            
            print(f"  Generated Grad-CAM analysis {samples_generated+1}/{num_samples}")
            samples_generated += 1
    
    # Save aggregate statistics (if we exit loop early)
    save_aggregate_stats(all_subcarrier_importance, all_predictions, 
                        all_true_labels, save_dir)

def save_aggregate_stats(subcarrier_importance_list, predictions, true_labels, save_dir):
    """
    Save aggregate statistics across all analyzed samples
    """
    if len(subcarrier_importance_list) == 0:
        return
    
    # Average subcarrier importance across all samples
    avg_importance = np.mean(subcarrier_importance_list, axis=0)
    
    # Save to file
    stats_path = os.path.join(save_dir, 'aggregate_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Grad-CAM Aggregate Statistics\n")
        f.write("="*60 + "\n\n")
        
        # Accuracy
        correct = sum([p == t for p, t in zip(predictions, true_labels)])
        accuracy = correct / len(predictions) * 100
        f.write(f"Samples Analyzed: {len(predictions)}\n")
        f.write(f"Accuracy: {accuracy:.2f}% ({correct}/{len(predictions)})\n\n")
        
        # Top subcarriers
        f.write("Top 15 Most Important Subcarriers (Averaged Across All Samples):\n")
        f.write("-"*60 + "\n")
        top_indices = np.argsort(avg_importance)[-15:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            f.write(f"{rank:2d}. Subcarrier {idx:2d}: {avg_importance[idx]:.6f}\n")
        
        # Per-class analysis if multiple samples
        f.write("\n" + "="*60 + "\n")
        f.write("Per-Class Prediction Summary:\n")
        f.write("-"*60 + "\n")
        unique_classes = sorted(set(true_labels))
        for cls in unique_classes:
            cls_preds = [p for p, t in zip(predictions, true_labels) if t == cls]
            cls_correct = sum([p == cls for p in cls_preds])
            if len(cls_preds) > 0:
                f.write(f"Class {cls}: {cls_correct}/{len(cls_preds)} correct "
                       f"({cls_correct/len(cls_preds)*100:.1f}%)\n")
    
    # Save numpy array of average importance
    np.save(os.path.join(save_dir, 'avg_subcarrier_importance.npy'), avg_importance)
    
    print(f"  Saved aggregate statistics to: {stats_path}")

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
# Training Function with Grad-CAM
# =============================
def train_model(train_split, val_split, test_split, num_classes, config,
                epochs=150, batch_size=256, device='cpu', dataset_name=''):
    # Create model with given config
    model = ImprovedCNN_GRU_Attention(
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
    
    # Load best model
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

    # Generate Grad-CAM visualizations
    print("\n  Generating Grad-CAM visualizations...")
    safe_ds = dataset_name.replace(' ', '_')
    gradcam_dir = os.path.join('results', f'gradcam_{safe_ds}')
    generate_gradcam_samples(model, test_loader, device, num_samples=10, save_dir=gradcam_dir)

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

    # Improved configurations
    configs = [
        {
            'name': 'Bidirectional + Multi-Head',
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
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Max epochs: 150 (with early stopping, patience=20)")
    if HAS_MATPLOTLIB:
        print(f"Grad-CAM visualizations will be generated as images\n")
    else:
        print(f"Grad-CAM visualizations will be saved as numpy arrays and text summaries")
        print(f"(Install matplotlib for image visualizations: pip install matplotlib)\n")

    results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Running: {config['name']}")
        print(f"{'='*80}")
        print(f"Config: gru_hidden={config['gru_hidden']}, gru_layers={config['gru_layers']}, "
              f"bidirectional={config['bidirectional']}, dropout={config['dropout']}, "
              f"lr={config['lr']}, heads={config['heads']}")
        print(f"{'='*80}\n")
        
        for dataset_name, train_path, test_path in datasets:
            print(f"Processing dataset: {dataset_name}...")
            train_split, val_split, test_split, num_classes = load_and_split(train_path, test_path)

            test_acc, val_acc, history, param_count, sample_counts = train_model(
                train_split, val_split, test_split, num_classes,
                config, epochs=150, batch_size=256, device=device, 
                dataset_name=dataset_name)
            
            # Print summary info
            print(f"  Samples - train: {sample_counts['train']}, val: {sample_counts['val']}, test: {sample_counts['test']}")
            print(f"  Model params: {param_count:,}")
            print(f"  Best Val Accuracy: {val_acc*100:.2f}%")
            print(f"  Test Accuracy: {test_acc*100:.2f}%\n")

            # Save per-epoch metrics to CSV
            os.makedirs('results', exist_ok=True)
            safe_cfg = config['name'].replace(' ', '_')
            safe_ds = dataset_name.replace(' ', '_')
            csv_path = os.path.join('results', f"{safe_ds}_{safe_cfg}_metrics.csv")
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
            summary_path = os.path.join('results', 'final_summary.csv')
            write_header = not os.path.exists(summary_path)
            with open(summary_path, 'a', newline='') as sumfile:
                writer = csv.writer(sumfile)
                if write_header:
                    writer.writerow(['Config', 'Dataset', 'Val Acc', 'Test Acc', 'Params', 'Train Samples', 'Val Samples', 'Test Samples', 'Metric CSV'])
                writer.writerow([config['name'], dataset_name, f"{val_acc:.6f}", f"{test_acc:.6f}", param_count, sample_counts['train'], sample_counts['val'], sample_counts['test'], csv_path])

            results.append({
                'config': config['name'],
                'dataset': dataset_name,
                'test_acc': test_acc,
                'val_acc': val_acc,
                'params': param_count,
                'samples': sample_counts
            })

    # Final summary
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY")
    print("="*100)
    print(f"{'Configuration':<30} {'Dataset':<20} {'Val Acc (%)':<15} {'Test Acc (%)':<15} {'Params':>12} {'#Train':>8} {'#Val':>8} {'#Test':>8}")
    print("-"*120)
    for entry in results:
        cfg = entry['config']
        ds = entry['dataset']
        val_acc = entry['val_acc']
        test_acc = entry['test_acc']
        params = entry['params']
        samples = entry['samples']
        print(f"{cfg:<30} {ds:<20} {val_acc*100:<15.2f} {test_acc*100:<15.2f} {params:12,d} {samples['train']:8d} {samples['val']:8d} {samples['test']:8d}")
    print("="*120)
    if HAS_MATPLOTLIB:
        print("\nGrad-CAM visualizations saved in 'results/gradcam_*' directories as PNG images")
    else:
        print("\nGrad-CAM data saved in 'results/gradcam_*' directories as .npy arrays and .txt summaries")
        print("Install matplotlib (pip install matplotlib) to generate image visualizations")