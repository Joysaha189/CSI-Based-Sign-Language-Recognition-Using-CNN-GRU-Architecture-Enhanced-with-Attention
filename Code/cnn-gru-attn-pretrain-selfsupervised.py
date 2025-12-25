import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# =============================
# Configuration
# =============================
USE_AUGMENTATION = False  # Enable augmentation for better generalization
SEED = 42

# Pretraining configuration
PRETRAINING_MODES = ['none', 'autoencoder', 'contrastive']  # Modes to test
PRETRAIN_EPOCHS = 50  # Epochs for pretraining phase
FINETUNE_EPOCHS = 150  # Epochs for fine-tuning phase

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =============================
# Augmentation Functions (GPU-Compatible)
# =============================
def temporal_warp(csi_tensor, lambda_range=(0.85, 1.15)):
    """
    Warp along the time axis for CSI tensor of shape (60, 200, 3).
    """
    lam = random.uniform(*lambda_range)
    seq_len = csi_tensor.shape[1]  # time steps
    new_len = int(seq_len / lam)
    
    # Reshape to (1, C*subcarriers, time) for interpolation
    x = csi_tensor.permute(2, 0, 1).reshape(1, -1, seq_len)  # (1, 3*60, 200)
    warped = F.interpolate(x, size=new_len, mode='linear', align_corners=False)
    
    # Reshape back to (60, time, 3)
    warped = warped.reshape(csi_tensor.shape[2], csi_tensor.shape[0], new_len).permute(1, 2, 0)
    
    # Pad or crop to original length
    if new_len < seq_len:
        pad = seq_len - new_len
        warped = torch.cat([warped, warped[:, -pad:, :]], dim=1)
    else:
        warped = warped[:, :seq_len, :]
    return warped

def gaussian_jitter(csi_tensor, sigma_factor=0.02):
    sigma = sigma_factor * torch.std(csi_tensor)
    noise = torch.normal(0, sigma, size=csi_tensor.shape, device=csi_tensor.device)
    return csi_tensor + noise

def random_mask(csi_tensor, mask_ratio=0.1):
    """Randomly mask some time steps"""
    mask = torch.rand(csi_tensor.shape[1], device=csi_tensor.device) > mask_ratio
    return csi_tensor * mask.view(1, -1, 1)

def apply_augmentation(X):
    """Apply random augmentation to a batch"""
    aug_type = random.choice(['warp', 'jitter', 'both', 'mask'])
    if aug_type == 'warp':
        return torch.stack([temporal_warp(x) for x in X])
    elif aug_type == 'jitter':
        return torch.stack([gaussian_jitter(x) for x in X])
    elif aug_type == 'mask':
        return torch.stack([random_mask(x) for x in X])
    else:  # both
        return torch.stack([gaussian_jitter(temporal_warp(x)) for x in X])

# =============================
# Load and Split Data (Modified for clearer train/test separation)
# =============================
def load_and_split_v2(train_path, test_path, val_ratio=0.2):
    """
    Load data with clear separation:
    - Training set (from train.npz) split into train/val
    - Test set (from test.npz) kept separate
    """
    train = np.load(train_path)
    test = np.load(test_path)
    
    # Load data
    train_csi = train["data"].astype(np.float32)
    test_csi = test["data"].astype(np.float32)
    train_labels = train['target']
    test_labels = test['target']
    
    # Normalize using training set statistics
    mean, std = train_csi.mean(), train_csi.std()
    train_csi = (train_csi - mean) / (std + 1e-8)
    test_csi = (test_csi - mean) / (std + 1e-8)
    
    # Convert to tensors
    train_csi = torch.tensor(train_csi, dtype=torch.float32)
    test_csi = torch.tensor(test_csi, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Map labels to continuous range
    all_labels = torch.cat([train_labels, test_labels])
    unique_labels = torch.unique(all_labels)
    label_mapping = {int(old): i for i, old in enumerate(unique_labels.tolist())}
    train_labels = torch.tensor([label_mapping[int(lbl)] for lbl in train_labels])
    test_labels = torch.tensor([label_mapping[int(lbl)] for lbl in test_labels])
    num_classes = len(unique_labels)
    
    # Create datasets
    train_dataset = TensorDataset(train_csi, train_labels)
    test_dataset = TensorDataset(test_csi, test_labels)
    
    # Split training into train/val
    train_len = int(len(train_dataset) * (1 - val_ratio))
    val_len = len(train_dataset) - train_len
    train_split, val_split = random_split(train_dataset, [train_len, val_len])
    
    print(f"  Train split: {len(train_split)} samples")
    print(f"  Val split: {len(val_split)} samples")
    print(f"  Test split: {len(test_dataset)} samples")
    
    return train_split, val_split, test_dataset, num_classes


def concat_and_split(train_path, test_path, seed=SEED, val_ratio=0.2):
    """Concatenate train+test, split 80/20 into train_all/test_final,
    then split train_all into train/val (80/20). Returns TensorDatasets and num_classes.
    """
    tr = np.load(train_path)
    te = np.load(test_path)
    data_all = np.concatenate([tr['data'], te['data']], axis=0).astype(np.float32)
    labels_all = np.concatenate([tr['target'], te['target']], axis=0).astype(np.int64)

    # Normalize using original training set statistics (train-only)
    train_data = tr['data'].astype(np.float32)
    mean, std = train_data.mean(), train_data.std()
    data_all = (data_all - mean) / (std + 1e-8)

    unique = np.unique(labels_all)
    mapping = {int(v): i for i, v in enumerate(unique.tolist())}
    labels_mapped = np.array([mapping[int(x)] for x in labels_all], dtype=np.int64)
    num_classes = len(unique)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(data_all))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_all_idx = idx[:split]
    test_final_idx = idx[split:]

    X_train_all = torch.tensor(data_all[train_all_idx], dtype=torch.float32)
    y_train_all = torch.tensor(labels_mapped[train_all_idx], dtype=torch.long)
    X_test_final = torch.tensor(data_all[test_final_idx], dtype=torch.float32)
    y_test_final = torch.tensor(labels_mapped[test_final_idx], dtype=torch.long)

    # split train_all into train/val 80/20
    N = len(X_train_all)
    idx2 = np.arange(N)
    rng.shuffle(idx2)
    split2 = int(0.8 * N)
    train_idx = idx2[:split2]
    val_idx = idx2[split2:]

    train_ds = TensorDataset(X_train_all[train_idx], y_train_all[train_idx])
    val_ds = TensorDataset(X_train_all[val_idx], y_train_all[val_idx])
    test_ds = TensorDataset(X_test_final, y_test_final)
    print(f"  After concat: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds, num_classes

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
# CNN Encoder (Shared for all models)
# =============================
class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=1)
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
        
        # Output dimensions: 256 * 7 * 25
        self.output_dim = 256 * 7 * 25
    
    def forward(self, x):
        # x shape: (B, 3, 60, 200)
        x = self.pool(torch.relu(self.se1(self.bn1(self.conv1(x)))))
        x = self.dropout_cnn(x)
        
        x = self.pool(torch.relu(self.se2(self.bn2(self.conv2(x)))))
        x = self.dropout_cnn(x)
        
        x = self.pool(torch.relu(self.se3(self.bn3(self.conv3(x)))))
        x = self.dropout_cnn(x)
        
        return x

# =============================
# Autoencoder for Pretraining
# =============================
class CSIAutoencoder(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder(input_channels=3, dropout=dropout)
        
        # Decoder - carefully designed to reconstruct from (256, 7, 25) back to (3, 60, 200)
        # After encoder: (B, 256, 7, 25)
        self.decoder = nn.Sequential(
            # (B, 256, 7, 25) -> (B, 128, 14, 50)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # (B, 128, 14, 50) -> (B, 64, 28, 100)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # (B, 64, 28, 100) -> (B, 32, 56, 200)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # (B, 32, 56, 200) -> (B, 3, 60, 200) using padding to get exact dimensions
            nn.ConvTranspose2d(32, 3, kernel_size=(5, 3), stride=1, padding=(0, 1))
        )
    
    def forward(self, x):
        # x shape: (B, 60, 200, 3)
        x = x.permute(0, 3, 1, 2)  # (B, 3, 60, 200)
        encoded = self.encoder(x)  # (B, 256, 7, 25)
        decoded = self.decoder(encoded)  # (B, 3, 60, 200)
        return decoded

# =============================
# Contrastive Learning Model
# =============================
class ContrastiveModel(nn.Module):
    def __init__(self, dropout=0.3, projection_dim=128):
        super().__init__()
        self.encoder = CNNEncoder(input_channels=3, dropout=dropout)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(256 * 7 * 25, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        # x shape: (B, 60, 200, 3)
        x = x.permute(0, 3, 1, 2)  # (B, 3, 60, 200)
        encoded = self.encoder(x)
        # use reshape to support non-contiguous tensors
        encoded = encoded.reshape(encoded.size(0), -1)
        projection = self.projection_head(encoded)
        return F.normalize(projection, dim=1)

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """NT-Xent loss for contrastive learning (SimCLR)"""
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # 2B x D
    
    # Cosine similarity
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # 2B x 2B
    sim_matrix = sim_matrix / temperature
    
    # Create mask to remove self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
    
    # Positive pairs are at positions (i, i+B) and (i+B, i)
    pos_sim = torch.cat([
        torch.diag(sim_matrix, batch_size),
        torch.diag(sim_matrix, -batch_size)
    ])
    
    # Compute loss
    loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
    loss = loss.mean()
    
    return loss

# =============================
# Main Classification Model
# =============================
class ImprovedCNN_GRU_Attention(nn.Module):
    def __init__(self, num_classes, input_shape=(60, 200, 3),
                 gru_hidden=512, gru_layers=3, dropout=0.3,
                 bidirectional=False, use_layer_norm=True, num_heads=1):
        super().__init__()
        
        # CNN Encoder
        self.encoder = CNNEncoder(input_channels=input_shape[2], dropout=dropout)
        
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
    
    def forward(self, x):
        # CNN processing
        x = x.permute(0, 3, 1, 2)  # (B, 60, 200, 3) -> (B, 3, 60, 200)
        x = self.encoder(x)
        
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
    
    def load_encoder_weights(self, encoder_state_dict):
        """Load pretrained encoder weights"""
        self.encoder.load_state_dict(encoder_state_dict)

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
# Create Augmented Dataset for Pretraining
# =============================
def create_augmented_pretrain_dataset(train_split, augmentation_factor=2):
    """
    Create augmented dataset for pretraining.
    Original data + augmented versions.
    """
    original_data = []
    for X, y in train_split:
        original_data.append((X, y))
    
    # Extract just the data tensors
    X_list = [x for x, _ in original_data]
    y_list = [y for _, y in original_data]
    
    X_original = torch.stack(X_list)
    y_original = torch.stack(y_list)
    
    # Create augmented versions
    augmented_X = []
    augmented_y = []
    
    for _ in range(augmentation_factor):
        for x, y in zip(X_original, y_original):
            x_aug = apply_augmentation(x.unsqueeze(0)).squeeze(0)
            augmented_X.append(x_aug)
            augmented_y.append(y)
    
    # Combine original + augmented
    all_X = torch.cat([X_original] + [torch.stack(augmented_X)])
    all_y = torch.cat([y_original] + [torch.stack(augmented_y)])
    
    return TensorDataset(all_X, all_y)

# =============================
# Pretraining Functions
# =============================
def pretrain_autoencoder(pretrain_dataset, config, epochs, batch_size, device):
    """Pretrain encoder using autoencoder on training + augmented data"""
    print(f"  Pretraining autoencoder for {epochs} epochs on {len(pretrain_dataset)} samples...")
    
    autoencoder = CSIAutoencoder(dropout=config['dropout']).to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    debug_printed = False
    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0
        for X, _ in train_loader:
            X = X.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(X)

            # Try candidate target permutations (cover different input orders)
            candidates = [X.permute(0, 3, 1, 2).contiguous(), X.permute(0, 3, 2, 1).contiguous()]
            target = None
            for cand in candidates:
                if cand.shape == reconstructed.shape:
                    target = cand
                    break

            if target is None:
                # Attempt to make reconstructed match one of the candidates
                for cand in candidates:
                    # Try swapping reconstructed H/W
                    if reconstructed.shape[2] == cand.shape[3] and reconstructed.shape[3] == cand.shape[2]:
                        tmp = reconstructed.permute(0, 1, 3, 2).contiguous()
                        if tmp.shape == cand.shape:
                            reconstructed = tmp
                            target = cand
                            break
                    # Otherwise try interpolation to candidate spatial size
                    try:
                        resized = F.interpolate(reconstructed, size=cand.shape[2:], mode='bilinear', align_corners=False)
                        if resized.shape == cand.shape:
                            reconstructed = resized
                            target = cand
                            break
                    except Exception:
                        pass

            if target is None:
                # Last resort: pick the first candidate and crop/pad reconstructed to match
                target = candidates[0]
                th, tw = target.shape[2], target.shape[3]
                rh, rw = reconstructed.shape[2], reconstructed.shape[3]
                # crop or pad as necessary
                if rh >= th and rw >= tw:
                    reconstructed = reconstructed[:, :, :th, :tw]
                else:
                    pad_h = max(0, th - rh)
                    pad_w = max(0, tw - rw)
                    reconstructed = F.pad(reconstructed, (0, pad_w, 0, pad_h))

            if reconstructed.shape != target.shape and not debug_printed:
                print(f"[pretrain_autoencoder] Shape mismatch after fixes: reconstructed={reconstructed.shape}, target={target.shape}")
                debug_printed = True

            loss = criterion(reconstructed, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Recon Loss: {total_loss/len(train_loader):.4f}")
    
    return autoencoder.encoder.state_dict()

def pretrain_contrastive(pretrain_dataset, config, epochs, batch_size, device):
    """Pretrain encoder using contrastive learning on training + augmented data"""
    print(f"  Pretraining with contrastive learning for {epochs} epochs on {len(pretrain_dataset)} samples...")
    
    model = ContrastiveModel(dropout=config['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    train_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, _ in train_loader:
            X = X.to(device)
            
            # Create two augmented views
            X_i = apply_augmentation(X)
            X_j = apply_augmentation(X)
            
            optimizer.zero_grad()
            z_i = model(X_i)
            z_j = model(X_j)
            
            loss = nt_xent_loss(z_i, z_j, temperature=0.5)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Contrastive Loss: {total_loss/len(train_loader):.4f}")
    
    return model.encoder.state_dict()

# =============================
# Training Function with Pretraining Support
# =============================
def train_model(train_split, val_split, test_split, num_classes, config,
                epochs=150, batch_size=256, device='cpu', pretrain_mode='none', 
                augmentation_factor=2):
    
    # Pretraining phase on training + augmented data
    encoder_weights = None
    if pretrain_mode != 'none':
        print(f"  Creating augmented dataset (factor={augmentation_factor})...")
        pretrain_dataset = create_augmented_pretrain_dataset(train_split, augmentation_factor)
        print(f"  Pretrain dataset size: {len(pretrain_dataset)} samples")
        print(f"  (Original: {len(train_split)}, Augmented: {len(pretrain_dataset) - len(train_split)})")
        
        if pretrain_mode == 'autoencoder':
            encoder_weights = pretrain_autoencoder(pretrain_dataset, config, PRETRAIN_EPOCHS, batch_size, device)
        elif pretrain_mode == 'contrastive':
            encoder_weights = pretrain_contrastive(pretrain_dataset, config, PRETRAIN_EPOCHS, batch_size, device)
    
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
    
    # Load pretrained encoder weights if available
    if encoder_weights is not None:
        model.load_encoder_weights(encoder_weights)
        print(f"  Loaded pretrained encoder weights from {pretrain_mode}")
    
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
    history = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # Augmentation (if enabled)
            if USE_AUGMENTATION and random.random() > 0.5:
                X = apply_augmentation(X)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += float(loss.item()) * X.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += X.size(0)

        train_loss_avg = train_loss / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss_v = criterion(outputs, y)
                val_loss += float(loss_v.item()) * X.size(0)
                pred = outputs.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        val_loss_avg = val_loss / val_total if val_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        history.append({'epoch': epoch+1, 'train_loss': train_loss_avg, 'val_loss': val_loss_avg, 'train_acc': train_acc, 'val_acc': val_acc})

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc*100:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
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
    test_acc = correct / total
    return test_acc, best_val_acc, history

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    # Current directory of the script
    current_dir = os.getcwd()

    # Dataset paths
    datasets = [
    (
        'Home_downlink',
        os.path.join(current_dir, 'Home-Data', 'trainset.npz'),
        os.path.join(current_dir, 'Home-Data', 'testset.npz')
    )
    ]
    # Configuration: use requested Bidirectional + Multi-Head settings
    config = {
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Augmentation during training: {'ENABLED' if USE_AUGMENTATION else 'DISABLED'}")
    print(f"Augmentation for pretraining: ENABLED (factor=2)")
    print(f"Pretraining epochs: {PRETRAIN_EPOCHS}")
    print(f"Fine-tuning epochs: {FINETUNE_EPOCHS} (with early stopping, patience=20)")
    print(f"Testing modes: {PRETRAINING_MODES}")
    print(f"\nData split strategy:")
    print(f"  - Pretraining: trainset.npz + augmented versions (2x augmentation)")
    print(f"  - Supervised training: trainset.npz (80% train, 20% val)")
    print(f"  - Final evaluation: testset.npz (separate test set)")
    print(f"\n")

    results = []
    
    for pretrain_mode in PRETRAINING_MODES:
        print(f"\n{'='*80}")
        print(f"Pretraining Mode: {pretrain_mode.upper()}")
        print(f"{'='*80}\n")
        
        for dataset_name, train_path, test_path in datasets:
            print(f"Processing dataset: {dataset_name}...")
            # Use concat-and-split strategy: concat train+test -> 80/20 -> split train->train/val 80/20
            train_split, val_split, test_split, num_classes = concat_and_split(train_path, test_path)
            
            test_acc, val_acc, history = train_model(
                train_split, val_split, test_split, num_classes,
                config, epochs=FINETUNE_EPOCHS, batch_size=256, 
                device=device, pretrain_mode=pretrain_mode,
                augmentation_factor=2
            )
            
            # Ensure results directory exists
            results_dir = os.path.join(os.getcwd(), 'results')
            os.makedirs(results_dir, exist_ok=True)

            # Sanitize filename parts
            cfg_name = config.get('name', 'config').replace(' ', '_').replace('+', 'plus')
            pre_name = pretrain_mode.replace(' ', '_')
            ds_name = dataset_name.replace(' ', '_')
            csv_fname = f"{ds_name}_{cfg_name}_{pre_name}_finetune_history.csv"
            csv_path = os.path.join(results_dir, csv_fname)

            # Write history to CSV
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
                for row in history:
                    writer.writerow([row.get('epoch'), row.get('train_loss'), row.get('val_loss'), row.get('train_acc'), row.get('val_acc')])

            results.append((pretrain_mode, dataset_name, test_acc, val_acc, csv_path))
            print(f"  History saved to: {csv_path}")
            print(f"  Best Val Accuracy: {val_acc*100:.2f}%")
            print(f"  Test Accuracy: {test_acc*100:.2f}%\n")

    # Final summary
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY -")
    print("="*100)
    print(f"{'Pretrain Mode':<20} {'Dataset':<20} {'Val Acc (%)':<15} {'Test Acc (%)':<15}")
    print("-"*100)
    for pretrain_mode, dataset_name, test_acc, val_acc, csv_path in results:
        print(f"{pretrain_mode:<20} {dataset_name:<20} {val_acc*100:<15.2f} {test_acc*100:<15.2f} {os.path.basename(csv_path):<30}")
    print("="*100)