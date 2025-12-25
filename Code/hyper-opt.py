
import os
import time
import json
import itertools
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ray

# ============================
# Initialize Ray for multi-GPU
# ============================
ray.init()

# ============================
# Original load function
# ============================
def load_and_normalize_data(train_path, test_path):
    train = np.load(train_path)
    test = np.load(test_path)
    train_csi = train["data"].astype(np.float32)
    train_labels = train["target"].astype(np.int64)
    test_csi = test["data"].astype(np.float32)
    test_labels = test["target"].astype(np.int64)
    mean, std = train_csi.mean(), train_csi.std()
    train_csi = (train_csi - mean) / (std + 1e-8)
    test_csi = (test_csi - mean) / (std + 1e-8)
    train_csi = torch.tensor(train_csi)
    test_csi = torch.tensor(test_csi)
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)
    unique_labels = torch.unique(train_labels)
    label_mapping = {int(old): i for i, old in enumerate(unique_labels.tolist())}
    train_labels = torch.tensor([label_mapping[int(lbl)] for lbl in train_labels])
    test_labels = torch.tensor([label_mapping[int(lbl)] for lbl in test_labels])
    num_classes = len(unique_labels)
    return train_csi, train_labels, test_csi, test_labels, num_classes

# ============================
# Models (unchanged)
# ============================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
    def forward(self, gru_output):
        gru_output = self.layer_norm(gru_output)
        weights = torch.softmax(self.attention(gru_output), dim=1)
        context = torch.sum(weights * gru_output, dim=1)
        return context, weights

class ImprovedGRU(nn.Module):
    def __init__(self, num_classes, input_shape=(60, 200, 3),
                 gru_hidden=256, gru_layers=3, dropout=0.3,
                 bidirectional=False, use_layer_norm=True):
        super().__init__()
        self.input_size = input_shape[0] * input_shape[2]
        self.seq_len = input_shape[1]
        self.input_norm = nn.LayerNorm(self.input_size) if use_layer_norm else nn.Identity()
        self.gru = nn.GRU(self.input_size, gru_hidden, gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0,
                          bidirectional=bidirectional)
        out_size = gru_hidden * 2 if bidirectional else gru_hidden
        self.output_norm = nn.LayerNorm(out_size) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        b = x.size(0)
        x = x.permute(0, 2, 1, 3).reshape(b, self.seq_len, -1)
        x = self.input_norm(x)
        out, _ = self.gru(x)
        out = self.output_norm(out)
        x = out[:, -1, :]
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ImprovedCNN_GRU(nn.Module):
    def __init__(self, num_classes, input_shape=(60, 200, 3),
                 gru_hidden=256, gru_layers=3, dropout=0.3,
                 bidirectional=False, use_layer_norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        self.gru_input_size = 64 * 15
        self.gru_seq_len = 50
        self.input_norm = nn.LayerNorm(self.gru_input_size) if use_layer_norm else nn.Identity()
        self.gru = nn.GRU(self.gru_input_size, gru_hidden, gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0,
                          bidirectional=bidirectional)
        out_size = gru_hidden * 2 if bidirectional else gru_hidden
        self.output_norm = nn.LayerNorm(out_size) if use_layer_norm else nn.Identity()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout_cnn(x)
        b = x.size(0)
        x = x.permute(0, 3, 1, 2).reshape(b, self.gru_seq_len, -1)
        x = self.input_norm(x)
        out, _ = self.gru(x)
        out = self.output_norm(out)
        x = out[:, -1, :]
        x = self.dropout_fc(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)

class ImprovedCNN_GRU_Attention(ImprovedCNN_GRU):
    def __init__(self, num_classes, input_shape=(60, 200, 3),
                 gru_hidden=256, gru_layers=3, dropout=0.3,
                 bidirectional=False, use_layer_norm=True):
        super().__init__(num_classes, input_shape, gru_hidden, gru_layers,
                         dropout, bidirectional, use_layer_norm)
        out_size = gru_hidden * 2 if bidirectional else gru_hidden
        self.attention = AttentionLayer(out_size)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout_cnn(x)
        b = x.size(0)
        x = x.permute(0, 3, 1, 2).reshape(b, self.gru_seq_len, -1)
        x = self.input_norm(x)
        out, _ = self.gru(x)
        context, _ = self.attention(out)
        x = self.dropout_fc(context)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)

# ============================
# Training function
# ============================
def train_model(model, train_loader, val_loader, config, device):
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(config['epochs']):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    # Simple validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# ============================
# Ray remote function
# ============================
@ray.remote(num_gpus=1)
def run_experiment(model_class, config, data, labels, num_classes, best_gru_arch=None):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    split = int(0.8 * len(data))
    train_ds = TensorDataset(data[:split], labels[:split])
    val_ds = TensorDataset(data[split:], labels[split:])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    if best_gru_arch:
        model = model_class(num_classes,
                            gru_hidden=best_gru_arch['gru_hidden'],
                            gru_layers=best_gru_arch['gru_layers'],
                            dropout=config['dropout'],
                            bidirectional=best_gru_arch['bidirectional'],
                            use_layer_norm=config['use_layer_norm'])
    else:
        model = model_class(num_classes,
                            gru_hidden=config['gru_hidden'],
                            gru_layers=config['gru_layers'],
                            dropout=config['dropout'],
                            bidirectional=config['bidirectional'],
                            use_layer_norm=True)
    acc = train_model(model, train_loader, val_loader, config, device)
    print(f"Completed: {config} | Accuracy: {acc:.4f}")
    return {**config, 'accuracy': acc}

# ============================
# Main logic
# ============================
if __name__ == "__main__":
    # Load data
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    train_path = os.path.join(parent_dir, 'Home_downlink', 'trainset.npz')
    test_path = os.path.join(parent_dir, 'Home_downlink', 'testset.npz')
    train_csi, train_labels, test_csi, test_labels, num_classes = load_and_normalize_data(train_path, test_path)
    data = torch.cat([train_csi, test_csi], dim=0)
    labels = torch.cat([train_labels, test_labels], dim=0)

    # Phase 1 configs
    phase1_configs = [{'gru_hidden': h, 'gru_layers': l, 'bidirectional': b,
                       'dropout': 0.3, 'use_layer_norm': True, 'lr': 5e-4,
                       'optimizer': 'AdamW', 'weight_decay': 1e-4, 'epochs': 100}
                      for h, l, b in itertools.product([128, 256, 512], [2, 3, 4], [False, True])]

    futures = [run_experiment.remote(ImprovedGRU, cfg, data, labels, num_classes) for cfg in phase1_configs]
    results_phase1 = ray.get(futures)
    top5_phase1 = sorted(results_phase1, key=lambda x: x['accuracy'], reverse=True)[:5]
    best_gru_arch = {'gru_hidden': top5_phase1[0]['gru_hidden'],
                     'gru_layers': top5_phase1[0]['gru_layers'],
                     'bidirectional': top5_phase1[0]['bidirectional']}
    print("Top 5 Phase 1:", top5_phase1)

    # Phase 2 configs
    phase2_configs = [{'dropout': d, 'use_layer_norm': ln, 'lr': lr,
                       'optimizer': opt, 'weight_decay': wd, 'epochs': 100}
                      for d in [0.2, 0.3, 0.4]
                      for ln in [True, False]
                      for lr in [1e-4, 5e-4]
                      for opt in ['AdamW', 'Adam']
                      for wd in [1e-5, 1e-4]]

    futures = [run_experiment.remote(ImprovedCNN_GRU, cfg, data, labels, num_classes, best_gru_arch) for cfg in phase2_configs]
    results_phase2 = ray.get(futures)
    top5_phase2 = sorted(results_phase2, key=lambda x: x['accuracy'], reverse=True)[:5]
    best_config_phase2 = top5_phase2[0]
    print("Top 5 Phase 2:", top5_phase2)

    # Phase 3 (reuse best Phase 2 config)
    futures = [run_experiment.remote(ImprovedCNN_GRU_Attention, best_config_phase2, data, labels, num_classes, best_gru_arch)]
    results_phase3 = ray.get(futures)
    print("Phase 3 result:", results_phase3)
