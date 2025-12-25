# -*- coding: utf-8 -*-
"""Multi-Classifier Comparison with 5-Fold Cross-Validation for CSI Data"""

import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import entropy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_statistical_features(csi_sample):
    """Extract comprehensive handcrafted statistical features from CSI sample"""
    features = []
    
    time_steps = csi_sample.shape[1]
    subcarriers = csi_sample.shape[0]
    antennas = csi_sample.shape[2]
    
    # ========== 1. STANDARD DEVIATION FEATURES ==========
    std_temporal = np.std(csi_sample, axis=1)
    features.extend(std_temporal.flatten())
    
    # FIXED: Summarize spatial std instead of flattening (was 12000 features)
    std_spatial = np.std(csi_sample, axis=2)
    features.append(np.mean(std_spatial))
    features.append(np.std(std_spatial))
    features.append(np.max(std_spatial))
    features.append(np.min(std_spatial))
    
    features.append(np.std(csi_sample))
    
    # ========== 2. ENTROPY FEATURES ==========
    for ant in range(antennas):
        ant_data = csi_sample[:, :, ant].flatten()
        hist, _ = np.histogram(ant_data, bins=50, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        ent = entropy(hist)
        features.append(ent)
    
    for ant in range(antennas):
        for sub in range(0, subcarriers, 10):
            temporal_slice = csi_sample[sub, :, ant]
            hist, _ = np.histogram(temporal_slice, bins=20, density=True)
            hist = hist + 1e-10
            hist = hist / hist.sum()
            features.append(entropy(hist))
    
    # ========== 3. SPECTRAL ENERGY FEATURES ==========
    for ant in range(antennas):
        ant_data = csi_sample[:, :, ant]
        avg_temporal = np.mean(ant_data, axis=0)
        fft_vals = np.fft.fft(avg_temporal)
        power_spectrum = np.abs(fft_vals) ** 2
        
        features.append(np.sum(power_spectrum))
        
        n = len(power_spectrum)
        features.append(np.sum(power_spectrum[:n//4]))
        features.append(np.sum(power_spectrum[n//4:n//2]))
        features.append(np.sum(power_spectrum[n//2:3*n//4]))
    
    for ant in range(antennas):
        ant_data = csi_sample[:, :, ant]
        avg_temporal = np.mean(ant_data, axis=0)
        fft_vals = np.abs(np.fft.fft(avg_temporal))
        freqs = np.fft.fftfreq(len(avg_temporal))
        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)
        features.append(spectral_centroid)
    
    # ========== 4. CORRELATION FEATURES ==========
    for ant1 in range(antennas):
        for ant2 in range(ant1 + 1, antennas):
            sig1 = np.mean(csi_sample[:, :, ant1], axis=0)
            sig2 = np.mean(csi_sample[:, :, ant2], axis=0)
            
            corr = np.corrcoef(sig1, sig2)[0, 1]
            features.append(corr if not np.isnan(corr) else 0.0)
    
    for ant in range(antennas):
        sig = np.mean(csi_sample[:, :, ant], axis=0)
        for lag in [1, 5, 10, 20]:
            if lag < len(sig):
                autocorr = np.corrcoef(sig[:-lag], sig[lag:])[0, 1]
                features.append(autocorr if not np.isnan(autocorr) else 0.0)
            else:
                features.append(0.0)
    
    # ========== 5. BASIC STATISTICS ==========
    features.append(np.mean(csi_sample))
    features.append(np.var(csi_sample))
    features.append(np.ptp(csi_sample))
    
    flattened = csi_sample.flatten()
    mean_val = np.mean(flattened)
    std_val = np.std(flattened)
    if std_val > 0:
        skewness = np.mean(((flattened - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((flattened - mean_val) / std_val) ** 4)
    else:
        skewness = 0.0
        kurtosis = 0.0
    features.append(skewness)
    features.append(kurtosis)
    
    # ========== 6. ZERO-CROSSING RATE ==========
    for ant in range(antennas):
        sig = np.mean(csi_sample[:, :, ant], axis=0)
        sig_centered = sig - np.mean(sig)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(sig_centered)))) / (2 * len(sig))
        features.append(zero_crossings)
    
    # ========== 7. SIGNAL ENERGY DISTRIBUTION ==========
    for ant in range(antennas):
        sig = np.mean(csi_sample[:, :, ant], axis=0)
        n = len(sig)
        early_energy = np.sum(sig[:n//3] ** 2)
        middle_energy = np.sum(sig[n//3:2*n//3] ** 2)
        late_energy = np.sum(sig[2*n//3:] ** 2)
        total_energy = np.sum(sig ** 2) + 1e-10
        
        features.append(early_energy / total_energy)
        features.append(middle_energy / total_energy)
        features.append(late_energy / total_energy)
    
    # ========== 8. AMPLITUDE ENVELOPE ==========
    for ant in range(antennas):
        sig = np.mean(csi_sample[:, :, ant], axis=0)
        try:
            analytic_signal = signal.hilbert(sig)
            amplitude_envelope = np.abs(analytic_signal)
            
            features.append(np.mean(amplitude_envelope))
            features.append(np.std(amplitude_envelope))
            features.append(np.max(amplitude_envelope))
            features.append(np.min(amplitude_envelope))
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    # ========== 9. SPECTRAL ROLLOFF ==========
    for ant in range(antennas):
        sig = np.mean(csi_sample[:, :, ant], axis=0)
        fft_vals = np.abs(np.fft.fft(sig))
        power = fft_vals ** 2
        cumulative_power = np.cumsum(power)
        total_power = cumulative_power[-1]
        
        rolloff_idx = np.where(cumulative_power >= 0.85 * total_power)[0]
        if len(rolloff_idx) > 0:
            spectral_rolloff = rolloff_idx[0] / len(power)
        else:
            spectral_rolloff = 1.0
        features.append(spectral_rolloff)
    
    # ========== 10. SPECTRAL FLATNESS ==========
    for ant in range(antennas):
        sig = np.mean(csi_sample[:, :, ant], axis=0)
        fft_vals = np.abs(np.fft.fft(sig))
        power = fft_vals ** 2 + 1e-10
        
        geometric_mean = np.exp(np.mean(np.log(power)))
        arithmetic_mean = np.mean(power)
        spectral_flatness = geometric_mean / arithmetic_mean
        features.append(spectral_flatness)
    
    # ========== 11. PERCENTILE FEATURES ==========
    for percentile in [10, 25, 50, 75, 90]:
        features.append(np.percentile(flattened, percentile))
    
    # ========== 12. IQR ==========
    features.append(np.percentile(flattened, 75) - np.percentile(flattened, 25))
    
    # ========== 13. RMS ENERGY ==========
    for ant in range(antennas):
        sig = np.mean(csi_sample[:, :, ant], axis=0)
        rms = np.sqrt(np.mean(sig ** 2))
        features.append(rms)
    
    # ========== 14. CREST FACTOR ==========
    for ant in range(antennas):
        sig = np.mean(csi_sample[:, :, ant], axis=0)
        rms = np.sqrt(np.mean(sig ** 2)) + 1e-10
        peak = np.max(np.abs(sig))
        crest_factor = peak / rms
        features.append(crest_factor)
    
    return np.array(features)

def extract_features_from_dataset(data):
    """Extract features from entire dataset (silently)"""
    num_samples = data.shape[0]
    first_features = extract_statistical_features(data[0])
    num_features = len(first_features)
    
    feature_matrix = np.zeros((num_samples, num_features))
    
    for i in range(num_samples):
        feature_matrix[i] = extract_statistical_features(data[i])
    
    return feature_matrix, num_features

# ============================================================================
# Data Loading
# ============================================================================

def load_data(train_path, test_path):
    """Load CSI data from npz files (silently)"""
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    train_csi = train_data['data']
    train_labels = train_data['target']
    test_csi = test_data['data']
    test_labels = test_data['target']
    
    unique_labels = np.unique(train_labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    
    train_labels = np.array([label_mapping[label] for label in train_labels])
    test_labels = np.array([label_mapping[label] for label in test_labels])
    
    return train_csi, train_labels, test_csi, test_labels, num_classes

# ============================================================================
# PyTorch ANN Model with Batch Normalization
# ============================================================================

class ANN_BatchNorm(nn.Module):
    """Custom ANN with Batch Normalization for CSI classification"""
    def __init__(self, input_size, num_classes):
        super(ANN_BatchNorm, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 552)
        self.bn1 = nn.BatchNorm1d(552)
        
        self.fc2 = nn.Linear(552, 276)
        self.bn2 = nn.BatchNorm1d(276)
        
        self.fc3 = nn.Linear(276, 276)
        self.bn3 = nn.BatchNorm1d(276)
        
        self.fc4 = nn.Linear(276, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

class PyTorchANNWrapper:
    """Wrapper to make PyTorch ANN work like sklearn classifiers"""
    def __init__(self, input_size, num_classes, epochs=100, batch_size=64, lr=0.001, device='cpu'):
        self.input_size = input_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = None
    
    def fit(self, X, y):
        """Train the model"""
        # Initialize model
        self.model = ANN_BatchNorm(self.input_size, self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()

# ============================================================================
# Classifier Configuration Printing
# ============================================================================

def print_classifier_configurations():
    """Print detailed configurations for all classifiers"""
    print("="*90)
    print("CLASSIFIER CONFIGURATIONS")
    print("="*90)
    print()
    
    print("1. Decision Tree")
    print("   - criterion: 'gini'")
    print("   - max_depth: 20")
    print("   - min_samples_split: 10")
    print("   - min_samples_leaf: 5")
    print("   - max_features: 'sqrt'")
    print("   - random_state: 42")
    print()
    
    print("2. Random Forest")
    print("   - n_estimators: 100")
    print("   - max_depth: 15")
    print("   - random_state: 42")
    print("   - n_jobs: -1 (use all processors)")
    print()
    
    print("3. Naive Bayes (Gaussian)")
    print("   - priors: None (learned from data)")
    print("   - var_smoothing: 1e-09 (default)")
    print()
    
    print("4. Logistic Regression")
    print("   - penalty: 'l2' (default)")
    print("   - C: 1.0 (default, inverse regularization strength)")
    print("   - solver: 'lbfgs' (default)")
    print("   - max_iter: 1000")
    print("   - random_state: 42")
    print("   - n_jobs: -1 (use all processors)")
    print()
    
    print("5. Support Vector Machine (SVM)")
    print("   - kernel: 'rbf' (Radial Basis Function)")
    print("   - C: 10.0 (regularization parameter)")
    print("   - gamma: 'scale' (1 / (n_features * X.var()))")
    print("   - random_state: 42")
    print()
    
    print("6. k-Nearest Neighbors (k-NN)")
    print("   - n_neighbors: 5")
    print("   - weights: 'distance' (weight by inverse distance)")
    print("   - metric: 'euclidean'")
    print("   - n_jobs: -1 (use all processors)")
    print()
    
    print("7. AdaBoost")
    print("   - estimator: DecisionTreeClassifier (default)")
    print("   - n_estimators: 50")
    print("   - learning_rate: 1.0 (default)")
    print("   - algorithm: 'SAMME.R' (default)")
    print("   - random_state: 42")
    print()
    
    print("8. ANN (Multi-Layer Perceptron)")
    print("   - Architecture: Input -> 552 -> 276 -> 276 -> Output")
    print("   - Activation: ReLU")
    print("   - Batch Normalization: Applied after each hidden layer")
    print("   - Dropout: 0.3 (applied after each hidden layer)")
    print("   - Optimizer: Adam (lr=0.001, weight_decay=1e-4)")
    print("   - Loss Function: CrossEntropyLoss")
    print("   - Epochs: 100")
    print("   - Batch Size: 64")
    print()
    print("="*90)
    print()

# ============================================================================
# 5-Fold Cross-Validation
# ============================================================================

def evaluate_with_cv(clf, X, y, n_splits=5):
    """Perform 5-fold cross-validation and return average metrics"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_val)
        
        # Metrics
        accuracies.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_val, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
    
    return {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1_score': np.mean(f1_scores),
        'accuracy_std': np.std(accuracies),
        'precision_std': np.std(precisions),
        'recall_std': np.std(recalls),
        'f1_std': np.std(f1_scores)
    }

# ============================================================================
# Classifier Training and Evaluation
# ============================================================================

def run_all_classifiers(dataset_name, train_path, test_path, use_scaling=True):
    """Run all classifiers on a single dataset with 5-fold CV"""
    
    # Load data
    train_csi, train_labels, test_csi, test_labels, num_classes = load_data(train_path, test_path)
    
    # Extract features
    train_features, num_features = extract_features_from_dataset(train_csi)
    test_features, _ = extract_features_from_dataset(test_csi)
    
    # Handle NaN/Inf
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=1e10, neginf=-1e10)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Feature scaling
    if use_scaling:
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
    
    # Setup device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define classifiers
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(
            criterion='gini',
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            random_state=42
        ),
        'k-NN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        ),
        'ANN (MLP)': PyTorchANNWrapper(
            input_size=num_features,
            num_classes=num_classes,
            epochs=100,
            batch_size=64,
            lr=0.001,
            device=device
        )
    }
    
    # Evaluate all classifiers with 5-fold CV
    results = []
    for clf_name, clf in classifiers.items():
        cv_results = evaluate_with_cv(clf, train_features, train_labels, n_splits=5)
        cv_results['classifier'] = clf_name
        cv_results['dataset'] = dataset_name
        cv_results['num_features'] = num_features
        cv_results['train_samples'] = len(train_labels)
        cv_results['test_samples'] = len(test_labels)
        cv_results['num_classes'] = num_classes
        cv_results['scaling'] = 'Enabled' if use_scaling else 'Disabled'
        results.append(cv_results)
    
    return results

# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    # Setup paths
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    
    # Define datasets
    datasets = [
        ('Lab_downlink', 
         os.path.join(parent_dir, 'Lab_downlink', 'trainset.npz'),
         os.path.join(parent_dir, 'Lab_downlink', 'testset.npz')),
        
        ('Lab_150_down', 
         os.path.join(parent_dir, 'Lab_150_down', 'trainset.npz'),
         os.path.join(parent_dir, 'Lab_150_down', 'testset.npz')),
        
        ('Home_downlink', 
         os.path.join(parent_dir, 'Home_downlink', 'trainset.npz'),
         os.path.join(parent_dir, 'Home_downlink', 'testset.npz'))
    ]
    
    # Print classifier configurations
    print_classifier_configurations()
    
    # Run experiments for BOTH configurations
    all_results = []
    dataset_info_printed = False
    
    for use_scaling in [True, False]:
        scaling_label = "WITH Feature Scaling" if use_scaling else "WITHOUT Feature Scaling"
        
        print("="*90)
        print(f"MULTI-CLASSIFIER COMPARISON - {scaling_label}")
        print("5-Fold Cross-Validation")
        print("="*90)
        print()
        
        # Run experiments
        for dataset_name, train_path, test_path in datasets:
            print(f"Processing {dataset_name} ({scaling_label})...", end=' ')
            try:
                results = run_all_classifiers(dataset_name, train_path, test_path, use_scaling)
                all_results.extend(results)
                
                # Print dataset info only once
                if not dataset_info_printed and results:
                    print("\n")
                    print("="*90)
                    print("DATASET INFORMATION")
                    print("="*90)
                    for r in results[:1]:
                        print(f"Number of Features Extracted: {r['num_features']}")
                    print()
                    dataset_info_printed = True
                
                print("✓ Done")
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                continue
        
        print()
    
    # ========== SUMMARY TABLES ==========
    
    datasets_list = ['Lab_downlink', 'Lab_150_down', 'Home_downlink']
    classifiers_list = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 
                       'SVM', 'k-NN', 'AdaBoost', 'ANN (MLP)']
    
    # Print results for each configuration
    for use_scaling in [True, False]:
        scaling_label = "WITH Feature Scaling" if use_scaling else "WITHOUT Feature Scaling"
        config_results = [r for r in all_results if r['scaling'] == ('Enabled' if use_scaling else 'Disabled')]
        
        print("="*90)
        print(f"RESULTS: {scaling_label}")
        print("="*90)
        print()
        
        for dataset in datasets_list:
            dataset_results = [r for r in config_results if r['dataset'] == dataset]
            
            if not dataset_results:
                continue
            
            # Print dataset info
            print("="*90)
            print(f"DATASET: {dataset}")
            if dataset_results:
                print(f"Train Samples: {dataset_results[0]['train_samples']}, "
                      f"Test Samples: {dataset_results[0]['test_samples']}, "
                      f"Classes: {dataset_results[0]['num_classes']}")
            print("="*90)
            print(f"{'Classifier':<25} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
            print("-"*90)
            
            for clf_name in classifiers_list:
                result = next((r for r in dataset_results if r['classifier'] == clf_name), None)
                if result:
                    print(f"{result['classifier']:<25} "
                          f"{result['accuracy']*100:>6.2f}±{result['accuracy_std']*100:>4.2f}%  "
                          f"{result['precision']*100:>6.2f}±{result['precision_std']*100:>4.2f}%  "
                          f"{result['recall']*100:>6.2f}±{result['recall_std']*100:>4.2f}%  "
                          f"{result['f1_score']*100:>6.2f}±{result['f1_std']*100:>4.2f}%")
            print()
    
    # ========== COMPARISON TABLE ==========
    print("="*90)
    print("SUMMARY: ALL CLASSIFIERS ACROSS ALL DATASETS AND CONFIGURATIONS")
    print("="*90)
    print(f"{'Classifier':<25} {'Dataset':<20} {'Scaling':<12} {'Accuracy':<15} {'F1-Score':<15}")
    print("-"*90)
    
    for scaling_config in ['Enabled', 'Disabled']:
        for clf_name in classifiers_list:
            for dataset in datasets_list:
                result = next((r for r in all_results 
                             if r['classifier'] == clf_name 
                             and r['dataset'] == dataset 
                             and r['scaling'] == scaling_config), None)
                if result:
                    print(f"{result['classifier']:<25} "
                          f"{result['dataset']:<20} "
                          f"{result['scaling']:<12} "
                          f"{result['accuracy']*100:>6.2f}±{result['accuracy_std']*100:>4.2f}%  "
                          f"{result['f1_score']*100:>6.2f}±{result['f1_std']*100:>4.2f}%")
    
    print("="*90)
    print("EXPERIMENT COMPLETED")
    print("="*90)