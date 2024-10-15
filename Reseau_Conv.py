#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:06:18 2024

@author: achillegausseres
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('/Users/achillegausseres/OneDrive/PRO/ACO/3A/Conf_Machine_Learning/Conf_Machine_Learning_repo/dfplusIQA.csv')

# Encodage One-Hot pour 'station' et 'wd'
categorical_vars = ['station', 'wd']
df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Convertir les colonnes quantitatives en numériques
quantitative_vars = ['TEMP', 'PRES', 'DEWP', 'WSPM', 'RAIN']
for var in quantitative_vars:
    df[var] = pd.to_numeric(df[var], errors='coerce')

# Supprimer les lignes avec des NaN
df.dropna(inplace=True)

# Normalisation des variables quantitatives
mean = df[quantitative_vars].mean()
std = df[quantitative_vars].std()
df[quantitative_vars] = (df[quantitative_vars] - mean) / std

# Encodage des variables temporelles
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df = df.drop(['year', 'month', 'day', 'hour'], axis=1)

# Encodage de la cible
df['IQA_encoded'] = df['IQA'].astype('category').cat.codes

# Vérifier les types de données après prétraitement
print(df.dtypes)

# Préparer les séquences
sequence_length = 24
features = df.drop(['IQA', 'IQA_encoded'], axis=1).values
target = df['IQA_encoded'].values

def create_sequences(features, target, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(features)):
        X.append(features[i - sequence_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)

X, y = create_sequences(features, target, sequence_length)

# Vérifier les types et la forme
print(f'X dtype: {X.dtype}, y dtype: {y.dtype}')
print(f'X shape: {X.shape}, y shape: {y.shape}')

# Assurez-vous que X est float32 et y est int64
X = X.astype(np.float32)
y = y.astype(np.int64)

print(f'X dtype after conversion: {X.dtype}, y dtype after conversion: {y.dtype}')

# Diviser les données
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f'Train X shape: {X_train.shape}')
print(f'Train y shape: {y_train.shape}')
print(f'Test X shape: {X_test.shape}')
print(f'Test y shape: {y_test.shape}')

# Définir le Dataset personnalisé
class AQADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Créer les datasets et les DataLoaders
batch_size = 32
train_dataset = AQADataset(X_train, y_train)
test_dataset = AQADataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Définir le modèle Conv1D
class Conv1DModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        # Calculer la taille après les convolutions
        # Chaque convolution sans padding réduit la longueur de la séquence de (kernel_size - 1)
        # Après deux convolutions : sequence_length - 2*(3-1) = sequence_length - 4
        self.fc1 = nn.Linear(64 * (sequence_length - 4), 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, n_features)
        x = x.permute(0, 2, 1)  # (batch_size, n_features, sequence_length) pour Conv1D
        x = self.relu(self.conv1(x))  # (batch_size, 64, L1)
        x = self.relu(self.conv2(x))  # (batch_size, 64, L2)
        x = self.flatten(x)            # (batch_size, 64 * L2)
        x = self.relu(self.fc1(x))     # (batch_size, 100)
        x = self.dropout(x)
        x = self.fc2(x)                # (batch_size, num_classes)
        return x

# Initialiser le modèle
n_features = X.shape[2]
num_classes = len(np.unique(y))
model = Conv1DModel(input_channels=n_features, num_classes=num_classes)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Déplacer le modèle vers le device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Entraîner le modèle
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total

    # Évaluation sur l'ensemble de validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0

    with torch.no_grad():
        for val_inputs, val_labels in test_loader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_inputs)
            loss = criterion(val_outputs, val_labels)

            val_loss += loss.item() * val_inputs.size(0)
            _, val_preds = torch.max(val_outputs, 1)
            val_corrects += torch.sum(val_preds == val_labels.data)
            val_total += val_labels.size(0)

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_corrects.double() / val_total

    print(f'Époque {epoch+1}/{num_epochs}:')
    print(f'Perte Entraînement: {epoch_loss:.4f} | Précision Entraînement: {epoch_acc:.4f}')
    print(f'Perte Validation: {val_epoch_loss:.4f} | Précision Validation: {val_epoch_acc:.4f}')
    print('-' * 30)

# Évaluer le modèle
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for val_inputs, val_labels in test_loader:
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)

        val_outputs = model(val_inputs)
        loss = criterion(val_outputs, val_labels)

        val_loss += loss.item() * val_inputs.size(0)
        _, val_preds = torch.max(val_outputs, 1)
        val_corrects += torch.sum(val_preds == val_labels.data)
        val_total += val_labels.size(0)

        all_preds.extend(val_preds.cpu().numpy())
        all_labels.extend(val_labels.cpu().numpy())

val_epoch_loss = val_loss / val_total
val_epoch_acc = val_corrects.double() / val_total

print(f'Époque {epoch + 1}/{num_epochs}:')
print(f'Perte Entraînement: {epoch_loss:.4f} | Précision Entraînement: {epoch_acc:.4f}')
print(f'Perte Validation: {val_epoch_loss:.4f} | Précision Validation: {val_epoch_acc:.4f}')
print('-' * 30)

# Évaluer le modèle
model.eval()



all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Définir les fonctions de métriques manuelles
def confusion_matrix_manual(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm


def precision_recall_f1(cm):
    precision = {}
    recall = {}
    f1_score = {}
    num_classes = cm.shape[0]

    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp

        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls) if (
                                                                                                precision_cls + recall_cls) > 0 else 0

        precision[cls] = precision_cls
        recall[cls] = recall_cls
        f1_score[cls] = f1_cls

    return precision, recall, f1_score


def accuracy_manual(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

# Calcul des métriques manuelles
num_classes = len(np.unique(y))
cm = confusion_matrix_manual(all_labels, all_preds, num_classes)
precision, recall, f1 = precision_recall_f1(cm)
accuracy = accuracy_manual(all_labels, all_preds)

categories = df['IQA'].astype('category').cat.categories

# Afficher les catégories avec leurs codes
for code, category in enumerate(categories):
    print(f"Code {code}: {category}")

# Afficher les résultats
print("Matrice de Confusion:")
print(cm)
print("\nPrécision par classe:")
for cls in range(num_classes):
    print(f"Classe {cls}: {precision[cls]:.4f}")
print("\nRappel par classe:")
for cls in range(num_classes):
    print(f"Classe {cls}: {recall[cls]:.4f}")
print("\nF1-Score par classe:")
for cls in range(num_classes):
    print(f"Classe {cls}: {f1[cls]:.4f}")
print(f"\nExactitude Globale: {accuracy:.4f}")



def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Vérité')
    plt.xlabel('Prédiction')
    plt.title('Matrice de Confusion')
    plt.show()

# Obtenir les noms des classes
class_names = df['IQA'].astype('category').cat.categories.tolist()

# Afficher la matrice de confusion
plot_confusion_matrix(cm, class_names)

plt.plot(a,b)

plt.show()

