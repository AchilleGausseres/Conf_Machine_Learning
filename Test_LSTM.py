#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:39:47 2024

@author: achillegausseres
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Chargement des données
data = pd.read_csv('dfplusIQA.csv')
data = data.sort_values(by=['year', 'month', 'day', 'hour']).reset_index(drop=True)

# 2. Gestion des Colonnes de Temps
data['Datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
data['Sin_Heure'] = np.sin(2 * np.pi * data['hour'] / 24)
data['Cos_Heure'] = np.cos(2 * np.pi * data['hour'] / 24)
data['Sin_Mois'] = np.sin(2 * np.pi * data['month'] / 12)
data['Cos_Mois'] = np.cos(2 * np.pi * data['month'] / 12)
data['JourSemaine'] = data['Datetime'].dt.dayofweek
data['Sin_JourSemaine'] = np.sin(2 * np.pi * data['JourSemaine'] / 7)
data['Cos_JourSemaine'] = np.cos(2 * np.pi * data['JourSemaine'] / 7)

# 3. Gestion des Données de 12 Stations
data = pd.get_dummies(data, columns=['Station'], prefix='Station')

# 4. Encodage de la Variable Cible
label_encoder = LabelEncoder()
data['Target_encoded'] = label_encoder.fit_transform(data['Target'])
y = tf.keras.utils.to_categorical(data['Target_encoded'], num_classes=6)

# 5. Normalisation des Variables Prédictives
feature_columns = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM',
                  'Sin_Heure', 'Cos_Heure', 'Sin_Mois', 'Cos_Mois',
                  'Sin_JourSemaine', 'Cos_JourSemaine'] + \
                 [col for col in data.columns if col.startswith('Station_')]

features = data[feature_columns]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 6. Création des Séquences Temporelles
sequence_length = 24

def create_sequences(features, targets, seq_length):
    X = []
    y_seq = []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y_seq.append(targets[i + seq_length])
    return np.array(X), np.array(y_seq)

X, y_seq = create_sequences(features_scaled, y, sequence_length)

# 7. Division des Données
X_train, X_temp, y_train, y_temp = train_test_split(X, y_seq, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

print(f'Entraînement: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}')

# 8. Construction du Modèle
model = Sequential()
model.add(LSTM(units=100, input_shape=(sequence_length, X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=6, activation='softmax'))
model.summary()

# 9. Compilation du Modèle
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 10. Définition des Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# 11. Gestion des Classes Déséquilibrées
y_true_train = np.argmax(y_train, axis=1)
class_weights_values = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_true_train),
    y=y_true_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_values)}
print("Poids des classes:", class_weights_dict)

# 12. Entraînement du Modèle
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)

# 13. Évaluation du Modèle
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')

# 14. Prédictions et Analyse
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Prédictions')
plt.ylabel('Vérités Terrain')
plt.title('Matrice de Confusion')
plt.show()

# Rapport de classification
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# 15. Visualisation des Performances de l'Entraînement
# Courbe de perte
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Perte Entraînement')
plt.plot(history.history['val_loss'], label='Perte Validation')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.title('Courbe de Perte')
plt.legend()
plt.show()

# Courbe de précision
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Précision Entraînement')
plt.plot(history.history['val_accuracy'], label='Précision Validation')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.title('Courbe de Précision')
plt.legend()
plt.show()


#Pas obligatoire
# 16. Interprétabilité avec SHAP
# Note: SHAP pour les modèles LSTM est encore un domaine de recherche. DeepExplainer peut ne pas fonctionner parfaitement, mais voici un exemple de base.

import shap
# Initialiser SHAP
explainer = shap.DeepExplainer(model, X_train[:100])

# Calculer les valeurs SHAP
shap_values = explainer.shap_values(X_test[:100])

# Visualiser les valeurs SHAP pour la première classe
shap.summary_plot(shap_values[0], X_test[:100])




