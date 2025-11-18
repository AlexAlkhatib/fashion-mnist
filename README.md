# 👗 Fashion MNIST Classification

Réseaux de neurones denses, CNN, Data Augmentation & Transfer Learning


## 📌 Description du Projet

Ce projet a pour objectif de **classifier les images du dataset Fashion-MNIST** en 10 catégories (T-shirt, Trouser, Pullover, etc.).
Plusieurs architectures de Deep Learning ont été implémentées pour comparer leurs performances :

* **Réseau Dense (Fully Connected Network)**
* **CNN (Convolutional Neural Network)**
* **CNN avec Data Augmentation**
* **Transfer Learning** via **VGG16** (ImageNet)

Le projet inclut :

✔️ Prétraitement complet des données
✔️ Visualisation des images
✔️ Construction et entraînement de différents modèles
✔️ Comparaison des performances
✔️ Visualisation des courbes d'apprentissage
✔️ Analyse des prédictions


## 📂 Dataset

Dataset : **Fashion MNIST**
Format : fichiers CSV (train + test)
Dimensions : 28 × 28 pixels, niveaux de gris

Catégories disponibles :

| Label | Classe      |
| ----- | ----------- |
| 0     | T-shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |


## 🧹 Prétraitement des Données

🔹 Chargement du train et test depuis des fichiers CSV
🔹 Séparation du train en :

* **50 000 images pour l'entraînement**
* **10 000 images pour la validation**

🔹 Normalisation :

```python
X_train_norm = X_train / 255
```

🔹 Encodage One-Hot des labels :

```python
y_train_cat = to_categorical(y_train, num_classes=10)
```

🔹 Reshape des images pour CNN :

* Dense : `(N, 784)`
* CNN : `(N, 28, 28, 1)`
* VGG16 : `(N, 28, 28, 3)` + resizing interne en `32×32`


## 🧠 Modèles Implémentés

### 🔹 1. Réseau Dense (Fully Connected Network)

Architecture :

* Input 784
* Dense(100, relu)
* Plusieurs couches Dense(20, relu)
* Output Softmax(10)

Optimiseur : **Adam (lr=0.0001)**
Perte : categorical_crossentropy

Résultat :
Accuracy ~ 87–88% sur validation


### 🔹 2. CNN Convolutionnel

Architecture :

* Conv2D(8) → MaxPooling
* Conv2D(16) → MaxPooling
* Conv2D(16)
* Flatten
* Dense(16)
* Softmax(10)

Optimiseur : **Adam**

Résultat :
Accuracy ~ 88–89% sur validation


### 🔹 3. CNN avec Data Augmentation

Transformations appliquées :

* RandomFlip
* RandomRotation
* RandomZoom
* RandomTranslation

Résultat :
Accuracy ~ 76–78% (modèle simple mais robuste, limité par architecture)


### 🔹 4. Transfer Learning — VGG16 (ImageNet)

Adaptations :

* Duplication des canaux pour passer en RGB
* Redimensionnement automatique en 32×32
* Couches VGG16 gelées
* Dense(16) → Dense(10)

Paramètres trainables : seulement **8k paramètres**
Contrairement aux **14M** de VGG16.

Résultat :
Accuracy ~ 80% en seulement quelques epochs


## 📊 Évaluation des Modèles

Chaque modèle est évalué via :

* **Loss**
* **Categorical Accuracy**
* Courbes :

  * Loss vs Val Loss
  * Accuracy vs Val Accuracy

Exemple de dictionnaire de performances :

```python
performances = {
    "Réseau dense": {"Loss": ..., "Accuracy": ...},
    "CNN": {"Loss": ..., "Accuracy": ...},
    "CNN augmenté": {"Loss": ..., "Accuracy": ...},
    "VGG16": {"Loss": ..., "Accuracy": ...}
}
```


## 🔍 Observation des Prédictions

Visualisation de quelques prédictions :

```python
for i in range(10):
    print("Classe prédite:", labels[np.argmax(y_pred[i])])
    print("Classe vraie  :", labels[np.argmax(y_test_cat[i])])
```

Affichage de l’image correspondante avec Matplotlib.


## 🚀 Technologies Utilisées

* Python 3.x
* NumPy
* Pandas
* Matplotlib / Seaborn
* TensorFlow / Keras
* Scikit-learn


## ▶️ Exécuter le Projet

1. **Importer les datasets**
2. **Exécuter le notebook**

ou lancer les scripts :

```bash
python train_dense.py
python train_cnn.py
python train_cnn_aug.py
python train_vgg16.py
```


## ✨ Améliorations Futures

* Ajout de **ResNet50**, **EfficientNet**, **MobileNetV2**
* Optimisation hyperparamètres (Optuna, Keras Tuner)
* Visualisation avancée (Grad-CAM)
* Déploiement Streamlit
* Comparaison avec MNIST classique
* Recherche d’architecture automatique (NAS)


## 👤 Auteur

Alex Alkhatib
Deep Learning — Classification d’images (Fashion MNIST)


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
Je peux te le générer directement.

