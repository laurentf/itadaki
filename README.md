# 🍜 Itadaki - Recipe Image Retrieval System

_"Itadaki" signifie "bon appétit" en japonais_

Un système d'intelligence artificielle qui **trouve des recettes similaires** à partir d'images de nourriture. Il utilise la vision par ordinateur et l'apprentissage profond pour analyser des photos de plats et proposer des recettes correspondantes depuis une base de données de 13,000+ recettes.

## 🎯 Vue d'ensemble

Ce projet implémente **3 approches différentes** pour la recherche d'images de recettes :

1. **🔸 Raw Model** - EfficientNetB0 pré-entraîné (sans fine-tuning)
2. **🔥 Transfer Learning** - Entraînement avec Triplet Loss personnalisé
3. **⚡ Fine-tuning** - Dégelage intelligent de couches EfficientNet

Chaque approche a ses propres notebooks, modèles et fichiers de données pour des comparaisons de performance détaillées.

## ✨ Fonctionnalités

- **📸 Analyse d'images** : Extraction de features visuelles avec EfficientNetB0
- **🔍 Recherche par similarité** : Top-k recettes les plus similaires via similarité cosinus
- **📊 Exploration de données** : Analyse complète de 13,463 recettes avec visualisations
- **🧠 Apprentissage métrique** : Triplet Loss pour optimiser la similarité
- **🎨 Visualisations** : Interface graphique avec scores de similarité
- **💾 Base de données optimisée** : Embeddings précalculés pour recherche rapide

## 🛠️ Installation

### Prérequis

- **Python 3.12+** installé sur votre système
- **Au moins 8GB de RAM** (recommandé: 16GB+)
- **Espace disque** : ~5GB pour les données et modèles
- **Connexion internet** pour télécharger le dataset Kaggle

### Installation rapide

```bash
# 1. Cloner/naviguer vers le projet
cd itadaki

# 2. Créer l'environnement virtuel
python -m venv itadaki_env

# 3. Activer l'environnement
# Windows:
itadaki_env\Scripts\activate
# Linux/Mac:
source itadaki_env/bin/activate

# 4. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 5. Lancer Jupyter
jupyter notebook
```

## 📊 Structure du Projet

```
itadaki/
├── 📓 NOTEBOOKS
│   ├── recipe_image_retrieval_raw.ipynb          # ✅ Raw EfficientNet (prêt)
│   ├── recipe_image_retrieval_tl.ipynb           # ✅ Transfer Learning (prêt)
│   └── recipe_image_retrieval_ft.ipynb           # 🚧 Fine-tuning (en développement)
├── 📁 MODÈLES & DONNÉES
│   ├── raw/                                      # Raw model files
│   │   ├── recipe_image_retrieval_model_raw.keras
│   │   ├── recipe_embeddings_database_raw.npy
│   │   └── recipe_embeddings_database_raw_metadata_raw.pkl
│   ├── tl/                                       # Transfer learning files
│   │   ├── best_embedding_recipe_image_retrieval_model_tl.keras
│   │   ├── best_triplet_recipe_image_retrieval_model_tl.keras
│   │   ├── recipe_embeddings_database_tl.npy
│   │   └── recipe_embeddings_database_metadata_tl.pkl
│   ├── ft/                                       # Fine-tuning files (à venir)
│   └── data/                                     # Dataset principal
│       ├── recipes_with_images_dataframe.pkl     # DataFrame principal
│       └── data.csv                              # Données CSV
├── 🖼️ TESTS
│   └── test_recipes/                             # Images de test variées
├── 📋 CONFIGURATION
│   ├── requirements.txt                          # Dépendances Python
│   ├── README.md                                 # Ce fichier
│   └── ERREURS_TRANSFER_LEARNING.txt             # Notes de débogage
└── 🔧 ENV
    └── itadaki_env/                              # Environnement virtuel
```

## 🚀 Guide d'utilisation

### 1. Modèle Raw (Débutant) ✅

**Le plus simple - Prêt à utiliser immédiatement**

```bash
# Ouvrir le notebook
jupyter notebook recipe_image_retrieval_raw.ipynb
```

**Caractéristiques :**

- EfficientNetB0 pré-entraîné (sans modification)
- Features natives 1280 dimensions
- Temps de setup : ~5 minutes
- Précision : Bonne pour tests rapides

### 2. Transfer Learning (Recommandé) ✅

**Le plus performant - Production ready**

```bash
# Ouvrir le notebook
jupyter notebook recipe_image_retrieval_tl.ipynb
```

**Caractéristiques :**

- Triplet Loss personnalisé avec EfficientNet
- Embeddings optimisés 512 dimensions
- Couches L2 normalisées pour similarité cosinus
- Temps d'entraînement : ~30-60 minutes
- Précision : Excellente pour similarité visuelle

### 3. Fine-tuning (Avancé) 🚧

**Le plus sophistiqué - En développement**

```bash
# Ouvrir le notebook
jupyter notebook recipe_image_retrieval_ft.ipynb
```

**Caractéristiques :**

- 20 couches EfficientNet dégelées
- Learning rates différentiés
- Fine-tuning intelligent
- Temps d'entraînement : ~1-2 heures
- Précision : Maximale (en cours d'optimisation)

## 🎯 Utilisation du Système

### Interface simple

```python
# Dans n'importe quel notebook
# 1. Charger votre image
query_image = "path/to/your/food_image.jpg"

# 2. Rechercher les recettes similaires
results = retrieval_system.search_similar_recipes(query_image, top_k=3)

# 3. Afficher les résultats avec visualisations
retrieval_system.display_results(query_image, results)
```

### Résultats obtenus

Le système retourne pour chaque image :

- **📸 Images similaires** avec scores de similarité
- **📖 Recettes complètes** (titre, ingrédients, instructions)
- **📊 Métriques de confiance** (similarité cosinus 0-1)
- **🎨 Visualisations graphiques** côte à côte

## 📥 Dataset

**Food Ingredients and Recipe Dataset with Images** (Kaggle)

- **13,463 recettes uniques** avec images HD
- **Ingrédients détaillés** et instructions complètes
- **Images haute qualité** (224x224 minimum)
- **Téléchargement automatique** via `kagglehub`
- **Taille totale** : ~2GB

## 🏗️ Architecture Technique

### Raw Model

```
EfficientNetB0 (ImageNet) → GlobalAveragePooling → Features (1280D)
```

### Transfer Learning

```
EfficientNetB0 (frozen) → Custom Head (1024→512) → L2 Norm → Triplet Loss
```

### Fine-tuning

```
EfficientNetB0 (20 layers unfrozen) → Custom Head → Differential Learning Rates
```

## 📈 Comparaison des Performances

| Modèle                | Temps Setup | Précision     | Dimensions | Cas d'usage   |
| --------------------- | ----------- | ------------- | ---------- | ------------- |
| **Raw**               | ~5 min      | 🟡 Bonne      | 1280D      | Tests rapides |
| **Transfer Learning** | ~45 min     | 🟢 Excellente | 512D       | Production    |
| **Fine-tuning**       | ~90 min     | 🟢 Maximale   | 512D       | Recherche     |

## 🖼️ Images de Test

Le dossier `test_recipes/` contient 13 images variées :

### Desserts 🍰

- `fraisier-matcha.jpg` - Pâtisserie française
- `Chichi-recette-végane.jpg` - Dessert végan
- `fedessert.jpg`, `fedessert2.jpg` - Variations

### Plats principaux 🍽️

- `steak-entrecôte-poivre-frites.jpg` - Viande et frites
- `pates piment persil WEB.jpg` - Pâtes aux herbes

### Images d'entraînement 📸

- `training1.jpg`, `training2.jpg`, `training3.jpg`

**Utilisation :**

```python
# Tester avec une image du dossier
test_image = "./test_recipes/fraisier-matcha.jpg"
results = retrieval_system.search_similar_recipes(test_image, top_k=3)
```

## 🔧 Configuration Avancée

### Optimisations Transfer Learning

```python
CONFIG_TL = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'EMBEDDING_DIM': 512,
    'TRIPLET_MARGIN': 0.3,
    'TRANSFER_EPOCHS': 10,
    'VALIDATION_SPLIT': 0.2
}
```

### Custom Layers

Le projet utilise des couches personnalisées sérialisables :

- `L2NormalizationLayer` - Normalisation L2 intégrée
- `ExtractTripletComponent` - Extraction des composantes triplet
- `TripletStackLayer` - Assemblage des embeddings

## 🔍 Résolution des Problèmes

### ❌ Erreur de mémoire

```python
# Réduire le batch size
CONFIG_TL['BATCH_SIZE'] = 8

# Ou utiliser un échantillon plus petit
sample_recipes = recipes_df.sample(n=1000)
```

### ❌ Modèle non trouvé

```bash
# Vérifier les fichiers
ls -la raw/     # Pour raw model
ls -la tl/      # Pour transfer learning
ls -la data/    # Pour le dataset
```

### ❌ Custom objects error

```python
# Lors du chargement, utiliser les custom_objects
custom_objects = {
    'L2NormalizationLayer': L2NormalizationLayer,
    'triplet_loss_fn': lambda y_true, y_pred: ...,
    # ... autres objets personnalisés
}
model = tf.keras.models.load_model(path, custom_objects=custom_objects)
```

### ❌ Kaggle dataset

```bash
# Si téléchargement échoue
pip install kagglehub --upgrade
# Le cache Kaggle est automatiquement géré
```

## 📊 Métriques et Évaluation

### Transfer Learning

- **Triplet Accuracy** : >85% (% triplets correctement classés)
- **Positive Similarity** : 0.75-0.90 (similarité même recette)
- **Negative Similarity** : 0.10-0.30 (différentes recettes)
- **Temps recherche** : <2 secondes par image

### Similarité Cosinus

```python
# Score 1.0 = Images identiques
# Score 0.8+ = Très similaires
# Score 0.6+ = Similaires
# Score <0.5 = Différentes
```

## 🚧 Développements Futurs

### Court terme

- [ ] Finaliser le fine-tuning avec 20 couches
- [ ] Optimiser les learning rates différentiés
- [ ] Interface web avec Flask/Streamlit

### Moyen terme

- [ ] Support GPU avec CUDA optimisations
- [ ] API REST pour intégration externe
- [ ] Déploiement cloud (Azure/AWS)

### Long terme

- [ ] Modèles Vision Transformers (ViT)
- [ ] Search multi-modale (texte + image)
- [ ] Recommandations personnalisées

## 🤝 Contribution

Contributions bienvenues ! Domaines prioritaires :

1. **Performance** : Optimisations CUDA, quantification modèles
2. **Interface** : UI/UX améliorée, visualisations interactives
3. **Données** : Augmentation dataset, nettoyage annotations
4. **Documentation** : Tutorials, exemples d'usage

```bash
# Fork → Clone → Branch → Commit → Pull Request
git checkout -b feature/awesome-improvement
git commit -m "Add awesome feature"
git push origin feature/awesome-improvement
```

## 📄 Licence & Crédits

- **Code** : Open source (usage libre)
- **Dataset** : Kaggle - "Food Ingredients and Recipe Dataset with Images"
- **Modèle** : EfficientNet (Google Research)
- **Inspiration** : Recherche académique en metric learning

## 🍽️ Bon Appétit !

_Transformez vos photos de nourriture en découvertes culinaires avec l'IA !_

---

**🔥 Créé avec passion, TensorFlow et beaucoup de recettes testées ☕**

### 📞 Support

- **Issues** : Utilisez GitHub Issues pour bugs/questions
- **Discussions** : GitHub Discussions pour idées/améliorations
- **Email** : Pour collaborations professionnelles

---

_Dernière mise à jour : Décembre 2024 | Version 2.0_
