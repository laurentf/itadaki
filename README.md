# 🍜 Itadaki - Recipe Image Retrieval System

_"Itadaki" signifie "bon appétit" en japonais_

Ce projet utilise l'intelligence artificielle pour **retrouver des recettes similaires** à partir d'images de nourriture. Il combine la vision par ordinateur avec le traitement de langage naturel pour créer un système capable de reconnaître des plats et de proposer des recettes correspondantes depuis une base de données.

## 🎯 Fonctionnalités

- **📸 Analyse d'images** : Extraction de caractéristiques visuelles des photos de nourriture avec EfficientNet
- **🔍 Recherche de recettes** : Récupération automatique de recettes similaires depuis une base de données d'embeddings
- **📊 Exploration de données** : Analyse approfondie de 13,000+ recettes avec visualisations
- **🧠 Correspondance intelligente** : Mapping entre images et recettes par similarité cosinus
- **📈 Visualisations** : Graphiques, nuages de mots, statistiques détaillées
- **💾 Base de données d'embeddings** : Stockage optimisé des représentations vectorielles des recettes

## 🛠️ Installation

### Prérequis

- **Python 3.12+** installé sur votre système
- **Au moins 4GB de RAM** (recommandé: 8GB+)
- **Connexion internet** pour télécharger les datasets

### Installation

1. **Cloner le projet**

```bash
cd C:\DEV\itadaki  # ou votre dossier de choix
```

2. **Créer l'environnement virtuel**

```bash
# Windows
python -m venv itadaki_env
itadaki_env\Scripts\activate

# Linux/Mac
python3 -m venv itadaki_env
source itadaki_env/bin/activate
```

3. **Installer les dépendances**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Utilisation

### 1. Activer l'environnement virtuel

```bash
# Windows
itadaki_env\Scripts\activate

# Linux/Mac
source itadaki_env/bin/activate
```

### 2. Lancer Jupyter Notebook

```bash
jupyter notebook
```

### 3. Ouvrir le notebook

Ouvrez `recipe_image_retrieval.ipynb` dans votre navigateur.

### 4. Exécuter les cellules

Exécutez les cellules dans l'ordre pour:

1. **Explorer les données** - Analyses détaillées et visualisations
2. **Construire la base d'embeddings** - Création des représentations vectorielles
3. **Entraîner le modèle de retrieval** - Système de correspondance image-recette
4. **Tester le système** - Recherche de recettes depuis des images

## 📊 Structure du Projet

```
itadaki/
├── recipe_image_retrieval.ipynb          # Notebook principal
├── recipe_image_retrieval_model_raw.keras # Modèle entraîné
├── recipe_embeddings_database.npy        # Base de données d'embeddings
├── recipe_embeddings_database_metadata.pkl # Métadonnées des embeddings
├── requirements.txt                      # Dépendances Python
├── README.md                            # Ce fichier
├── test_recipes/                        # Images de test
└── itadaki_env/                         # Environnement virtuel (créé automatiquement)
```

## 📥 Données Requises

Le projet utilise le dataset **"Food Ingredients and Recipe Dataset with Images"** de Kaggle:

- **13,000+ recettes** avec noms, ingrédients et instructions
- **13,000+ images** de plats correspondants
- Téléchargement automatique via `kagglehub`

## 🔧 Fonctionnalités du Notebook

### Cellules d'Exploration

1. **📊 Exploration détaillée** - Statistiques générales, valeurs manquantes
2. **🖼️ Analyse des images** - Correspondances recette-image, propriétés des images
3. **🍽️ Exemples visuels** - Affichage des recettes avec leurs images
4. **🥕 Analyse des ingrédients** - Top ingrédients, catégorisation, visualisations
5. **📝 Analyse textuelle** - Longueurs, corrélations, recettes remarquables
6. **🎨 Nuages de mots** - Visualisation des mots-clés et techniques de cuisson

### Système de Retrieval

- **Extracteur de caractéristiques** basé sur EfficientNet
- **Vectorisation TF-IDF** des textes de recettes
- **Base de données d'embeddings** pour recherche rapide
- **Modèle de correspondance** image → recette par similarité
- **Interface de recherche** avec exemples et images de test

## 🎯 Utilisation du Système de Retrieval

```python
# Rechercher des recettes similaires à partir d'une image
similar_recipes = search_similar_recipes('path/to/your/food_image.jpg')

# Le système retourne:
# - Recettes les plus similaires
# - Scores de similarité
# - Noms des recettes
# - Ingrédients et instructions
# - Images correspondantes
```

## 📁 Images de Test

Le dossier `test_recipes/` contient des images de test variées:

- Desserts français (fraisier-matcha, chichi)
- Plats principaux (steak-entrecôte, pâtes)
- Images d'entraînement diverses
- Utilisez ces images pour tester le système de retrieval

## 🔍 Résolution des Problèmes

### Erreur: Module non trouvé

```bash
# Vérifiez que l'environnement virtuel est activé
# Windows: itadaki_env\Scripts\activate
# Linux/Mac: source itadaki_env/bin/activate

# Réinstallez les dépendances
pip install -r requirements.txt
```

### Erreur: Mémoire insuffisante

- **Réduisez la taille du batch** dans les paramètres d'entraînement
- **Fermez les autres applications** pour libérer de la RAM
- **Utilisez un échantillon plus petit** pour les tests

### Erreur: Fichiers manquants

- **Vérifiez la présence des fichiers** :
  - `recipe_image_retrieval_model_raw.keras` (modèle entraîné)
  - `recipe_embeddings_database.npy` (base d'embeddings)
  - `recipe_embeddings_database_metadata.pkl` (métadonnées)
- **Ré-entraînez le modèle** si les fichiers sont manquants

### Erreur: Téléchargement du dataset

- **Vérifiez votre connexion internet**
- **Le téléchargement peut prendre plusieurs minutes** (dataset ~2GB)
- **Kagglehub télécharge automatiquement** et met en cache les données

## 🏗️ Architecture Technique

- **Vision**: EfficientNetB0 (pré-entraîné ImageNet) pour extraction de features
- **NLP**: TF-IDF avec scikit-learn pour traitement textuel
- **Correspondance**: Similarité cosinus entre embeddings
- **Stockage**: NumPy arrays pour les embeddings, pickle pour métadonnées
- **Interface**: Jupyter Notebook avec visualisations matplotlib/seaborn

### 🎯 Modèles Implémentés

**3 phases de développement :**

1. **Phase 1 - Raw Model** ✅ **FONCTIONNEL**

   - EfficientNetB0 pré-entraîné sans fine-tuning
   - Extraction de features natives (1280 dimensions)
   - Aucun entraînement supplémentaire

2. **Phase 2 - Transfer Learning** 🚧 **EN COURS**

   - Head personnalisé avec 1-2 couches entraînables
   - 10 epochs d'entraînement
   - Sortie 512 dimensions

3. **Phase 3 - Fine-tuning** 🚧 **EN COURS**
   - 20 couches dégelées d'EfficientNetB0
   - 15 epochs d'entraînement complet
   - Sortie 512 dimensions

## 📈 Performance

- **13,463 recettes** avec images correspondantes
- **Embeddings**: 1280 dimensions par recette
- **Temps d'entraînement**: ~15-30 minutes (CPU)
- **Temps de recherche**: ~1-2 secondes par image
- **Précision**: Dépend de la qualité/similarité des images et du contenu de la base

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:

- Signaler des bugs
- Proposer des améliorations
- Ajouter de nouvelles fonctionnalités
- Améliorer la documentation

## 📄 Licence

Ce projet est open source. Utilisez-le librement pour vos projets personnels ou éducatifs.

## 🍽️ Bon Appétit!

_Retrouvez vos recettes préférées grâce à vos photos de nourriture et l'IA!_

---

**Créé avec ❤️ et beaucoup de café ☕**
