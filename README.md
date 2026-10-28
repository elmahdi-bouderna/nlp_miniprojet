# 📚 BookFinder — Moteur de Recherche de Livres Basé sur le Contenu

**Projet 7 — Domaine 2 : Recherche d'Information et Extraction**  
**Auteur :** BOUDERNA EL MAHDI

## Description

BookFinder est un moteur de recherche qui retrouve des livres à partir d'une description en langage naturel. Le système compare deux approches de représentation du texte :

- **TF-IDF** (Term Frequency–Inverse Document Frequency) avec index inversé et similarité cosinus
- **Sentence Embeddings** (Sentence-BERT : `all-MiniLM-L6-v2`) avec recherche de voisins proches

Le corpus provient du dataset public **Goodreads Books Metadata** (sous-ensemble Young Adult, Wan & McAuley), filtré à **10 000 livres** en anglais.

## Structure du Projet

```
book-search-nlp/
├── src/book_search_nlp/          # Package Python principal
│   ├── __init__.py               # Exports du package
│   ├── prepare_dataset.py        # Étape 1 : Acquisition et nettoyage du corpus
│   ├── preprocessing.py          # Étape 2 : Tokenisation, stopwords, stemming
│   ├── tfidf_search.py           # Étape 3-4 : TF-IDF, index inversé, recherche
│   ├── semantic_search.py        # Étape 3-4 : Embeddings, NearestNeighbors
│   ├── evaluation.py             # Étape 5 : Precision@K, Recall@K, MAP@K
│   ├── app.py                    # Étape 6 : Application Streamlit
│   ├── data/                     # Données
│   │   ├── raw/                  # Fichier source brut (.json.gz)
│   │   └── books.csv             # Corpus nettoyé (10 000 livres)
│   └── notebooks/
│       └── exploration.ipynb     # Exploration des données
├── nlp_outputs/                  # Sorties générées par le notebook
│   ├── tfidf_vectorizer.joblib   # Vectoriseur TF-IDF sauvegardé
│   ├── tfidf_matrix.npz          # Matrice TF-IDF creuse
│   ├── semantic_embeddings.npy   # Embeddings des 10 000 livres
│   ├── annotation_pool.csv       # Ground truth pour l'évaluation
│   └── similarity_*.csv/png      # Matrices de similarité (échantillon)
├── BookFinder_NLP_Projet_Complet_CPU.ipynb  # Notebook complet (pipeline entier)
├── pyproject.toml                # Configuration du projet et dépendances
└── README.md                     # Ce fichier
```

## Pipeline NLP (6 étapes)

| Étape | Module | Description |
|-------|--------|-------------|
| 1. Dataset | `prepare_dataset.py` | Lecture du JSON Goodreads, filtrage (≥30 mots, anglais), export CSV |
| 2. Prétraitement | `preprocessing.py` | Nettoyage HTML, tokenisation, suppression stopwords, stemming Porter |
| 3. Représentation | `tfidf_search.py` + `semantic_search.py` | TF-IDF (uni+bigrammes) et Sentence Embeddings (384 dim) |
| 4. Modélisation | `tfidf_search.py` + `semantic_search.py` | Index inversé (CSC) + NearestNeighbors (cosinus) |
| 5. Évaluation | `evaluation.py` | Precision@10, Recall@10, MAP@10 sur 12 requêtes annotées |
| 6. Application | `app.py` | Interface Streamlit (recherche + comparaison) |

## Installation

```bash
# Cloner le projet
git clone https://github.com/elmahdi-bouderna/nlp_miniprojet.git
cd nlp_miniprojet

# Créer l'environnement virtuel et installer les dépendances
pip install -e .

# Télécharger les données NLTK (si nécessaire)
python -c "import nltk; nltk.download('punkt')"
```

## Utilisation

### 1. Préparer le dataset

```bash
python -m book_search_nlp.prepare_dataset
```

### 2. Exécuter le notebook complet

Ouvrir `BookFinder_NLP_Projet_Complet_CPU.ipynb` et exécuter toutes les cellules.
Le notebook génère les index (TF-IDF + embeddings) dans `nlp_outputs/`.

### 3. Lancer l'application Streamlit

```bash
streamlit run src/book_search_nlp/app.py
```

## Dépendances Principales

- **pandas** — Manipulation du corpus
- **scikit-learn** — TF-IDF, NearestNeighbors, métriques
- **nltk** — Stemming de Porter
- **sentence-transformers** — Encodage sémantique (all-MiniLM-L6-v2)
- **streamlit** — Interface web
- **matplotlib** — Visualisations

## Évaluation

Le système est évalué sur **12 requêtes de test** (`Q01`–`Q12`) avec un pool d'annotations
(`nlp_outputs/annotation_pool.csv`). Les métriques utilisées sont :

- **Precision@10** : proportion de résultats pertinents dans le top 10
- **Recall@10** : proportion de documents pertinents retrouvés
- **MAP@10** : Mean Average Precision (qualité du classement)

## Licence

Projet académique — Usage éducatif uniquement.
