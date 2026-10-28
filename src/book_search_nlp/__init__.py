"""
book_search_nlp — Moteur de Recherche de Livres Basé sur le Contenu.

Projet 7 (Domaine 2 : Recherche d'Information et Extraction)
Auteur : BOUDERNA EL MAHDI

Modules :
    - prepare_dataset  : Acquisition et nettoyage du corpus Goodreads.
    - preprocessing     : Tokenisation, stopwords, stemming (Porter).
    - tfidf_search      : Vectorisation TF-IDF, index inversé, recherche cosinus.
    - semantic_search   : Embeddings Sentence-BERT + voisins proches.
    - evaluation        : Precision@K, Recall@K, MAP@K.
    - app               : Interface web Streamlit.
"""

from .preprocessing import clean_text, preprocess, get_stats
from .tfidf_search import build_index, load_index
from .tfidf_search import search as search_tfidf
from .semantic_search import build_embeddings, load_embeddings, build_nn_index
from .semantic_search import search as search_semantic
from .evaluation import precision_at_k, recall_at_k, average_precision_at_k, compare_methods

__all__ = [
    "clean_text",
    "preprocess",
    "get_stats",
    "build_index",
    "load_index",
    "search_tfidf",
    "build_embeddings",
    "load_embeddings",
    "build_nn_index",
    "search_semantic",
    "precision_at_k",
    "recall_at_k",
    "average_precision_at_k",
    "compare_methods",
]
