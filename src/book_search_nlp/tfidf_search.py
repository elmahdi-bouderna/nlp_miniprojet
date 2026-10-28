"""
tfidf_search.py — Vectorisation TF-IDF, index inversé et recherche cosinus.

Ce module construit un index TF-IDF (uni + bigrammes) sur les descriptions
de livres et expose une fonction `search` qui renvoie les K meilleurs résultats
classés par similarité cosinus.

Auteur : BOUDERNA EL MAHDI
"""

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import preprocess


# ── Construction de l'index ──────────────────────────────────────────────────

def build_index(corpus: list[str], output_dir: Path) -> tuple:
    """Construit et sauvegarde le vectoriseur TF-IDF et la matrice creuse.

    Parameters
    ----------
    corpus : list[str]
        Textes déjà nettoyés par `preprocessing.preprocess`.
    output_dir : Path
        Dossier où sauvegarder les fichiers (vectorizer + matrix).

    Returns
    -------
    tuple : (vectorizer, tfidf_matrix, inverted_index)
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # unigrammes + bigrammes
        min_df=2,            # ignorer les termes trop rares
        max_df=0.95,         # ignorer les termes présents partout
        sublinear_tf=True,   # appliquer log(1 + tf)
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    save_npz(output_dir / "tfidf_matrix.npz", tfidf_matrix)

    # Index inversé = matrice CSC (colonnes = termes → lignes = documents)
    inverted_index = tfidf_matrix.tocsc()

    return vectorizer, tfidf_matrix, inverted_index


def load_index(output_dir: Path) -> tuple:
    """Charge un index TF-IDF depuis le disque.

    Returns
    -------
    tuple : (vectorizer, tfidf_matrix, inverted_index)
    """
    vectorizer = joblib.load(output_dir / "tfidf_vectorizer.joblib")
    tfidf_matrix = load_npz(output_dir / "tfidf_matrix.npz")
    inverted_index = tfidf_matrix.tocsc()
    return vectorizer, tfidf_matrix, inverted_index


# ── Recherche ────────────────────────────────────────────────────────────────

def search(query: str, vectorizer, tfidf_matrix, inverted_index, k: int = 10) -> list[tuple[int, float]]:
    """Recherche les K livres les plus similaires à la requête (TF-IDF + cosinus).

    Utilise l'index inversé pour ne scorer que les documents candidats
    (ceux qui partagent au moins un terme avec la requête).

    Parameters
    ----------
    query : str
        Requête en langage naturel.
    vectorizer : TfidfVectorizer
        Le vectoriseur ajusté sur le corpus.
    tfidf_matrix : sparse matrix
        Matrice TF-IDF du corpus (CSR).
    inverted_index : sparse matrix
        Matrice TF-IDF en format CSC (index inversé).
    k : int
        Nombre de résultats à retourner.

    Returns
    -------
    list[tuple[int, float]]
        Paires (index_du_livre, score_cosinus), triées par score décroissant.
    """
    q = vectorizer.transform([preprocess(query)])

    if q.nnz == 0:
        return []

    # Récupérer les documents candidats via l'index inversé
    posting_lists = [
        inverted_index.indices[inverted_index.indptr[t]:inverted_index.indptr[t + 1]]
        for t in q.indices
    ]
    candidates = np.unique(np.concatenate(posting_lists))

    # Similarité cosinus = produit scalaire (vecteurs déjà normalisés L2)
    scores = (tfidf_matrix[candidates] @ q.T).toarray().ravel()
    order = np.argsort(-scores)[:k]

    return [(int(candidates[i]), float(scores[i])) for i in order if scores[i] > 0]
